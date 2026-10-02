"""``watch_video`` — local-first frame + transcript retrieval, keyed off fake YouTube URLs.

Design
------
At SFT-collection time ``watch_video`` fetches frames and transcripts from
the real YouTube. For RL rollouts that same pipe would murder throughput
(yt-dlp latency + 403s + captcha). We therefore do a **bridge-mapped
local-first** lookup:

1. **URL resolution**

   - If the URL matches a ``fake_url`` built by the corpus pipeline, look it
     up in ``url_to_path.json`` to get an absolute local ``.mp4`` path.
   - Otherwise, fall back to the shared ``VideoDownloader`` (real
     yt-dlp against YouTube) — this keeps the tool usable for real URLs
     during mixed SFT/RL training, and can be disabled via
     ``enable_remote_fallback: false`` in the tool config.

2. **Frame extraction** (always local)

   - ``sparse`` mode: uniform ``n_frames`` samples across the whole video,
     via ``FrameExtractor.extract_sparse``.
   - ``dense`` mode: ``[start_time, end_time]`` window at the requested
     ``fps``, via ``FrameExtractor.extract_dense``.
   - Frames are JPEG-encoded at 720p cap by ``FrameExtractor``, base64'd
     for transport, then decoded back to ``PIL.Image.Image`` before being
     handed to ``ToolResponse.image`` (verl's ``tool_agent_loop`` expects
     PIL/numpy, not base64).

3. **Transcript**

   - For **local fake_url**: read from ``videos.parquet.subtitle`` loaded
     once at tool startup from the corpus subtitle field.
   - For **remote URL**: ``TranscriptFetcher.fetch(url)`` reads available
     subtitles through yt-dlp.
   - For both: format through ``TranscriptFetcher.format_for_prompt`` with
     a ``max_transcript_chars`` cap (default 4000) to keep the tool
     response under ~8 KB.

4. **verl return shape**

   - ``ToolResponse.image = [PIL.Image, ...]``: the verl tool_agent_loop
     (``_handle_tool_calling_state``) appends each frame into
     ``agent_data.image_data`` and emits one ``{"type": "image"}`` slot in
     the tool response message. Qwen3.5-4B then sees them via the image
     processor.
   - ``ToolResponse.text``: a JSON envelope with the duration, mode, the
     per-frame timestamps, the (possibly truncated) transcript, and a
     few debug flags. **Never** embed base64 into ``text`` — that would
     tokenise into a useless blob.

XML parser alignment (qwen3_coder)
----------------------------------
``Qwen3XMLToolParser`` converts ``<parameter>`` values by looking up each
name's ``type`` in the tool schema:

- ``mode`` → ``string``, one of ``sparse`` / ``dense``
- ``n_frames`` → ``integer``
- ``start_time`` / ``end_time`` → ``number``
- ``fps`` → ``number``
- ``url`` → ``string``

We **do not** use array parameters here: the parser would fall through to
``eval()`` for them (see ``Qwen3XMLToolParser._parse_xml_function_call``)
which is fragile under even minor model drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from ._common import (
    decode_base64_frames,
    dumps_tool_text,
    ensure_video_agent_on_path,
    init_execution_pool,
    resolve_youtube_cookies,
    truncate_text,
)
from ._common import get_corpus_bridge as _get_corpus_bridge

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIDEO_SEARCH_SIM_LOG_LEVEL", "WARNING"))


class _CorpusBridge:
    """In-memory side-channel that maps fake URLs back to local files + local subtitles.

    Loaded once at tool startup so every ``execute`` call is pure memory
    lookup. Only the subtitle column is kept (a few hundred MB for 45K
    videos worst case) — the full videos.parquet is not retained.

    Also reads ``manifest.json`` to learn how cached mp4s are stored:
    legacy ``loose_files`` layout (one mp4 per file) is consumed via the
    same path the bridge already returned, while ``sharded_tar`` layout
    surfaces ``shard://...`` URIs that callers must route through
    :func:`shard_store.materialise_to_tempfile` before handing off to
    decode-by-path consumers.
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self._url_to_path: dict[str, str] = {}
        self._url_to_subtitle: dict[str, str] = {}
        self._url_to_duration: dict[str, float] = {}
        self._cache_layout: str = "none"
        self._shard_root: Path | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        mapping_path = self.corpus_dir / "url_to_path.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(f"corpus bridge mapping not found at {mapping_path}; run vss-build-corpus first.")
        with open(mapping_path, encoding="utf-8") as f:
            self._url_to_path = json.load(f)

        # Pull cache metadata from manifest.json so we know whether
        # local_path strings are real paths, fineVideo:// virtuals, or
        # shard:// URIs that need byte-range materialisation.
        manifest_path = self.corpus_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                cache = manifest.get("cache") or {}
                self._cache_layout = str(cache.get("layout") or "none")
                shard_dir = cache.get("dir")
                if shard_dir:
                    self._shard_root = Path(shard_dir).expanduser()
            except Exception as e:  # noqa: BLE001
                logger.warning("manifest.json parse failed: %s", e)

        parquet_path = self.corpus_dir / "videos.parquet"
        if parquet_path.is_file():
            import pandas as pd  # noqa: PLC0415

            df = pd.read_parquet(parquet_path, columns=["fake_url", "subtitle", "duration"])
            for _, row in df.iterrows():
                fake_url = row["fake_url"]
                sub = row.get("subtitle", "")
                dur = row.get("duration", 0.0)
                if isinstance(sub, str) and sub:
                    self._url_to_subtitle[fake_url] = sub
                if dur is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        self._url_to_duration[fake_url] = float(dur)
        self._loaded = True
        logger.info(
            "_CorpusBridge loaded: %d urls, %d with subtitles, cache_layout=%s",
            len(self._url_to_path),
            len(self._url_to_subtitle),
            self._cache_layout,
        )

    def local_path(self, url: str) -> str | None:
        self.load()
        return self._url_to_path.get(url)

    def local_subtitle(self, url: str) -> str | None:
        self.load()
        return self._url_to_subtitle.get(url)

    def duration(self, url: str) -> float | None:
        self.load()
        return self._url_to_duration.get(url)

    @property
    def cache_layout(self) -> str:
        self.load()
        return self._cache_layout

    @property
    def shard_root(self) -> Path | None:
        self.load()
        return self._shard_root


class WatchVideoTool(BaseTool):
    """Local-first ``watch_video`` implementation.

    Expected ``config`` keys:

    ``corpus_dir`` (str, **required** unless ``enable_local_lookup=False``)
        Directory that contains ``url_to_path.json`` and ``videos.parquet``
        (the ``vss-build-corpus`` output).
    ``enable_local_lookup`` (bool, default True)
        When False, every URL goes through ``VideoDownloader`` (the full SFT
        behaviour). Useful for eval / debugging the fallback path.
    ``enable_remote_fallback`` (bool, default True)
        When False, any URL that misses the local bridge returns an error,
        instead of hitting yt-dlp. Recommended ``False`` for pure local
        simulation training.
    ``downloader_cache_dir`` (str, default ``data/cache/videos``)
    ``transcript_cache_dir`` (str, default ``data/cache/transcripts``)
    ``max_n_frames_sparse`` (int, default 16)
    ``max_n_frames_dense`` (int, default 32)
    ``max_transcript_chars`` (int, default 4000)
    ``jpeg_quality`` (int, default 85)
    ``num_workers`` (int, default 16)
    ``rate_limit`` (int, default 16)
    ``enable_global_rate_limit`` (bool, default True)
    ``execution_backend`` (str, default ``"local"``)
        ``"local"`` keeps frame extraction in this Ray tool worker;
        ``"remote_server"`` forwards work to a video_search_sim
        ``/watch_video`` endpoint.
    ``watch_service_url`` (str, optional)
        Required when ``execution_backend == "remote_server"``.
    ``watch_service_timeout`` (float, default ``exec_timeout_sec``)
    ``type`` (str, default ``"native"``)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        ensure_video_agent_on_path()
        try:
            from video_agent.tools.frame_extractor import FrameExtractor  # noqa: PLC0415
            from video_agent.tools.transcript import TranscriptFetcher  # noqa: PLC0415
            from video_agent.tools.video_download import VideoDownloader  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - environmental
            # Surface the *actual* missing module: this except also fires when
            # video_agent itself is importable but one of ITS deps (yt_dlp,
            # decord, …) is missing — in which case pointing at
            # VSS_VIDEO_AGENT_PATH is misleading.
            raise RuntimeError(
                f"WatchVideoTool failed to import video_agent tools: {e!r}. "
                "If the missing module is 'video_agent.*', set VSS_VIDEO_AGENT_PATH; "
                "otherwise pip install the missing dependency (e.g. yt-dlp, decord) "
                "into the training env."
            ) from e

        self.enable_local_lookup = bool(config.get("enable_local_lookup", True))
        self.enable_remote_fallback = bool(config.get("enable_remote_fallback", True))
        self.max_n_frames_sparse = int(config.get("max_n_frames_sparse", 16))
        self.max_n_frames_dense = int(config.get("max_n_frames_dense", 32))
        self.max_transcript_chars = int(config.get("max_transcript_chars", 4000))
        self.jpeg_quality = int(config.get("jpeg_quality", 85))
        self.execution_backend = str(config.get("execution_backend", "local")).lower().strip() or "local"
        if self.execution_backend not in {"local", "remote_server"}:
            raise ValueError(
                "WatchVideoTool: execution_backend must be 'local' or 'remote_server', "
                f"got {self.execution_backend!r}"
            )
        self.watch_service_url = str(config.get("watch_service_url") or "").strip()

        # Where to materialise mp4 bytes pulled out of tar shards. Default to
        # a per-tool subdirectory under /tmp so concurrent tools (e.g. mixed
        # eval + train) don't trip over each other. Override this if /tmp is
        # tight or shared across containers.
        self.shard_materialise_dir = Path(
            config.get("shard_materialise_dir", "/tmp/vss_shard_cache")
        ).expanduser()
        # Optional override: if the manifest's cache dir is not the path
        # actually mounted on this host, the user can pin it here.
        self.shard_root_override: Path | None = None
        shard_root_cfg = config.get("shard_root")
        if shard_root_cfg:
            self.shard_root_override = Path(shard_root_cfg).expanduser()

        if self.enable_local_lookup:
            corpus_dir = config.get("corpus_dir")
            if not corpus_dir:
                raise ValueError("WatchVideoTool: config.corpus_dir is required when enable_local_lookup=True")
            self._bridge = _get_corpus_bridge(corpus_dir)
        else:
            self._bridge = None

        self._frame_extractor = FrameExtractor(jpeg_quality=self.jpeg_quality)
        # YouTube cookies for the remote path (real yt-dlp video download +
        # subtitle fetch during eval). resolve_youtube_cookies validates the
        # file is Netscape format and degrades to anonymous (with a warning) on
        # a missing/malformed file, so a bad cookies.txt cannot break every
        # remote download/transcript fetch.
        cookies_file = resolve_youtube_cookies(config.get("youtube_cookies_file"))

        # --- wall-clock timeouts for the remote (yt-dlp / YouTube) path -------
        # YouTube throttles bot-flagged IPs to a slow trickle: the socket keeps
        # receiving a few KB/s, so yt-dlp's `socket_timeout` (which only fires
        # on a *fully stalled* socket) never trips and a single watch_video can
        # run for 5-36 min (observed) or hang indefinitely. Since validation is
        # a sync barrier (the step waits for ALL rollouts), one such hang idles
        # every GPU at 0%. We bound the remote work at two layers:
        #   * `remote_download_timeout_sec` -> inside VideoDownloader, aborted
        #     via a yt-dlp progress hook so the WORKER THREAD returns by itself
        #     and frees its rate-limit token (Ray threaded-actor tasks cannot be
        #     ray.cancel-ed once running, so the abort must be self-inflicted).
        #   * `exec_timeout_sec` -> an asyncio.wait_for backstop around the pool
        #     await, so even an extraction/JS-challenge hang (which fires no
        #     progress hook) can never block the rollout / val barrier forever.
        self.exec_timeout_sec = float(config.get("exec_timeout_sec", 240) or 0) or None
        self.remote_download_timeout_sec = int(config.get("remote_download_timeout_sec", 120) or 0) or None
        self.watch_service_timeout = float(
            config.get("watch_service_timeout") or self.exec_timeout_sec or 240
        )
        if self.execution_backend == "remote_server" and not self.watch_service_url:
            raise ValueError("WatchVideoTool: watch_service_url is required for execution_backend=remote_server")

        self._transcript_fetcher = TranscriptFetcher(
            cache_dir=config.get("transcript_cache_dir", "data/cache/transcripts"),
            cookies_file=cookies_file,
        )
        self._downloader = VideoDownloader(
            cache_dir=config.get("downloader_cache_dir", "data/cache/videos"),
            cookies_file=cookies_file,
            download_timeout_sec=self.remote_download_timeout_sec,
        )

        self.num_workers = int(config.get("num_workers", 16))
        self.rate_limit = int(config.get("rate_limit", 16))
        self.enable_global_rate_limit = bool(config.get("enable_global_rate_limit", True))
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            limiter_name="vss-watch-video-rate-limiter",
        )
        logger.info(
            "WatchVideoTool ready (local=%s remote_fallback=%s execution_backend=%s)",
            self.enable_local_lookup,
            self.enable_remote_fallback,
            self.execution_backend,
        )

    # ------------------------------------------------------------- verl API

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Select local-corpus training or live-web evaluation for this sample."""
        if instance_id is None:
            instance_id = str(uuid4())
        ck = (kwargs.get("create_kwargs") or {})
        backend = str(ck.get("backend", "local")).lower().strip() or "local"
        if backend not in ("local", "remote"):
            logger.warning("WatchVideoTool: unknown backend=%r, falling back to 'local'", backend)
            backend = "local"
        self._instance_dict[instance_id] = {
            "calls": 0,
            "urls": [],
            "backend": backend,
        }
        return instance_id, ToolResponse()

    # ----------------------------------------------------------- helpers

    def _resolve_local_path(
        self,
        url: str,
        backend: str = "local",
    ) -> tuple[str | None, bool]:
        """Resolve a simulator corpus URL or download a live-web URL."""
        if backend == "remote":
            candidate = self._maybe_local_file(url)
            if candidate is not None:
                return candidate, False
            # Treat as real YouTube URL (or any URL ytdlp can resolve).
            try:
                path = self._downloader.download(url)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "watch_video remote download failed (url=%r): %s",
                    url, e,
                )
                return None, False
            return path, False

        # --- backend=local: original corpus-bridge-first path --------------
        if self._bridge is not None:
            path = self._bridge.local_path(url)
            if path:
                # Case 2: shard URI from sharded_tar layout.
                if path.startswith("shard://"):
                    shard_root = self.shard_root_override or self._bridge.shard_root
                    if shard_root is None:
                        logger.warning(
                            "shard URI %s but no shard_root configured (manifest.cache.dir missing)",
                            path,
                        )
                    else:
                        try:
                            from ..video_corpus.shard_store import (  # noqa: PLC0415
                                materialise_to_tempfile,
                            )

                            tmp_path = materialise_to_tempfile(
                                path,
                                shard_root=shard_root,
                                tmp_dir=self.shard_materialise_dir,
                            )
                            return str(tmp_path), True
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "Failed to materialise shard URI %s under %s: %s",
                                path,
                                shard_root,
                                e,
                            )
                # Case 1: real file on disk (loose_files layout, or path
                # produced by the legacy non-sharded code path).
                elif Path(path).is_file():
                    return path, True
        if self.enable_remote_fallback:
            path = self._downloader.download(url)
            return path, False
        return None, False

    @staticmethod
    def _maybe_local_file(s: str) -> str | None:
        """Return a real on-disk path if ``s`` looks like one, else None.

        The agent often echoes the path straight out of the (markdown) prompt,
        so it can arrive wrapped in backticks / quotes or with a trailing
        sentence period, e.g. ``` `/abs/1.mp4`. ```. We strip those wrappers
        before the ``is_file`` check so such lightly-decorated paths still
        resolve as local files instead of being shipped off to yt-dlp.
        """
        if not s:
            return None
        s = s.strip()
        # Strip a trailing sentence period the model may have appended.
        s = s.rstrip(".").strip()
        # Strip surrounding markdown backticks / single / double quotes
        # (possibly several, e.g. ``"`path`"``).
        while len(s) >= 2 and s[0] == s[-1] and s[0] in "`'\"":
            s = s[1:-1].strip()
        if s.startswith("file://"):
            s = s[len("file://"):]
        # Absolute path? Or already a relative path that exists on disk?
        try:
            p = Path(s)
        except (TypeError, ValueError):
            return None
        if p.is_file():
            return str(p)
        return None

    def _resolve_transcript_text(self, url: str, local_path: str | None, is_local_hit: bool) -> str:
        """Fetch transcript — local subtitle table first, then yt-dlp path as fallback."""
        if is_local_hit and self._bridge is not None:
            local = self._bridge.local_subtitle(url)
            if local:
                return local
        try:
            segs = self._transcript_fetcher.fetch(url)
        except Exception as e:  # noqa: BLE001
            logger.warning("TranscriptFetcher error on %s: %s", url, e)
            return ""
        return self._transcript_fetcher.format_for_prompt(segs, max_chars=self.max_transcript_chars)

    def _do_watch(
        self,
        url: str,
        mode: str,
        start_time: float | None,
        end_time: float | None,
        n_frames: int,
        fps: float,
        backend: str = "local",
    ) -> dict[str, Any]:
        """Blocking work: resolve URL, extract frames, fetch transcript. Runs in Ray worker."""
        local_path, is_local_hit = self._resolve_local_path(
            url, backend=backend
        )
        if local_path is None:
            return {
                "error": (
                    f"video could not be resolved for url={url} (backend={backend}): "
                    "not in corpus bridge, not a local file, and remote download failed."
                ),
            }

        if mode == "dense":
            if start_time is None or end_time is None:
                return {"error": "dense mode requires `start_time` and `end_time`"}
            frames = self._frame_extractor.extract_dense(
                local_path,
                start=float(start_time),
                end=float(end_time),
                fps=float(fps),
                max_frames=self.max_n_frames_dense,
            )
        else:
            frames = self._frame_extractor.extract_sparse(
                local_path,
                n_frames=int(n_frames),
            )

        transcript_text = self._resolve_transcript_text(url, local_path, is_local_hit)
        transcript_text, transcript_truncated = truncate_text(transcript_text, self.max_transcript_chars)

        duration = None
        if self._bridge is not None and is_local_hit:
            duration = self._bridge.duration(url)
        if duration is None:
            try:
                duration = self._downloader.get_duration(local_path)
            except Exception:  # noqa: BLE001
                duration = None

        return {
            "url": url,
            "mode": mode,
            "duration_sec": float(duration or 0.0),
            "local_hit": is_local_hit,
            "timestamps": [f.timestamp for f in frames],
            "frames_b64": [f.image_b64 for f in frames],
            "transcript": transcript_text,
            "transcript_truncated": transcript_truncated,
        }

    def _do_remote_watch(
        self,
        url: str,
        mode: str,
        start_time: float | None,
        end_time: float | None,
        n_frames: int,
        fps: float,
        backend: str = "local",
    ) -> dict[str, Any]:
        """Blocking HTTP client path for offloading frame extraction to the service."""
        payload = {
            "url": url,
            "mode": mode,
            "n_frames": int(n_frames),
            "start_time": start_time,
            "end_time": end_time,
            "fps": float(fps),
            "backend": backend,
        }
        try:
            with httpx.Client(timeout=self.watch_service_timeout) as client:
                resp = client.post(self.watch_service_url, json=payload)
                resp.raise_for_status()
                body = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("WatchVideoTool remote service error: %s", e)
            return {"error": f"watch_video remote service error: {e!r}"}

        if body.get("error"):
            return {"error": body["error"], "url": url, "backend": backend}
        return {
            "url": body.get("url", url),
            "mode": body.get("mode", mode),
            "duration_sec": float(body.get("duration_sec", 0.0) or 0.0),
            "local_hit": bool(body.get("local_hit", False)),
            "timestamps": body.get("timestamps", []) or [],
            "frames_b64": body.get("frames_b64", []) or [],
            "transcript": body.get("transcript", "") or "",
            "transcript_truncated": bool(body.get("transcript_truncated", False)),
            "remote_latency_ms": float(body.get("latency_ms", 0.0) or 0.0),
        }

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        url = parameters.get("url", "")
        rec = self._instance_dict.get(instance_id) or {}
        backend = rec.get("backend", "local")
        if not isinstance(url, str) or not url.strip():
            msg = "Error: `url` must be a non-empty string."
            return ToolResponse(text=msg), 0.0, {"error": msg}

        mode = str(parameters.get("mode", "sparse")).lower()
        if mode not in {"sparse", "dense"}:
            return (
                ToolResponse(text=f"Error: `mode` must be 'sparse' or 'dense', got {mode!r}."),
                0.0,
                {"error": "bad_mode"},
            )

        n_frames = int(parameters.get("n_frames", self.max_n_frames_sparse))
        n_frames = max(1, min(n_frames, self.max_n_frames_sparse))
        fps = float(parameters.get("fps", 1.0) or 1.0)
        start_time = parameters.get("start_time")
        end_time = parameters.get("end_time")
        try:
            start_time = None if start_time in (None, "", "null") else float(start_time)
            end_time = None if end_time in (None, "", "null") else float(end_time)
        except (TypeError, ValueError):
            return (
                ToolResponse(text="Error: `start_time` / `end_time` must be numeric seconds."),
                0.0,
                {"error": "bad_time"},
            )

        use_remote_service = self.execution_backend == "remote_server" and backend == "local"
        effective_execution_backend = "remote_server" if use_remote_service else "local"

        try:
            # Live-web evaluation uses the rollout worker; corpus watches may use the service.
            worker_fn = self._do_remote_watch if use_remote_service else self._do_watch
            ref = self.execution_pool.execute.remote(
                worker_fn, url, mode, start_time, end_time, n_frames, fps,
                backend,
            )
            if self.exec_timeout_sec:
                body = await asyncio.wait_for(asyncio.shield(ref), timeout=self.exec_timeout_sec)
            else:
                body = await ref
        except asyncio.TimeoutError:
            # Wall-clock backstop hit (e.g. yt-dlp extraction / JS-challenge
            # hang that fires no progress hook). Best-effort cancel the Ray
            # task (may be a no-op for an already-running threaded-actor call)
            # and return an error so the agent moves on and the validation
            # barrier is never blocked indefinitely.
            with contextlib.suppress(Exception):
                import ray  # noqa: PLC0415
                ray.cancel(ref, force=True)
            logger.warning(
                "WatchVideoTool exec timeout after %.0fs (url=%r backend=%r execution_backend=%r)",
                self.exec_timeout_sec, url, backend, effective_execution_backend,
            )
            body = {"error": f"watch_video timed out after {self.exec_timeout_sec:.0f}s (url={url})"}
        except Exception as e:  # noqa: BLE001
            logger.warning("WatchVideoTool pool error: %s", e)
            body = {"error": repr(e)}

        if body.get("error"):
            return (
                ToolResponse(text=dumps_tool_text({"url": url, "error": body["error"]})),
                0.0,
                {"error": body["error"], "url": url, "backend": backend},
            )

        frames_b64 = body.pop("frames_b64", [])
        images = decode_base64_frames(frames_b64)

        if instance_id in self._instance_dict:
            rec = self._instance_dict[instance_id]
            rec["calls"] += 1
            rec["urls"].append(url)

        payload = {
            "url": body["url"],
            "mode": body["mode"],
            "duration_sec": body["duration_sec"],
            "local_hit": body["local_hit"],
            "n_frames": len(images),
            "timestamps": body["timestamps"],
            "transcript": body["transcript"],
            "transcript_truncated": body["transcript_truncated"],
        }
        tool_text = dumps_tool_text(payload)

        metrics = {
            "n_frames": len(images),
            "duration_sec": body["duration_sec"],
            "local_hit": body["local_hit"],
            "mode": mode,
            "backend": backend,
            "execution_backend": effective_execution_backend,
        }
        if "remote_latency_ms" in body:
            metrics["remote_latency_ms"] = body["remote_latency_ms"]
        return (
            ToolResponse(text=tool_text, image=images if images else None),
            0.0,
            metrics,
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
