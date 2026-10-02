"""Local-corpus client for offline video task generation.

This module provides the *local* counterpart to the online yt-dlp / Serper
helpers used by Stage 2.  When ``runtime.mode == 'local'``, every call that
the pipeline previously routed through YouTube goes through this layer
instead:

* :func:`search_local_corpus`  — HTTP POST to the video_search_sim
  ``/video_search`` endpoint (BM25 + CLIP fused retrieval), returning
  :class:`VideoSearchResult` objects shape-compatible with
  ``video_agent.tools.video_search.VideoSearchResult`` so that all the
  downstream graph-construction code stays unchanged.

* :class:`CorpusBridge`        — vendored, dependency-free port of
  ``video_search_sim.verl_tools.watch_video_tool._CorpusBridge`` (we cannot
  import it directly because that module hard-imports ``verl.tools``).
  Resolves a fake YouTube URL back to a local mp4 path, transparently
  materialising shard:// URIs through the corpus's shard_store when the
  cache layout is ``sharded_tar``.

Why HTTP instead of in-process retrieval
----------------------------------------
The user explicitly requested HTTP-only retrieval to support cross-machine
deployments (synthesis on box A, corpus on box B). It also keeps the
synthesis driver free of the heavy CLIP / FAISS dependencies — the
retrieval service is the only process that needs them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Shape-compatible search result
# ──────────────────────────────────────────────────────────────────────


@dataclass
class VideoSearchResult:
    """Structurally identical to ``video_agent.tools.video_search.VideoSearchResult``.

    The Stage 2 graph-construction code consumes this dataclass purely by
    attribute access (``r.url``, ``r.duration`` …), so as long as we keep
    the same field names and types every existing call site keeps working.
    """

    video_id: str
    url: str
    title: str
    description: str
    duration: Optional[float]
    view_count: Optional[int]
    upload_date: Optional[str]
    channel: Optional[str]
    thumbnail_url: Optional[str]


# ──────────────────────────────────────────────────────────────────────
# CorpusBridge — vendored from video_search_sim/verl_tools/watch_video_tool.py
# ──────────────────────────────────────────────────────────────────────


class CorpusBridge:
    """In-memory mapping of fake_url → local mp4 path / subtitle / duration.

    Loaded once per process; subsequent lookups are pure dict reads. The
    bridge also understands the ``sharded_tar`` cache layout: when a
    ``local_path`` starts with ``shard://``, callers should hand it to
    :meth:`materialise` to get a real on-disk mp4.

    This class purposely avoids importing anything from the
    ``video_search_sim`` package's verl-flavoured modules so that the
    task_generation pipeline does not pull in ``verl``/``ray``.
    """

    def __init__(self, corpus_dir: str, shard_root_override: Optional[str] = None,
                 shard_materialise_dir: str = "/tmp/vss_shard_cache"):
        self.corpus_dir = Path(corpus_dir).expanduser().resolve()
        self._url_to_path: Dict[str, str] = {}
        self._url_to_subtitle: Dict[str, str] = {}
        self._url_to_duration: Dict[str, float] = {}
        self._cache_layout: str = "none"
        self._manifest_shard_root: Optional[Path] = None
        self._shard_root_override: Optional[Path] = (
            Path(shard_root_override).expanduser() if shard_root_override else None
        )
        self.shard_materialise_dir = Path(shard_materialise_dir).expanduser()
        self._loaded = False
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            mapping_path = self.corpus_dir / "url_to_path.json"
            if not mapping_path.is_file():
                raise FileNotFoundError(
                    f"corpus bridge mapping not found at {mapping_path}; "
                    "run vss-build-corpus first."
                )
            with open(mapping_path, encoding="utf-8") as f:
                self._url_to_path = json.load(f)

            manifest_path = self.corpus_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        manifest = json.load(f)
                    cache = manifest.get("cache") or {}
                    self._cache_layout = str(cache.get("layout") or "none")
                    shard_dir = cache.get("dir")
                    if shard_dir:
                        self._manifest_shard_root = Path(shard_dir).expanduser()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[corpus_bridge] manifest.json parse failed: %s", exc)

            parquet_path = self.corpus_dir / "videos.parquet"
            if parquet_path.is_file():
                try:
                    import pandas as pd  # local import — pandas is already a project dep
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError(
                        "pandas is required to read videos.parquet; pip install pandas"
                    ) from e
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
                "[corpus_bridge] loaded: %d urls, %d with subtitles, cache_layout=%s",
                len(self._url_to_path),
                len(self._url_to_subtitle),
                self._cache_layout,
            )

    # ── Lookups ───────────────────────────────────────────────────────

    def local_path(self, url: str) -> Optional[str]:
        self.load()
        return self._url_to_path.get(url)

    def subtitle(self, url: str) -> Optional[str]:
        self.load()
        return self._url_to_subtitle.get(url)

    def duration(self, url: str) -> Optional[float]:
        self.load()
        return self._url_to_duration.get(url)

    @property
    def shard_root(self) -> Optional[Path]:
        self.load()
        return self._shard_root_override or self._manifest_shard_root

    @property
    def cache_layout(self) -> str:
        self.load()
        return self._cache_layout

    # ── shard:// materialisation ──────────────────────────────────────

    def materialise(self, path_or_uri: str) -> Optional[str]:
        """Resolve a bridge ``local_path`` value to a real on-disk mp4 path.

        - Real on-disk path → returned as-is (after ``Path.is_file()``).
        - ``shard://...`` URI → byte-range materialised via
          ``video_search_sim.video_corpus.shard_store.materialise_to_tempfile``.
        - Anything else → ``None``.
        """
        if not path_or_uri:
            return None
        if path_or_uri.startswith("shard://"):
            shard_root = self.shard_root
            if shard_root is None:
                logger.warning(
                    "[corpus_bridge] shard URI %s but no shard_root configured", path_or_uri,
                )
                return None
            try:
                # Lazy import: shard_store does not require verl/ray.
                _ensure_video_search_sim_on_path()
                from video_search_sim.video_corpus.shard_store import (  # type: ignore
                    materialise_to_tempfile,
                )
                tmp_path = materialise_to_tempfile(
                    path_or_uri,
                    shard_root=shard_root,
                    tmp_dir=self.shard_materialise_dir,
                )
                return str(tmp_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[corpus_bridge] failed to materialise %s under %s: %s",
                    path_or_uri, shard_root, exc,
                )
                return None
        # Plain on-disk path
        if Path(path_or_uri).is_file():
            return path_or_uri
        return None


def _ensure_video_search_sim_on_path() -> None:
    """Inject ``verl-video/video_search_sim`` into sys.path if not already there.

    The module is imported lazily so the synthesis driver never has to
    touch CLIP / FAISS / verl unless a shard URI actually needs to be
    materialised.
    """
    here = Path(__file__).resolve()
    # task_generation/video_task_generation/shared/local_corpus.py → repo root
    repo_root = here.parents[3]
    candidate = repo_root / "verl-video" / "video_search_sim"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


# ──────────────────────────────────────────────────────────────────────
# HTTP retrieval client
# ──────────────────────────────────────────────────────────────────────


def _seconds_to_int_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if f <= 0:
            return None
        return f
    except (TypeError, ValueError):
        return None


def search_local_corpus(
    query: str,
    topk: int,
    service_url: str,
    timeout: int = 30,
    bridge: Optional[CorpusBridge] = None,
) -> List[VideoSearchResult]:
    """Query the video_search_sim ``/video_search`` HTTP endpoint.

    Returns shape-compatible :class:`VideoSearchResult` objects so that
    Stage 2's existing helpers (``rank_videos``, etc.) can consume them
    without modification.

    Failures (timeout, 5xx, connection error) are logged and surfaced as
    an empty list — Stage 2 already treats "no candidates" as a soft
    skip and moves on to the next seed.
    """
    if not query or not query.strip():
        return []
    payload = {"query": query.strip(), "topk": int(topk)}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(service_url, json=payload)
        if resp.status_code != 200:
            logger.warning(
                "[local_corpus] search '%s' returned status=%d body=%s",
                query, resp.status_code, resp.text[:300],
            )
            return []
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.error("[local_corpus] search '%s' failed: %s", query, exc)
        return []

    hits = data.get("results") or []
    out: List[VideoSearchResult] = []
    for h in hits:
        fake_url = h.get("fake_url") or ""
        if not fake_url:
            continue
        # Recover the 11-char video id from the fake URL tail.
        video_id = fake_url.rsplit("=", 1)[-1] if "=" in fake_url else fake_url[-11:]
        title = h.get("title") or ""
        snippet = h.get("snippet") or ""
        duration = _seconds_to_int_or_none(h.get("duration"))
        thumbnail = h.get("thumbnail")

        out.append(
            VideoSearchResult(
                video_id=video_id,
                url=fake_url,
                title=title,
                description=snippet,
                duration=duration,
                view_count=None,        # local corpus has no view counts
                upload_date=None,
                channel=None,
                thumbnail_url=thumbnail,
            )
        )
    logger.info("[local_corpus] '%s' → %d hits", query, len(out))
    return out


# ──────────────────────────────────────────────────────────────────────
# Module-level singletons (configured once via ``configure_local_corpus``)
# ──────────────────────────────────────────────────────────────────────


_BRIDGE: Optional[CorpusBridge] = None
_SERVICE_URL: str = "http://127.0.0.1:8000/video_search"
_SERVICE_TIMEOUT: int = 30


def configure_local_corpus(
    corpus_dir: str,
    service_url: str,
    service_timeout: int = 30,
    shard_root: str = "",
    shard_materialise_dir: str = "/tmp/vss_shard_cache",
) -> CorpusBridge:
    """Initialise the module-level CorpusBridge + HTTP retrieval target."""
    global _BRIDGE, _SERVICE_URL, _SERVICE_TIMEOUT
    _SERVICE_URL = service_url
    _SERVICE_TIMEOUT = int(service_timeout)
    _BRIDGE = CorpusBridge(
        corpus_dir=corpus_dir,
        shard_root_override=(shard_root or None),
        shard_materialise_dir=shard_materialise_dir,
    )
    # Eager-load once so any url_to_path.json error surfaces at startup,
    # not deep inside Stage 2 worker threads.
    _BRIDGE.load()
    logger.info(
        "[local_corpus] configured: corpus_dir=%s service_url=%s n_urls=%d layout=%s",
        corpus_dir, service_url, len(_BRIDGE._url_to_path), _BRIDGE.cache_layout,
    )
    return _BRIDGE


def get_bridge() -> Optional[CorpusBridge]:
    return _BRIDGE


def get_service_url() -> str:
    return _SERVICE_URL


def get_service_timeout() -> int:
    return _SERVICE_TIMEOUT
