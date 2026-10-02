"""FastAPI application factory + request handlers.

Separating app construction from ``server.py`` lets tests spin up the app with
``httpx.AsyncClient(transport=ASGITransport(app))`` without ever binding a
port, while ``server.py`` handles the uvicorn-launch glue.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import anyio
from fastapi import FastAPI, HTTPException

from ..common.config import ServiceConfig
from ..common.fake_url import thumbnail_url
from ..common.schemas import (
    SearchHit,
    VideoRecord,
    VideoSearchRequest,
    VideoSearchResponse,
    WatchVideoRequest,
    WatchVideoResponse,
)
from ..verl_tools._common import ensure_video_agent_on_path, truncate_text
from ..video_corpus.embedder import CLIPEmbedder
from .fusion import reciprocal_rank_fusion
from .loader import Corpus, load_corpus
from .retrievers import BM25Retriever, DenseRetriever

logger = logging.getLogger(__name__)


class _FrameData(NamedTuple):
    timestamp: float
    image_b64: str


class _ServiceFrameExtractor:
    """Small service-side fallback when ``video_agent`` is not installed."""

    def __init__(self, jpeg_quality: int = 85, max_short_side: int = 720):
        self.jpeg_quality = int(jpeg_quality)
        self.max_short_side = int(max_short_side)

    def extract_sparse(self, video_path: str, n_frames: int = 16) -> list[_FrameData]:
        try:
            import decord  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total = len(vr)
            if total <= 0:
                return self._extract_sparse_cv2(video_path, n_frames)
            fps = float(vr.get_avg_fps() or 0.0)
            indices = np.linspace(0, total - 1, min(int(n_frames), total), dtype=int).tolist()
            frames = vr.get_batch(indices).asnumpy()
            return [
                _FrameData(
                    timestamp=round((indices[i] / fps) if fps > 0 else 0.0, 2),
                    image_b64=self._encode_frame(frame[:, :, ::-1]),
                )
                for i, frame in enumerate(frames)
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("service decord sparse failed for %s: %s; falling back to cv2", video_path, e)
            return self._extract_sparse_cv2(video_path, n_frames)

    def extract_dense(
        self,
        video_path: str,
        start: float,
        end: float,
        fps: float = 1.0,
        max_frames: int = 32,
    ) -> list[_FrameData]:
        try:
            import decord  # noqa: PLC0415

            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            video_fps = float(vr.get_avg_fps() or 0.0)
            total = len(vr)
            if total <= 0 or video_fps <= 0:
                return self._extract_dense_cv2(video_path, start, end, fps, max_frames)
            timestamps: list[float] = []
            t = float(start)
            while t <= float(end) and len(timestamps) < int(max_frames):
                timestamps.append(t)
                t += 1.0 / float(fps)
            indices = [min(int(ts * video_fps), total - 1) for ts in timestamps]
            if not indices:
                return []
            frames = vr.get_batch(indices).asnumpy()
            return [
                _FrameData(timestamp=round(timestamps[i], 2), image_b64=self._encode_frame(frame[:, :, ::-1]))
                for i, frame in enumerate(frames)
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("service decord dense failed for %s: %s; falling back to cv2", video_path, e)
            return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

    def _extract_sparse_cv2(self, video_path: str, n_frames: int) -> list[_FrameData]:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0 or fps <= 0:
            cap.release()
            return []
        out: list[_FrameData] = []
        for idx in np.linspace(0, total - 1, min(int(n_frames), total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                out.append(_FrameData(timestamp=round(float(idx) / fps, 2), image_b64=self._encode_frame(frame)))
        cap.release()
        return out

    def _extract_dense_cv2(
        self,
        video_path: str,
        start: float,
        end: float,
        fps: float,
        max_frames: int,
    ) -> list[_FrameData]:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if video_fps <= 0:
            cap.release()
            return []
        out: list[_FrameData] = []
        t = float(start)
        while t <= float(end) and len(out) < int(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_fps))
            ok, frame = cap.read()
            if ok:
                out.append(_FrameData(timestamp=round(t, 2), image_b64=self._encode_frame(frame)))
            t += 1.0 / float(fps)
        cap.release()
        return out

    def _encode_frame(self, frame_bgr) -> str:
        import base64  # noqa: PLC0415

        import cv2  # noqa: PLC0415

        h, w = frame_bgr.shape[:2]
        short = min(h, w)
        if self.max_short_side > 0 and short > self.max_short_side:
            scale = self.max_short_side / short
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buf).decode("utf-8")


@dataclass
class _WatchRuntime:
    """Long-lived objects for the optional remote watch endpoint."""

    frame_extractor: Any
    transcript_fetcher: Any
    downloader: Any
    limiter: anyio.Semaphore
    url_to_record: dict[str, VideoRecord]


@dataclass
class _ServiceState:
    """Opaque bag of long-lived singletons, stored on ``app.state``."""

    cfg: ServiceConfig
    corpus: Corpus
    bm25: BM25Retriever | None
    dense: DenseRetriever | None
    watch: _WatchRuntime | None = None
    watch_error: str | None = None


def _build_snippet(record, query_tokens: set[str]) -> str:
    """Assemble a short snippet; try to surface query-adjacent text first."""
    text = record.description or record.subtitle or record.title
    if not query_tokens or not text:
        return (text or "").strip()

    # Find the first sentence-ish chunk containing any query token.
    lowered = text.lower()
    for token in query_tokens:
        idx = lowered.find(token.lower())
        if idx >= 0:
            start = max(0, idx - 40)
            end = min(len(text), idx + 160)
            return text[start:end].strip()
    return text[:200].strip()


def _assemble_hits(
    corpus: Corpus,
    fused: list[tuple[int, float]],
    query: str,
    topk: int,
) -> list[SearchHit]:
    """Translate ``(video_idx, rrf_score)`` tuples into client-facing hits."""
    query_tokens = {t for t in query.lower().split() if t}
    hits: list[SearchHit] = []
    for v_idx, score in fused[:topk]:
        if v_idx < 0 or v_idx >= corpus.num_videos:
            continue
        rec = corpus.videos[v_idx]
        hits.append(
            SearchHit(
                fake_url=rec.fake_url,
                title=rec.title or f"Untitled ({rec.video_id})",
                snippet=_build_snippet(rec, query_tokens),
                duration=rec.duration,
                thumbnail=thumbnail_url(rec.video_id),
                score=float(score),
            )
        )
    return hits


def _maybe_local_file(s: str) -> str | None:
    """Return a real on-disk path if ``s`` looks like one, else None."""
    if not s:
        return None
    s = s.strip().rstrip(".").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "`'\"":
        s = s[1:-1].strip()
    if s.startswith("file://"):
        s = s[len("file://"):]
    try:
        p = Path(s)
    except (TypeError, ValueError):
        return None
    if p.is_file():
        return str(p)
    return None


def _shard_root(corpus: Corpus, cfg: ServiceConfig) -> Path | None:
    if cfg.watch.shard_root is not None:
        return cfg.watch.shard_root
    cache = (corpus.manifest or {}).get("cache") or {}
    root = cache.get("dir")
    return Path(root).expanduser().resolve() if root else None


def _resolve_watch_path(
    req: WatchVideoRequest,
    state: _ServiceState,
) -> tuple[str | None, bool, VideoRecord | None]:
    """Resolve a watch request to a local mp4 path plus corpus-hit metadata."""
    assert state.watch is not None
    if req.backend == "remote":
        candidate = _maybe_local_file(req.url)
        if candidate is not None:
            return candidate, False, None
        try:
            if state.watch.downloader is None:
                return None, False, None
            return state.watch.downloader.download(req.url), False, None
        except Exception as e:  # noqa: BLE001
            logger.warning("watch_video service remote download failed url=%r: %s", req.url, e)
            return None, False, None

    record = state.watch.url_to_record.get(req.url)
    path = state.corpus.url_to_path.get(req.url) or (record.local_path if record else "")
    if path:
        if path.startswith("shard://"):
            root = _shard_root(state.corpus, state.cfg)
            if root is None:
                logger.warning("watch_video service got shard URI but no shard_root: %s", path)
            else:
                try:
                    from ..video_corpus.shard_store import materialise_to_tempfile  # noqa: PLC0415

                    tmp_path = materialise_to_tempfile(
                        path,
                        shard_root=root,
                        tmp_dir=state.cfg.watch.shard_materialise_dir,
                    )
                    return str(tmp_path), True, record
                except Exception as e:  # noqa: BLE001
                    logger.warning("watch_video service failed to materialise %s: %s", path, e)
        elif Path(path).is_file():
            return path, True, record

    candidate = _maybe_local_file(req.url)
    if candidate is not None:
        return candidate, False, None
    return None, False, record


def _watch_video_blocking(req: WatchVideoRequest, state: _ServiceState) -> WatchVideoResponse:
    """Blocking watch implementation, intended to run in anyio's threadpool."""
    assert state.watch is not None
    t0 = time.perf_counter()
    local_path, is_local_hit, record = _resolve_watch_path(req, state)
    if local_path is None:
        return WatchVideoResponse(
            url=req.url,
            mode=req.mode,
            error=(
                f"video could not be resolved for url={req.url} "
                f"(backend={req.backend}) on watch service"
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if req.mode == "dense":
        if req.start_time is None or req.end_time is None:
            return WatchVideoResponse(
                url=req.url,
                mode=req.mode,
                error="dense mode requires `start_time` and `end_time`",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        frames = state.watch.frame_extractor.extract_dense(
            local_path,
            start=float(req.start_time),
            end=float(req.end_time),
            fps=float(req.fps),
            max_frames=state.cfg.watch.max_n_frames_dense,
        )
    else:
        frames = state.watch.frame_extractor.extract_sparse(
            local_path,
            n_frames=min(int(req.n_frames), state.cfg.watch.max_n_frames_sparse),
        )

    transcript_text = ""
    if is_local_hit and record is not None and record.subtitle:
        transcript_text = record.subtitle
    else:
        try:
            if state.watch.transcript_fetcher is None:
                raise RuntimeError("TranscriptFetcher unavailable")
            segs = state.watch.transcript_fetcher.fetch(req.url)
            transcript_text = state.watch.transcript_fetcher.format_for_prompt(
                segs,
                max_chars=state.cfg.watch.max_transcript_chars,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("watch_video service transcript fetch failed for %s: %s", req.url, e)

    transcript_text, transcript_truncated = truncate_text(
        transcript_text,
        state.cfg.watch.max_transcript_chars,
    )

    duration = record.duration if is_local_hit and record is not None else None
    if duration is None:
        try:
            if state.watch.downloader is None:
                raise RuntimeError("VideoDownloader unavailable")
            duration = state.watch.downloader.get_duration(local_path)
        except Exception:  # noqa: BLE001
            duration = 0.0

    return WatchVideoResponse(
        url=req.url,
        mode=req.mode,
        duration_sec=float(duration or 0.0),
        local_hit=is_local_hit,
        timestamps=[float(f.timestamp) for f in frames],
        frames_b64=[f.image_b64 for f in frames],
        transcript=transcript_text,
        transcript_truncated=transcript_truncated,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def _init_watch_runtime(cfg: ServiceConfig, corpus: Corpus) -> tuple[_WatchRuntime | None, str | None]:
    """Initialise watch endpoint dependencies without making service startup fatal."""
    if not cfg.watch.enable:
        return None, "watch endpoint disabled by config"
    try:
        ensure_video_agent_on_path()
        from video_agent.tools.frame_extractor import FrameExtractor  # noqa: PLC0415
        from video_agent.tools.transcript import TranscriptFetcher  # noqa: PLC0415
        from video_agent.tools.video_download import VideoDownloader  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        logger.warning("video_agent tools unavailable for watch service: %s; using local frame extractor only", e)
        FrameExtractor = None  # type: ignore[assignment]
        TranscriptFetcher = None  # type: ignore[assignment]
        VideoDownloader = None  # type: ignore[assignment]

    try:
        frame_extractor = (
            FrameExtractor(jpeg_quality=cfg.watch.jpeg_quality)
            if FrameExtractor is not None
            else _ServiceFrameExtractor(jpeg_quality=cfg.watch.jpeg_quality)
        )
        transcript_fetcher = (
            TranscriptFetcher(cache_dir=str(cfg.watch.transcript_cache_dir))
            if TranscriptFetcher is not None
            else None
        )
        downloader = (
            VideoDownloader(
                cache_dir=str(cfg.watch.downloader_cache_dir),
                download_timeout_sec=cfg.watch.remote_download_timeout_sec,
            )
            if VideoDownloader is not None
            else None
        )
        url_to_record = {rec.fake_url: rec for rec in corpus.videos}
        return (
            _WatchRuntime(
                frame_extractor=frame_extractor,
                transcript_fetcher=transcript_fetcher,
                downloader=downloader,
                limiter=anyio.Semaphore(max(1, int(cfg.watch.concurrency))),
                url_to_record=url_to_record,
            ),
            None,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"watch endpoint unavailable: failed to initialise runtime: {e!r}"


def create_app(cfg: ServiceConfig) -> FastAPI:
    """Build a ready-to-serve FastAPI app from the given ``ServiceConfig``.

    Loading the corpus (parquet, FAISS, BM25) happens inside the lifespan
    handler, so importing this module is side-effect free (important for
    tests + autoreload).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Pin per-request compute to a single thread so that many CONCURRENT
        # requests (run in the threadpool) map cleanly to one core each, instead
        # of every request spawning core-count intra-op threads and oversubscribing
        # the CPU (which throttled throughput to ~1/7th of a 142-core box). True
        # parallelism comes from request-level concurrency, not intra-op threads.
        try:
            import torch

            torch.set_num_threads(1)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not pin torch threads: %s", e)
        try:
            import faiss

            faiss.omp_set_num_threads(1)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not pin faiss threads: %s", e)

        # Widen the anyio threadpool that runs the blocking searches; the default
        # (40) caps concurrency well below a many-core host's capacity.
        try:
            limiter = anyio.to_thread.current_default_thread_limiter()
            limiter.total_tokens = max(1, cfg.server.threadpool_size)
            logger.info("anyio threadpool size set to %d", limiter.total_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not set threadpool size: %s", e)

        corpus = load_corpus(cfg.corpus_dir)

        bm25: BM25Retriever | None = None
        if cfg.retrieval.enable_bm25 and corpus.bm25_state is not None:
            bm25 = BM25Retriever(corpus)

        dense: DenseRetriever | None = None
        if cfg.retrieval.enable_dense and corpus.faiss_index is not None:
            embedder = CLIPEmbedder(cfg.clip)
            # Warm the model so the first request isn't a cold-start outlier.
            embedder.encode_texts([""])
            dense = DenseRetriever(
                corpus=corpus,
                clip_cfg=cfg.clip,
                embedder=embedder,
                aggregation=cfg.retrieval.dense_aggregation,
            )

        if bm25 is None and dense is None:
            raise RuntimeError("Both BM25 and dense retrievers are disabled/unavailable; nothing to serve.")

        watch_runtime, watch_error = _init_watch_runtime(cfg, corpus)
        if watch_error:
            logger.warning(watch_error)

        app.state.service = _ServiceState(
            cfg=cfg,
            corpus=corpus,
            bm25=bm25,
            dense=dense,
            watch=watch_runtime,
            watch_error=watch_error,
        )
        logger.info("Service ready: bm25=%s dense=%s", bm25 is not None, dense is not None)
        try:
            yield
        finally:
            logger.info("Shutting down retrieval service")

    app = FastAPI(
        title="video-search-sim",
        version="0.1.0",
        description="Local video_search simulation for veRL RL training.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        state: _ServiceState | None = getattr(app.state, "service", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Corpus not loaded yet")
        return {
            "status": "ok",
            "num_videos": state.corpus.num_videos,
            "num_keyframes": state.corpus.num_keyframes,
            "bm25": state.bm25 is not None,
            "dense": state.dense is not None,
            "watch": state.watch is not None,
            "watch_error": state.watch_error,
            "manifest": state.corpus.manifest,
        }

    @app.post("/video_search", response_model=VideoSearchResponse)
    async def video_search(req: VideoSearchRequest) -> VideoSearchResponse:
        state: _ServiceState | None = getattr(app.state, "service", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Corpus not loaded yet")

        topk = min(req.topk, state.cfg.retrieval.max_topk)
        t0 = time.perf_counter()

        ranked_inputs: list[list[int]] = []

        # CRITICAL: bm25.search (BLAS) and dense.search (CLIP GPU encode + FAISS
        # over ~1.4M keyframes) are BLOCKING, synchronous calls. Running them
        # directly in this ``async def`` handler would block the single event
        # loop (workers=1) for the whole duration, serialising every request and
        # — under the rollout fan-out (100s of concurrent agents) — making the
        # tail of the queue ReadTimeout en masse. Offload to the threadpool so
        # requests actually overlap and a slow/stuck query can't freeze the loop.
        if state.bm25 is not None:
            bm25_hits = await anyio.to_thread.run_sync(
                functools.partial(
                    state.bm25.search, req.query, topk=state.cfg.retrieval.bm25_candidate_pool
                )
            )
            ranked_inputs.append([v for v, _ in bm25_hits])

        if state.dense is not None:
            dense_hits = await anyio.to_thread.run_sync(
                functools.partial(
                    state.dense.search,
                    req.query,
                    topk=state.cfg.retrieval.dense_candidate_pool,
                    candidate_pool=state.cfg.retrieval.dense_candidate_pool,
                )
            )
            ranked_inputs.append([v for v, _ in dense_hits])

        if not ranked_inputs:
            return VideoSearchResponse(
                query=req.query,
                results=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                debug={"note": "no retrievers active"},
            )

        fused = reciprocal_rank_fusion(ranked_inputs, k=state.cfg.retrieval.rrf_k)
        hits = _assemble_hits(state.corpus, fused, req.query, topk=topk)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Surface per-query latency server-side so a fundamentally slow retriever
        # (GPU contention, oversized candidate pool, ...) is diagnosable even when
        # the client has already given up with a ReadTimeout.
        if elapsed_ms > 5000:
            logger.warning("slow video_search: %.0f ms for query=%r", elapsed_ms, req.query[:80])

        return VideoSearchResponse(
            query=req.query,
            results=hits,
            latency_ms=elapsed_ms,
            debug={
                "num_candidate_lists": len(ranked_inputs),
                "num_fused_docs": len(fused),
            },
        )

    @app.post("/watch_video", response_model=WatchVideoResponse)
    async def watch_video(req: WatchVideoRequest) -> WatchVideoResponse:
        state: _ServiceState | None = getattr(app.state, "service", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Corpus not loaded yet")
        if state.watch is None:
            return WatchVideoResponse(
                url=req.url,
                mode=req.mode,
                error=state.watch_error or "watch endpoint unavailable",
            )

        async with state.watch.limiter:
            return await anyio.to_thread.run_sync(
                functools.partial(_watch_video_blocking, req, state)
            )

    return app
