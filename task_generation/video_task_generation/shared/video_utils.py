"""Video sourcing helpers for Stage 2 — supports ``online`` and ``local`` backends.

Public API (stable across both backends, so Stage 2 / 3 / 4 stay backend-agnostic):
    * ``search_videos(query, topk)``      → list[VideoSearchResult]
    * ``rank_videos(...)`` / ``pick_video(...)``   — same as before
    * ``download_video(url)``             → local mp4 path
    * ``extract_frames(path, n)``         → list[FrameData]

Backends:
    * ``online`` (default): yt-dlp + YouTube + local cache, identical to the
      pre-existing behaviour. Uses the shared ``video_agent`` package.
    * ``local`` : queries the ``video_search_sim`` ``/video_search`` HTTP
      service for retrieval, and resolves each fake YouTube URL back to a
      real local mp4 via ``url_to_path.json`` (with optional shard://
      materialisation through the corpus's shard_store). FrameExtractor is
      reused unmodified — frame extraction is local-only in both modes.

Concurrency note (online only):
    yt-dlp reads AND writes the cookies file (to persist refreshed cookies),
    so running many downloads concurrently with a single shared
    ``cookies_file`` races on the file and can produce
    ``'... does not look like a Netscape format cookies file'`` errors.
    To make graph_workers > 1 safe we snapshot the cookies file to a
    per-call temp copy and hand that to a fresh ``VideoDownloader``;
    search / extract never write cookies, so they stay on the shared
    singleton.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _runtime_cache_root() -> Path:
    """Return the on-disk directory used for per-run scratch files.

    Kept under ``<project-root>/cache/`` by default so we never pollute the
    system ``/tmp``.  Override with ``VIDEO_SEARCHER_CACHE_ROOT`` if you want
    to point it at a fast local disk.
    """
    override = os.environ.get("VIDEO_SEARCHER_CACHE_ROOT")
    root = Path(override) if override else _PROJECT_ROOT / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root

from video_agent.tools.video_search import YouTubeSearchTool, VideoSearchResult as _OnlineVideoSearchResult  # type: ignore  # noqa: E402
from video_agent.tools.video_download import VideoDownloader  # type: ignore  # noqa: E402
from video_agent.tools.frame_extractor import FrameExtractor, FrameData  # type: ignore  # noqa: E402

# Local-corpus path: shape-compatible VideoSearchResult + bridge utilities.
from video_task_generation.shared.local_corpus import (  # noqa: E402
    VideoSearchResult as _LocalVideoSearchResult,
    configure_local_corpus,
    get_bridge,
    get_service_timeout,
    get_service_url,
    search_local_corpus,
)

# Stage 2 / 3 / 4 import this name — both backends produce the same shape.
VideoSearchResult = _OnlineVideoSearchResult  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────
# Singletons, configured via ``configure_video(...)``
# ──────────────────────────────────────────────────────────────────────


_BACKEND: str = "online"   # "online" | "local"

_SEARCH: Optional[YouTubeSearchTool] = None
_DOWNLOADER: Optional[VideoDownloader] = None
_EXTRACTOR: Optional[FrameExtractor] = None

_CACHE_DIR: str = "data/cache/videos_taskgen"
_MAX_RES: str = "360"
_COOKIES_FILE: Optional[str] = None
_COOKIES_LOCK = threading.Lock()


def configure_video(
    cache_dir: str = "data/cache/videos_taskgen",
    max_resolution: str = "360",
    cookies_file: Optional[str] = None,
    ytsearch_topk: int = 8,
    backend: str = "online",
    local_corpus_dir: str = "",
    local_service_url: str = "http://127.0.0.1:8000/video_search",
    local_service_timeout: int = 30,
    local_shard_root: str = "",
    local_shard_materialise_dir: str = "/tmp/vss_shard_cache",
    local_top_k: int = 10,
) -> None:
    """Configure module-level video tooling for the chosen backend.

    The same function handles both backends so the runner only needs one
    call. ``backend == 'local'`` skips constructing the YouTube / yt-dlp
    singletons and instead initialises the local corpus bridge + records
    the HTTP retrieval target. The frame extractor is shared across both
    backends (it operates purely on local mp4 paths).
    """
    global _SEARCH, _DOWNLOADER, _EXTRACTOR, _CACHE_DIR, _MAX_RES, _COOKIES_FILE, _BACKEND
    backend = backend.lower().strip() or "online"
    if backend not in ("online", "local"):
        raise ValueError(f"Unknown video backend: {backend!r}")
    _BACKEND = backend
    _CACHE_DIR = cache_dir
    _MAX_RES = max_resolution

    # Frame extractor is always-on (both backends decode local mp4s).
    _EXTRACTOR = FrameExtractor()

    if backend == "online":
        cookies = cookies_file or os.environ.get("YOUTUBE_COOKIES_FILE") or None
        _COOKIES_FILE = cookies
        # NOTE: YouTube ytsearch: endpoint does not require auth — giving the
        # search tool the same cookies file would force it to contend with the
        # per-call downloader cookies races under high concurrency.  Skip.
        _SEARCH = YouTubeSearchTool(max_results=ytsearch_topk, cookies_file=None)
        # Shared fallback downloader — only used when cookies are not configured,
        # so racing is impossible here.
        _DOWNLOADER = VideoDownloader(
            cache_dir=cache_dir,
            max_resolution=max_resolution,
            cookies_file=cookies,
        )
        logger.info(
            "[video_utils] backend=online cache_dir=%s max_res=%s cookies=%s topk=%d",
            cache_dir, max_resolution, bool(cookies), ytsearch_topk,
        )
    else:
        # Tear down any online singletons so accidental cross-backend calls fail loudly.
        _SEARCH = None
        _DOWNLOADER = None
        _COOKIES_FILE = None
        if not local_corpus_dir:
            raise ValueError(
                "configure_video(backend='local') requires local_corpus_dir; "
                "point it at the directory produced by vss-build-corpus."
            )
        configure_local_corpus(
            corpus_dir=local_corpus_dir,
            service_url=local_service_url,
            service_timeout=local_service_timeout,
            shard_root=local_shard_root,
            shard_materialise_dir=local_shard_materialise_dir,
        )
        # Stash topk for the search helper.
        global _LOCAL_TOPK_DEFAULT
        _LOCAL_TOPK_DEFAULT = int(local_top_k)
        logger.info(
            "[video_utils] backend=local corpus_dir=%s service=%s topk=%d",
            local_corpus_dir, local_service_url, local_top_k,
        )


# Default top-k for local search when caller doesn't pass one explicitly.
_LOCAL_TOPK_DEFAULT: int = 10


def _ensure() -> None:
    if _EXTRACTOR is None:
        configure_video()
    if _BACKEND == "online" and (_SEARCH is None or _DOWNLOADER is None):
        configure_video()
    # local backend doesn't need anything beyond the bridge, which
    # configure_local_corpus already loaded.


def _snapshot_cookies_for_call() -> Optional[str]:
    """Write a per-call private copy of the cookies file so yt-dlp's post-
    download cookie refresh never races across threads.

    The snapshot is placed under ``<project-root>/cache/runtime_cookies/``
    (overridable via ``VIDEO_SEARCHER_CACHE_ROOT``).  Even though this lives
    on ceph-FUSE, we avoid the "half-written cookies file" issue that broke
    ``shutil.copyfile`` by (1) reading the source fully into memory under a
    lock, and (2) materialising the copy through ``mkstemp + fsync`` which
    yields a freshly-created, fully-written file before yt-dlp opens it.

    Returns the path to the copy (caller deletes it) or ``None`` when no
    cookies are configured / snapshot fails.
    """
    if not _COOKIES_FILE:
        return None
    try:
        tmp_dir = _runtime_cache_root() / "runtime_cookies"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with _COOKIES_LOCK:
            # Lock prevents two snapshotters from interleaving reads while
            # the source is being refreshed (by a legacy code path).
            src = Path(_COOKIES_FILE)
            if not src.exists() or src.stat().st_size == 0:
                return None
            data = src.read_bytes()
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"cookies_{threading.get_ident()}_",
            suffix=".txt",
            dir=str(tmp_dir),
        )
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return tmp_path
    except Exception as exc:
        logger.warning("[video_utils] cookies snapshot failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def search_videos(query: str, topk: int = 8) -> List[VideoSearchResult]:
    """Backend-dispatched video search.

    * ``online`` → YouTube via ``YouTubeSearchTool``.
    * ``local``  → ``video_search_sim`` HTTP ``/video_search`` (BM25+CLIP),
      returning shape-compatible :class:`VideoSearchResult` records whose
      ``url`` field is the corpus's fake YouTube URL (the same URL used as
      ground-truth in the eventual reward computation).
    """
    _ensure()
    if _BACKEND == "local":
        effective_topk = topk if topk and topk > 0 else _LOCAL_TOPK_DEFAULT
        try:
            return search_local_corpus(
                query,
                topk=effective_topk,
                service_url=get_service_url(),
                timeout=get_service_timeout(),
                bridge=get_bridge(),
            )  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network/service errors
            logger.error("local_corpus search failed for '%s': %s", query, exc)
            return []

    assert _SEARCH is not None
    try:
        return _SEARCH.search(query, max_results=topk)
    except Exception as exc:  # pragma: no cover
        logger.error("YouTube search failed for '%s': %s", query, exc)
        return []


def pick_video(
    results: List[VideoSearchResult],
    min_duration: float = 30.0,
    max_duration: float = 1500.0,
    strategy: str = "random",
) -> Optional[VideoSearchResult]:
    """Filter by duration, then pick one.

    strategy: ``random`` (default) or ``top1``/``top_views``.
    """
    ranked = rank_videos(results, min_duration, max_duration, strategy)
    return ranked[0] if ranked else None


def rank_videos(
    results: List[VideoSearchResult],
    min_duration: float = 30.0,
    max_duration: float = 1500.0,
    strategy: str = "random",
) -> List[VideoSearchResult]:
    """Filter by duration and return an ordered candidate list.

    The caller can iterate through this list to try multiple videos when
    earlier ones fail to download (e.g. age-gated, region-blocked, geo-blocked).
    """
    if not results:
        return []
    pool: List[VideoSearchResult] = []
    for r in results:
        dur = r.duration or 0.0
        if dur <= 0:
            pool.append(r)
            continue
        if min_duration <= dur <= max_duration:
            pool.append(r)
    if not pool:
        pool = list(results)
    if strategy == "random":
        pool = list(pool)
        random.shuffle(pool)
    elif strategy == "top_views":
        pool = sorted(pool, key=lambda x: -(x.view_count or 0))
    # strategy == "top1" → keep original order
    return pool


# Hard wall-clock cap per download — yt-dlp/ffmpeg can hang indefinitely on
# certain streams which would otherwise stall a worker thread forever and
# wedge Stage 2's long tail.
_DOWNLOAD_HARD_TIMEOUT_SEC = int(os.environ.get("VIDEO_DOWNLOAD_HARD_TIMEOUT", "180"))


def _download_with_timeout(url: str, cookies_file: Optional[str]) -> Optional[str]:
    """Run ``VideoDownloader.download`` in a helper thread with a hard timeout.

    yt-dlp's ``socket_timeout`` applies to network sockets, but when it
    delegates to an external downloader (ffmpeg/curl/aria2) that timeout is
    NOT enforced end-to-end — the child process can hang on stalled streams.
    We guard against that by waiting on a helper thread; if the timeout
    elapses we return ``None`` (letting the caller try the next candidate)
    and leave the helper thread to finish in the background.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _do() -> Optional[str]:
        downloader = VideoDownloader(
            cache_dir=_CACHE_DIR,
            max_resolution=_MAX_RES,
            cookies_file=cookies_file,
        )
        return downloader.download(url)

    # A fresh single-worker executor per call — the thread will simply exit
    # once the (potentially stuck) yt-dlp subprocess it spawned returns. We
    # don't block on shutdown.
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yt-dlp-guard")
    fut = ex.submit(_do)
    try:
        try:
            return fut.result(timeout=_DOWNLOAD_HARD_TIMEOUT_SEC)
        except FuturesTimeout:
            logger.warning(
                "[video_utils] download hard-timeout after %ds: %s — skipping",
                _DOWNLOAD_HARD_TIMEOUT_SEC, url,
            )
            return None
        except Exception as exc:
            logger.error("[video_utils] download raised: %s (%s)", exc, url)
            return None
    finally:
        # wait=False so a still-running helper thread doesn't keep us here.
        ex.shutdown(wait=False)


def download_video(url: str) -> Optional[str]:
    """Resolve a video URL to a local mp4 path.

    * ``online`` → ``yt-dlp`` download (cookies snapshot + hard timeout).
    * ``local``  → ``CorpusBridge.local_path`` lookup, with shard:// URIs
      materialised to a real on-disk file. No network, no yt-dlp.
    """
    _ensure()

    if _BACKEND == "local":
        bridge = get_bridge()
        if bridge is None:
            logger.error("[video_utils] local backend requested but bridge not configured")
            return None
        raw = bridge.local_path(url)
        if not raw:
            logger.warning("[video_utils] no local path for url=%s", url)
            return None
        resolved = bridge.materialise(raw)
        if not resolved:
            logger.warning("[video_utils] failed to materialise %s (raw=%s)", url, raw)
            return None
        return resolved

    assert _DOWNLOADER is not None

    # No cookies configured → use the shared downloader, still with timeout.
    if not _COOKIES_FILE:
        return _download_with_timeout(url, cookies_file=None)

    tmp_cookies = _snapshot_cookies_for_call()
    if not tmp_cookies:
        return _download_with_timeout(url, cookies_file=None)

    try:
        return _download_with_timeout(url, cookies_file=tmp_cookies)
    finally:
        try:
            os.unlink(tmp_cookies)
        except OSError:
            pass


def extract_frames(video_path: str, n_frames: int = 24) -> List[FrameData]:
    _ensure()
    assert _EXTRACTOR is not None
    try:
        return _EXTRACTOR.extract_sparse(video_path, n_frames=n_frames)
    except Exception as exc:
        logger.error("frame extract failed for %s: %s", video_path, exc)
        return []
