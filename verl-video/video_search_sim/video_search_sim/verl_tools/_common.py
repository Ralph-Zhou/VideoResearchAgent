"""Shared internals for the verl_tools layer.

Responsibilities
----------------
- **Ray execution pool + token-bucket rate limiter** — one shared pattern,
  reused by every HTTP-backed or GPU-heavy tool.
- **``video_agent`` package path injection** — the shared tools live in the
  co-located ``video_agent`` package. Rather than vendor-copy that code, we
  append its path to ``sys.path`` at first import so we can reuse
  ``YouTubeSearchTool`` / ``WebSearchTool`` / ``FrameExtractor`` /
  ``TranscriptFetcher`` / ``VideoDownloader`` / ``VisualGroundingTool``
  directly.
- **Tool-text formatting helpers** — a single place that produces the JSON
  envelope every tool returns, so SFT/RL prompt tokenisation stays aligned.

Why lazy imports
----------------
``ray`` / ``verl`` are runtime-only dependencies (rollout time, not index-build
time). Any module that imports from this file must be importable in a plain
``pip install -e '.'`` env **without** verl installed — otherwise we'd break
unit tests, the FastAPI service, and the index pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- path injection

# Environment variable pointing at the repository root that contains the
# co-located ``video_agent`` package. When set, that directory is prepended to
# ``sys.path`` so ``import video_agent.tools.xxx`` works.
_VIDEO_AGENT_ENV = "VSS_VIDEO_AGENT_PATH"
_DEFAULT_VIDEO_AGENT_PATH = str(Path(__file__).resolve().parents[4])
_video_agent_injected = False


def ensure_video_agent_on_path() -> str | None:
    """Ensure the co-located ``video_agent`` package is importable.

    Looks up ``VSS_VIDEO_AGENT_PATH`` in the environment, falling back to the
    co-located repository root. Returns the injected path (or
    ``None`` if the path is not a directory, in which case callers should
    handle ``ImportError`` themselves).

    Safe to call multiple times — the actual injection happens at most once.
    """
    global _video_agent_injected
    if _video_agent_injected:
        return os.environ.get(_VIDEO_AGENT_ENV) or _DEFAULT_VIDEO_AGENT_PATH

    root = os.environ.get(_VIDEO_AGENT_ENV, _DEFAULT_VIDEO_AGENT_PATH)
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        logger.warning(
            "video_agent path not found at %s; set %s to the repository root, or install video_agent as a package.",
            root_path,
            _VIDEO_AGENT_ENV,
        )
        return None

    # The package is ``video_agent`` under ``root_path``; we inject ``root_path``
    # itself so ``import video_agent.xxx`` works.
    path_str = str(root_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.info("Injected %s into sys.path for video_agent imports", path_str)
    _video_agent_injected = True
    return path_str


def resolve_youtube_cookies(config_value: str | None = None) -> str | None:
    """Resolve & validate a YouTube cookies.txt for the remote (yt-dlp) path.

    Resolution order: explicit ``config_value`` → ``YOUTUBE_COOKIES_FILE`` env
    → ``None`` (anonymous).

    yt-dlp hard-fails the *entire* request if ``cookiefile`` points at a file
    that is not in Netscape format (``"... does not look like a Netscape
    format cookies file"``), which would otherwise break every remote search
    and download. So we validate the file here and, if it is missing or not in
    Netscape format, log a warning and fall back to anonymous (``None``) rather
    than poisoning every tool call.

    A valid Netscape cookies.txt starts with a ``# Netscape HTTP Cookie File``
    (or ``# HTTP Cookie File``) magic comment on its first non-empty line.
    """
    path = config_value or os.getenv("YOUTUBE_COOKIES_FILE") or None
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        logger.warning("YouTube cookies file not found at %s; using anonymous yt-dlp.", p)
        return None
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            head = ""
            for line in f:
                if line.strip():
                    head = line.strip()
                    break
    except OSError as e:
        logger.warning("Could not read YouTube cookies file %s (%s); using anonymous yt-dlp.", p, e)
        return None
    if not head.startswith("# Netscape HTTP Cookie File") and not head.startswith("# HTTP Cookie File"):
        logger.warning(
            "YouTube cookies file %s is not in Netscape format (first line: %r); "
            "ignoring it and using anonymous yt-dlp. Regenerate with e.g. "
            "`yt-dlp --cookies-from-browser chrome --cookies %s` or a browser "
            "'Get cookies.txt' export.",
            p, head[:60], p,
        )
        return None
    return str(p)


# --------------------------------------------------------------------------- Ray execution pool


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


def _load_ray():
    """Lazy ``import ray``; raises ``RuntimeError`` with a clear message if unavailable."""
    try:
        import ray  # noqa: PLC0415
        import ray.actor  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - declared in pyproject but optional for tests
        raise RuntimeError(
            "Ray is not installed; the verl_tools layer is only usable inside an environment with ray + verl installed."
        ) from e
    return ray


def make_token_bucket_actor_class():
    """Build the ``TokenBucketWorker`` class, lazily, so Ray is only imported when needed.

    Each call returns the same class (ray decorates it), but the decorator is
    re-applied on every call so we don't force Ray at module import.
    """
    ray = _load_ray()

    @ray.remote(concurrency_groups={"acquire": 1, "release": 10})
    class TokenBucketWorker:
        """Counting semaphore actor that caps concurrent tool calls."""

        def __init__(self, rate_limit: int):
            self.rate_limit = rate_limit
            self.current_count = 0
            self._semaphore = threading.Semaphore(rate_limit)

        @ray.method(concurrency_group="acquire")
        def acquire(self):
            self._semaphore.acquire()
            self.current_count += 1

        @ray.method(concurrency_group="release")
        def release(self):
            self._semaphore.release()
            self.current_count -= 1

        def get_current_count(self):
            return self.current_count

    return TokenBucketWorker


class ExecutionWorker:
    """Actor-side executor that runs a callable under an optional rate-limit token.

    The callable is expected to be **blocking** (HTTP POST, ffmpeg, CLIP infer,
    etc.) and will run inside Ray's thread pool, so one actor can execute
    ``max_concurrency`` blocking calls in parallel.
    """

    def __init__(
        self,
        enable_global_rate_limit: bool = True,
        rate_limit: int = 64,
        limiter_name: str = "vss-default-rate-limiter",
    ):
        # This actor runs in a SEPARATE Ray process that never executes a
        # tool's __init__, so it has not yet put the ``video_agent``
        # package on sys.path. Any blocking fn we run (or its pickled
        # args/results) may reference ``video_agent.*`` classes, which would
        # otherwise fail to (de)serialise with "No module named 'video_agent'".
        # Inject it here so every ExecutionWorker can import it.
        ensure_video_agent_on_path()
        self.rate_limit_worker = None
        if enable_global_rate_limit:
            _load_ray()  # eager-check Ray is importable; actor decorator uses it below
            cls = make_token_bucket_actor_class()
            self.rate_limit_worker = cls.options(
                name=limiter_name,
                get_if_exists=True,
                # CRITICAL: without a detached lifetime this named actor is
                # *owned* by whichever transient worker happens to create it
                # first. When that owner (an ExecutionWorker / AgentLoopWorker)
                # is recycled or dies, Ray garbage-collects the rate limiter
                # too, and every other tool call then fails with
                # "ActorDiedError: ... owner has died" — a cluster-wide storm.
                # `lifetime="detached"` unbinds it from any owner so it lives
                # as a cluster-global singleton keyed by `name`.
                lifetime="detached",
            ).remote(rate_limit)

    def ping(self) -> bool:
        return True

    def execute(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if self.rate_limit_worker is not None:
            ray = _load_ray()
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)


def init_execution_pool(
    num_workers: int,
    enable_rate_limit: bool,
    rate_limit: int,
    limiter_name: str,
):
    """Create a Ray execution pool with ``num_workers`` blocking-call slots.

    ``limiter_name`` is used as the Ray actor name of the global token-bucket
    limiter — giving each tool type its own name prevents cross-tool head-of-line
    blocking (e.g. a surge of ``watch_video`` decoder calls shouldn't stall
    ``video_search`` HTTP calls).
    """
    ray = _load_ray()
    return (
        ray.remote(ExecutionWorker)
        .options(max_concurrency=num_workers)
        .remote(
            enable_global_rate_limit=enable_rate_limit,
            rate_limit=rate_limit,
            limiter_name=limiter_name,
        )
    )


# --------------------------------------------------------------------------- output formatting


def dumps_tool_text(payload: dict[str, Any]) -> str:
    """Render a tool's structured result as a single JSON line.

    We emit a machine-parseable string (rather than free-form prose) so:

    - downstream reward / scoring code can regex the trajectory deterministically,
    - follow-up tool calls (e.g. ``watch_video`` picking a URL from
      ``video_search`` results) can lift fields unambiguously, and
    - SFT-time and RL-time tool response tokenisation stays identical.

    ``ensure_ascii=False`` preserves CJK text for Chinese queries.
    """
    return json.dumps(payload, ensure_ascii=False)


def truncate_text(text: str, max_chars: int, *, from_: str = "right") -> tuple[str, bool]:
    """Truncate ``text`` to at most ``max_chars``; returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    if from_ == "left":
        return "...(truncated)..." + text[-max_chars:], True
    return text[:max_chars] + "...(truncated)...", True


# --------------------------------------------------------------------------- image decoding


def decode_base64_to_pil(b64: str):
    """Decode a base-64-encoded JPEG/PNG string to a ``PIL.Image.Image``.

    Raised exceptions are swallowed and ``None`` is returned — callers should
    drop ``None`` results, because the verl tool_agent_loop will crash if we
    feed a ``None`` into ``ToolResponse.image``. We intentionally keep the
    ``PIL`` import lazy so ``_common.py`` stays cheap to import (Pillow is
    otherwise a hard dep of the whole video_corpus pipeline).
    """
    if not b64:
        return None
    try:
        import base64 as _b64  # noqa: PLC0415
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        data = _b64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        img.load()  # force-decode now so later multiprocessing doesn't re-read bytes
        return img.convert("RGB")
    except Exception as e:  # noqa: BLE001 - defensive: never crash rollouts on bad frames
        logger.warning("Failed to decode base64 frame: %s", e)
        return None


def decode_base64_frames(b64_list: list[str]):
    """Batch decode helper — drops entries that fail to decode."""
    out = []
    for b in b64_list:
        img = decode_base64_to_pil(b)
        if img is not None:
            out.append(img)
    return out


# --------------------------------------------------------------------------- corpus bridge cache

_corpus_bridge_cache: dict[str, Any] = {}


def get_corpus_bridge(corpus_dir: str):
    """Return a shared ``_CorpusBridge`` instance for ``corpus_dir``.

    The bridge is lazily constructed and loaded on first access, and reused
    by every tool that references the same corpus_dir. This matters when
    both ``WatchVideoTool`` and ``VisualGroundingTool`` live in the same
    rollout process — otherwise we'd re-read ``videos.parquet`` twice for
    no reason.
    """
    from .watch_video_tool import _CorpusBridge  # noqa: PLC0415 - late to avoid circular import

    key = str(Path(corpus_dir).expanduser().resolve())
    bridge = _corpus_bridge_cache.get(key)
    if bridge is None:
        bridge = _CorpusBridge(key)
        _corpus_bridge_cache[key] = bridge
    return bridge
