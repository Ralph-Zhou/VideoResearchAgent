"""Video downloader with caching, ffmpeg probe, and robust filename resolution.

Dependencies:
    pip install -U "yt-dlp[default]"

    The "yt-dlp[default]" extra installs **yt-dlp-ejs** (the JS challenge-solver
    scripts that YouTube now requires).  Without it, the "n parameter challenge"
    fails, no video format URLs can be decrypted, and yt-dlp falls back to
    image-only thumbnails → "Requested format is not available".

    A JavaScript runtime reachable from PATH is also required (one of):
      - Node.js >= 20   (most common)
      - Deno >= 2.0     (yt-dlp default preference)
      - Bun >= 1.0.31
"""

import re
import os
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import yt_dlp

from video_agent.tools.ytdlp_utils import cookiefile_snapshot, use_cookie_player_clients

logger = logging.getLogger(__name__)


class _DownloadTimeout(Exception):
    """Raised from a yt-dlp progress hook to abort a download that has blown
    its wall-clock budget (e.g. YouTube throttling the IP to a slow trickle,
    which keeps the socket alive so ``socket_timeout`` never fires)."""

# Detect an available JavaScript runtime once at import time. YouTube's
# n-challenge requires one; without it downloads fail outright.
_JS_RUNTIMES: list[str] = []
for _rt in ("deno", "node", "bun"):
    if shutil.which(_rt):
        _JS_RUNTIMES.append(_rt)

if not _JS_RUNTIMES:
    logger.warning(
        "No supported JavaScript runtime found on PATH (deno / node / bun). "
        "YouTube downloads will fail with 'n challenge solving failed'. "
        "Install Node.js >= 20 or Deno >= 2.0."
    )


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


class VideoDownloader:
    """Download YouTube (and other) videos to local cache, with ffmpeg metadata support."""

    def __init__(
        self,
        cache_dir: str = "data/cache/videos",
        max_resolution: str = "480",
        timeout: int = 240,
        cookies_file: Optional[str] = None,
        download_timeout_sec: Optional[int] = None,
        max_download_attempts: int = 5,
        retry_backoff_sec: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_resolution = max_resolution
        self.timeout = timeout
        self.cookies_file = cookies_file
        # Wall-clock budget for the whole download phase. None = unlimited
        # (legacy behaviour, used by the scaffold). The RL/eval watch_video
        # path sets this so a throttled/hanging YouTube download self-aborts
        # instead of stalling the rollout (and, via the validation barrier,
        # idling every GPU).
        self.download_timeout_sec = download_timeout_sec
        self.max_download_attempts = max(1, max_download_attempts)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)
        self._ffmpeg_ok = check_ffmpeg()
        if not self._ffmpeg_ok:
            logger.warning("ffmpeg not found — duration probe and merge-download will be limited")

    def download(self, url: str, force: bool = False) -> Optional[str]:
        """
        Download video to cache. Returns local file path, or None on failure.
        Uses extract_info + prepare_filename for reliable path resolution.
        """
        video_id = self._extract_video_id(url)
        cache_path = self.cache_dir / f"{video_id}.mp4"

        if cache_path.exists() and not force:
            logger.info("Cache hit: %s", cache_path)
            return str(cache_path)

        # Wall-clock guard: abort from a progress hook once the budget is blown.
        # This is the only reliable way to bound a throttled-trickle download —
        # raising inside a progress hook propagates out of ydl.download() and
        # lets the worker thread return (freeing its rate-limit token), which
        # ray.cancel() cannot do for an already-running threaded-actor task.
        deadline = (
            time.monotonic() + self.download_timeout_sec
            if self.download_timeout_sec
            else None
        )
        last_error: Optional[BaseException] = None

        for attempt in range(self.max_download_attempts):
            ydl_opts = self._build_ydl_opts(video_id, attempt=attempt)
            if deadline is not None:
                def _deadline_hook(d, _deadline=deadline):
                    if time.monotonic() > _deadline:
                        raise _DownloadTimeout(
                            f"download exceeded {self.download_timeout_sec}s wall-clock budget"
                        )

                hooks = list(ydl_opts.get("progress_hooks") or [])
                hooks.append(_deadline_hook)
                ydl_opts["progress_hooks"] = hooks

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(
                        "Downloading (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_download_attempts,
                        url,
                    )

                    # Phase 1: extract info WITHOUT downloading to verify formats.
                    # A new YoutubeDL instance on each outer attempt also forces
                    # YouTube to issue fresh signed media URLs.
                    info = ydl.extract_info(url, download=False)
                    if info is None:
                        raise yt_dlp.utils.DownloadError("extract_info returned None")

                    formats = info.get("formats") or []
                    video_formats = [
                        f for f in formats
                        if f.get("vcodec", "none") != "none"
                        or f.get("acodec", "none") != "none"
                    ]
                    if not video_formats:
                        raise yt_dlp.utils.DownloadError(
                            "No video/audio formats available; the YouTube "
                            "n-parameter challenge may have failed"
                        )

                    # Phase 2 re-extracts immediately before transfer, preventing
                    # stale signed URLs from being reused across attempts.
                    ydl.download([url])

                    downloaded_path = ydl.prepare_filename(info)
                    final_path = self._resolve_downloaded_file(downloaded_path, video_id)
                    if final_path:
                        if Path(final_path) != cache_path:
                            shutil.move(final_path, cache_path)
                        logger.info("Downloaded → %s", cache_path)
                        return str(cache_path)
                    raise yt_dlp.utils.DownloadError(
                        f"Downloaded file could not be resolved for video_id={video_id}"
                    )

            except _DownloadTimeout as exc:
                last_error = exc
                logger.error("Download wall-clock timeout for %s: %s", url, exc)
                self._cleanup_partial_files(video_id)
                break
            except yt_dlp.utils.DownloadError as exc:
                last_error = exc
                if (
                    attempt + 1 >= self.max_download_attempts
                    or not self._is_retryable_download_error(exc)
                ):
                    break
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s; retrying",
                    attempt + 1,
                    self.max_download_attempts,
                    url,
                    str(exc).splitlines()[0][:300],
                )
                self._cleanup_partial_files(video_id)
                self._sleep_before_retry(attempt)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_download_attempts:
                    break
                logger.warning(
                    "Unexpected download error on attempt %d/%d for %s: %s; retrying",
                    attempt + 1,
                    self.max_download_attempts,
                    url,
                    str(exc).splitlines()[0][:300],
                )
                self._cleanup_partial_files(video_id)
                self._sleep_before_retry(attempt)

        self._cleanup_partial_files(video_id)
        if isinstance(last_error, yt_dlp.utils.DownloadError):
            logger.error("DownloadError for %s after %d attempt(s): %s",
                         url, attempt + 1, last_error)
        elif last_error is not None and not isinstance(last_error, _DownloadTimeout):
            logger.error("Unexpected download error for %s after %d attempt(s): %s",
                         url, attempt + 1, last_error)

        return None
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_duration(self, video_path: str) -> Optional[float]:
        """Probe video duration in seconds via ffmpeg. Returns None if unavailable."""
        if not self._ffmpeg_ok:
            return self._get_duration_cv2(video_path)
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except Exception:
            return self._get_duration_cv2(video_path)

    def get_video_bytes(self, path_or_url: str) -> Optional[bytes]:
        """Read video bytes from local path or download from URL first."""
        if path_or_url.startswith(("http://", "https://")):
            path_or_url = self.download(path_or_url)
            if path_or_url is None:
                return None
        try:
            with open(path_or_url, "rb") as f:
                return f.read()
        except OSError as e:
            logger.error("Failed to read video file: %s", e)
            return None

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_ydl_opts(self, video_id: str, attempt: int = 0) -> dict:
        """Build yt-dlp option dict with JS runtime, format fallback, and cookie support."""

        res = self.max_resolution
        # Format string with graceful degradation chain. PROGRESSIVE-FIRST:
        # a progressive stream already muxes audio+video into ONE file, so
        # yt-dlp does NOT invoke ffmpeg to merge — this is what avoids the
        # "Postprocessing: Stream #1:0 -> #0:1 (copy)" merge failures that flood
        # the logs under concurrent downloads. We only fall back to separate
        # video+audio streams (which require an ffmpeg merge) when no
        # progressive stream is available at this resolution.
        #   1st: direct-HTTP progressive mp4 (audio+video, no merge)
        #   2nd: any direct-HTTP progressive stream (audio+video, no merge)
        #   3rd: any progressive stream, including HLS
        #   4th: H.264 mp4 + AAC m4a (old-ffmpeg-compatible merge)
        #   5th: any video + audio (needs merge)
        #   6th: best single stream (last resort)
        # Note: YouTube progressive caps around 360p (itag 18) / sometimes 720p,
        # which is plenty for frame extraction — we trade a bit of resolution
        # for a large drop in download/merge failures.
        if attempt >= 2:
            # web_safari's HLS formats use a different delivery path from the
            # direct Googlevideo URLs that produced the observed intermittent
            # 403s. Keep direct/progressive formats as later fallbacks.
            fmt = (
                f"best[height<={res}][protocol*=m3u8][acodec!=none][vcodec!=none]/"
                f"bestvideo[height<={res}][protocol*=m3u8]+bestaudio/"
                f"best[height<={res}][ext=mp4][acodec!=none][vcodec!=none]/"
                f"bestvideo[height<={res}][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={res}]+bestaudio/"
                f"best[height<={res}]/best"
            )
        else:
            fmt = (
                f"best[height<={res}][ext=mp4][protocol^=http][acodec!=none][vcodec!=none]/"
                f"best[height<={res}][protocol^=http][acodec!=none][vcodec!=none]/"
                f"best[height<={res}][acodec!=none][vcodec!=none]/"
                f"bestvideo[height<={res}][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={res}]+bestaudio/"
                f"best[height<={res}]/best"
            )

        opts: dict = {
            "format": fmt,
            "outtmpl": str(self.cache_dir / f"{video_id}.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": False,           # keep warnings visible for debugging
            "socket_timeout": self.timeout,
            "retries": 5,
            "noprogress": True,
            # force_generic_extractor is deliberately omitted: it bypasses the
            # YouTube-specific extractor.
        }

        # JS runtime — required for YouTube's n-challenge. The yt-dlp Python API
        # takes a dict-of-dicts, e.g. {"node": {}, "deno": {}}.
        if _JS_RUNTIMES:
            opts["js_runtimes"] = {rt: {} for rt in _JS_RUNTIMES}

        # Keep authentication on every retry. Each YoutubeDL instance receives
        # a fresh in-memory snapshot so concurrent workers and yt-dlp's cookie
        # write-back cannot mutate the shared on-disk cookie file.
        cookie_snapshot = cookiefile_snapshot(self.cookies_file, logger=logger)
        if cookie_snapshot is not None:
            opts["cookiefile"] = cookie_snapshot
            use_cookie_player_clients(opts)
        elif attempt >= 2:
            opts["extractor_args"] = {
                "youtube": {"player_client": ["web_safari", "android_vr"]}
            }

        # Without ffmpeg, disable merging of separate video/audio streams.
        if not self._ffmpeg_ok:
            opts["format"] = (
                f"best[height<={res}][ext=mp4][protocol^=http][acodec!=none][vcodec!=none]/"
                f"best[height<={res}][protocol^=http][acodec!=none][vcodec!=none]/"
                f"best[height<={res}][acodec!=none][vcodec!=none]/best"
            )
            opts.pop("merge_output_format", None)

        return opts
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_sec * (2 ** attempt)
        if delay:
            time.sleep(delay)

    def _cleanup_partial_files(self, video_id: str) -> None:
        """Remove only yt-dlp partial metadata/fragments for this video."""
        for pattern in (f"{video_id}*.part*", f"{video_id}*.ytdl"):
            for path in self.cache_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError as exc:
                    logger.debug("Could not remove partial download %s: %s", path, exc)

    @staticmethod
    def _is_retryable_download_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        permanent_markers = (
            "video unavailable",
            "private video",
            "has been removed",
            "not made this video available in your country",
            "members-only",
            "this live event will begin",
            "premieres in",
        )
        return not any(marker in message for marker in permanent_markers)

    def _resolve_downloaded_file(self, prepared_path: str, video_id: str) -> Optional[str]:
        """
        Find the actual downloaded file. yt-dlp's prepare_filename may not
        reflect the final extension after merge, so we also scan the cache dir.
        """
        if Path(prepared_path).exists():
            return prepared_path

        mp4_path = Path(prepared_path).with_suffix(".mp4")
        if mp4_path.exists():
            return str(mp4_path)

        for f in self.cache_dir.iterdir():
            if f.stem == video_id and f.suffix in (".mp4", ".webm", ".mkv"):
                return str(f)

        logger.warning("Could not locate downloaded file for video_id=%s", video_id)
        return None

    @staticmethod
    def _get_duration_cv2(video_path: str) -> Optional[float]:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and count > 0:
                return count / fps
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_video_id(url: str) -> str:
        # Accepts the /shorts/ path in addition to v= and youtu.be/.
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else hashlib.md5(url.encode()).hexdigest()[:11]
