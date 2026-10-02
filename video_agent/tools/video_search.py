"""YouTube video search using yt-dlp."""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import yt_dlp

from video_agent.tools.ytdlp_utils import cookiefile_snapshot

logger = logging.getLogger(__name__)


@dataclass
class VideoSearchResult:
    video_id: str
    url: str
    title: str
    description: str
    duration: Optional[float]
    view_count: Optional[int]
    upload_date: Optional[str]
    channel: Optional[str]
    thumbnail_url: Optional[str]


class YouTubeSearchTool:
    """Search YouTube videos via yt-dlp and return structured results."""

    def __init__(
        self,
        max_results: int = 10,
        cookies_file: Optional[str] = None,
        max_attempts: int = 5,
        retry_backoff_sec: float = 1.0,
    ):
        self.max_results = max_results
        self.cookies_file = cookies_file
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)

    def search(self, query: str, max_results: Optional[int] = None) -> List[VideoSearchResult]:
        n = max_results or self.max_results
        info = None
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_attempts):
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": f"ytsearch{n}",
                "socket_timeout": 30,
                "retries": 5,
            }
            if attempt == 0:
                cookie_snapshot = cookiefile_snapshot(self.cookies_file, logger=logger)
                if cookie_snapshot is not None:
                    ydl_opts["cookiefile"] = cookie_snapshot
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    logger.warning(
                        "YouTube search attempt %d/%d failed for query '%s': %s; retrying",
                        attempt + 1,
                        self.max_attempts,
                        query,
                        str(exc).splitlines()[0][:300],
                    )
                    delay = self.retry_backoff_sec * (2 ** attempt)
                    if delay:
                        time.sleep(delay)
        else:
            logger.error(
                "YouTube search failed for query '%s' after %d attempts: %s",
                query,
                self.max_attempts,
                last_error,
            )
            return []

        results = []
        for entry in (info or {}).get("entries", [])[:n]:
            if entry is None:
                continue
            vid = entry.get("id", "")
            results.append(VideoSearchResult(
                video_id=vid,
                url=f"https://www.youtube.com/watch?v={vid}",
                title=entry.get("title", ""),
                description=entry.get("description") or "",
                duration=entry.get("duration"),
                view_count=entry.get("view_count"),
                upload_date=entry.get("upload_date"),
                channel=entry.get("channel") or entry.get("uploader", ""),
                thumbnail_url=entry.get("thumbnail"),
            ))
        logger.info("YouTube search '%s' returned %d results", query, len(results))
        return results
