"""Transcript fetching: yt-dlp subtitles with caching."""

import json
import re
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from video_agent.tools.ytdlp_utils import cookiefile_snapshot, use_cookie_player_clients

logger = logging.getLogger(__name__)

DEFAULT_SUBTITLE_LANGS = [
    "en", "en-US", "en-GB", "en-orig",
    "zh", "zh-Hans", "zh-CN", "zh-Hant", "zh-TW",
    "ja", "ja-JP",
]


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptFetcher:
    """Fetch available video subtitles with yt-dlp."""

    def __init__(
        self,
        cache_dir: str = "data/cache/transcripts",
        cookies_file: Optional[str] = None,
        subtitle_languages: Optional[List[str]] = None,
        max_ytdlp_attempts: int = 5,
        retry_backoff_sec: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file
        self.subtitle_languages = subtitle_languages or DEFAULT_SUBTITLE_LANGS
        self.max_ytdlp_attempts = max(1, max_ytdlp_attempts)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)

    def fetch(self, url: str) -> List[TranscriptSegment]:
        video_id = self._extract_id(url)

        cached = self._load_cache(video_id)
        if cached is not None:
            logger.info("Transcript cache hit: %s", video_id)
            return cached

        segments = self._fetch_ytdlp(url)
        if segments:
            self._save_cache(video_id, segments)
            return segments

        logger.warning("No transcript available for %s", video_id)
        return []

    def format_for_prompt(self, segments: List[TranscriptSegment],
                          max_chars: int = 25000) -> str:
        lines = [f"[{s.start:.1f}s - {s.end:.1f}s] {s.text}" for s in segments]
        text = "\n".join(lines)
        return text[:max_chars]

    # ── Provider implementations ──


    def _fetch_ytdlp(self, url: str) -> List[TranscriptSegment]:
        import yt_dlp
        import tempfile
        import os

        last_error: Optional[BaseException] = None
        for attempt in range(self.max_ytdlp_attempts):
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": self.subtitle_languages,
                    "subtitlesformat": "json3/vtt/srv3/best",
                    "skip_download": True,
                    # Subtitle extraction does not need a playable media format.
                    # This avoids false "Requested format is not available"
                    # failures when captions are still accessible.
                    "ignore_no_formats_error": True,
                    "outtmpl": os.path.join(tmpdir, "%(id)s"),
                    "socket_timeout": 30,
                    "retries": 5,
                }
                if attempt == 0:
                    cookie_snapshot = cookiefile_snapshot(self.cookies_file, logger=logger)
                    if cookie_snapshot is not None:
                        ydl_opts["cookiefile"] = cookie_snapshot
                        use_cookie_player_clients(ydl_opts)
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self.max_ytdlp_attempts:
                        logger.info(
                            "yt-dlp subtitle attempt %d/%d failed for %s (%s: %s); retrying",
                            attempt + 1,
                            self.max_ytdlp_attempts,
                            url,
                            type(exc).__name__,
                            str(exc).splitlines()[0][:200],
                        )
                        delay = self.retry_backoff_sec * (2 ** attempt)
                        if delay:
                            time.sleep(delay)
                        continue
                    break

                files = sorted(Path(tmpdir).iterdir())
                # Prefer json3 first (most accurate timing), then srv3, then vtt.
                preferred = (
                    [f for f in files if f.suffix == ".json3"]
                    + [f for f in files if f.suffix == ".srv3"]
                    + [f for f in files if f.suffix == ".vtt"]
                )
                if not preferred and files:
                    logger.info("yt-dlp produced unexpected files for %s: %s",
                                url, [f.name for f in files])

                for f in preferred:
                    try:
                        if f.suffix in (".json3", ".srv3"):
                            segments = self._parse_json3(f.read_text(encoding="utf-8"))
                        elif f.suffix == ".vtt":
                            segments = self._parse_vtt(f.read_text(encoding="utf-8"))
                        else:
                            continue
                        if segments:
                            logger.info("yt-dlp subtitles (%s): got %d segments from %s",
                                        f.suffix.lstrip("."), len(segments), f.name)
                            return segments
                    except Exception as exc:
                        logger.info("yt-dlp parse error on %s: %s", f.name, exc)
                if not preferred:
                    logger.info("yt-dlp: no subtitles available for %s", url)
                return []

        if last_error is not None:
            logger.info(
                "yt-dlp subtitle download failed for %s after %d attempts (%s: %s)",
                url,
                self.max_ytdlp_attempts,
                type(last_error).__name__,
                str(last_error).splitlines()[0][:200],
            )
        return []

    @staticmethod
    def _parse_json3(text: str) -> List[TranscriptSegment]:
        data = json.loads(text)
        events = data.get("events", [])
        segments: List[TranscriptSegment] = []
        for ev in events:
            segs = ev.get("segs", [])
            line = "".join(s.get("utf8", "") for s in segs).strip()
            if not line or line == "\n":
                continue
            start_ms = ev.get("tStartMs", 0)
            dur_ms = ev.get("dDurationMs", 0)
            segments.append(TranscriptSegment(
                start=start_ms / 1000.0,
                end=(start_ms + dur_ms) / 1000.0,
                text=line,
            ))
        return segments

    @staticmethod
    def _parse_vtt(text: str) -> List[TranscriptSegment]:
        # Minimal WEBVTT parser: cue header line "HH:MM:SS.mmm --> HH:MM:SS.mmm"
        # then payload lines until a blank line. Strips inline tags like <c>.
        cue_re = re.compile(
            r"(\d+):(\d+):(\d+)\.(\d+)\s+-->\s+(\d+):(\d+):(\d+)\.(\d+)"
        )
        tag_re = re.compile(r"<[^>]+>")
        segments: List[TranscriptSegment] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = cue_re.search(lines[i])
            if not m:
                i += 1
                continue
            sh, sm, ss, sms, eh, em, es, ems = m.groups()
            start = int(sh) * 3600 + int(sm) * 60 + int(ss) + int(sms) / 1000.0
            end = int(eh) * 3600 + int(em) * 60 + int(es) + int(ems) / 1000.0
            i += 1
            payload: List[str] = []
            while i < len(lines) and lines[i].strip():
                payload.append(tag_re.sub("", lines[i]).strip())
                i += 1
            line = " ".join(p for p in payload if p)
            if line:
                segments.append(TranscriptSegment(start=start, end=end, text=line))
        # YouTube auto-captions duplicate adjacent cues; deduplicate consecutive
        # identical text to keep transcripts compact.
        deduped: List[TranscriptSegment] = []
        for seg in segments:
            if deduped and deduped[-1].text == seg.text:
                deduped[-1] = TranscriptSegment(
                    start=deduped[-1].start, end=seg.end, text=seg.text
                )
            else:
                deduped.append(seg)
        return deduped


    # ── Cache ──

    def _load_cache(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        p = self.cache_dir / f"{video_id}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return [TranscriptSegment(**s) for s in data]
        except Exception:
            return None

    def _save_cache(self, video_id: str, segments: List[TranscriptSegment]):
        p = self.cache_dir / f"{video_id}.json"
        p.write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _extract_id(url: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else url
