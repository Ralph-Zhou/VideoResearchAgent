"""Local cache management for videos and transcripts."""

import json
from pathlib import Path
from typing import Optional, List


class CacheManager:
    def __init__(self, videos_dir: str, transcripts_dir: str, enabled: bool = True):
        self.videos_dir = Path(videos_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.enabled = enabled
        if enabled:
            self.videos_dir.mkdir(parents=True, exist_ok=True)
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def get_video_path(self, video_id: str) -> Optional[str]:
        if not self.enabled:
            return None
        for ext in (".mp4", ".webm", ".mkv"):
            p = self.videos_dir / f"{video_id}{ext}"
            if p.exists():
                return str(p)
        return None

    def get_transcript(self, video_id: str) -> Optional[List[dict]]:
        if not self.enabled:
            return None
        p = self.transcripts_dir / f"{video_id}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_transcript(self, video_id: str, segments: List[dict]):
        if not self.enabled:
            return
        p = self.transcripts_dir / f"{video_id}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
