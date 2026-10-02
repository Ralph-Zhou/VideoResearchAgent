"""Shared utilities: schemas, fake-url generation, config loading."""

from .fake_url import fake_url_from_path, fake_video_id, parse_fake_url, thumbnail_url
from .schemas import SearchHit, VideoRecord, VideoSearchRequest, VideoSearchResponse

__all__ = [
    "SearchHit",
    "VideoRecord",
    "VideoSearchRequest",
    "VideoSearchResponse",
    "fake_url_from_path",
    "fake_video_id",
    "parse_fake_url",
    "thumbnail_url",
]
