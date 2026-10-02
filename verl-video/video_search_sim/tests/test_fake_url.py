"""Unit tests for fake URL / ID generation."""

from __future__ import annotations

import re

from video_search_sim.common.fake_url import (
    fake_url_from_path,
    fake_video_id,
    parse_fake_url,
    thumbnail_url,
)


class TestFakeVideoId:
    def test_deterministic(self) -> None:
        a = fake_video_id("/foo/bar/video.mp4")
        b = fake_video_id("/foo/bar/video.mp4")
        assert a == b

    def test_different_paths_yield_different_ids(self) -> None:
        a = fake_video_id("/path/a.mp4")
        b = fake_video_id("/path/b.mp4")
        assert a != b

    def test_length_and_alphabet(self) -> None:
        ids = [fake_video_id(f"/x/{i}.mp4") for i in range(100)]
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}$")
        assert all(pattern.match(vid) for vid in ids), "IDs must match YouTube alphabet"
        assert len({*ids}) == 100, "IDs should be ~unique for distinct inputs"


class TestFakeUrl:
    def test_url_shape(self) -> None:
        url = fake_url_from_path("/data/foo.mp4")
        assert url.startswith("https://www.youtube.com/watch?v=")
        assert len(url) == len("https://www.youtube.com/watch?v=") + 11

    def test_parse_roundtrip(self) -> None:
        url = fake_url_from_path("/data/foo.mp4")
        vid = parse_fake_url(url)
        assert vid is not None
        assert url.endswith(vid)

    def test_parse_rejects_bogus(self) -> None:
        assert parse_fake_url("https://youtu.be/short") is None
        assert parse_fake_url("https://www.youtube.com/watch?v=TOOSHORT") is None
        assert parse_fake_url("random text") is None


class TestThumbnail:
    def test_thumbnail_url(self) -> None:
        vid = fake_video_id("/any/path.mp4")
        thumb = thumbnail_url(vid)
        assert thumb.startswith("https://www.youtube.com/vi/")
        assert thumb.endswith("/hqdefault.jpg")
        assert vid in thumb
