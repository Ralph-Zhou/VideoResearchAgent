"""Schema contract tests.

These lock down the public surface of ``VideoSearchResponse`` so that future
refactors cannot silently drift away from what the SFT trajectories expect.
"""

from __future__ import annotations

import pytest

from video_search_sim.common.schemas import (
    SearchHit,
    VideoRecord,
    VideoSearchRequest,
    VideoSearchResponse,
)


def test_search_hit_required_fields() -> None:
    hit = SearchHit(
        fake_url="https://www.youtube.com/watch?v=abcDEFghi12",
        title="t",
        snippet="s",
        duration=10.0,
        thumbnail="https://www.youtube.com/vi/abcDEFghi12/hqdefault.jpg",
        score=0.9,
    )
    d = hit.model_dump()
    assert set(d.keys()) == {"fake_url", "title", "snippet", "duration", "thumbnail", "score"}


def test_search_hit_snippet_cap() -> None:
    long = "x" * 2000
    hit = SearchHit(
        fake_url="https://www.youtube.com/watch?v=abcDEFghi12",
        title="t",
        snippet=long,
        duration=0.0,
        thumbnail="https://www.youtube.com/vi/abcDEFghi12/hqdefault.jpg",
        score=0.0,
    )
    # validator caps at 512; we only assert <= 512 so the specific cap can evolve.
    assert len(hit.snippet) <= 512


def test_video_search_request_bounds() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        VideoSearchRequest(query="", topk=1)
    with pytest.raises(Exception):  # noqa: B017
        VideoSearchRequest(query="ok", topk=0)
    with pytest.raises(Exception):  # noqa: B017
        VideoSearchRequest(query="ok", topk=9999)


def test_video_search_response_roundtrip() -> None:
    resp = VideoSearchResponse(query="dogs", results=[], latency_ms=1.23)
    j = resp.model_dump_json()
    restored = VideoSearchResponse.model_validate_json(j)
    assert restored.query == "dogs"
    assert restored.results == []
    assert restored.latency_ms == pytest.approx(1.23)


def test_video_record_defaults() -> None:
    r = VideoRecord(
        fake_url="https://www.youtube.com/watch?v=abcDEFghi12",
        video_id="abcDEFghi12",
        local_path="/tmp/a.mp4",
    )
    assert r.title == ""
    assert r.duration == 0.0
    assert r.tags == []
    assert r.scene_splits == []
