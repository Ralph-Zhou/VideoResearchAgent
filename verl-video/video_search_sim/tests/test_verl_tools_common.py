"""Tests for the verl-agnostic parts of ``verl_tools/_common.py``.

The full module requires ray + verl to be installed in the runtime env, but
these helpers are pure Python / Pillow / env var manipulation and should
round-trip cleanly in any test env with ``video_search_sim[dev]`` extras.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _encode_jpeg(color: tuple[int, int, int] = (255, 0, 0)) -> str:
    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_dumps_tool_text_is_single_line_json() -> None:
    from video_search_sim.verl_tools._common import dumps_tool_text

    payload = {"query": "中文 q", "results": [{"url": "https://x/y", "title": "t"}]}
    text = dumps_tool_text(payload)

    assert "\n" not in text, "tool response text must be a single line to keep token count stable"
    parsed = json.loads(text)
    assert parsed == payload
    # ensure_ascii=False so CJK queries stay verbatim (same tokens as SFT)
    assert "中文" in text


def test_decode_base64_to_pil_returns_pil_image() -> None:
    from video_search_sim.verl_tools._common import decode_base64_to_pil

    img = decode_base64_to_pil(_encode_jpeg())
    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.size == (4, 4)
    assert img.mode == "RGB"


def test_decode_base64_frames_drops_bad_entries() -> None:
    from video_search_sim.verl_tools._common import decode_base64_frames

    good = _encode_jpeg()
    frames = decode_base64_frames([good, "", "not-base64"])
    # Bad entries are silently dropped so the rollout loop doesn't choke.
    assert len(frames) == 1
    assert isinstance(frames[0], Image.Image)


def test_truncate_text_returns_tuple() -> None:
    from video_search_sim.verl_tools._common import truncate_text

    short = "hi"
    full, truncated = truncate_text(short, 100)
    assert (full, truncated) == ("hi", False)

    long = "x" * 50
    trimmed, truncated = truncate_text(long, 10)
    assert truncated is True
    assert trimmed.startswith("xxxxxxxxxx")
    assert "truncated" in trimmed


def test_ensure_video_agent_on_path_noop_when_path_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the env points nowhere valid we must not crash — callers fail gracefully."""
    import video_search_sim.verl_tools._common as common

    monkeypatch.setenv("VSS_VIDEO_AGENT_PATH", str(tmp_path / "does_not_exist"))
    # reset the one-shot flag so the test exercises the real code path
    common._video_agent_injected = False

    result = common.ensure_video_agent_on_path()
    assert result is None, "missing path should be flagged via return None, not via exception"


def test_ensure_video_agent_on_path_injects_existing_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    import video_search_sim.verl_tools._common as common

    (tmp_path / "video_agent").mkdir()
    monkeypatch.setenv("VSS_VIDEO_AGENT_PATH", str(tmp_path))
    common._video_agent_injected = False

    path = common.ensure_video_agent_on_path()
    assert path == str(tmp_path)
    assert str(tmp_path) in sys.path

    # Idempotency: second call should not double-insert.
    n_before = sys.path.count(str(tmp_path))
    common.ensure_video_agent_on_path()
    assert sys.path.count(str(tmp_path)) == n_before

    # Cleanup sys.path so we don't leak into later tests.
    while str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
