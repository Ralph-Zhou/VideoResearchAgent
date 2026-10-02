"""Tests for ``_parse_timestamps_param`` used by VisualGroundingTool.

This helper is the safety net we added to avoid qwen3_coder's XML parser
falling through to ``eval()`` on array parameters. It runs on the tool side
and must gracefully accept every reasonable model output the agent might
emit.

Note: importing the full ``visual_grounding_tool`` module pulls in verl, ray
and the co-located ``video_agent`` package. We import just the helper to keep
this test portable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_helper():
    """Load only ``_parse_timestamps_param`` without importing verl / ray."""
    src = Path(__file__).resolve().parents[1] / "video_search_sim" / "verl_tools" / "visual_grounding_tool.py"
    # Read the source and extract just the function; evaluating the full
    # module would import `verl` and friends. This is a targeted approach
    # that tolerates additions to the module.
    text = src.read_text(encoding="utf-8")
    # Locate ``def _parse_timestamps_param`` and copy until the next top-level
    # ``def`` or ``class``.
    start = text.index("def _parse_timestamps_param")
    rest = text[start:]
    # End: next line starting with ``def `` or ``class `` at column 0.
    lines = rest.splitlines(keepends=True)
    out_lines = [lines[0]]
    for line in lines[1:]:
        if (line.startswith("def ") or line.startswith("class ")) and not line[0].isspace():
            break
        out_lines.append(line)
    func_src = "".join(out_lines)
    # We need `from typing import Any`.
    wrapper = "from typing import Any\n" + func_src
    spec = importlib.util.spec_from_loader("_vg_helper", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(wrapper, str(src), "exec"), mod.__dict__)  # noqa: S102
    return mod._parse_timestamps_param


parse = _load_helper()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("1.0", [1.0]),
        ("1.0, 2.5, 3.75", [1.0, 2.5, 3.75]),
        ("1,2,3", [1.0, 2.0, 3.0]),
        ("[1.0, 2.5]", [1.0, 2.5]),  # tolerates XML-encoded list literal
        ("1.0,,garbage,2.5", [1.0, 2.5]),  # drops unparseable tokens
        ([1.0, 2.5], [1.0, 2.5]),
        ([1, "2.0", None], [1.0, 2.0]),  # mixed types ok
        (3.14, [3.14]),  # scalar ok
        ("abc", []),
    ],
)
def test_parse_timestamps_param(raw, expected) -> None:
    assert parse(raw) == expected
