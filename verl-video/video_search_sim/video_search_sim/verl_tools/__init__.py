"""verl-side tool adapters for the video-research agent.

Exposes five tools via ``verl.tools.base_tool.BaseTool``:

- ``VideoSearchTool`` — hits the local FastAPI ``/video_search`` backend.
- ``WebSearchTool`` — Serper text search.
- ``WatchVideoTool`` — local-first frame + transcript retrieval keyed off
  the fake YouTube URLs produced by ``vss-build-corpus``.
- ``VisualGroundingTool`` — Grounding DINO on local frames.

.. important::

    Importing a specific tool class (e.g. ``from video_search_sim.verl_tools
    import VideoSearchTool``) requires ``verl`` on the path, since every
    tool extends ``BaseTool``. This package uses **lazy resolution** via
    ``__getattr__`` so that ``from video_search_sim.verl_tools._common import
    ...`` — which is useful in pure-Python unit tests — works even in envs
    without verl installed. Only attribute access of a concrete tool
    symbol triggers the verl import.
"""

from __future__ import annotations

__all__ = [
    "VideoSearchTool",
    "VisualGroundingTool",
    "WatchVideoTool",
    "WebSearchTool",
]


_LAZY_MAP = {
    "VideoSearchTool": ("video_search_sim.verl_tools.video_search_tool", "VideoSearchTool"),
    "WebSearchTool": ("video_search_sim.verl_tools.web_search_tool", "WebSearchTool"),
    "WatchVideoTool": ("video_search_sim.verl_tools.watch_video_tool", "WatchVideoTool"),
    "VisualGroundingTool": (
        "video_search_sim.verl_tools.visual_grounding_tool",
        "VisualGroundingTool",
    ),
}


def __getattr__(name: str):
    """Lazily import the concrete tool class only when requested."""
    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'video_search_sim.verl_tools' has no attribute {name!r}")
    mod_path, sym = target
    import importlib

    module = importlib.import_module(mod_path)
    return getattr(module, sym)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_MAP))
