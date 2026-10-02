"""Tool registry: OpenAI function-calling schemas + execution dispatch.

Wraps existing tool classes (YouTubeSearchTool, WebSearchTool, VideoDownloader,
FrameExtractor, TranscriptFetcher) into a unified interface that the
VideoResearchAgent can use via the OpenAI tool-calling API.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from video_agent.config import AppConfig
from video_agent.tools.video_search import YouTubeSearchTool
from video_agent.tools.web_search import WebSearchTool
from video_agent.tools.video_download import VideoDownloader
from video_agent.tools.frame_extractor import FrameExtractor, FrameData
from video_agent.tools.transcript import TranscriptFetcher
from video_agent.tools.visual_grounding import VisualGroundingTool

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_youtube",
            "description": (
                "Search YouTube for videos matching a query. "
                "Returns a list of video metadata (title, url, duration, description). "
                "Use targeted keywords for best results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "YouTube search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for general information. "
                "Useful for finding context about a video, channel, or topic "
                "before searching YouTube directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "watch_video",
            "description": (
                "Download a video and extract frames + transcript for visual analysis. "
                "Two modes:\n"
                "- 'sparse': Extract N evenly-spaced frames from the entire video "
                "  plus the full transcript. Use this first to get an overview.\n"
                "- 'dense': Extract frames at higher FPS from a specific time window "
                "  [start_time, end_time]. Use this after sparse viewing to zoom into "
                "  a region of interest.\n"
                "The extracted frames will be shown to you as images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube video URL or local file path",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sparse", "dense"],
                        "description": "Viewing mode: 'sparse' for overview, 'dense' for time-window zoom",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "(dense mode only) Start time in seconds",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "(dense mode only) End time in seconds",
                    },
                    "n_frames": {
                        "type": "integer",
                        "description": "(sparse mode) Number of evenly-spaced frames to extract",
                        "default": 16,
                    },
                    "fps": {
                        "type": "number",
                        "description": "(dense mode) Frames per second to extract",
                        "default": 1.0,
                    },
                },
                "required": ["url", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visual_grounding",
            "description": (
                "Detect and locate specific objects in video frames using text descriptions "
                "(powered by Grounding DINO). Use this when you need to precisely identify, "
                "count, or locate objects, text, logos, landmarks, or other visual elements "
                "in a video.\n"
                "Input a video path and a text prompt describing what to find (separate "
                "multiple objects with periods, e.g. 'a red car. a person. a blue sign.').\n"
                "Returns: bounding box coordinates + confidence for each detection, plus "
                "annotated frames with boxes drawn for visual verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to video file (local path or YouTube URL that was previously watched)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Object descriptions to detect, separated by periods. "
                            "E.g. 'a red ribbon. a sports car. text on a sign.'"
                        ),
                    },
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Specific timestamps (seconds) to analyze. "
                            "If omitted, frames are sampled uniformly from the video."
                        ),
                    },
                    "max_frames": {
                        "type": "integer",
                        "description": "Maximum number of frames to analyze (default: 4)",
                        "default": 4,
                    },
                },
                "required": ["video_path", "prompt"],
            },
        },
    },
]


@dataclass
class ToolResult:
    """Result from a tool execution."""
    text: str
    frames: List[FrameData] = field(default_factory=list)
    is_terminal: bool = False
    terminal_data: Optional[Dict] = None


class ToolRegistry:
    """Manages tool schemas and dispatches tool calls to underlying implementations."""

    def __init__(self, cfg: AppConfig):
        cookies = cfg.search.youtube_cookies
        self.yt_search = YouTubeSearchTool(
            max_results=cfg.search.max_results,
            cookies_file=cookies,
        )
        self.web_search = WebSearchTool(
            provider=cfg.search.text_provider or "serper",
            serper_api_key=None,
            serper_api_url=None,
        )
        self.downloader = VideoDownloader(
            cache_dir=cfg.cache.videos_dir,
            max_resolution=cfg.watcher.max_video_resolution,
            timeout=cfg.watcher.download_timeout,
            cookies_file=cookies,
        )
        self.frame_extractor = FrameExtractor()
        self.transcript_fetcher = TranscriptFetcher(
            cache_dir=cfg.cache.transcripts_dir,
            cookies_file=cookies,
        )
        self.visual_grounding = VisualGroundingTool(
            model_id=cfg.grounding.model_id,
            box_threshold=cfg.grounding.box_threshold,
            text_threshold=cfg.grounding.text_threshold,
            device=cfg.grounding.device,
        )
        self.cfg = cfg
        self._dispatchers: Dict[str, Callable] = {
            "search_youtube": self._exec_search_youtube,
            "web_search": self._exec_web_search,
            "watch_video": self._exec_watch_video,
            "visual_grounding": self._exec_visual_grounding,
        }

        enabled = cfg.agent.enabled_tools
        if enabled is not None:
            keep = set(enabled)
            self._dispatchers = {k: v for k, v in self._dispatchers.items() if k in keep}
            self._tool_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in keep]
        else:
            self._tool_schemas = list(TOOL_SCHEMAS)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return self._tool_schemas

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        handler = self._dispatchers.get(name)
        if handler is None:
            return ToolResult(text=f"Error: unknown tool '{name}'")
        try:
            return handler(**arguments)
        except Exception as e:
            logger.error("Tool '%s' execution error: %s", name, e, exc_info=True)
            return ToolResult(text=f"Error executing {name}: {e}")

    # ── Tool implementations ──

    def _exec_search_youtube(self, query: str, max_results: int = 10) -> ToolResult:
        results = self.yt_search.search(query, max_results=max_results)
        if not results:
            return ToolResult(text="No YouTube results found for this query.")
        items = []
        for r in results:
            item = {
                "video_id": r.video_id,
                "title": r.title,
                "url": r.url,
                "duration_seconds": r.duration,
                "channel": r.channel or "",
                "description": (r.description[:200] + "...") if r.description and len(r.description) > 200 else (r.description or ""),
            }
            items.append(item)
        text = json.dumps(items, ensure_ascii=False, indent=2)
        return ToolResult(text=f"Found {len(items)} videos:\n{text}")

    def _exec_web_search(self, query: str, max_results: int = 5) -> ToolResult:
        results = self.web_search.search(query, max_results=max_results)
        if not results:
            return ToolResult(text="No web search results found.")
        text = json.dumps(results, ensure_ascii=False, indent=2)
        return ToolResult(text=f"Found {len(results)} web results:\n{text}")

    def _exec_watch_video(
        self,
        url: str,
        mode: str = "sparse",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        n_frames: int = 16,
        fps: float = 1.0,
    ) -> ToolResult:
        is_local = not url.startswith(("http://", "https://"))

        if is_local:
            from pathlib import Path as _P
            local_path = _P(url)
            if not local_path.exists():
                return ToolResult(text=f"Local video not found: {url}")
            video_path = str(local_path)
        else:
            video_path = self.downloader.download(url)
            if not video_path:
                return ToolResult(text=f"Failed to download video: {url}")

        duration = self.downloader.get_duration(video_path)
        duration_str = f"{duration:.1f}s" if duration else "unknown"

        # Fetch transcript (skip for local files without a YouTube URL)
        if is_local:
            transcript_segs = []
        else:
            transcript_segs = self.transcript_fetcher.fetch(url)
        transcript_text = self.transcript_fetcher.format_for_prompt(
            transcript_segs, max_chars=self.cfg.watcher.transcript_max_chars
        )

        if mode == "sparse":
            frames = self.frame_extractor.extract_sparse(video_path, n_frames=n_frames)
            frame_summary = ", ".join(f"{f.timestamp:.1f}s" for f in frames)

            text_parts = [
                f"Video downloaded (duration: {duration_str}). Extracted {len(frames)} sparse frames at timestamps: [{frame_summary}].",
            ]
            if transcript_text:
                text_parts.append(f"\nTranscript:\n{transcript_text[:8000]}")
            else:
                text_parts.append("\nNo transcript available.")
            return ToolResult(text="\n".join(text_parts), frames=frames)

        elif mode == "dense":
            if start_time is None or end_time is None:
                return ToolResult(
                    text="Error: dense mode requires start_time and end_time parameters."
                )
            frames = self.frame_extractor.extract_dense(
                video_path, start=start_time, end=end_time,
                fps=fps, max_frames=self.cfg.watcher.max_dense_frames_per_window,
            )
            frame_summary = ", ".join(f"{f.timestamp:.1f}s" for f in frames)
            n_dense = len(frames)

            relevant_segs = [
                s for s in transcript_segs
                if s.start >= start_time - 2 and s.end <= end_time + 2
            ]
            seg_text = "\n".join(
                f"[{s.start:.1f}s] {s.text}" for s in relevant_segs
            ) if relevant_segs else "No transcript for this segment."

            text = (
                f"Dense extraction [{start_time:.1f}s - {end_time:.1f}s]: "
                f"{n_dense} frames at timestamps: [{frame_summary}].\n"
                f"\nTranscript segment:\n{seg_text}"
            )
            return ToolResult(text=text, frames=frames)

        return ToolResult(text=f"Error: unknown mode '{mode}'. Use 'sparse' or 'dense'.")

    def _exec_visual_grounding(
        self,
        video_path: str,
        prompt: str,
        timestamps: Optional[List[float]] = None,
        max_frames: int = 4,
    ) -> ToolResult:
        is_local = not video_path.startswith(("http://", "https://"))
        if is_local:
            from pathlib import Path as _P
            if not _P(video_path).exists():
                return ToolResult(text=f"Video not found: {video_path}")
            local_path = video_path
        else:
            local_path = self.downloader.download(video_path)
            if not local_path:
                return ToolResult(text=f"Failed to download video: {video_path}")

        frame_results = self.visual_grounding.detect_in_video(
            video_path=local_path,
            prompt=prompt,
            timestamps=timestamps,
            max_frames=max_frames,
        )

        text_parts = []
        annotated_frames: List[FrameData] = []

        for fr in frame_results:
            if fr.detections:
                det_lines = []
                for d in fr.detections:
                    x1, y1, x2, y2 = d.box
                    det_lines.append(
                        f"  - {d.label} (conf={d.score:.2f}) at [{x1},{y1},{x2},{y2}]"
                    )
                text_parts.append(
                    f"Frame @{fr.timestamp:.1f}s — {len(fr.detections)} detections:\n"
                    + "\n".join(det_lines)
                )
                if fr.annotated_b64:
                    annotated_frames.append(FrameData(
                        timestamp=fr.timestamp,
                        image_b64=fr.annotated_b64,
                    ))
            else:
                text_parts.append(f"Frame @{fr.timestamp:.1f}s — no detections.")

        summary = (
            f"Visual grounding for \"{prompt}\" across {len(frame_results)} frames:\n\n"
            + "\n\n".join(text_parts)
        )
        return ToolResult(text=summary, frames=annotated_frames)

