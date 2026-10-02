"""``visual_grounding`` — Grounding DINO over local video frames, returned as images.

This tool wraps ``VisualGroundingTool.detect_in_video``: given a
video URL (fake or real) and a text prompt (``"red car. wall."``), it
extracts frames at specified timestamps, runs open-vocabulary detection,
and returns **annotated frames + textual detection list**.

Local-first URL resolution
--------------------------
Same bridge as ``WatchVideoTool``: fake URL → ``url_to_path.json`` lookup.
If the URL is not in the bridge and ``enable_remote_fallback=True``, we
delegate to ``VideoDownloader``.

XML parser alignment (qwen3_coder)
----------------------------------
- ``url`` → string (required)
- ``prompt`` → string (required) — Grounding DINO accepts ``"obj1. obj2."``
  syntax natively; we pass it through unchanged.
- ``timestamps`` → **string** (comma-separated floats), NOT array. We parse
  on this side; see the module-level docstring on ``watch_video_tool`` for
  the rationale (qwen3_coder's XML parser routes arrays through ``eval()``,
  which is fragile).
- ``max_frames`` → integer.

The tool's output is:

- ``ToolResponse.image`` — list of PIL frames (annotated if detections are
  non-empty; raw frames if no detections, so the agent can at least see
  what it looked at).
- ``ToolResponse.text`` — JSON envelope ``{"url", "prompt", "frames": [
    {"timestamp", "detections": [{"label", "score", "box": [x1,y1,x2,y2]}]}]}``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from ._common import (
    decode_base64_to_pil,
    dumps_tool_text,
    ensure_video_agent_on_path,
    get_corpus_bridge,
    init_execution_pool,
    resolve_youtube_cookies,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIDEO_SEARCH_SIM_LOG_LEVEL", "WARNING"))


def _parse_timestamps_param(raw: Any) -> list[float]:
    """Turn the model-emitted ``timestamps`` parameter into ``list[float]``.

    Accepts:
      - an already-parsed ``list[float]`` / ``list[int]`` (defensive),
      - a comma- or space-separated string ``"1.0, 2.5, 3.0"``,
      - an empty value → ``[]``.

    Non-parseable entries are silently dropped. We do NOT fall through to
    ``eval()``: see the parser alignment section in the module docstring.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        out = []
        for x in raw:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, str):
        out = []
        for tok in raw.replace("[", " ").replace("]", " ").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                continue
        return out
    try:
        return [float(raw)]
    except (TypeError, ValueError):
        return []


class VisualGroundingTool(BaseTool):
    """Open-vocabulary visual grounding on local video frames.

    Expected ``config`` keys:

    ``corpus_dir`` (str, **required** unless ``enable_local_lookup=False``)
    ``enable_local_lookup`` (bool, default True)
    ``enable_remote_fallback`` (bool, default True)
    ``model_id`` (str, default ``"IDEA-Research/grounding-dino-tiny"``)
    ``device`` (str, default ``"auto"``) — ``cuda`` / ``cpu`` / ``auto``
    ``box_threshold`` (float, default 0.35)
    ``text_threshold`` (float, default 0.25)
    ``jpeg_quality`` (int, default 85)
    ``max_frames_cap`` (int, default 8)
    ``downloader_cache_dir`` (str, default ``data/cache/videos``)
    ``num_workers`` (int, default 4)
    ``rate_limit`` (int, default 4)
    ``enable_global_rate_limit`` (bool, default True)
    ``type`` (str, default ``"native"``)

    Grounding DINO forward pass is GPU-heavy; we keep the ``rate_limit``
    conservative (4 by default) so it doesn't compete with the rollout
    vLLM worker for GPU. Raise once you know the GPU footprint on H20.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        ensure_video_agent_on_path()
        try:
            from video_agent.tools.video_download import VideoDownloader  # noqa: PLC0415
            from video_agent.tools.visual_grounding import (  # noqa: PLC0415
                VisualGroundingTool as _UserGrounding,
            )
        except ImportError as e:  # pragma: no cover - environmental
            raise RuntimeError(
                "VisualGroundingTool requires the video_agent package; set VSS_VIDEO_AGENT_PATH."
            ) from e

        self.enable_local_lookup = bool(config.get("enable_local_lookup", True))
        self.enable_remote_fallback = bool(config.get("enable_remote_fallback", True))
        self.max_frames_cap = int(config.get("max_frames_cap", 8))

        if self.enable_local_lookup:
            corpus_dir = config.get("corpus_dir")
            if not corpus_dir:
                raise ValueError("VisualGroundingTool: config.corpus_dir is required when enable_local_lookup=True")
            self._bridge = get_corpus_bridge(corpus_dir)
        else:
            self._bridge = None

        self._impl = _UserGrounding(
            model_id=config.get("model_id", "IDEA-Research/grounding-dino-tiny"),
            box_threshold=float(config.get("box_threshold", 0.35)),
            text_threshold=float(config.get("text_threshold", 0.25)),
            device=config.get("device", "auto"),
            jpeg_quality=int(config.get("jpeg_quality", 85)),
        )
        self._downloader = VideoDownloader(
            cache_dir=config.get("downloader_cache_dir", "data/cache/videos"),
            # Validated YouTube cookies (Netscape-format check + anonymous
            # fallback); keeps the remote eval path consistent with watch_video.
            cookies_file=resolve_youtube_cookies(config.get("youtube_cookies_file")),
        )

        self.num_workers = int(config.get("num_workers", 4))
        self.rate_limit = int(config.get("rate_limit", 4))
        self.enable_global_rate_limit = bool(config.get("enable_global_rate_limit", True))
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            limiter_name="vss-visual-grounding-rate-limiter",
        )
        logger.info(
            "VisualGroundingTool ready (model=%s cap=%d rate_limit=%d)",
            self._impl.model_id,
            self.max_frames_cap,
            self.rate_limit,
        )

    # ------------------------------------------------------------- verl API

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"calls": 0}
        return instance_id, ToolResponse()

    def _resolve_local_path(self, url: str) -> str | None:
        if self._bridge is not None:
            path = self._bridge.local_path(url)
            if path and Path(path).is_file():
                return path
        if self.enable_remote_fallback:
            return self._downloader.download(url)
        return None

    def _do_ground(
        self,
        url: str,
        prompt: str,
        timestamps: list[float],
        max_frames: int,
    ) -> dict[str, Any]:
        local_path = self._resolve_local_path(url)
        if local_path is None:
            return {"error": f"cannot resolve local video for url={url}"}

        try:
            results = self._impl.detect_in_video(
                video_path=local_path,
                prompt=prompt,
                timestamps=timestamps or None,
                max_frames=max_frames,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Grounding DINO error: %s", e)
            return {"error": repr(e)}

        frames_json = []
        annotated_b64 = []
        for r in results:
            frames_json.append(
                {
                    "timestamp": r.timestamp,
                    "detections": [{"label": d.label, "score": d.score, "box": list(d.box)} for d in r.detections],
                }
            )
            # Fallback to raw-frame bytes is not available here — user code
            # returns empty string when there are no detections. In that case
            # we skip the image slot for that frame so the agent still sees
            # the textual detection summary.
            if r.annotated_b64:
                annotated_b64.append(r.annotated_b64)

        return {
            "url": url,
            "prompt": prompt,
            "frames": frames_json,
            "annotated_b64": annotated_b64,
        }

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        # Accept both `video_path` (SFT-aligned) and `url` (legacy) parameter names
        url = parameters.get("video_path") or parameters.get("url", "")
        prompt = parameters.get("prompt", "")
        if not isinstance(url, str) or not url.strip():
            return ToolResponse(text="Error: `video_path` must be a non-empty string."), 0.0, {}
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResponse(text="Error: `prompt` must be a non-empty string."), 0.0, {}

        timestamps = _parse_timestamps_param(parameters.get("timestamps"))
        max_frames = int(parameters.get("max_frames", self.max_frames_cap))
        max_frames = max(1, min(max_frames, self.max_frames_cap))

        try:
            ref = self.execution_pool.execute.remote(self._do_ground, url, prompt, timestamps, max_frames)
            body = await ref
        except Exception as e:  # noqa: BLE001
            logger.warning("VisualGroundingTool pool error: %s", e)
            body = {"error": repr(e)}

        if body.get("error"):
            return (
                ToolResponse(text=dumps_tool_text({"url": url, "error": body["error"]})),
                0.0,
                {"error": body["error"]},
            )

        images = []
        for b in body.get("annotated_b64", []):
            img = decode_base64_to_pil(b)
            if img is not None:
                images.append(img)

        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["calls"] += 1

        payload = {
            "url": body["url"],
            "prompt": body["prompt"],
            "frames": body["frames"],
        }
        tool_text = dumps_tool_text(payload)

        metrics = {
            "n_frames": len(body["frames"]),
            "n_annotated": len(images),
        }
        return (
            ToolResponse(text=tool_text, image=images if images else None),
            0.0,
            metrics,
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
