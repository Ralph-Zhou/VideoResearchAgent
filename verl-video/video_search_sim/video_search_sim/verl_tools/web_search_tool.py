"""``web_search`` — real Serper-backed text search, kept as verl ``BaseTool``.

Text search does *not* need a local simulation (Serper's 300 QPS headroom is
already enough for RL rollouts, per the design doc). We therefore keep this
tool as a thin adapter over the ``video_agent``
``video_agent.tools.web_search.WebSearchTool``:

- **provider**: official ``serper``; configure ``SERPER_API_KEY``.
- **concurrency**: every request goes through the shared Ray-backed pool in
  ``_common.py`` so rollout-side parallelism is bounded (name-spaced limiter
  ``vss-web-search-rate-limiter`` so it does not starve ``video_search``).
- **serialisation**: responses are rendered through
  ``dumps_tool_text(...)`` → identical JSON envelope used by every other
  tool to keep SFT/RL tokenisation aligned.

XML parser alignment (qwen3_coder)
----------------------------------
The tool schema defined in ``configs/tool_config.yaml`` must list parameters
with string/integer/boolean types only — the ``Qwen3XMLToolParser`` dispatches
on those names. ``max_results`` is declared as ``integer`` here, and
validated again in ``execute`` against a hard ceiling so a misbehaving agent
can't starve the downstream API.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from ._common import (
    dumps_tool_text,
    ensure_video_agent_on_path,
    init_execution_pool,
    truncate_text,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIDEO_SEARCH_SIM_LOG_LEVEL", "WARNING"))


def _format_tool_text(query: str, results: list[dict[str, Any]]) -> str:
    """Render web search results as a JSON envelope.

    The field names are kept short and identical to the ``video_search``
    envelope (``url``, ``title``, ``snippet``) so the model learns one
    consistent output schema across the two text-first search paths.

    Snippets are truncated to 400 chars to keep the tool response under the
    typical 1-2 KB budget per turn, which empirically is where Qwen3.5's
    XML-style tool follow-up stays coherent.
    """
    payload = {
        "query": query,
        "results": [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": truncate_text(r.get("snippet", "") or "", 400)[0],
            }
            for r in results
        ],
    }
    return dumps_tool_text(payload)


class WebSearchTool(BaseTool):
    """Serper-backed web search.

    Expected ``config`` keys (all optional):

    ``provider`` (str, default ``"serper"``)
        Only ``serper`` is supported.
    ``serper_api_key`` (str)
        Overrides ``SERPER_API_KEY`` env var.
    ``serper_api_url`` (str)
        Overrides ``SERPER_API_URL`` env var / library default.
    ``serper_api_format`` (str, ``"official"``)
        Uses public Serper with X-API-KEY authentication.
    ``max_results_cap`` (int, default 10)
        Hard ceiling on ``max_results`` regardless of what the agent asks.
    ``timeout`` (int seconds, default 60)
    ``num_workers`` (int, default 32)
    ``rate_limit`` (int, default 32)
    ``enable_global_rate_limit`` (bool, default True)
    ``type`` (str, default ``"native"``)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # Import the shared implementation lazily, keeping module import cheap.
        ensure_video_agent_on_path()
        try:
            from video_agent.tools.web_search import WebSearchTool as _UserWebSearch  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - environmental
            raise RuntimeError(
                "WebSearchTool requires the video_agent package on sys.path; set "
                "VSS_VIDEO_AGENT_PATH to the repository root."
            ) from e

        self.provider = config.get("provider", "serper")
        self.max_results_cap = int(config.get("max_results_cap", 10))
        self.timeout = int(config.get("timeout", 60))
        self._impl = _UserWebSearch(
            provider=self.provider,
            serper_api_key=config.get("serper_api_key"),
            serper_api_url=config.get("serper_api_url"),
            serper_api_format=config.get("serper_api_format"),
            serper_timeout=self.timeout,
        )

        self.num_workers = int(config.get("num_workers", 32))
        self.rate_limit = int(config.get("rate_limit", 32))
        self.enable_global_rate_limit = bool(config.get("enable_global_rate_limit", True))
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            limiter_name="vss-web-search-rate-limiter",
        )
        logger.info(
            "WebSearchTool ready (provider=%s cap=%d rate_limit=%d)",
            self.provider,
            self.max_results_cap,
            self.rate_limit,
        )

    # ------------------------------------------------------------- verl API

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"queries": [], "num_calls": 0}
        return instance_id, ToolResponse()

    def _do_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        try:
            return self._impl.search(query, max_results=max_results) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("WebSearchTool backend error: %s", e)
            return []

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        query = parameters.get("query", "")
        if not isinstance(query, str) or not query.strip():
            msg = "Error: `query` must be a non-empty string."
            return ToolResponse(text=msg), 0.0, {"error": msg}

        max_results = int(parameters.get("max_results", 5))
        max_results = max(1, min(max_results, self.max_results_cap))

        try:
            ref = self.execution_pool.execute.remote(self._do_search, query, max_results)
            results = await ref
        except Exception as e:  # noqa: BLE001
            logger.warning("WebSearchTool pool error: %s", e)
            results = []

        if instance_id in self._instance_dict:
            rec = self._instance_dict[instance_id]
            rec["queries"].append(query)
            rec["num_calls"] += 1

        tool_text = _format_tool_text(query, results)
        metrics = {
            "num_results": len(results),
            "provider": self.provider,
        }
        return ToolResponse(text=tool_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
