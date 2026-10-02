"""Text-modality search + visit helpers (wraps Serper + Jina-style visit).

* `search_web(query)`  -> snippet-level search results (Serper)
* `visit_url(url, goal)` -> fetch the full page, then summarize with an LLM
  (with enable_visit=False this degrades to returning the snippet only)

Behaviour matches video_agent/tools/web_search.py and uses the same
SERPER_API_URL / SERPER_API_KEY. The visit path talks to an OpenAI-compatible
endpoint directly via httpx plus the openai client.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Reuse WebSearchTool from video_agent/tools/web_search.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from video_agent.tools.web_search import WebSearchTool  # type: ignore
except ImportError as e:  # pragma: no cover
    logger.warning("Failed to import WebSearchTool from video_agent: %s", e)
    WebSearchTool = None  # type: ignore

from video_task_generation.shared.llm_client import call_llm


# ──────────────────────────────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────────────────────────────

_SEARCH_TOOL: Optional[Any] = None
_VISIT_CFG: Dict[str, Any] = {
    "enable_visit": True,
    "visit_timeout": 30,
    "max_results": 5,
    "provider": "serper",
}

# "disabled" => search_web / visit_url short-circuit to empty without any
# network calls. This is what the local-corpus pipeline uses when its
# fake YouTube URLs can never appear in Serper results, so visiting them
# would only waste time and pollute the synthesis trace.
_MODE: str = "keep_serper"


def configure_search(
    provider: str = "serper",
    max_results: int = 5,
    enable_visit: bool = True,
    visit_timeout: int = 30,
    mode: str = "keep_serper",
) -> None:
    """Set global search/visit configuration.

    ``mode`` controls whether text-modality search is active at all:

    * ``keep_serper`` (default) — original behaviour, uses Serper+Jina.
    * ``disabled`` — :func:`search_web` and :func:`visit_url` short-circuit
      to empty results. Use this for the local-corpus pipeline where the
      ground-truth URLs are fake and not on the open web.
    """
    global _SEARCH_TOOL, _MODE
    mode = (mode or "keep_serper").lower().strip()
    if mode not in ("keep_serper", "disabled"):
        raise ValueError(f"Unknown text_search mode: {mode!r}")
    _MODE = mode

    _VISIT_CFG["provider"] = provider
    _VISIT_CFG["max_results"] = max_results
    _VISIT_CFG["enable_visit"] = enable_visit
    _VISIT_CFG["visit_timeout"] = visit_timeout

    if mode == "disabled":
        _SEARCH_TOOL = None
        logger.info("[text_search] mode=disabled — search_web/visit_url will return empty")
        return

    if WebSearchTool is None:
        logger.error(
            "WebSearchTool unavailable — text search will return empty results. "
            "Check that video_agent/tools/web_search.py is importable."
        )
        _SEARCH_TOOL = None
        return

    _SEARCH_TOOL = WebSearchTool(
        provider=provider,
        serper_api_key=os.getenv("SERPER_API_KEY"),
        serper_api_url=os.getenv("SERPER_API_URL"),
        serper_timeout=visit_timeout,
    )
    logger.info(
        "Search configured: provider=%s max_results=%d enable_visit=%s mode=%s",
        provider, max_results, enable_visit, mode,
    )


def is_text_search_enabled() -> bool:
    """Whether text-modality search/visit is active in this run."""
    return _MODE != "disabled"


def _ensure_tool():
    if _MODE == "disabled":
        return None
    if _SEARCH_TOOL is None:
        configure_search(**_VISIT_CFG)  # type: ignore[arg-type]
    return _SEARCH_TOOL


# ──────────────────────────────────────────────────────────────────────
# Public APIs
# ──────────────────────────────────────────────────────────────────────


def search_web(query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
    """Return list of ``{title, url, snippet}`` dicts."""
    if _MODE == "disabled":
        return []
    tool = _ensure_tool()
    if tool is None:
        return []
    n = max_results or _VISIT_CFG["max_results"]
    try:
        return tool.search(query, max_results=n)
    except Exception as exc:  # pragma: no cover
        logger.error("search_web failed for '%s': %s", query, exc)
        return []


# Webpage retrieval and LLM extraction.

_EXTRACTOR_PROMPT = (
    "Please process the following webpage content and user goal to extract relevant information:\n\n"
    "## Webpage Content\n{content}\n\n"
    "## User Goal\n{goal}\n\n"
    "## Task Guidelines\n"
    "1. Locate the specific sections/data directly related to the user's goal.\n"
    "2. Extract the most relevant information — quote the full original context when helpful.\n"
    "3. Summarise into a concise paragraph.\n\n"
    "Output format:\n"
    "EVIDENCE:\n<evidence>\n\n"
    "SUMMARY:\n<summary>\n"
)


def _jina_readpage(url: str, timeout: int = 30) -> str:
    """Fetch page content directly without forwarding API credentials."""
    # Fallback: direct GET (will often fail for JS-heavy pages; best-effort only)
    try:
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
    except Exception as exc:
        logger.warning("[visit] direct GET error for %s: %s", url, exc)
    return ""


def visit_url(
    url: str,
    goal: str,
    max_retries: int = 2,
    timeout: Optional[int] = None,
) -> str:
    """Fetch a page, then ask LLM to extract evidence related to ``goal``.

    Returns a formatted string; empty string on failure.
    """
    if _MODE == "disabled":
        return ""
    if not _VISIT_CFG.get("enable_visit", True):
        return ""
    t = timeout or _VISIT_CFG.get("visit_timeout", 30)
    content = ""
    for attempt in range(1, max_retries + 1):
        content = _jina_readpage(url, timeout=t)
        if content:
            break
        time.sleep(1)
    if not content:
        return ""

    content = content[:80_000]  # cap to avoid blowing up LLM context
    prompt = _EXTRACTOR_PROMPT.format(content=content, goal=goal)
    summary = call_llm([{"role": "user", "content": prompt}])
    if not summary:
        return ""
    return (
        f"Useful information from {url} (goal: {goal}):\n\n{summary}"
    )
