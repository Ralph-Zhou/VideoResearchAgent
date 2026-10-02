"""Stage 2 tail — text-search driven entity enrichment.

For each real hop in a freshly-built ``VideoEntityGraph``, we run one Serper
search + a couple of page visits, then ask an LLM to distill 5-8 factual
``properties`` and 3-6 named ``relations`` about the entity.  These are
written back onto ``VideoEntity.properties`` / ``.relations`` so Stage 3's
task prompt can leverage them to narrow the target entity down to a single
unambiguous match (mimicking the entity_dict style used by text-only
deep-research task synthesis).

Design notes:
    * Enrichment is **best-effort**: any failure (empty search, visit errors,
      JSON parse failure) degrades silently to empty ``properties`` / ``relations``
      and never blocks the pipeline.
    * Per hop we cap the network budget to ``1 search + N visits`` (default N=2).
    * Visual / frame-level information is explicitly excluded from the prompt
      to prevent leaking answers that must come from watching the video.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from video_task_generation.data_structures import VideoEntity, VideoEntityGraph
from video_task_generation.prompts import (
    ENTITY_ENRICHMENT_SYSTEM_EN,
    ENTITY_ENRICHMENT_SYSTEM_ZH,
    ENTITY_ENRICHMENT_USER,
)
from video_task_generation.shared.llm_client import call_llm
from video_task_generation.shared.text_search import (
    is_text_search_enabled,
    search_web,
    visit_url,
)

logger = logging.getLogger(__name__)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _enrichment_system(language: str) -> str:
    lang = (language or "zh").lower()
    if lang.startswith("en"):
        return ENTITY_ENRICHMENT_SYSTEM_EN
    return ENTITY_ENRICHMENT_SYSTEM_ZH


def _parse_enrichment_json(resp: str) -> Tuple[List[str], Dict[str, str]]:
    """Extract (properties, relations) from an LLM response; lenient parsing."""
    if not resp:
        return [], {}

    candidates: List[str] = []
    m = _JSON_FENCE_RE.search(resp)
    if m:
        candidates.append(m.group(1))
    m = _BARE_JSON_RE.search(resp)
    if m:
        candidates.append(m.group(1))

    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        props_raw = data.get("properties", [])
        rels_raw = data.get("relations", {})
        props: List[str] = []
        if isinstance(props_raw, list):
            for p in props_raw:
                if isinstance(p, str) and p.strip():
                    props.append(p.strip())
        rels: Dict[str, str] = {}
        if isinstance(rels_raw, dict):
            for k, v in rels_raw.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    rels[k.strip()] = v.strip()
        if props or rels:
            return props, rels
    return [], {}


def _collect_snippets(
    entity_name: str,
    search_topk: int,
    visit_topk: int,
) -> str:
    """Run one Serper search and up to ``visit_topk`` page visits, concatenated."""
    # Local-corpus mode: no usable text-modality search. Skip the network
    # round-trips entirely and let the LLM enrich from the video summary alone.
    if not is_text_search_enabled():
        return "(text search disabled — local-corpus mode)"

    try:
        results = search_web(entity_name, max_results=search_topk)
    except Exception as exc:
        logger.warning("[enrich] search failed for '%s': %s", entity_name, exc)
        results = []

    blocks: List[str] = []
    for i, r in enumerate(results[:search_topk], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        blocks.append(f"[{i}] {title}\n    {url}\n    {snippet}")

    # Page visits for the top-N results (best-effort).
    for r in results[:visit_topk]:
        url = r.get("url", "")
        if not url:
            continue
        try:
            page_summary = visit_url(
                url,
                goal=(
                    f"Extract factual background properties about '{entity_name}' "
                    "— official ownership, location, founding year, core specs, "
                    "affiliations and named relations to other entities."
                ),
            )
        except Exception as exc:
            logger.warning("[enrich] visit failed for %s: %s", url, exc)
            page_summary = ""
        if page_summary:
            blocks.append(f"[visit {url}]\n{page_summary[:1500]}")

    return "\n\n".join(blocks) if blocks else "(no search results)"


def enrich_entity(
    name: str,
    video_summary: Optional[str],
    language: str = "zh",
    search_topk: int = 3,
    visit_topk: int = 2,
) -> Tuple[List[str], Dict[str, str]]:
    """Best-effort enrichment for a single entity name.

    Returns ``(properties, relations)``.  On any failure returns ``([], {})``.
    """
    if not name:
        return [], {}

    snippets = _collect_snippets(name, search_topk=search_topk, visit_topk=visit_topk)
    user_prompt = ENTITY_ENRICHMENT_USER.format(
        entity_name=name,
        video_summary=(video_summary or "(no video summary)")[:1500],
        snippets=snippets[:8000],
    )
    messages = [
        {"role": "system", "content": _enrichment_system(language)},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = call_llm(messages)
    except Exception as exc:
        logger.warning("[enrich] LLM failed for '%s': %s", name, exc)
        return [], {}

    props, rels = _parse_enrichment_json(resp or "")
    if not props and not rels:
        logger.info("[enrich] empty parse for '%s' (resp_len=%d)", name, len(resp or ""))
    return props, rels


def enrich_graph_in_place(
    graph: VideoEntityGraph,
    language: str = "zh",
    search_topk: int = 3,
    visit_topk: int = 2,
    workers: int = 4,
) -> None:
    """Populate ``properties`` / ``relations`` on every real hop of ``graph``.

    Runs enrichment concurrently across hops; failures are swallowed.
    """
    targets: List[Tuple[int, VideoEntity]] = list(enumerate(graph.entities))
    if not targets:
        return

    def _one(idx: int, ent: VideoEntity) -> Tuple[int, List[str], Dict[str, str]]:
        props, rels = enrich_entity(
            name=ent.name,
            video_summary=ent.video_summary,
            language=language,
            search_topk=search_topk,
            visit_topk=visit_topk,
        )
        return idx, props, rels

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets)))) as pool:
        futures = [pool.submit(_one, idx, ent) for idx, ent in targets]
        for fut in as_completed(futures):
            try:
                idx, props, rels = fut.result()
            except Exception as exc:
                logger.warning("[enrich] worker error: %s", exc)
                continue
            graph.entities[idx].properties = props
            graph.entities[idx].relations = rels
