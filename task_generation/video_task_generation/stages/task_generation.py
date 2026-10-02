"""Stage 3 — Task generation from VideoEntityGraph + double verification.

Per graph:
    1. LLM designs a candidate task anchored on a specific frame-level fact
       (see TASK_GENERATION_PROMPT).
    2. Text-only search-agent tries to solve the task (search + visit tools).
       If the agent can solve it → reject, the task isn't video-dependent.
    3. Optional self-check: LLM tries to answer from internal knowledge → too easy.
    4. Obfuscate to soften any over-specific clues (minimum edit).
Final accepted task is appended to JSONL with full provenance.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_task_generation.data_structures import VideoEntityGraph
from video_task_generation.prompts import (
    OBFUSCATION_PROMPT,
    SELF_CHECK_CORRECTNESS_PROMPT,
    SELF_CHECK_SYSTEM,
    TASK_GENERATION_PROMPT,
    TEXT_SEARCH_AGENT_SYSTEM,
    TEXT_SEARCH_AGENT_USER,
    obfuscation_language_instruction,
    task_generation_language_instruction,
)
from video_task_generation.stages.seed_generation import detect_language
from video_task_generation.shared.checkpoint import StageCheckpoint
from video_task_generation.shared.jsonl_utils import append_jsonl_to_path, load_jsonl_safe
from video_task_generation.shared.llm_client import call_llm, extract_tag
from video_task_generation.shared.text_search import search_web, visit_url, is_text_search_enabled

logger = logging.getLogger(__name__)


REJECT_FILENAME = "stage3_rejected.jsonl"


# ──────────────────────────────────────────────────────────────────────
# Text-only search agent
# ──────────────────────────────────────────────────────────────────────


_VISIT_RE = re.compile(
    r'<visit\s+url\s*=\s*"(?P<url>[^"]+)"\s*>(?P<goal>.*?)</visit>',
    re.DOTALL | re.IGNORECASE,
)


class TextSearchAgent:
    """Minimal agent loop: search / visit / final_answer. All via LLM tags."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Return (final_answer, trace)."""
        trace: List[Dict[str, Any]] = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TEXT_SEARCH_AGENT_SYSTEM},
            {"role": "user", "content": TEXT_SEARCH_AGENT_USER.format(question=question)},
        ]
        for turn in range(self.max_turns):
            resp = call_llm(messages)
            if not resp:
                trace.append({"turn": turn, "kind": "empty_llm"})
                break
            messages.append({"role": "assistant", "content": resp})

            # First check for final_answer
            final = extract_tag(resp, "final_answer")
            if final is not None:
                trace.append({"turn": turn, "kind": "final_answer", "content": final})
                return final.strip(), trace

            # Then check for search
            query = extract_tag(resp, "search")
            if query:
                results = search_web(query)
                # prune to compact snippet block
                compact = [
                    {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
                    for r in results[:5]
                ]
                trace.append({"turn": turn, "kind": "search", "query": query, "results": compact})
                messages.append({
                    "role": "user",
                    "content": f"Search results for '{query}':\n{_format_search(compact)}",
                })
                continue

            # Then check for visit
            mv = _VISIT_RE.search(resp)
            if mv:
                url, goal = mv.group("url"), mv.group("goal").strip()
                page = visit_url(url, goal)
                trace.append({"turn": turn, "kind": "visit", "url": url, "goal": goal,
                              "ok": bool(page)})
                snippet = page[:3000] if page else "(page unreachable)"
                messages.append({
                    "role": "user",
                    "content": f"Visit summary for {url} (goal: {goal}):\n{snippet}",
                })
                continue

            # No recognised tag — prompt to recover
            trace.append({"turn": turn, "kind": "malformed", "raw": resp[:400]})
            messages.append({
                "role": "user",
                "content": (
                    "Your last message did not contain a valid <search>, <visit> or "
                    "<final_answer> tag. Please follow the protocol exactly."
                ),
            })

        trace.append({"turn": self.max_turns, "kind": "max_turns_reached"})
        return "UNANSWERABLE_WITHOUT_VIDEO", trace


def _format_search(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title','')}\n    {r.get('url','')}\n    {r.get('snippet','')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _check_correctness(question: str, gold: str, pred: str) -> bool:
    if not pred:
        return False
    if pred.strip() == "UNANSWERABLE_WITHOUT_VIDEO":
        return False
    prompt = SELF_CHECK_CORRECTNESS_PROMPT.format(question=question, gold=gold, pred=pred)
    resp = call_llm([{"role": "user", "content": prompt}])
    return "yes" in (resp or "").lower()


def _self_check_too_easy(question: str, gold: str) -> bool:
    resp = call_llm(
        [
            {"role": "system", "content": SELF_CHECK_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    if not resp:
        return False
    if "don't know" in resp.lower() or "不知道" in resp:
        return False
    return _check_correctness(question, gold, resp)


def _obfuscate(
    question: str, answer: str, frame_evidence: str, language: str = "zh",
) -> Tuple[str, str]:
    resp = call_llm([{"role": "user", "content": OBFUSCATION_PROMPT.format(
        question=question, answer=answer, frame_evidence=frame_evidence,
        language_instruction=obfuscation_language_instruction(language),
    )}])
    rq = extract_tag(resp, "refined_question")
    ra = extract_tag(resp, "answer")
    return (rq or question).strip(), (ra or answer).strip()


# ──────────────────────────────────────────────────────────────────────
# Stage orchestrator
# ──────────────────────────────────────────────────────────────────────


class TaskGenerationStage:

    def __init__(
        self,
        workers: int = 4,
        max_agent_turns: int = 10,
        oversampling_factor: int = 2,
        max_qa_retries: int = 5,
        self_check: bool = True,
        require_frame_evidence: bool = True,
        target_tasks: int = 50,
    ):
        self.workers = workers
        self.max_agent_turns = max_agent_turns
        self.oversampling_factor = oversampling_factor
        self.max_qa_retries = max_qa_retries
        self.self_check = self_check
        self.require_frame_evidence = require_frame_evidence
        self.target_tasks = target_tasks

    # ── Candidate task from graph ─────────────────────────────────────

    def _draft_task(
        self, graph: VideoEntityGraph, language: str = "zh",
    ) -> Optional[Dict[str, str]]:
        prompt = TASK_GENERATION_PROMPT.format(
            graph_text=graph.format_for_prompt(),
            language_instruction=task_generation_language_instruction(language),
        )
        for attempt in range(self.max_qa_retries):
            resp = call_llm([{"role": "user", "content": prompt}])
            if not resp:
                continue
            q = extract_tag(resp, "question")
            a = extract_tag(resp, "answer")
            fe = extract_tag(resp, "frame_evidence")
            if q and a:
                if self.require_frame_evidence and not fe:
                    logger.info(
                        "[draft_task] missing frame_evidence tag for seed=%s (attempt %d)",
                        graph.seed_name, attempt + 1,
                    )
                    continue
                # Language compliance — reject if the question drifted away from
                # the requested language. Heuristic: the narration (question) must
                # match the target language.
                detected = detect_language(q)
                if detected != language:
                    logger.info(
                        "[draft_task] language drift for seed=%s (want=%s, got=%s) attempt %d",
                        graph.seed_name, language, detected, attempt + 1,
                    )
                    continue
                used_props_raw = extract_tag(resp, "used_properties") or ""
                used_properties = [
                    line.strip(" \t-•·*")
                    for line in used_props_raw.splitlines()
                    if line.strip(" \t-•·*")
                ]
                return {
                    "question": q.strip(),
                    "answer": a.strip(),
                    "frame_evidence": (fe or "").strip(),
                    "thinking": extract_tag(resp, "thinking") or "",
                    "used_properties": used_properties,
                }
        return None

    # ── Verification pipeline ─────────────────────────────────────────

    def _verify(
        self, task: Dict[str, str],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Returns (accepted, reject_reason, verification_metadata)."""
        meta: Dict[str, Any] = {}

        # (1) text-only search agent — skipped entirely in local-corpus mode,
        # where Serper cannot resolve the corpus's fake URLs anyway. We still
        # log a marker so downstream analysis can tell the two regimes apart.
        if is_text_search_enabled():
            agent = TextSearchAgent(max_turns=self.max_agent_turns)
            agent_answer, trace = agent.answer(task["question"])
            meta["text_agent_answer"] = agent_answer
            meta["text_agent_trace"] = trace
            if _check_correctness(task["question"], task["answer"], agent_answer):
                return False, "text_agent_solved", meta
        else:
            meta["text_agent_answer"] = ""
            meta["text_agent_trace"] = []
            meta["text_agent_skipped"] = "text_search_disabled"

        # (2) self-check: LLM internal knowledge
        if self.self_check:
            if _self_check_too_easy(task["question"], task["answer"]):
                meta["self_check"] = "too_easy"
                return False, "too_easy_by_internal_knowledge", meta
            meta["self_check"] = "ok"

        # (3) must contain frame-based reasoning — agent must have failed
        # AND we captured evidence pointing to a frame detail.
        if self.require_frame_evidence and not task.get("frame_evidence"):
            return False, "no_frame_evidence_cited", meta

        return True, "", meta

    # ── Single graph processor ────────────────────────────────────────

    def _process_graph(self, graph_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        graph = VideoEntityGraph.from_dict(graph_record)
        if graph.depth == 0:
            return None
        tid = threading.current_thread().name
        t0 = time.monotonic()

        # Resolve the language of this graph. Prefer the explicit seed_meta.language
        # written by Stage 1; fall back to detecting on the initial seed name
        # (which is the LLM-fabricated inducement, stored in seed_meta).
        seed_meta = graph_record.get("seed_meta") or {}
        initial_seed = str(seed_meta.get("name") or "").strip()
        language = str(seed_meta.get("language") or "").strip().lower()
        if language not in ("zh", "en"):
            language = detect_language(initial_seed or graph.seed_name)

        logger.info(
            "[task_gen] [%s] START seed=%s depth=%d lang=%s",
            tid, graph.seed_name, graph.depth, language,
        )

        draft = self._draft_task(graph, language=language)
        if not draft:
            logger.info("[task_gen] [%s] draft failed seed=%s", tid, graph.seed_name)
            return {
                "seed": graph.seed_name,
                "initial_seed": initial_seed,
                "language": language,
                "status": "reject",
                "reject_reason": "draft_failed",
                "graph": graph_record,
            }

        accepted, reason, meta = self._verify(draft)
        if not accepted:
            logger.info(
                "[task_gen] [%s] REJECT (%s) seed=%s (%.1fs)",
                tid, reason, graph.seed_name, time.monotonic() - t0,
            )
            return {
                "seed": graph.seed_name,
                "initial_seed": initial_seed,
                "language": language,
                "status": "reject",
                "reject_reason": reason,
                "draft": draft,
                "verification": meta,
                "graph": graph_record,
            }

        # Obfuscate as final polish (language-aware)
        refined_q, refined_a = _obfuscate(
            draft["question"], draft["answer"], draft["frame_evidence"],
            language=language,
        )

        out = {
            "seed": graph.seed_name,               # first REAL hop (per v5 semantics)
            "initial_seed": initial_seed,          # LLM-fabricated inducement only
            "language": language,
            "status": "accept",
            "question": refined_q,
            "answer": refined_a,
            "question_raw": draft["question"],
            "answer_raw": draft["answer"],
            "frame_evidence": draft["frame_evidence"],
            "thinking": draft.get("thinking", ""),
            "used_properties": draft.get("used_properties", []),
            "graph_depth": graph.depth,
            "target_depth": graph_record.get("target_depth"),
            "verification": meta,
            "graph": graph_record,
        }
        logger.info(
            "[task_gen] [%s] ACCEPT seed=%s lang=%s (%.1fs)",
            tid, graph.seed_name, language, time.monotonic() - t0,
        )
        return out

    # ── Public run ────────────────────────────────────────────────────

    def run(
        self,
        graphs_path: Path,
        output_path: Path,
        reject_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reject_path = reject_path or output_path.parent / REJECT_FILENAME

        all_graphs = load_jsonl_safe(graphs_path)
        if not all_graphs:
            raise RuntimeError(f"no graphs found in {graphs_path}")

        existing_accept = load_jsonl_safe(output_path)
        done_keys = {r.get("seed", "") for r in existing_accept}
        existing_reject = load_jsonl_safe(reject_path)
        done_keys |= {r.get("seed", "") for r in existing_reject}

        ckpt = StageCheckpoint(checkpoint_path) if checkpoint_path else None

        pending = []
        for g in all_graphs:
            name = (g.get("entities") or [{}])[0].get("name", "")
            if not name:
                continue
            if name in done_keys:
                continue
            if ckpt is not None and ckpt.is_processed(name):
                continue
            pending.append(g)

        # Oversample cap: only run enough to try hit target
        needed = max(self.target_tasks - len(existing_accept), 0) if self.target_tasks > 0 else 0
        if needed > 0:
            cap = needed * self.oversampling_factor
            if len(pending) > cap:
                random.shuffle(pending)
                pending = pending[:cap]

        logger.info(
            "[Stage 3] %d graphs pending (already accept=%d, target=%d, cap=%s), workers=%d",
            len(pending), len(existing_accept), self.target_tasks,
            (needed * self.oversampling_factor) if needed > 0 else "∞", self.workers,
        )

        results: List[Dict[str, Any]] = []
        reject_count = 0
        accept_count = len(existing_accept)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._process_graph, g): g for g in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                g = futures[fut]
                name = (g.get("entities") or [{}])[0].get("name", "")
                try:
                    rec = fut.result()
                except Exception as exc:
                    logger.error("[Stage 3] [%d/%d] error for '%s': %s", i, len(pending), name, exc)
                    if ckpt is not None:
                        ckpt.mark(name, "error")
                    continue

                if rec is None:
                    if ckpt is not None:
                        ckpt.mark(name, "empty")
                    continue

                if rec["status"] == "accept":
                    append_jsonl_to_path(rec, output_path)
                    results.append(rec)
                    accept_count += 1
                    if ckpt is not None:
                        ckpt.mark(name, "accept")
                    logger.info(
                        "[Stage 3] [%d/%d] ACCEPT seed=%s (total accept=%d/%d)",
                        i, len(pending), name, accept_count, self.target_tasks,
                    )
                    if self.target_tasks > 0 and accept_count >= self.target_tasks:
                        logger.info(
                            "[Stage 3] target %d reached, stopping early", self.target_tasks,
                        )
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                else:
                    append_jsonl_to_path(rec, reject_path)
                    reject_count += 1
                    if ckpt is not None:
                        ckpt.mark(name, f"reject:{rec.get('reject_reason','unknown')}")

        logger.info(
            "[Stage 3] done — %d new accepted (total=%d), %d rejected. output=%s rejects=%s",
            len(results), accept_count, reject_count, output_path, reject_path,
        )
        return results
