"""Stage 4 — Search-agent-driven difficulty enhancement.

Per accepted Stage 3 task:

  Round r = 0 .. max_rounds-1:
      1. Run the text-only search agent (reuse `TextSearchAgent` from Stage 3)
         → (predicted_answer, trace).
      2. Ask the LLM auditor for a verdict using the (question, gold, trace):
            * pass_task_is_hard       → accept as-is.
            * reject_wrong_answer     → drop (gold likely wrong).
            * reject_junk_entity      → drop (unfixable anchor).
            * too_easy_rewrite        → pull over-specific clues and rewrite
                                         the question; loop back to step 1
                                         (keeping the answer unchanged).
  If after ``max_rounds`` the verdict is still too_easy_rewrite → enhance_fail.

Outputs:
    - Accepted tasks  → ``stage4_tasks.jsonl`` (superset of Stage 3 record with
      extra ``stage4`` block).
    - Rejected tasks  → ``stage4_rejected.jsonl``.

Design notes:
    * The answer is NEVER modified in this stage (all mechanical refinement
      operates on question wording).  Changing gold answers is out of scope
      per the current user spec.
    * All prompts are built in the task's own language (zh / en), reusing
      the language tag written by Stage 3.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_task_generation.prompts import (
    STAGE4_REWRITE_SYSTEM,
    STAGE4_REWRITE_USER,
    STAGE4_VERIFY_SYSTEM,
    STAGE4_VERIFY_USER,
)
from video_task_generation.shared.checkpoint import StageCheckpoint
from video_task_generation.shared.jsonl_utils import append_jsonl_to_path, load_jsonl_safe
from video_task_generation.shared.llm_client import call_llm, extract_tag
from video_task_generation.shared.text_search import is_text_search_enabled
from video_task_generation.stages.task_generation import TextSearchAgent

logger = logging.getLogger(__name__)


REJECT_FILENAME = "stage4_rejected.jsonl"


_ALLOWED_VERDICTS = {
    "pass_task_is_hard",
    "too_easy_rewrite",
    "reject_wrong_answer",
    "reject_junk_entity",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _abridge_trace(trace: List[Dict[str, Any]], max_events: int = 6) -> str:
    """Compact textual summary of a TextSearchAgent trace."""
    if not trace:
        return "(empty trace)"
    lines: List[str] = []
    for ev in trace[:max_events]:
        kind = ev.get("kind", "?")
        if kind == "search":
            q = ev.get("query", "")
            results = ev.get("results", []) or []
            lines.append(f"- search({q!r}) → {len(results)} results")
            for r in results[:3]:
                lines.append(
                    f"    • {r.get('title','')} — {r.get('url','')}"
                )
                snip = (r.get("snippet") or "").strip().replace("\n", " ")
                if snip:
                    lines.append(f"      {snip[:200]}")
        elif kind == "visit":
            lines.append(
                f"- visit {ev.get('url','')} goal={ev.get('goal','')!r} "
                f"ok={ev.get('ok', False)}"
            )
        elif kind == "final_answer":
            lines.append(f"- final_answer: {str(ev.get('content',''))[:200]}")
        elif kind == "malformed":
            lines.append("- malformed turn")
        elif kind == "empty_llm":
            lines.append("- empty LLM reply")
        elif kind == "max_turns_reached":
            lines.append("- max_turns_reached")
        else:
            lines.append(f"- {kind}")
    omitted = max(0, len(trace) - max_events)
    if omitted:
        lines.append(f"... ({omitted} more events omitted)")
    return "\n".join(lines)


def _parse_clues(raw: str) -> List[str]:
    """Extract the bullet list inside a <over_specific_clues> block."""
    if not raw:
        return []
    clues: List[str] = []
    for line in raw.splitlines():
        s = line.strip(" \t-•·*")
        if s:
            clues.append(s)
    return clues


def _collect_properties(graph_record: Dict[str, Any]) -> List[str]:
    props: List[str] = []
    for ent in graph_record.get("entities", []) or []:
        for p in ent.get("properties", []) or []:
            if isinstance(p, str) and p.strip():
                props.append(p.strip())
    return props


# ──────────────────────────────────────────────────────────────────────
# Single-task enhancer
# ──────────────────────────────────────────────────────────────────────


class DifficultyEnhancer:

    def __init__(self, max_rounds: int = 2, max_agent_turns: int = 8):
        self.max_rounds = max(1, int(max_rounds))
        self.max_agent_turns = int(max_agent_turns)

    # ── Verify ────────────────────────────────────────────────────────

    def _verify(
        self,
        question: str,
        answer: str,
        frame_evidence: str,
    ) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """Return (verdict, reason, over_specific_clues, attempt_meta)."""
        # In local-corpus mode the text agent has no Serper access (and even
        # if it did, the URLs it knows about cannot match the corpus's fake
        # IDs). We bypass the agent loop and let the LLM auditor decide on
        # difficulty purely from (question, gold, frame_evidence) — matching
        # the disabled-agent semantics in Stage 3.
        if is_text_search_enabled():
            agent = TextSearchAgent(max_turns=self.max_agent_turns)
            agent_answer, trace = agent.answer(question)
        else:
            agent_answer = ""
            trace = []

        user_prompt = STAGE4_VERIFY_USER.format(
            question=question,
            answer=answer,
            frame_evidence=frame_evidence or "(none)",
            agent_answer=agent_answer or "(empty)",
            trace=_abridge_trace(trace),
        )
        resp = call_llm(
            [
                {"role": "system", "content": STAGE4_VERIFY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        verdict = (extract_tag(resp, "verdict") or "").strip().lower()
        reason = (extract_tag(resp, "reason") or "").strip()
        clues = _parse_clues(extract_tag(resp, "over_specific_clues") or "")

        if verdict not in _ALLOWED_VERDICTS:
            logger.info("[stage4.verify] unknown verdict %r, defaulting to pass", verdict)
            verdict = "pass_task_is_hard"

        meta = {
            "agent_answer": agent_answer,
            "agent_trace": trace,
            "raw_verdict_response": resp,
        }
        return verdict, reason, clues, meta

    # ── Rewrite ───────────────────────────────────────────────────────

    def _rewrite(
        self,
        question: str,
        answer: str,
        frame_evidence: str,
        clues: List[str],
        properties: List[str],
    ) -> Optional[str]:
        if not clues:
            clues = ["(auditor did not specify; reduce obvious long-tail names)"]
        user_prompt = STAGE4_REWRITE_USER.format(
            question=question,
            answer=answer,
            frame_evidence=frame_evidence or "(none)",
            clues="\n".join(f"- {c}" for c in clues),
            properties="\n".join(f"- {p}" for p in properties[:12]) or "(none available)",
        )
        resp = call_llm(
            [
                {"role": "system", "content": STAGE4_REWRITE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        refined = extract_tag(resp, "refined_question")
        if not refined:
            return None
        refined = refined.strip()
        return refined or None

    # ── Public entry ──────────────────────────────────────────────────

    def enhance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the verify/rewrite loop on a Stage 3 accepted task.

        The input dict is expected to contain at least ``question``, ``answer``,
        ``frame_evidence``, and ``graph``.  The returned record is a copy of
        ``task`` with an additional ``stage4`` block and ``status`` in
        ``{accept, reject_wrong_answer, reject_junk_entity, enhance_fail}``.
        """
        base_question = (task.get("question") or "").strip()
        answer = (task.get("answer") or "").strip()
        frame_evidence = (task.get("frame_evidence") or "").strip()
        properties = _collect_properties(task.get("graph") or {})

        rounds: List[Dict[str, Any]] = []
        question_current = base_question
        status = "enhance_fail"
        final_verdict = ""

        for r in range(self.max_rounds):
            verdict, reason, clues, meta = self._verify(
                question_current, answer, frame_evidence,
            )
            rounds.append({
                "round": r,
                "question": question_current,
                "verdict": verdict,
                "reason": reason,
                "over_specific_clues": clues,
                "meta": meta,
            })
            final_verdict = verdict

            if verdict == "pass_task_is_hard":
                status = "accept"
                break
            if verdict in ("reject_wrong_answer", "reject_junk_entity"):
                status = verdict
                break
            # too_easy_rewrite
            if r == self.max_rounds - 1:
                # out of budget → give up
                status = "enhance_fail"
                break
            refined = self._rewrite(
                question_current, answer, frame_evidence, clues, properties,
            )
            if not refined:
                status = "enhance_fail"
                rounds[-1]["rewrite"] = "empty"
                break
            rounds[-1]["rewrite"] = refined
            question_current = refined

        out = dict(task)
        out["stage4"] = {
            "final_verdict": final_verdict,
            "rounds": rounds,
            "question_before_stage4": base_question,
        }
        if status == "accept":
            # Only replace question if at least one rewrite happened
            if question_current != base_question:
                out["question_before_stage4"] = base_question
                out["question"] = question_current
            out["status"] = "accept"
        else:
            out["status"] = status
            out["reject_reason"] = status
        return out


# ──────────────────────────────────────────────────────────────────────
# Stage orchestrator
# ──────────────────────────────────────────────────────────────────────


class DifficultyEnhancementStage:
    """Batch difficulty enhancement over a JSONL of Stage 3 accepted tasks."""

    def __init__(
        self,
        workers: int = 4,
        max_rounds: int = 2,
        max_agent_turns: int = 8,
    ):
        self.workers = workers
        self.max_rounds = max_rounds
        self.max_agent_turns = max_agent_turns

    def _key(self, rec: Dict[str, Any]) -> str:
        return (
            rec.get("seed")
            or rec.get("initial_seed")
            or (rec.get("question") or "")[:80]
        )

    def _enhance_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        enhancer = DifficultyEnhancer(
            max_rounds=self.max_rounds,
            max_agent_turns=self.max_agent_turns,
        )
        return enhancer.enhance(rec)

    def run(
        self,
        input_path: Path,
        output_path: Path,
        reject_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reject_path = reject_path or output_path.parent / REJECT_FILENAME

        records = load_jsonl_safe(input_path)
        if not records:
            logger.info("[Stage 4] no input tasks at %s", input_path)
            return []

        existing_accept = load_jsonl_safe(output_path)
        existing_reject = load_jsonl_safe(reject_path)
        done_keys = {self._key(r) for r in existing_accept} | {
            self._key(r) for r in existing_reject
        }
        ckpt = StageCheckpoint(checkpoint_path) if checkpoint_path else None

        pending = [
            r for r in records
            if self._key(r) not in done_keys
            and not (ckpt and ckpt.is_processed(self._key(r)))
        ]

        logger.info(
            "[Stage 4] %d tasks pending (of %d total; already accept=%d reject=%d)",
            len(pending), len(records), len(existing_accept), len(existing_reject),
        )

        accept_new: List[Dict[str, Any]] = []
        reject_new = 0
        reasons: Dict[str, int] = {}
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._enhance_one, r): r for r in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                orig = futures[fut]
                key = self._key(orig)
                t0 = time.monotonic()
                try:
                    enhanced = fut.result()
                except Exception as exc:
                    logger.error(
                        "[Stage 4] [%d/%d] '%s' error: %s", i, len(pending), key, exc,
                    )
                    if ckpt is not None:
                        ckpt.mark(key, "error")
                    continue

                status = enhanced.get("status", "enhance_fail")
                with lock:
                    if status == "accept":
                        append_jsonl_to_path(enhanced, output_path)
                        accept_new.append(enhanced)
                    else:
                        append_jsonl_to_path(enhanced, reject_path)
                        reject_new += 1
                        reasons[status] = reasons.get(status, 0) + 1
                if ckpt is not None:
                    ckpt.mark(key, status)
                logger.info(
                    "[Stage 4] [%d/%d] %s seed=%s (%.1fs)",
                    i, len(pending), status.upper(), key, time.monotonic() - t0,
                )

        logger.info(
            "[Stage 4] done — %d accepted, %d rejected. reasons=%s output=%s rejects=%s",
            len(accept_new), reject_new, reasons, output_path, reject_path,
        )
        return accept_new
