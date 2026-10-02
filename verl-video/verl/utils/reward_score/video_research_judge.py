# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Terminal reward: binary semantic answer correctness + 0.1 answer-format credit.

The last <answer>...</answer> span is judged against the reference answer.
Configure JUDGER_MODEL, JUDGER_API_KEY and JUDGER_BASE_URL for an
OpenAI-compatible judge. JUDGER_TIMEOUT, JUDGER_MAX_RETRIES,
JUDGER_TEMPERATURE and JUDGER_FORMAT_WEIGHT are optional overrides.
JUDGER_FAIL_OPEN=1 returns zero answer credit on judge errors; set it to 0
to raise those errors. The same reward is used for training and validation.
"""

from __future__ import annotations

import logging
import os
import re
import string
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIDEO_RESEARCH_REWARD_LOG_LEVEL", "WARNING"))

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_DEFAULT_FORMAT_WEIGHT = 0.1

# OpenAI client is created lazily so importing this module on a worker that
# never actually scores (e.g. a critic-only worker) does not require the
# package to be installed.
_OPENAI_CLIENT: Any = None


# ────────────────────────────────────────────────────────────────────────────
# Trajectory parsing helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, drop punctuation + articles, collapse whitespace.

    Used only for the cheap ``exact_match`` shortcut so we can skip the LLM
    judge when the model already produced an obviously-correct answer.
    """
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract_answer(solution_str: str) -> str | None:
    """Pull the *last* ``<answer>...</answer>`` span (most stable target)."""
    matches = list(_ANSWER_RE.finditer(solution_str or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _coerce_ground_truth(ground_truth) -> dict:
    """Accept either a dict or a raw string and normalise to the dict shape."""
    if isinstance(ground_truth, dict):
        gt_dict = dict(ground_truth)
    elif isinstance(ground_truth, str):
        gt_dict = {"answer": ground_truth}
    else:
        gt_dict = {"answer": str(ground_truth)}
    gt_dict.setdefault("gold_urls", [])
    gt_dict.setdefault("source", "")
    return gt_dict


# ────────────────────────────────────────────────────────────────────────────
# Judge LLM client
# ────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert grader for an open-domain video question-answering "
    "system. Your only job is to decide whether a candidate answer is "
    "semantically equivalent to the gold answer for the given question.\n"
    "Match if the candidate conveys the same key fact (e.g. the same person, "
    "object, count, year, place), even if it is phrased differently, more "
    "verbose, in a different language, or includes extra context that does "
    "not contradict the gold answer.\n"
    "Do NOT match if the candidate is empty, refuses to answer, hedges with "
    "'I don't know', or asserts a different fact.\n"
    "Reply with EXACTLY one token, no explanation: either `MATCH` or `MISMATCH`."
)

JUDGE_USER_TMPL = (
    "Question:\n{question}\n\n"
    "Gold answer:\n{gold}\n\n"
    "Candidate answer:\n{cand}\n\n"
    "Decision:"
)


def _get_openai_client():
    """Construct (lazily) an OpenAI-compatible client from env vars."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("JUDGER_API_KEY", "").strip()
    base_url = os.getenv("JUDGER_BASE_URL", "").strip()
    if not api_key:
        raise RuntimeError("JUDGER_API_KEY env var is required for video_research_judge reward")
    if not base_url:
        raise RuntimeError("JUDGER_BASE_URL env var is required for video_research_judge reward")

    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "openai>=1.0 is not installed; install with `pip install openai` to use the judge reward."
        ) from e

    _OPENAI_CLIENT = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.getenv("JUDGER_TIMEOUT", "30")),
        max_retries=int(os.getenv("JUDGER_MAX_RETRIES", "2")),
    )
    logger.info("video_research_judge: OpenAI client ready (base_url=%s)", base_url)
    return _OPENAI_CLIENT


def _llm_judge(question: str, gold: str, cand: str) -> tuple[float, str]:
    """Return ``(score ∈ {0,1}, raw_decision_str)``.

    Robustness:

    - If the JUDGER_* env vars are missing, raise — we want the operator to
      notice this misconfiguration immediately, not silently get all-zero
      rewards.
    - If the LLM call itself fails (network, 429, timeout), respect
      ``JUDGER_FAIL_OPEN``: default behaviour is judge=0 (so the trajectory
      still gets format-credit but not answer-credit), letting training stay
      alive across transient outages.
    - We parse the response very leniently: any of MATCH/CORRECT/YES/TRUE in
      the first 32 chars (case-insensitive) counts as a match. The system
      prompt asks for one token but models occasionally hedge.
    """
    if not gold:
        # No ground-truth answer to compare against → cannot score the
        # answer, so only format credit can be assigned.
        return 0.0, "no_gold"

    if cand and _normalize(cand) == _normalize(gold):
        # Cheap shortcut: avoid an LLM round-trip when the agent already
        # produced an exact normalised match. This costs nothing for English
        # benchmarks where many answers are short canonical strings.
        return 1.0, "exact_match"

    try:
        client = _get_openai_client()
    except Exception as e:  # noqa: BLE001 - fail loudly on misconfig
        logger.error("judge client init failed: %s", e)
        if os.getenv("JUDGER_FAIL_OPEN", "1") == "1":
            return 0.0, f"client_error:{e}"
        raise

    user_msg = JUDGE_USER_TMPL.format(
        question=(question or "").strip()[:4000],
        gold=str(gold).strip()[:2000],
        cand=str(cand or "").strip()[:2000],
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("JUDGER_MODEL", ""),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=float(os.getenv("JUDGER_TEMPERATURE", "0")),
            max_tokens=8,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("judge call failed: %s", e)
        if os.getenv("JUDGER_FAIL_OPEN", "1") == "1":
            return 0.0, f"call_error:{e}"
        raise

    raw = (resp.choices[0].message.content or "").strip().upper()
    head = raw[:32]
    matched = any(tok in head for tok in ("MATCH", "CORRECT", "YES", "TRUE"))
    # Disambiguate against MISMATCH which contains "MATCH" — handle it
    # explicitly so we don't false-positive.
    if "MISMATCH" in head or "INCORRECT" in head or "WRONG" in head:
        matched = False

    return (1.0 if matched else 0.0), raw or "empty"


# ────────────────────────────────────────────────────────────────────────────
# Public reward fn (verl ``custom_reward_function`` contract)
# ────────────────────────────────────────────────────────────────────────────

def compute_score(
    data_source: str | None,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
) -> dict:
    """Return the terminal reward and its answer/format components."""
    gt = _coerce_ground_truth(ground_truth)
    gold_answer = gt.get("answer", "")
    question = (extra_info or {}).get("question", "")

    format_weight = float((extra_info or {}).get("format_weight", _DEFAULT_FORMAT_WEIGHT))
    if "JUDGER_FORMAT_WEIGHT" in os.environ:
        with _suppress(ValueError, TypeError):
            format_weight = float(os.environ["JUDGER_FORMAT_WEIGHT"])

    answer = _extract_answer(solution_str)
    r_format = 1.0 if answer is not None else 0.0

    if answer is None:
        r_judge, decision = 0.0, "no_answer_tag"
    else:
        r_judge, decision = _llm_judge(question, gold_answer, answer)

    total = r_judge + format_weight * r_format

    # Real-time green line: one trajectory finished and was judged. This is the
    # only place the judge score exists (rollout completion in the agent loop
    # has no score yet), so it fires during the reward phase of each step.
    info = extra_info or {}
    rollout_tokens = info.get("rollout_token_len", "?")
    total_tokens = info.get("total_token_len", "?")
    print(
        f"\033[32m[rollout judged] score={total:.2f} (judge={r_judge:.0f} fmt={r_format:.0f}) "
        f"tokens={rollout_tokens}/{total_tokens} decision={decision} src={data_source or ''}\033[0m",
        flush=True,
    )
    return {
        "score": total,
        "answer_score": r_judge,
        "format_score": r_format,
        "judge_decision": decision,
        "has_answer_tag": answer is not None,
        "data_source": data_source or "",
    }




class _suppress:  # tiny stdlib-free contextlib.suppress, avoid import for one-line use
    def __init__(self, *excs): self._excs = excs
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return exc_type is not None and issubclass(exc_type, self._excs)
