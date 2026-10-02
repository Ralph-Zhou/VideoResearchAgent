"""LLM Judge for evaluating answer correctness via semantic equivalence."""

import re
import json
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

from openai import OpenAI

from video_agent.config import AppConfig

logger = logging.getLogger(__name__)

JudgeStatus = Literal["correct", "incorrect", "evaluation_error"]
JudgeMethod = Literal["llm_judge", "quick_judge"]


@dataclass(frozen=True)
class JudgeResult:
    """Structured judge outcome that keeps API failures separate from answers."""

    is_correct: Optional[bool]
    status: JudgeStatus
    method: JudgeMethod
    attempts: int
    error: Optional[str] = None


JUDGE_PROMPT = """Question: {question}

Ground Truth: {ground_truth}

Model Prediction: {prediction}

Evaluate if the Prediction matches the Ground Truth semantically.
A match means the prediction conveys the same factual information as the ground truth,
even if phrased differently (e.g., "20 points" vs "20 pts" vs "twenty points" should all match "20 points").
Singular/plural differences (e.g. "peanut" vs "peanuts") count as a match.
If the prediction contains the ground truth answer among extra details, it is still correct.
Do NOT accept predictions that are clearly wrong or factually different.

You MUST respond with ONLY this JSON, nothing else: {{"is_correct": true}} or {{"is_correct": false}}"""


class LLMJudge:
    def __init__(self, cfg: AppConfig):
        self.client = OpenAI(
            api_key=cfg.judger.api_key,
            base_url=cfg.judger.base_url,
        )
        self.model = cfg.judger.model
        self.temperature = cfg.judger.temperature
        self.max_tokens = cfg.judger.max_tokens
        self.enable_quick_judge = cfg.judger.enable_quick_judge
        self.max_attempts = cfg.judger.max_attempts
        self.retry_backoff_sec = cfg.judger.retry_backoff_sec
        logger.info(
            "Judge mode: %s (max attempts: %d)",
            "quick judge with LLM fallback"
            if self.enable_quick_judge
            else "LLM judge only",
            self.max_attempts,
        )

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> JudgeResult:
        clean_prediction = self._extract_answer(prediction)

        if self.enable_quick_judge:
            if clean_prediction.startswith("ERROR"):
                return JudgeResult(
                    is_correct=False,
                    status="incorrect",
                    method="quick_judge",
                    attempts=0,
                )
            if self._quick_match(clean_prediction, ground_truth):
                return JudgeResult(
                    is_correct=True,
                    status="correct",
                    method="quick_judge",
                    attempts=0,
                )

        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question,
                            ground_truth=ground_truth,
                            prediction=clean_prediction,
                        ),
                    }],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                parsed = self._parse_judge_response(content)
                if parsed is None:
                    raise ValueError(
                        "Judge response did not contain a parseable is_correct value"
                    )
                return JudgeResult(
                    is_correct=parsed,
                    status="correct" if parsed else "incorrect",
                    method="llm_judge",
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = self._format_error(exc)
                if attempt < self.max_attempts:
                    delay = self.retry_backoff_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "Judge attempt %d/%d failed: %s; retrying in %.1fs",
                        attempt,
                        self.max_attempts,
                        last_error,
                        delay,
                    )
                    if delay:
                        time.sleep(delay)
                else:
                    logger.error(
                        "Judge evaluation failed after %d attempts: %s",
                        self.max_attempts,
                        last_error,
                    )

        return JudgeResult(
            is_correct=None,
            status="evaluation_error",
            method="llm_judge",
            attempts=self.max_attempts,
            error=last_error,
        )

    @staticmethod
    def _quick_match(prediction: str, ground_truth: str) -> bool:
        """Fast string-based pre-check before calling LLM."""
        def norm(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        p, g = norm(prediction), norm(ground_truth)
        if not p or not g:
            return False
        if p == g:
            return True
        if g in p or p in g:
            return True
        if p + 's' == g or g + 's' == p:
            return True
        p_tokens = set(re.split(r'[\s(),]+', p)) - {'', 'the', 'a', 'an'}
        g_tokens = set(re.split(r'[\s(),]+', g)) - {'', 'the', 'a', 'an'}
        if g_tokens and g_tokens.issubset(p_tokens):
            return True
        return False

    @staticmethod
    def _parse_judge_response(content: str) -> Optional[bool]:
        """Robust JSON parsing with regex fallback."""
        try:
            result = json.loads(content)
            value = result.get("is_correct")
            if isinstance(value, bool):
                return value
        except (json.JSONDecodeError, ValueError):
            pass
        m = re.search(r'"is_correct"\s*:\s*(true|false)', content, re.IGNORECASE)
        if m:
            return m.group(1).lower() == 'true'
        lower = content.lower()
        if 'true' in lower and 'false' not in lower:
            return True
        if 'false' in lower and 'true' not in lower:
            return False
        return None

    @staticmethod
    def _format_error(exc: BaseException) -> str:
        text = str(exc).strip().replace("\n", " ")
        if not text:
            text = exc.__class__.__name__
        return text[:1000]

    @staticmethod
    def _extract_answer(prediction: str) -> str:
        try:
            data = json.loads(prediction)
            if isinstance(data, dict) and "Answer" in data:
                return data["Answer"]
        except Exception:
            pass
        return prediction
