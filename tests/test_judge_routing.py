from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eval.judge import LLMJudge
from video_agent.config import AppConfig


def _response(llm_result: bool):
    content = '{"is_correct": %s}' % str(llm_result).lower()
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _judge(
    *,
    enable_quick_judge: bool,
    llm_result: bool = False,
    side_effect=None,
    max_attempts: int = 5,
) -> LLMJudge:
    cfg = AppConfig.model_validate(
        {
            "judger": {
                "api_key": "test-key",
                "base_url": "http://localhost:1/v1",
                "enable_quick_judge": enable_quick_judge,
                "max_attempts": max_attempts,
                "retry_backoff_sec": 0,
            }
        }
    )
    create = Mock(return_value=_response(llm_result), side_effect=side_effect)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    with patch("eval.judge.OpenAI", return_value=client):
        judge = LLMJudge(cfg)
    return judge


class JudgeRoutingTests(unittest.TestCase):
    def test_llm_judge_is_used_by_default_even_for_exact_match(self):
        judge = _judge(enable_quick_judge=False, llm_result=False)

        result = judge.evaluate("question", "same answer", "same answer")

        self.assertFalse(result.is_correct)
        self.assertEqual(result.status, "incorrect")
        self.assertEqual(result.method, "llm_judge")
        judge.client.chat.completions.create.assert_called_once()

    def test_llm_judge_is_used_by_default_for_error_prediction(self):
        judge = _judge(enable_quick_judge=False, llm_result=False)

        result = judge.evaluate("question", "answer", "ERROR: model failed")

        self.assertFalse(result.is_correct)
        self.assertEqual(result.status, "incorrect")
        judge.client.chat.completions.create.assert_called_once()

    def test_quick_judge_can_be_enabled_for_debugging(self):
        judge = _judge(enable_quick_judge=True, llm_result=False)

        result = judge.evaluate("question", "same answer", "same answer")

        self.assertTrue(result.is_correct)
        self.assertEqual(result.status, "correct")
        self.assertEqual(result.method, "quick_judge")
        self.assertEqual(result.attempts, 0)
        judge.client.chat.completions.create.assert_not_called()

    def test_quick_judge_falls_back_to_llm_for_non_match(self):
        judge = _judge(enable_quick_judge=True, llm_result=True)

        result = judge.evaluate(
            "question",
            "ground truth",
            "different prediction",
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.status, "correct")
        self.assertEqual(result.method, "llm_judge")
        judge.client.chat.completions.create.assert_called_once()

    def test_llm_judge_retries_until_fifth_attempt(self):
        judge = _judge(
            enable_quick_judge=False,
            side_effect=[
                RuntimeError("temporary failure 1"),
                RuntimeError("temporary failure 2"),
                RuntimeError("temporary failure 3"),
                RuntimeError("temporary failure 4"),
                _response(True),
            ],
        )

        result = judge.evaluate("question", "answer", "prediction")

        self.assertTrue(result.is_correct)
        self.assertEqual(result.status, "correct")
        self.assertEqual(result.attempts, 5)
        self.assertIsNone(result.error)
        self.assertEqual(judge.client.chat.completions.create.call_count, 5)

    def test_llm_judge_returns_evaluation_error_after_five_failures(self):
        judge = _judge(
            enable_quick_judge=False,
            side_effect=RuntimeError("service unavailable"),
        )

        result = judge.evaluate("question", "answer", "prediction")

        self.assertIsNone(result.is_correct)
        self.assertEqual(result.status, "evaluation_error")
        self.assertEqual(result.method, "llm_judge")
        self.assertEqual(result.attempts, 5)
        self.assertEqual(result.error, "service unavailable")
        self.assertEqual(judge.client.chat.completions.create.call_count, 5)

    def test_unparseable_response_is_retried(self):
        bad_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="I cannot decide"),
                )
            ]
        )
        judge = _judge(
            enable_quick_judge=False,
            side_effect=[bad_response, _response(False)],
        )

        result = judge.evaluate("question", "answer", "prediction")

        self.assertFalse(result.is_correct)
        self.assertEqual(result.status, "incorrect")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(judge.client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
