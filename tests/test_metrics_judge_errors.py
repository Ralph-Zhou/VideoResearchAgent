import unittest

from eval.metrics import compute_metrics


class JudgeErrorMetricsTests(unittest.TestCase):
    def test_evaluation_errors_are_excluded_from_accuracy(self):
        results = [
            {
                "row_id": "1",
                "level": 1,
                "category": "Film",
                "is_correct": True,
                "judge_status": "correct",
            },
            {
                "row_id": "2",
                "level": 1,
                "category": "Film",
                "is_correct": False,
                "judge_status": "incorrect",
            },
            {
                "row_id": "3",
                "level": 1,
                "category": "Film",
                "is_correct": None,
                "judge_status": "evaluation_error",
            },
        ]

        report = compute_metrics(results)

        self.assertEqual(report["overall_accuracy"], 50.0)
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["total_count"], 3)
        self.assertEqual(report["judged_count"], 2)
        self.assertEqual(report["evaluation_error_count"], 1)
        self.assertEqual(report["judge_coverage"], 66.67)
        self.assertEqual(report["l_1_count"], 3)
        self.assertEqual(report["l_1_judged_count"], 2)

    def test_legacy_boolean_results_remain_supported(self):
        report = compute_metrics(
            [
                {"row_id": "1", "is_correct": True},
                {"row_id": "2", "is_correct": False},
            ]
        )

        self.assertEqual(report["overall_accuracy"], 50.0)
        self.assertEqual(report["judged_count"], 2)
        self.assertEqual(report["evaluation_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
