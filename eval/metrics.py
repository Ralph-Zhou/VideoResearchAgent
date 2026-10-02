"""Evaluation metrics: Overall Accuracy (by level), Calibration Error."""

import numpy as np
from typing import List, Dict


def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {
            "overall_accuracy": 0,
            "correct_count": 0,
            "total_count": 0,
            "judged_count": 0,
            "evaluation_error_count": 0,
            "judge_coverage": 0,
        }

    total = len(results)
    judged_results = [r for r in results if _is_judged(r)]
    judged_count = len(judged_results)
    evaluation_error_count = total - judged_count
    correct = sum(1 for r in judged_results if r.get("is_correct") is True)
    overall_acc = correct / judged_count * 100 if judged_count else 0
    judge_coverage = judged_count / total * 100 if total else 0

    # Per-level accuracy (supports both int levels 1/2/3 and string levels High/Mid/Low)
    level_stats = {}
    all_levels = set(r.get("level") for r in results if r.get("level") is not None)
    for level in sorted(all_levels, key=str):
        level_results = [r for r in results if r.get("level") == level]
        level_judged = [r for r in level_results if _is_judged(r)]
        if level_results:
            lc = sum(1 for r in level_judged if r.get("is_correct") is True)
            level_key = str(level).lower().replace(" ", "_")
            level_stats[f"l_{level_key}_accuracy"] = (
                lc / len(level_judged) * 100 if level_judged else 0
            )
            level_stats[f"l_{level_key}_count"] = len(level_results)
            level_stats[f"l_{level_key}_judged_count"] = len(level_judged)
            level_stats[f"l_{level_key}_evaluation_error_count"] = (
                len(level_results) - len(level_judged)
            )

    # Per-category accuracy
    categories = set(r.get("category", "Unknown") for r in results)
    category_stats = {}
    for cat in categories:
        if cat is None:
            continue
        cat_results = [r for r in results if r.get("category") == cat]
        cat_judged = [r for r in cat_results if _is_judged(r)]
        cc = sum(1 for r in cat_judged if r.get("is_correct") is True)
        category_stats[cat] = (
            round(cc / len(cat_judged) * 100, 2) if cat_judged else 0
        )

    # Calibration Error (ECE)
    ce = _compute_calibration_error(judged_results)

    # Token stats
    total_tokens = sum(r.get("metric_total_tokens", 0) for r in results)
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_duration = (
        sum(r.get("duration_s", 0) for r in results) / total if total > 0 else 0
    )

    return {
        "overall_accuracy": round(overall_acc, 2),
        "correct_count": correct,
        "total_count": total,
        "judged_count": judged_count,
        "evaluation_error_count": evaluation_error_count,
        "judge_coverage": round(judge_coverage, 2),
        **level_stats,
        "category_accuracy": category_stats,
        "calibration_error": round(ce, 4),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(avg_tokens, 0),
        "avg_duration_s": round(avg_duration, 1),
    }


def _is_judged(result: Dict) -> bool:
    if result.get("judge_status") == "evaluation_error":
        return False
    return isinstance(result.get("is_correct"), bool)


def _compute_calibration_error(results: List[Dict], n_bins: int = 5) -> float:
    pairs = []
    for r in results:
        conf_str = r.get("confidence", "0%")
        try:
            conf = float(str(conf_str).replace("%", "").strip()) / 100.0
        except (ValueError, TypeError):
            conf = 0.0
        is_correct = float(r.get("is_correct", False))
        pairs.append((conf, is_correct))

    if not pairs:
        return 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    ce = 0.0
    N = len(pairs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        bin_pairs = [(c, a) for c, a in pairs if lo <= c < hi]
        if not bin_pairs:
            continue
        n_i = len(bin_pairs)
        acc_i = np.mean([a for _, a in bin_pairs])
        conf_i = np.mean([c for c, _ in bin_pairs])
        ce += (n_i / N) * abs(acc_i - conf_i)

    return float(ce)
