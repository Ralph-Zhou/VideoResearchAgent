"""Benchmark evaluation entry point with checkpointing and concurrency."""

import json
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from video_agent.config import load_config, AppConfig
from video_agent.agent.video_research_agent import VideoResearchAgent
from video_agent.utils.colors import (
    header, key_value, success, error, dim, warning,
    BOLD, RESET, CYAN, GREEN, RED, YELLOW, BLUE, DIM,
)
from eval.judge import JudgeResult, LLMJudge
from eval.checkpoint import EvalCheckpoint
from eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _worker_task(cfg: AppConfig, row: Dict[str, Any]) -> Dict[str, Any]:
    """Single sample evaluation task (runs in worker thread)."""
    row_id = str(row.get("row_id", "unknown"))
    question = row["question"]
    ground_truth = row["answer"]
    level = row.get("level")
    category = row.get("category")

    agent = VideoResearchAgent(cfg)
    judge = LLMJudge(cfg)

    start_time = time.time()
    try:
        result = agent.run(
            query=question,
            row_id=row_id,
            ground_truth=ground_truth,
        )
        prediction = result.get("final_answer", "")
        explanation = result.get("explanation", "")
        confidence = result.get("confidence", "") or "low"
        metrics = result.get("metrics", {})
        model = result.get("model", cfg.llm.model)
        iterations = result.get("iterations", 0)
    except Exception as e:
        logger.error("Worker error for row %s: %s", row_id, e)
        prediction = f"ERROR: {e}"
        explanation = ""
        confidence = "low"
        metrics = {}
        model = cfg.llm.model
        iterations = 0

    try:
        judge_result = judge.evaluate(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
    except Exception as exc:
        logger.exception("Unexpected judge error for row %s", row_id)
        judge_result = JudgeResult(
            is_correct=None,
            status="evaluation_error",
            method="llm_judge",
            attempts=0,
            error=str(exc)[:1000],
        )

    duration = time.time() - start_time

    return {
        "row_id": row_id,
        "model": model,
        "level": level,
        "category": category,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "explanation": explanation,
        "confidence": confidence,
        "is_correct": judge_result.is_correct,
        "judge_status": judge_result.status,
        "judge_method": judge_result.method,
        "judge_attempts": judge_result.attempts,
        "judge_error": judge_result.error,
        "judge_model": cfg.judger.model,
        "iterations": iterations,
        "duration_s": round(duration, 2),
        **{f"metric_{k}": v for k, v in metrics.items()},
    }


def run_benchmark(
    config_path: str = "config/default.yaml",
    run_name: Optional[str] = None,
    level_filter: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
    max_samples: Optional[int] = None,
    row_ids: Optional[List[str]] = None,
):
    if cfg is None:
        cfg = load_config(config_path)

    if run_name is None:
        run_name = time.strftime("eval_%Y%m%d_%H%M%S")

    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{run_name}_results.jsonl"

    benchmark_file = Path(cfg.eval.benchmark_file)
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    data = []
    with open(benchmark_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            # Tolerate datasets that don't follow the strict benchmark schema
            # (e.g. distillation task files where each row has 'question'+'answer'
            # but lacks row_id/level/category). Synthesize sensible defaults so
            # the worker pipeline keeps working.
            if "row_id" not in row:
                row["row_id"] = row.get("seed") or row.get("id") or f"task_{idx}"
            data.append(row)

    if level_filter is not None:
        data = [r for r in data if str(r.get("level")) == str(level_filter)]

    if row_ids is not None:
        requested = {
            value
            for item in row_ids
            for value in (part.strip() for part in item.replace(",", " ").split())
            if value
        }
        available = {str(row.get("row_id")) for row in data}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested row IDs not found in benchmark: {', '.join(missing)}")
        data = [row for row in data if str(row.get("row_id")) in requested]

    if max_samples is not None:
        data = data[:max_samples]

    checkpoint = EvalCheckpoint(str(output_dir), run_name)
    pending = checkpoint.filter_pending(data)

    print(header("Benchmark Evaluation"))
    print(key_value("Total:", str(len(data)), CYAN))
    print(key_value("Pending:", str(len(pending)), YELLOW))
    print(key_value("Workers:", str(cfg.eval.max_workers), CYAN))
    print(key_value("Config:", config_path, CYAN))
    print()

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.eval.max_workers) as executor:
        future_to_row = {
            executor.submit(_worker_task, cfg, row): row for row in pending
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_row),
            total=len(pending),
            desc="Evaluating",
        ):
            try:
                result = future.result()
                all_results.append(result)

                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                checkpoint.mark_done(result["row_id"])

                if result.get("judge_status") == "evaluation_error":
                    mark = f"{YELLOW}{BOLD}!{RESET}"
                elif result["is_correct"]:
                    mark = f"{GREEN}{BOLD}✓{RESET}"
                else:
                    mark = f"{RED}{BOLD}✗{RESET}"
                print(
                    f"  {mark} {CYAN}row={result['row_id']}{RESET} "
                    f"pred={YELLOW}'{result['prediction'][:50]}'{RESET} "
                    f"gt={DIM}'{result['ground_truth'][:50]}'{RESET} "
                    f"{DIM}({result['duration_s']}s){RESET}"
                )
            except Exception as exc:
                logger.error("Worker exception: %s", exc)

    if results_file.exists():
        all_from_file = []
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_from_file.append(json.loads(line))
        report = compute_metrics(all_from_file)
    else:
        report = compute_metrics(all_results)

    report["run_name"] = run_name
    report["config_path"] = config_path
    report["total_samples"] = len(data)
    report["evaluated_samples"] = len(all_results)

    report_file = output_dir / f"{run_name}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(header(f"Evaluation Complete: {run_name}"))
    oa = report["overall_accuracy"]
    oa_color = GREEN if oa >= 30 else YELLOW if oa >= 15 else RED
    print(key_value("Overall Acc:", f"{oa_color}{BOLD}{oa:.2f}%{RESET}", GREEN))
    for k, v in sorted(report.items()):
        if k.startswith("l_") and k.endswith("_accuracy"):
            level_name = k[2:-9].replace("_", " ").title()
            print(key_value(f"  Level {level_name}:", f"{v:.2f}%", BLUE))
    print(key_value("Calib. Error:", f"{report.get('calibration_error', 0):.4f}", BLUE))
    print(key_value("Avg Tokens:", f"{report.get('avg_tokens_per_query', 0):.0f}/query", BLUE))
    print(dim(f"\n  Report saved to: {report_file}"))
    print()

    return report
