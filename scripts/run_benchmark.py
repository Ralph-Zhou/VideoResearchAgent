#!/usr/bin/env python3
"""
Run benchmark evaluation.

Usage:
  python scripts/run_benchmark.py --benchmark-name video_browsecomp
  python scripts/run_benchmark.py --benchmark-name video_browsecomp --level 1
  python scripts/run_benchmark.py --benchmark-name video_browsecomp --max-samples 5
  python scripts/run_benchmark.py --benchmark-name video_browsecomp --run-name exp_v1
  python scripts/run_benchmark.py --benchmark-name video_browsecomp --config config/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_agent.config import load_config
from eval.benchmark import run_benchmark

BENCHMARK_DIR = Path(__file__).parent.parent / "data" / "benchmark"


def resolve_benchmark(name: str) -> Path:
    """Resolve a benchmark name to its JSONL file path.

    Accepts either:
      - a bare benchmark name → resolved to data/benchmark/<name>.jsonl
      - a direct .jsonl path (absolute or relative) → returned as-is
    """
    direct = Path(name)
    if direct.suffix == ".jsonl" and direct.exists():
        return direct
    path = BENCHMARK_DIR / f"{name}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in BENCHMARK_DIR.glob("*.jsonl"))
        raise FileNotFoundError(
            f"Benchmark '{name}' not found at {path}\n"
            f"Available benchmarks: {', '.join(available) or '(none)'}\n"
            f"You can also pass a direct .jsonl path."
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--benchmark-name", type=str, required=True,
                        help="Benchmark name (e.g. video_browsecomp, video_browsecomp, sample50)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name. If set, all outputs (results/report/checkpoint/"
                             "trajectories) are saved under <results-root>/<run-name>/. "
                             "Overrides cfg.eval.output_dir and cfg.logging.trajectory_dir.")
    parser.add_argument("--results-root", type=str, default="data/results",
                        help="Root directory under which per-run folders are created "
                             "(default: data/results). Only used when --run-name is set.")
    parser.add_argument(
        "--level", type=str, default=None,
        help="Only evaluate specific difficulty level (e.g. 1, High, Mid, Low)",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument(
        "--row-ids", type=str, nargs="*", default=None,
        help="Only evaluate these benchmark row IDs. Accepts whitespace- or "
             "comma-separated values, e.g. --row-ids 67 107,120.",
    )
    parser.add_argument(
        "--tools", type=str, nargs="*", default=None,
        help="Tool list to enable (e.g. search_youtube web_search watch_video). "
             "Default: all tools.",
    )
    parser.add_argument(
        "--required-tools", type=str, nargs="*", default=None,
        help="Defer final answers until these tools have completed successfully. "
             "Intended for curated evidence/audit runs.",
    )
    parser.add_argument(
        "--force-required-tool-order", action="store_true",
        help="Use tool_choice to call required tools in their listed order "
             "before allowing a free-form final answer.",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of concurrent workers (overrides config)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model (overrides config llm.model)")
    parser.add_argument(
        "--num-instances", type=int, default=None,
        help="Override the number of local endpoints used from llm.start_port.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", action="store_true",
        help="Pass enable_thinking=true to Qwen-compatible chat templates.",
    )
    thinking_group.add_argument(
        "--disable-thinking", action="store_true",
        help="Pass enable_thinking=false to Qwen-compatible chat templates.",
    )
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model (overrides config judger.model)")
    parser.add_argument("--enable-full-trajectory", action="store_true",
                        help="Save unpruned image-bearing trajectories as "
                             "per-case OpenAI-style archives and a combined "
                             "ms-swift all.jsonl. Output goes to "
                             "<full-trajectory-dir> (default: <results-root>/<run-name>/distill).")
    parser.add_argument("--full-trajectory-dir", type=str, default=None,
                        help="Override directory for ms-swift exports. "
                             "If unset and --enable-full-trajectory is on, "
                             "defaults to <run-dir>/distill (or "
                             "data/distill/<run-name> if no --run-name).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    benchmark_path = resolve_benchmark(args.benchmark_name)

    cfg = load_config(args.config)
    cfg.eval.benchmark_file = str(benchmark_path)

    if args.tools is not None:
        # Accept both forms:
        #   --tools a b c                 -> ['a', 'b', 'c']
        #   --tools "a b c"               -> ['a b c']  (common bash mistake)
        #   --tools a,b,c                 -> ['a,b,c']
        # Normalize whitespace/comma-separated single strings into a proper list
        # so bash quoting slip-ups don't silently disable most tools.
        flat: list = []
        for item in args.tools:
            parts = [p.strip() for p in item.replace(",", " ").split()]
            flat.extend(p for p in parts if p)
        cfg.agent.enabled_tools = flat or None
        logging.info("Enabled tools: %s", cfg.agent.enabled_tools)

    if args.required_tools is not None:
        required: list = []
        for item in args.required_tools:
            parts = [p.strip() for p in item.replace(",", " ").split()]
            required.extend(p for p in parts if p)
        cfg.agent.required_tools_before_answer = required
        logging.info(
            "Required tools before final answer: %s",
            cfg.agent.required_tools_before_answer,
        )
    if args.force_required_tool_order:
        cfg.agent.force_required_tool_order = True

    if args.workers is not None:
        cfg.eval.max_workers = args.workers

    if args.model is not None:
        cfg.llm.model = args.model

    if args.num_instances is not None:
        if args.num_instances < 1:
            parser.error("--num-instances must be at least 1")
        cfg.llm.num_instances = args.num_instances

    if args.enable_thinking:
        cfg.llm.enable_thinking = True
    elif args.disable_thinking:
        cfg.llm.enable_thinking = False

    if args.judge_model is not None:
        cfg.judger.model = args.judge_model


    if args.run_name is not None:
        # Centralize everything for this run under <results-root>/<run-name>/.
        # This makes each run fully self-contained: results.jsonl, report.json,
        # checkpoint.json and per-sample trajectories all live in one folder.
        run_dir = Path(args.results_root) / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.eval.output_dir = str(run_dir)
        cfg.logging.trajectory_dir = str(run_dir / "trajectories")
        logging.info("Run outputs will be saved under: %s", run_dir)

    if args.enable_full_trajectory:
        cfg.agent.full_trajectory = True
        if args.full_trajectory_dir is not None:
            cfg.agent.full_trajectory_dir = args.full_trajectory_dir
        elif args.run_name is not None:
            cfg.agent.full_trajectory_dir = str(Path(args.results_root) / args.run_name / "distill")
        else:
            cfg.agent.full_trajectory_dir = "data/distill/default_run"
        Path(cfg.agent.full_trajectory_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Full trajectories (ms-swift format) will be saved under: %s",
                     cfg.agent.full_trajectory_dir)

    run_benchmark(
        config_path=args.config,
        run_name=args.run_name,
        level_filter=args.level,
        cfg=cfg,
        max_samples=args.max_samples,
        row_ids=args.row_ids,
    )


if __name__ == "__main__":
    main()
