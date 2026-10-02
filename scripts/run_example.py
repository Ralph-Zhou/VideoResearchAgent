#!/usr/bin/env python3
"""
Test the agent on a single example query.

Usage:
  python scripts/run_example.py --query "Your question here"
  python scripts/run_example.py --benchmark-name video_browsecomp --benchmark-id 42
  python scripts/run_example.py --query "..." --config config/default.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_agent.config import load_config
from video_agent.agent.video_research_agent import VideoResearchAgent
from video_agent.utils.colors import (
    header, key_value, warning, dim,
    CYAN, MAGENTA, DIM, RESET, BOLD,
)

BENCHMARK_DIR = Path(__file__).parent.parent / "data" / "benchmark"


def resolve_benchmark(name: str) -> Path:
    """Resolve a benchmark name to its JSONL file path."""
    path = BENCHMARK_DIR / f"{name}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in BENCHMARK_DIR.glob("*.jsonl"))
        raise FileNotFoundError(
            f"Benchmark '{name}' not found at {path}\n"
            f"Available benchmarks: {', '.join(available) or '(none)'}"
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Run VideoResearchAgent on a single query")
    parser.add_argument("--query", type=str, help="Direct query string")
    parser.add_argument("--benchmark-name", type=str, default=None,
                        help="Benchmark name (e.g. video_browsecomp)")
    parser.add_argument("--benchmark-id", type=str, help="Row ID from benchmark file")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output", type=str, default="data/results/examples")
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument(
        "--tools", type=str, nargs="*", default=None,
        help="Tool list to enable (e.g. search_youtube web_search watch_video). "
             "Default: all tools.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    if args.tools is not None:
        cfg.agent.enabled_tools = args.tools

    query = args.query
    ground_truth = args.ground_truth
    row_id = None

    if args.benchmark_id:
        if args.benchmark_name:
            benchmark_path = resolve_benchmark(args.benchmark_name)
        else:
            benchmark_path = Path(cfg.eval.benchmark_file)
        if benchmark_path.exists():
            with open(benchmark_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row.get("row_id")) == args.benchmark_id:
                        query = row["question"]
                        ground_truth = row.get("answer")
                        row_id = args.benchmark_id
                        print(f"  {CYAN}Loaded row {row_id}{RESET}")
                        print(f"  {DIM}Level: {row.get('level')}, Category: {row.get('category')}{RESET}")
                        if ground_truth:
                            print(f"  {MAGENTA}Ground Truth: {ground_truth}{RESET}")
                        break
        else:
            print(warning(f"Benchmark file not found: {benchmark_path}"))

    if not query:
        from video_agent.utils.colors import error
        print(error("Error: Please provide --query or --benchmark-id"))
        sys.exit(1)

    print(header("Example Task"))
    print(key_value("Query:", query[:120], CYAN))
    print(key_value("Config:", args.config, CYAN))
    print()

    cfg.logging.trajectory_dir = args.output
    cfg.logging.print_steps = True

    agent = VideoResearchAgent(cfg)

    result = agent.run(
        query=query,
        row_id=row_id,
        ground_truth=ground_truth,
    )

    if ground_truth:
        print(f"\n  {MAGENTA}{BOLD}Ground Truth:{RESET}  {ground_truth}")

    print(dim(f"\n  Trajectory saved to: {args.output}/"))
    print(dim(f"  Messages in trajectory: {len(result.get('messages', []))}"))


if __name__ == "__main__":
    main()
