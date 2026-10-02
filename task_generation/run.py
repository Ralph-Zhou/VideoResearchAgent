"""CLI entry point for the Video DeepResearch task generation pipeline.

Usage:
    python task_generation/run.py --config task_generation/config.toml
    python task_generation/run.py --config ... --skip-seeds --skip-graphs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Ensure the task_generation dir itself is on sys.path so that
# `import video_task_generation` works whether run from repo root or here.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_dotenv(env_path: Path) -> None:
    """Minimal .env loader (does not overwrite existing env vars)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(HERE / "config.toml"),
        help="Path to the pipeline config TOML file.",
    )
    parser.add_argument("--env", type=str, default=str(HERE.parent / ".env"), help="Path to .env file")
    parser.add_argument("--skip-seeds", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--skip-tasks", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument(
        "--only-stage4", action="store_true",
        help="Skip stages 1-3 and re-run Stage 4 only (reads stage3_tasks.jsonl).",
    )
    args = parser.parse_args()

    _load_dotenv(Path(args.env))

    from video_task_generation.config import PipelineConfig
    from video_task_generation.runner import VideoTaskGenerationWorkflow

    config = PipelineConfig.from_toml(args.config)
    _setup_logging(config.workflow.log_level)

    workflow = VideoTaskGenerationWorkflow(config)
    workflow.run(
        skip_seeds=args.skip_seeds,
        skip_graphs=args.skip_graphs,
        skip_tasks=args.skip_tasks,
        skip_stage4=args.skip_stage4,
        only_stage4=args.only_stage4,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
