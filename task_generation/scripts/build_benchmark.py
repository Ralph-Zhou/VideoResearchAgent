"""Convert Stage-4 accepted tasks into the benchmark jsonl schema used by
`scripts/run_benchmark.py`.

Input  : task_generation/results/<run>/stage4_tasks.jsonl
Output : data/benchmark/<name>.jsonl

Schema fields copied/derived:
    row_id          f"v5_{i+1}"
    question        task["question"]
    answer          task["answer"]
    category        "video_deepresearch"
    level           f"video_dr_{name}"
    language        task["language"]
    graph_depth     task["graph_depth"]
    _frame_evidence task["frame_evidence"]

Usage:
    python task_generation/scripts/build_benchmark.py \
        --run smoke10_v5 \
        --out data/benchmark/smoke10_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]


def build(run: str, stage: str, out_path: Path, level_tag: str | None = None) -> int:
    run_dir = PROJECT_ROOT / "task_generation" / "results" / run
    src = run_dir / f"{stage}_tasks.jsonl"
    if not src.exists():
        raise SystemExit(f"[build_benchmark] missing source file: {src}")

    level = level_tag or f"video_dr_{run}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("status") and t["status"] != "accept":
                continue
            question = t.get("question") or t.get("question_raw")
            answer = t.get("answer") or t.get("answer_raw")
            if not question or not answer:
                print(f"[build_benchmark] skip row {i}: empty question/answer")
                continue
            rows.append({
                "row_id": f"v5_{len(rows) + 1}",
                "question": question,
                "answer": answer,
                "category": t.get("category") or "video_deepresearch",
                "level": level,
                "language": t.get("language") or "",
                "graph_depth": t.get("graph_depth"),
                "_frame_evidence": t.get("frame_evidence") or "",
                "_used_properties": t.get("used_properties") or [],
                "_seed": t.get("seed"),
                "_initial_seed": t.get("initial_seed"),
            })

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build_benchmark] wrote {len(rows)} rows → {out_path}")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under task_generation/results/")
    ap.add_argument("--stage", default="stage4", choices=["stage3", "stage4"],
                    help="which stage's accepted tasks to use (default: stage4)")
    ap.add_argument("--out", required=True, help="output benchmark jsonl path")
    ap.add_argument("--level", default=None, help="override level tag")
    args = ap.parse_args(argv)
    n = build(args.run, args.stage, Path(args.out), args.level)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
