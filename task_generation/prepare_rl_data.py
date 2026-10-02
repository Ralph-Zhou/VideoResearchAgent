#!/usr/bin/env python3
"""Prepare 4B agent RL data: accepted stage-4 tasks and optional Video-BrowseComp validation.

Writes train.parquet and, when --val_browsecomp is supplied, val.parquet.
Training uses the local simulator with optional retrieval-domain randomization;
validation uses live search. Gold URLs remain outside the policy prompt.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "video_agent/prompts/default_system_prompt.md"
SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _gold_urls_from_graph(graph: dict | None) -> list[str]:
    if not graph:
        return []
    out: list[str] = []
    for ent in graph.get("entities", []) or []:
        url = (ent.get("video") or {}).get("url")
        if url:
            out.append(url)
    return out


def _stage4_to_row(raw: dict, dr_config: dict | None = None) -> dict | None:
    """Return a normalized row, or None if the sample should be dropped.

    ``dr_config`` is the light retrieval-domain-randomization config, passed
    through for training samples only.
    """
    if raw.get("status") and raw["status"] != "accept":
        return None
    question = (raw.get("question") or "").strip()
    if not question:
        return None
    answer = raw.get("answer") or ""
    gold_urls = _gold_urls_from_graph(raw.get("graph"))
    sample_id = raw.get("seed") or raw.get("initial_seed") or ""

    return _build_row(
        sample_id=str(sample_id),
        question=question,
        answer=answer,
        gold_urls=gold_urls,
        source="train",
        data_source="video_research_train",
        backend="local",
        dr_config=dr_config,
    )


def _browsecomp_row(raw: dict) -> dict | None:
    """video_browsecomp: real search-style task. No fixed video_path."""
    q = (raw.get("question") or "").strip()
    if not q:
        return None
    return _build_row(
        sample_id=f"bc_{raw.get('row_id', '')}",
        question=q,
        answer=raw.get("answer", ""),
        gold_urls=[],          # real-web eval; we don't have fake/real gold URLs upfront
        source="video_browsecomp",
        data_source="video_research_val_browsecomp",
        backend="remote",
    )


def _build_row(
    *,
    sample_id: str,
    question: str,
    answer: str,
    gold_urls: list[str],
    source: str,
    data_source: str,
    backend: str,
    dr_config: dict | None = None,
) -> dict:

    # Domain-randomization switch and gold identifiers, injected **for training
    # samples only** (backend=="local"). Validation and real-evaluation samples
    # (backend=="remote") omit them, so randomization is off by default there; a
    # second hard gate on backend=="local" inside VideoSearchTool.execute()
    # guarantees that the test-time path (live YouTube API) is never perturbed.
    search_youtube_kwargs: dict[str, Any] = {
        "backend": backend,
        "corpus_shard": "default",
    }
    if backend == "local" and dr_config is not None and dr_config.get("enable", False):
        search_youtube_kwargs["domain_randomization"] = dr_config
        search_youtube_kwargs["gold_urls"] = gold_urls

    tools_kwargs: dict[str, Any] = {
        "search_youtube": {
            "create_kwargs": search_youtube_kwargs
        },
        "web_search": {
            # pyarrow refuses to serialise empty struct columns ("Cannot write
            # struct type 'create_kwargs' with no child field"); the verl tool
            # loop keys off the dict's existence, not its contents, so we just
            # echo the backend marker (web_search itself ignores it).
            "create_kwargs": {"backend": backend}
        },
        "watch_video": {
            "create_kwargs": {
                "backend": backend,
            }
        },
        "visual_grounding": {
            "create_kwargs": {
                "backend": backend,
            }
        },
    }

    return {
        "data_source": data_source,
        "ability": "video_research",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": answer,
                "gold_urls": gold_urls,
                "source": source,           # extra hint for the reward fn / debug logs
            },
        },
        "extra_info": {
            "split": "train" if backend == "local" else "val",
            "index": sample_id,
            "question": question,
            "source": source,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True, help="Stage-4 tasks; accepted rows are retained.")
    parser.add_argument("--val_browsecomp", help="Optional Video-BrowseComp JSONL validation input.")
    parser.add_argument("--out_dir", required=True, help="Output directory for train.parquet and val.parquet.")
    parser.add_argument("--disable_domain_randomization", action="store_true")
    parser.add_argument("--dr_candidate_pool", type=int, default=50)
    parser.add_argument("--dr_seed", type=int, default=0)
    args = parser.parse_args()
    if args.dr_candidate_pool < 1:
        parser.error("--dr_candidate_pool must be positive")
    dr_config = {"enable": not args.disable_domain_randomization,
                 "candidate_pool": args.dr_candidate_pool, "seed": args.dr_seed}
    rows = {"train": [row for raw in _load_jsonl(Path(args.train_jsonl))
                      if (row := _stage4_to_row(raw, dr_config)) is not None]}
    if args.val_browsecomp:
        rows["val"] = [row for raw in _load_jsonl(Path(args.val_browsecomp))
                       if (row := _browsecomp_row(raw)) is not None]
    for split, records in rows.items():
        if not records:
            parser.error(f"No usable records in {split} input")
    # Align optional nested fields so both splits load as one dataset configuration.
    schema = pa.unify_schemas([pa.Table.from_pylist(r).schema for r in rows.values()])
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split, records in rows.items():
        path = output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd")
        print(f"Wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
