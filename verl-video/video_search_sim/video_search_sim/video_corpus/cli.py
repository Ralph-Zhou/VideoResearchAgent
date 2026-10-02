"""``vss-build-corpus`` CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..common.config import CorpusConfig, load_yaml_config
from .pipeline import build_corpus


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="vss-build-corpus",
        description="Build a local video_search corpus from one or more HuggingFace datasets.",
    )
    ap.add_argument("--config", type=Path, required=True, help="Path to a CorpusConfig YAML file")
    ap.add_argument("--output-dir", type=Path, default=None, help="Override config.output_dir")
    ap.add_argument("--max-videos", type=int, default=None, help="Override config.max_videos")
    ap.add_argument("--skip-dense", action="store_true", help="Skip CLIP/FAISS build (BM25 only)")
    ap.add_argument("--skip-bm25", action="store_true", help="Skip BM25 build (dense only)")
    ap.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Wipe <output_dir>/_partial/ and start from scratch. Default "
            "behaviour is to resume from the last committed batch."
        ),
    )
    ap.add_argument(
        "--keep-partial",
        action="store_true",
        help=(
            "Keep <output_dir>/_partial/ around after a successful "
            "finalize (useful for debugging the resume path). Overrides "
            "config.keep_partial when given on the command line."
        ),
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    cfg = load_yaml_config(args.config, CorpusConfig)
    assert isinstance(cfg, CorpusConfig)

    # CLI flag wins over config when explicitly given.
    if args.keep_partial:
        cfg = cfg.model_copy(update={"keep_partial": True})

    report = build_corpus(
        cfg,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        skip_dense=args.skip_dense,
        skip_bm25=args.skip_bm25,
        restart=args.restart,
    )
    print(
        "Build complete: "
        f"videos={report.num_videos} keyframes={report.num_keyframes} "
        f"skipped={report.skipped_videos} dim={report.embedding_dim} "
        f"bm25={report.bm25_backend or 'skipped'} "
        f"resumed_from={report.resumed_from} out={report.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
