"""End-to-end orchestration: HF source -> full on-disk corpus artefacts.

Steps (run in order; each can be skipped via ``skip_*`` flags for debugging):

    1. Iterate records from HF sources, resolve local paths.
    2. For each record, extract keyframes (decord).
    3. Encode keyframes with CLIP; append to an in-memory batch buffer.
    4. Every ``cfg.checkpoint_every`` successful videos, persist the batch
       under ``<output_dir>/_partial/`` (atomic, with a ``.done`` sentinel)
       so a crash leaves at most one partial batch unsaved.
    5. After the main loop ends, **finalize**: load every committed batch,
       rewrite local indices to global, build BM25 over the merged record
       set and FAISS over the merged embeddings, and write the canonical
       artefacts (``videos.parquet``, ``url_to_path.json``, ``bm25/``,
       ``clip/``, ``manifest.json``).
    6. Optionally clean up ``_partial/``.

Resume
------
By default a build is resumable: re-running ``vss-build-corpus`` with the
same ``output_dir`` picks up after the highest committed batch. Pass
``restart=True`` (or ``--restart`` on the CLI) to wipe ``_partial/`` and
start from scratch.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common.config import CorpusConfig
from ..common.schemas import VideoRecord
from . import checkpoint as ckpt
from . import indexer
from .embedder import CLIPEmbedder
from .ingest import iter_records
from .keyframe import extract_keyframes
from .shard_store import ShardedTarWriter

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


@dataclass
class BuildReport:
    """Summary of a completed build, useful for tests and for the manifest."""

    num_videos: int = 0
    num_keyframes: int = 0
    skipped_videos: int = 0
    embedding_dim: int = 0
    bm25_backend: str = ""
    output_dir: Path | None = None
    per_source_counts: dict[str, int] = field(default_factory=dict)
    cache_layout: str = "none"
    cache_dir: Path | None = None
    num_shards: int = 0
    resumed_from: int = 0  # number of videos already done at start


def _frames_to_pil(frames: np.ndarray) -> list[PILImage]:
    """Convert a ``(N, H, W, 3)`` uint8 numpy array to PIL images."""
    from PIL import Image

    return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]


def _doc_text(r: VideoRecord) -> str:
    """Assemble the BM25 document for one video."""
    parts = [r.title, r.description, r.subtitle, " ".join(r.tags)]
    return " ".join(p for p in parts if p)


def build_corpus(
    cfg: CorpusConfig,
    output_dir: Path | None = None,
    max_videos: int | None = None,
    *,
    skip_dense: bool = False,
    skip_bm25: bool = False,
    restart: bool = False,
) -> BuildReport:
    """Run the full corpus build and persist artefacts.

    Args:
        cfg: Parsed ``CorpusConfig``.
        output_dir: Override ``cfg.output_dir``; takes precedence if given.
        max_videos: Override ``cfg.max_videos``; takes precedence if given.
        skip_dense: If True, do not run CLIP / FAISS (for BM25-only smoke tests).
        skip_bm25: If True, do not run BM25 (for dense-only smoke tests).
        restart: If True, wipe ``_partial/`` (and any existing shard cache
            references in the manifest) and start from scratch. Default
            False = resume from the last committed batch.

    Returns:
        ``BuildReport`` with per-step counts. Also writes ``manifest.json``
        alongside the other artefacts.
    """
    out_dir = Path(output_dir) if output_dir is not None else cfg.output_dir
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building corpus into %s", out_dir)

    cap = max_videos if max_videos is not None else cfg.max_videos
    partial_dir = out_dir / "_partial"

    # ------------------------------------------------------------------ resume

    if restart and partial_dir.is_dir():
        logger.info("--restart given: wiping %s", partial_dir)
        shutil.rmtree(partial_dir)
    state = ckpt.discover_state(partial_dir)
    seen_ids = set(state.seen_video_ids)
    next_batch_id = state.next_batch_id

    # ------------------------------------------------------------------ cache dir

    if cfg.cache_dir is not None:
        cache_dir = Path(cfg.cache_dir).expanduser().resolve()
    else:
        cache_dir = out_dir / "downloads_cache"

    use_sharded_tar = bool(cfg.cache_videos) and cfg.cache_layout == "sharded_tar"
    use_loose_files = bool(cfg.cache_videos) and cfg.cache_layout == "loose_files"

    if cfg.cache_videos:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Caching mp4s under %s (layout=%s)", cache_dir, cfg.cache_layout,
        )

    ingest_writes_loose = use_loose_files

    # ------------------------------------------------------------------ models

    embedder: CLIPEmbedder | None = None
    if not skip_dense:
        embedder = CLIPEmbedder(cfg.clip)

    # ------------------------------------------------------------------ batch buffers (current uncommitted batch only)

    batch_records: list[VideoRecord] = []
    batch_embeddings: list[np.ndarray] = []
    batch_kfmeta: list[dict] = []   # video_idx LOCAL within the current batch
    batch_skipped: list[str] = []

    # Cumulative counters across the full run (spans resumes via state).
    skipped_total = len(state.skipped_video_ids)
    per_source: dict[str, int] = {}

    # Per-batch state
    current_batch_id = next_batch_id

    def flush_batch_if_any(force: bool = False) -> None:
        """Persist the current batch buffer if it is non-empty.

        ``force=True`` always commits (used at end-of-loop). When False, we
        only commit when at least one record is buffered — there is no need
        to write empty batches.
        """
        nonlocal current_batch_id, batch_records, batch_embeddings, batch_kfmeta, batch_skipped
        if not force and not batch_records and not batch_skipped:
            return
        if not batch_records and not batch_skipped:
            return
        ckpt.commit_batch(
            partial_dir,
            current_batch_id,
            ckpt.BatchPayload(
                records=batch_records,
                embeddings=batch_embeddings,
                keyframe_meta_local=batch_kfmeta,
                skipped_video_ids=batch_skipped,
            ),
        )
        # Reset buffers for the next batch.
        current_batch_id += 1
        batch_records = []
        batch_embeddings = []
        batch_kfmeta = []
        batch_skipped = []

    # ------------------------------------------------------------------ main loop

    with contextlib.ExitStack() as stack:
        shard_writer: ShardedTarWriter | None = None
        if use_sharded_tar:
            shard_writer = stack.enter_context(
                ShardedTarWriter(
                    cache_dir,
                    prefix="shard",
                    videos_per_shard=cfg.videos_per_shard,
                    max_shard_bytes=cfg.max_shard_bytes,
                    resume=not restart,
                )
            )

        # Total cap accounts for already-done videos in resume scenarios.
        total_cap_remaining = None
        if cap is not None:
            already_done = len(seen_ids)
            total_cap_remaining = max(0, cap - already_done)
            logger.info(
                "Cap=%d, already done=%d, remaining=%d",
                cap, already_done, total_cap_remaining,
            )

        iterator = iter_records(
            cfg.sources,
            cache_dir=cache_dir,
            max_videos=total_cap_remaining,
            cache_videos=ingest_writes_loose,
            skip_video_ids=seen_ids,
        )
        pbar = tqdm(
            iterator,
            desc="ingest+encode",
            total=cap,
            initial=len(seen_ids),
            unit="vid",
        )

        try:
            for record in pbar:
                # If sharding, pack mp4 first so record.local_path becomes the
                # shard:// URI before we drop the bytes.
                if shard_writer is not None and record.video_bytes is not None:
                    shard_uri = shard_writer.write(
                        record.video_id,
                        record.video_bytes,
                        source_id=record.source_id,
                    )
                    record.local_path = shard_uri

                if not skip_dense:
                    bundle = extract_keyframes(record, cfg.keyframe)
                    record.video_bytes = None
                    if bundle.frames.shape[0] == 0:
                        logger.warning(
                            "Skipping %s: no keyframes extracted", record.local_path,
                        )
                        batch_skipped.append(record.video_id)
                        skipped_total += 1
                        continue

                    assert embedder is not None
                    pil_frames = _frames_to_pil(bundle.frames)
                    try:
                        frame_emb = embedder.encode_images(pil_frames)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Skipping %s: CLIP encode failed (%s)",
                            record.local_path, e,
                        )
                        batch_skipped.append(record.video_id)
                        skipped_total += 1
                        continue

                    if frame_emb.shape[0] != bundle.frames.shape[0]:
                        logger.warning(
                            "Keyframe/embedding count mismatch for %s: %d vs %d",
                            record.local_path,
                            bundle.frames.shape[0],
                            frame_emb.shape[0],
                        )

                    # Per-record kfmeta uses LOCAL video_idx within the batch
                    # and LOCAL vector_row within the batch's concatenated
                    # embedding. Finalize rewrites both to globals.
                    local_video_idx = len(batch_records)
                    local_base_row = sum(e.shape[0] for e in batch_embeddings)
                    for i, ts in enumerate(bundle.timestamps):
                        batch_kfmeta.append({
                            "video_idx": local_video_idx,
                            "timestamp": float(ts),
                            "vector_row": local_base_row + i,
                        })
                    batch_embeddings.append(frame_emb)

                record.video_bytes = None
                batch_records.append(record)
                per_source[record.source_dataset] = per_source.get(record.source_dataset, 0) + 1
                seen_ids.add(record.video_id)

                # Periodic checkpoint commit.
                if len(batch_records) + len(batch_skipped) >= cfg.checkpoint_every:
                    flush_batch_if_any()

            pbar.close()

            # Final partial batch (whatever's still in the buffer).
            flush_batch_if_any(force=True)

        except BaseException:  # noqa: BLE001
            # On *any* exception (including KeyboardInterrupt), try to commit
            # what we have so the partial work is recoverable. Then rethrow
            # so the user sees the original error.
            logger.warning(
                "Build interrupted; flushing %d-video partial batch before exit.",
                len(batch_records) + len(batch_skipped),
            )
            try:
                flush_batch_if_any(force=True)
            except Exception as flush_err:  # noqa: BLE001
                logger.error("Failed to flush partial batch on exit: %s", flush_err)
            raise

        # Persist a per-run shard index so finalize can merge them.
        num_shards = 0
        if shard_writer is not None:
            run_index_name = f"shard_index_run_{current_batch_id:05d}.parquet"
            shard_writer.write_index(name=run_index_name)
            num_shards = shard_writer.num_shards
            logger.info(
                "Wrote run-scoped shard index %s; total shards now %d",
                run_index_name, num_shards,
            )

    # ------------------------------------------------------------------ finalize

    finalized = ckpt.load_committed_batches(partial_dir)
    if not finalized.records:
        raise RuntimeError(
            "No videos ingested (no committed batches); nothing to index."
        )
    logger.info(
        "Finalize: merged %d batches -> %d records, %d keyframes",
        len(state.committed_batches) + (current_batch_id - state.next_batch_id),
        len(finalized.records),
        finalized.embeddings.shape[0] if finalized.embeddings.ndim == 2 else 0,
    )

    # 1. videos.parquet + url_to_path.json
    parquet_path = indexer.write_videos_parquet(finalized.records, out_dir)
    url_map_path = indexer.write_url_map(finalized.records, out_dir)
    logger.info(
        "Wrote %s (%d rows) and %s",
        parquet_path, len(finalized.records), url_map_path,
    )

    # 2. BM25 over the merged record set.
    report_bm25_backend = ""
    if not skip_bm25:
        docs = [_doc_text(r) for r in finalized.records]
        bm25_art = indexer.build_bm25(docs, cfg.bm25)
        bm25_art.save(out_dir / "bm25")
        report_bm25_backend = bm25_art.backend
        logger.info(
            "Wrote BM25 index (%s backend) under %s/bm25",
            bm25_art.backend, out_dir,
        )

    # 3. FAISS Flat over the merged embeddings.
    dim = cfg.clip.embedding_dim
    num_keyframes = 0
    if not skip_dense and finalized.embeddings.size > 0:
        emb = finalized.embeddings
        dim = emb.shape[1]
        dense_art = indexer.build_dense_index(emb, finalized.keyframe_meta)
        dense_art.save(out_dir / "clip")
        num_keyframes = emb.shape[0]
        logger.info(
            "Wrote FAISS IndexFlatIP over %d keyframes (dim=%d)",
            num_keyframes, dim,
        )

    # 4. Merge per-run shard indices into one canonical file (sharded_tar only).
    if use_sharded_tar:
        _consolidate_shard_index(cache_dir)

    # 5. Manifest
    cache_layout_str = cfg.cache_layout if cfg.cache_videos else "none"
    indexer.write_manifest(
        out_dir,
        {
            "num_videos": len(finalized.records),
            "num_keyframes": num_keyframes,
            "skipped_videos": skipped_total,
            "embedding_dim": dim,
            "clip_model": cfg.clip.model_name if not skip_dense else None,
            "bm25_backend": report_bm25_backend or None,
            "per_source": per_source,
            "seed": cfg.seed,
            "version": 1,
            "cache": {
                "layout": cache_layout_str,
                "dir": str(cache_dir) if cfg.cache_videos else None,
                "num_shards": num_shards,
                "videos_per_shard": cfg.videos_per_shard if use_sharded_tar else None,
            },
        },
    )

    # 6. Clean up _partial/ unless asked to keep it.
    if not cfg.keep_partial:
        ckpt.cleanup_partial(partial_dir)
        logger.info("Removed %s", partial_dir)
    else:
        logger.info("keep_partial=True, leaving %s in place", partial_dir)

    return BuildReport(
        num_videos=len(finalized.records),
        num_keyframes=num_keyframes,
        skipped_videos=skipped_total,
        embedding_dim=dim,
        bm25_backend=report_bm25_backend,
        output_dir=out_dir,
        per_source_counts=per_source,
        cache_layout=cache_layout_str,
        cache_dir=cache_dir if cfg.cache_videos else None,
        num_shards=num_shards,
        resumed_from=len(state.seen_video_ids),
    )


def _consolidate_shard_index(cache_dir: Path) -> None:
    """Merge ``shard_index_run_*.parquet`` + any pre-existing ``shard_index.parquet``.

    Produces a single ``cache_dir / shard_index.parquet`` listing every entry
    written across all runs of the build (including resumed ones).
    Per-run files are removed after the merge succeeds.
    """
    pieces: list[pd.DataFrame] = []
    canonical = cache_dir / "shard_index.parquet"
    if canonical.is_file():
        with contextlib.suppress(Exception):
            pieces.append(pd.read_parquet(canonical))
    run_files = sorted(cache_dir.glob("shard_index_run_*.parquet"))
    for f in run_files:
        try:
            pieces.append(pd.read_parquet(f))
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping unreadable %s: %s", f, e)
    if not pieces:
        return
    merged = pd.concat(pieces, axis=0, ignore_index=True)
    # Drop duplicates: same video_id should only appear once. Keep the last
    # entry written (which is the most recent run's view).
    if "video_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)
    merged.to_parquet(canonical, index=False)
    for f in run_files:
        with contextlib.suppress(OSError):
            f.unlink()
    logger.info(
        "Consolidated shard index: %d entries written to %s",
        len(merged), canonical,
    )
