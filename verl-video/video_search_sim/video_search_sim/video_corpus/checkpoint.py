"""On-disk batch checkpoints for resumable corpus builds.

The corpus build is the long-pole of any new training run (hours on H20 for
~50K videos). To avoid losing work to OOMs, driver crashes, preemption,
power events, etc. we checkpoint per-batch under ``<output_dir>/_partial/``
so we can resume from the last fully-committed batch.

Layout
------
``<output_dir>/_partial/``::

    batch_00000.parquet            # VideoRecord rows (one per ingested video)
    batch_00000_emb.npy            # (sum_keyframes_in_batch, D) float32
    batch_00000_kfmeta.parquet     # video_idx (LOCAL within batch), timestamp, vector_row (LOCAL)
    batch_00000_skipped.txt        # video_ids that we attempted but skipped
    batch_00000.done               # zero-byte sentinel marking the batch as committed
    batch_00001.*
    ...

The ``.done`` file is created **last**, after every other artefact for the
batch is fsync'd. A batch without ``.done`` is considered partial and
discarded on resume.

Resumption
----------
On startup we walk ``_partial/``, treat batches with ``.done`` as committed,
and gather the union of their ``video_id`` columns plus the ``_skipped.txt``
entries. The pipeline then asks ingest to skip these IDs.

Finalisation
------------
After the main loop completes (no more new videos coming), the finalizer
loads every committed batch in order, concatenates the records and
embeddings, **rewrites local-to-global ``video_idx`` and ``vector_row``**,
and writes the canonical ``videos.parquet`` / ``clip/`` / ``bm25/`` /
``manifest.json``. Optionally deletes ``_partial/`` afterwards.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.schemas import VideoRecord

logger = logging.getLogger(__name__)


_BATCH_RE = re.compile(r"^batch_(\d{5})\.done$")


# --------------------------------------------------------------------------- types


@dataclass
class CheckpointState:
    """Summary of what's already on disk under ``_partial/``."""

    next_batch_id: int = 0
    seen_video_ids: set[str] = field(default_factory=set)
    skipped_video_ids: set[str] = field(default_factory=set)
    committed_batches: list[int] = field(default_factory=list)

    @property
    def num_committed_videos(self) -> int:
        return len(self.seen_video_ids)


# --------------------------------------------------------------------------- helpers


def _batch_files(partial_dir: Path, batch_id: int) -> dict[str, Path]:
    base = f"batch_{batch_id:05d}"
    return {
        "records": partial_dir / f"{base}.parquet",
        "emb": partial_dir / f"{base}_emb.npy",
        "kfmeta": partial_dir / f"{base}_kfmeta.parquet",
        "skipped": partial_dir / f"{base}_skipped.txt",
        "done": partial_dir / f"{base}.done",
    }


def _fsync(path: Path) -> None:
    """Flush a file's metadata to stable storage. Best-effort on FSes that NOOP."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # On some networked filesystems (NFS, lustre) fsync is a no-op or
        # may fail silently; we don't want to crash the build over it.
        pass


# --------------------------------------------------------------------------- read


def discover_state(partial_dir: Path) -> CheckpointState:
    """Scan ``partial_dir`` and return what's already committed."""
    state = CheckpointState()
    if not partial_dir.is_dir():
        return state

    committed: list[int] = []
    for p in partial_dir.iterdir():
        m = _BATCH_RE.match(p.name)
        if m:
            committed.append(int(m.group(1)))
    committed.sort()
    state.committed_batches = committed

    if not committed:
        return state

    state.next_batch_id = committed[-1] + 1

    # Collect video_ids from every committed batch (records parquet is the
    # source of truth; skipped.txt is auxiliary).
    for bid in committed:
        files = _batch_files(partial_dir, bid)
        if files["records"].is_file():
            try:
                ids = pd.read_parquet(files["records"], columns=["video_id"])["video_id"]
                state.seen_video_ids.update(map(str, ids.tolist()))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", files["records"], e)
        if files["skipped"].is_file():
            try:
                with files["skipped"].open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            state.skipped_video_ids.add(line)
            except OSError as e:
                logger.warning("Failed to read %s: %s", files["skipped"], e)

    state.seen_video_ids.update(state.skipped_video_ids)
    logger.info(
        "Checkpoint resume: %d committed batches, %d videos already done "
        "(%d successful + %d skipped). Next batch id = %d.",
        len(committed),
        len(state.seen_video_ids),
        len(state.seen_video_ids) - len(state.skipped_video_ids),
        len(state.skipped_video_ids),
        state.next_batch_id,
    )
    return state


# --------------------------------------------------------------------------- write


@dataclass
class BatchPayload:
    """Everything we need to commit a single batch."""

    records: list[VideoRecord]
    embeddings: list[np.ndarray]  # one (k_i, D) array per record (may be empty)
    keyframe_meta_local: list[dict]  # video_idx LOCAL, timestamp, vector_row LOCAL
    skipped_video_ids: list[str]


def commit_batch(partial_dir: Path, batch_id: int, payload: BatchPayload) -> None:
    """Atomically commit one batch's artefacts.

    Order matters: every data file must hit disk before the ``.done``
    sentinel is created. We write each file to ``<name>.tmp`` first, fsync,
    then rename — this guarantees that a crash mid-write leaves only stale
    ``.tmp`` files behind and never a half-written committed file.
    """
    partial_dir.mkdir(parents=True, exist_ok=True)
    files = _batch_files(partial_dir, batch_id)

    # Records parquet
    if payload.records:
        rows = []
        for r in payload.records:
            row = r.model_dump()
            row["scene_splits"] = [list(p) for p in r.scene_splits]
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()
    tmp = files["records"].with_suffix(files["records"].suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    _fsync(tmp)
    tmp.replace(files["records"])

    # Embeddings (concatenated within the batch)
    if payload.embeddings:
        emb = np.concatenate(payload.embeddings, axis=0).astype(np.float32, copy=False)
    else:
        emb = np.zeros((0, 0), dtype=np.float32)
    # ``np.save`` auto-appends ``.npy`` if the path doesn't end in it, which
    # would defeat our atomic-rename pattern. Pass an explicit file handle
    # so the bytes go exactly where we want.
    tmp = files["emb"].with_suffix(files["emb"].suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, emb, allow_pickle=False)
    _fsync(tmp)
    tmp.replace(files["emb"])

    # Keyframe meta (LOCAL indices; finalize will rewrite to global)
    if payload.keyframe_meta_local:
        kf = pd.DataFrame(payload.keyframe_meta_local)
    else:
        kf = pd.DataFrame(columns=["video_idx", "timestamp", "vector_row"])
    tmp = files["kfmeta"].with_suffix(files["kfmeta"].suffix + ".tmp")
    kf.to_parquet(tmp, index=False)
    _fsync(tmp)
    tmp.replace(files["kfmeta"])

    # Skipped IDs (one per line)
    if payload.skipped_video_ids:
        tmp = files["skipped"].with_suffix(files["skipped"].suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for vid in payload.skipped_video_ids:
                fh.write(vid + "\n")
        _fsync(tmp)
        tmp.replace(files["skipped"])
    elif files["skipped"].exists():
        files["skipped"].unlink()

    # Sentinel: created LAST, signals "everything above is durable".
    files["done"].touch()
    _fsync(files["done"])

    logger.info(
        "Committed batch %05d: %d videos, %d skipped, %d keyframes",
        batch_id,
        len(payload.records),
        len(payload.skipped_video_ids),
        emb.shape[0] if emb.ndim == 2 else 0,
    )


# --------------------------------------------------------------------------- finalize


@dataclass
class FinalizedCorpus:
    """In-memory result of merging all committed batches."""

    records: list[VideoRecord]
    embeddings: np.ndarray  # (N_keyframes_total, D) float32
    keyframe_meta: pd.DataFrame  # global video_idx and vector_row
    skipped_video_ids: list[str]


def load_committed_batches(partial_dir: Path) -> FinalizedCorpus:
    """Load every committed batch, rewrite local indices to global, return the merged whole.

    This walks committed batches in batch_id order so the resulting global
    ``video_idx`` is stable and matches the order videos were originally
    committed (across resumes).
    """
    state = discover_state(partial_dir)

    all_records: list[VideoRecord] = []
    emb_chunks: list[np.ndarray] = []
    meta_rows: list[dict] = []
    all_skipped: list[str] = []

    global_video_offset = 0
    global_vector_offset = 0

    for bid in state.committed_batches:
        files = _batch_files(partial_dir, bid)

        # Records
        if files["records"].is_file():
            df = pd.read_parquet(files["records"])
            for _, row in df.iterrows():
                d = row.to_dict()
                # Parquet round-trip can turn scene_splits / tags into numpy
                # arrays (or arrays-of-arrays), and into a list-of-lists.
                # ``or []`` doesn't work on ndarrays (truth value ambiguous),
                # so normalise explicitly here.
                ss = d.get("scene_splits")
                if ss is None or (hasattr(ss, "__len__") and len(ss) == 0):
                    d["scene_splits"] = []
                else:
                    d["scene_splits"] = [tuple(p) for p in ss]
                tags = d.get("tags")
                if tags is None:
                    d["tags"] = []
                elif not isinstance(tags, list):
                    d["tags"] = [str(t) for t in tags]
                all_records.append(VideoRecord.model_validate(d))
            num_records_in_batch = len(df)
        else:
            num_records_in_batch = 0

        # Embeddings
        if files["emb"].is_file():
            arr = np.load(files["emb"])
            if arr.ndim == 2 and arr.size > 0:
                emb_chunks.append(arr)

        # Keyframe meta — rewrite local indices to global.
        if files["kfmeta"].is_file():
            kf = pd.read_parquet(files["kfmeta"])
            if not kf.empty:
                kf = kf.copy()
                kf["video_idx"] = kf["video_idx"].astype(int) + global_video_offset
                kf["vector_row"] = kf["vector_row"].astype(int) + global_vector_offset
                meta_rows.extend(kf.to_dict("records"))
                global_vector_offset += len(kf)

        global_video_offset += num_records_in_batch

        # Skipped IDs
        if files["skipped"].is_file():
            try:
                with files["skipped"].open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            all_skipped.append(line)
            except OSError as e:
                logger.warning("Failed to read %s: %s", files["skipped"], e)

    if emb_chunks:
        emb = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
    else:
        emb = np.zeros((0, 0), dtype=np.float32)

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
    else:
        meta_df = pd.DataFrame(columns=["video_idx", "timestamp", "vector_row"])

    return FinalizedCorpus(
        records=all_records,
        embeddings=emb,
        keyframe_meta=meta_df,
        skipped_video_ids=all_skipped,
    )


def cleanup_partial(partial_dir: Path) -> None:
    """Remove the entire ``_partial/`` directory after a successful finalize."""
    if not partial_dir.is_dir():
        return
    for p in sorted(partial_dir.iterdir(), reverse=True):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                # Defensive: we don't expect subdirs, but be tolerant.
                for q in sorted(p.iterdir(), reverse=True):
                    q.unlink()
                p.rmdir()
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", p, e)
    try:
        partial_dir.rmdir()
    except OSError as e:
        logger.warning("Failed to remove %s: %s", partial_dir, e)
