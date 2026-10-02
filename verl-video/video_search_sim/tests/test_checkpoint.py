"""Unit tests for ``video_corpus.checkpoint``.

These exercise the resume / commit / finalize round-trip without touching GPU,
CLIP, HuggingFace or FAISS. They guard the most subtle invariants:

* a batch is only resumable iff its ``.done`` sentinel is present;
* ``load_committed_batches`` rewrites batch-local ``video_idx`` /
  ``vector_row`` to globals that match the order of ingestion;
* a crash mid-write (simulated by leaving ``.tmp`` files) leaves the previous
  ``.done`` batches intact and resume picks up cleanly;
* ``np.save`` is invoked via an explicit handle so the atomic-rename pattern
  isn't broken by ``.npy`` auto-suffixing (regression test for the bug we hit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from video_search_sim.common.schemas import VideoRecord
from video_search_sim.video_corpus import checkpoint as ckpt


def _mk_record(i: int) -> VideoRecord:
    return VideoRecord(
        fake_url=f"https://www.youtube.com/watch?v=vid{i:08d}",
        video_id=f"vid{i:08d}",
        local_path=f"shard://shard_00000.tar?o=0&n=1#vid{i:08d}",
        title=f"title {i}",
        description=f"desc {i}",
        subtitle="",
        duration=10.0 + i,
        tags=[f"tag{i}"],
        scene_splits=[(0.0, 5.0), (5.0, 10.0)],
        source_dataset="FineVideo-Local",
        source_id=f"vid{i:08d}",
    )


def _mk_batch(start: int, n: int, dim: int = 4, kfs_per_video: int = 2) -> ckpt.BatchPayload:
    records = [_mk_record(start + j) for j in range(n)]
    embeddings = []
    kf_meta = []
    cum_rows = 0
    for j in range(n):
        e = np.full((kfs_per_video, dim), float(start + j), dtype=np.float32)
        embeddings.append(e)
        for k in range(kfs_per_video):
            kf_meta.append({
                "video_idx": j,
                "timestamp": float(k),
                "vector_row": cum_rows + k,
            })
        cum_rows += kfs_per_video
    return ckpt.BatchPayload(
        records=records,
        embeddings=embeddings,
        keyframe_meta_local=kf_meta,
        skipped_video_ids=[],
    )


def test_commit_and_discover_roundtrip(tmp_path: Path) -> None:
    partial = tmp_path / "_partial"

    ckpt.commit_batch(partial, 0, _mk_batch(start=0, n=3))
    ckpt.commit_batch(partial, 1, _mk_batch(start=3, n=2))

    state = ckpt.discover_state(partial)
    assert state.committed_batches == [0, 1]
    assert state.next_batch_id == 2
    assert state.seen_video_ids == {f"vid{i:08d}" for i in range(5)}

    finalized = ckpt.load_committed_batches(partial)
    assert len(finalized.records) == 5
    # Global vector_row is contiguous and matches concat(emb_chunks).
    assert finalized.embeddings.shape == (5 * 2, 4)
    # video_idx must have been rewritten from local (0..n-1 per batch) to global (0..4).
    assert sorted(finalized.keyframe_meta["video_idx"].unique().tolist()) == list(range(5))
    # vector_row must be a contiguous 0..N-1 (in commit order).
    assert finalized.keyframe_meta["vector_row"].tolist() == list(range(10))
    # Embeddings are ordered: rows for video_idx==i should be all == float(i).
    for i in range(5):
        rows = finalized.embeddings[i * 2:(i + 1) * 2]
        assert np.allclose(rows, float(i))


def test_partial_batch_without_done_is_ignored(tmp_path: Path) -> None:
    """A crash that drops the .done sentinel must hide the batch from resume."""
    partial = tmp_path / "_partial"
    ckpt.commit_batch(partial, 0, _mk_batch(start=0, n=2))
    ckpt.commit_batch(partial, 1, _mk_batch(start=2, n=2))

    # Simulate a crash *after* writing data files for batch 2 but *before* the sentinel.
    files = ckpt._batch_files(partial, 2)
    pd.DataFrame([{"video_id": "vid00000004"}]).to_parquet(files["records"])
    np.save(files["emb"], np.zeros((0, 0), dtype=np.float32))
    pd.DataFrame(columns=["video_idx", "timestamp", "vector_row"]).to_parquet(files["kfmeta"])
    # No .done file -> batch 2 must be invisible.

    state = ckpt.discover_state(partial)
    assert state.committed_batches == [0, 1]
    assert state.next_batch_id == 2
    assert "vid00000004" not in state.seen_video_ids


def test_skipped_ids_are_persisted_and_replayed(tmp_path: Path) -> None:
    partial = tmp_path / "_partial"
    payload = _mk_batch(start=0, n=2)
    payload.skipped_video_ids = ["bad_vid_a", "bad_vid_b"]
    ckpt.commit_batch(partial, 0, payload)

    state = ckpt.discover_state(partial)
    # Skipped IDs are folded into seen_video_ids so resume short-circuits them.
    assert "bad_vid_a" in state.seen_video_ids
    assert "bad_vid_b" in state.seen_video_ids
    assert state.skipped_video_ids == {"bad_vid_a", "bad_vid_b"}

    finalized = ckpt.load_committed_batches(partial)
    assert set(finalized.skipped_video_ids) == {"bad_vid_a", "bad_vid_b"}


def test_emb_file_name_has_no_double_npy_suffix(tmp_path: Path) -> None:
    """Regression: np.save must not auto-append .npy to our atomic filename."""
    partial = tmp_path / "_partial"
    ckpt.commit_batch(partial, 0, _mk_batch(start=0, n=1))
    files = ckpt._batch_files(partial, 0)
    assert files["emb"].is_file()
    # Make sure no leftover ``batch_00000_emb.npy.npy`` exists.
    assert not (partial / "batch_00000_emb.npy.npy").exists()
    # Make sure no leftover ``.tmp`` exists.
    leftovers = sorted(partial.glob("*.tmp"))
    assert leftovers == []


def test_cleanup_partial_removes_directory(tmp_path: Path) -> None:
    partial = tmp_path / "_partial"
    ckpt.commit_batch(partial, 0, _mk_batch(start=0, n=1))
    assert partial.is_dir()
    ckpt.cleanup_partial(partial)
    assert not partial.exists()


def test_resume_with_empty_partial_dir(tmp_path: Path) -> None:
    """No _partial/ at all is a perfectly valid 'cold start' state."""
    partial = tmp_path / "_partial"
    state = ckpt.discover_state(partial)
    assert state.committed_batches == []
    assert state.next_batch_id == 0
    assert state.seen_video_ids == set()


def test_global_index_rewriting_across_three_batches(tmp_path: Path) -> None:
    """Three batches with different sizes -> global indices must stay monotone & dense."""
    partial = tmp_path / "_partial"
    ckpt.commit_batch(partial, 0, _mk_batch(start=0, n=2, kfs_per_video=3))
    ckpt.commit_batch(partial, 1, _mk_batch(start=2, n=1, kfs_per_video=3))
    ckpt.commit_batch(partial, 2, _mk_batch(start=3, n=4, kfs_per_video=3))

    finalized = ckpt.load_committed_batches(partial)
    assert len(finalized.records) == 7
    assert finalized.embeddings.shape == (21, 4)
    # Each video has exactly kfs_per_video=3 rows in keyframe_meta, ordered.
    counts = finalized.keyframe_meta.groupby("video_idx").size()
    assert counts.tolist() == [3, 3, 3, 3, 3, 3, 3]
    # vector_row is dense 0..20.
    assert finalized.keyframe_meta["vector_row"].tolist() == list(range(21))
