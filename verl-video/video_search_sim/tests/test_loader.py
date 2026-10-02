"""Round-trip tests for the corpus on-disk format.

These tests exercise the ``indexer -> loader`` path without touching CLIP,
FAISS or BM25. We write only the minimum set of files (``videos.parquet`` +
``url_to_path.json``) and assert the loader gracefully handles the missing
dense / BM25 indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from video_search_sim.common.schemas import VideoRecord
from video_search_sim.retrieval_service.loader import load_corpus
from video_search_sim.video_corpus import indexer


def _make_records(tmp: Path) -> list[VideoRecord]:
    mp = tmp / "videos"
    mp.mkdir()
    out = []
    for i in range(3):
        path = mp / f"v{i}.mp4"
        path.write_bytes(b"\x00")  # non-empty stub; decord not used here
        from video_search_sim.common.fake_url import fake_url_from_path, fake_video_id

        local = str(path)
        out.append(
            VideoRecord(
                fake_url=fake_url_from_path(local),
                video_id=fake_video_id(local),
                local_path=local,
                title=f"title {i}",
                description=f"desc {i}",
                subtitle="",
                duration=float(i * 10),
                tags=["a", "b"],
                scene_splits=[(0.0, 1.0), (1.0, 2.0)],
                source_dataset="unit-test",
                source_id=str(i),
            )
        )
    return out


def test_videos_parquet_roundtrip(tmp_path: Path) -> None:
    records = _make_records(tmp_path)
    out_dir = tmp_path / "corpus"
    out_dir.mkdir()

    parquet_path = indexer.write_videos_parquet(records, out_dir)
    url_map_path = indexer.write_url_map(records, out_dir)
    indexer.write_manifest(out_dir, {"num_videos": len(records), "version": 1})

    assert parquet_path.is_file()
    assert url_map_path.is_file()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 3
    # scene_splits must survive the parquet round-trip as list-of-lists.
    first_scene = list(df.iloc[0]["scene_splits"])
    assert len(first_scene) == 2

    corpus = load_corpus(out_dir)
    assert corpus.num_videos == 3
    assert corpus.bm25_state is None  # not built
    assert corpus.faiss_index is None  # not built
    assert corpus.embedding_dim == 0
    assert corpus.url_to_path == {r.fake_url: r.local_path for r in records}
    assert corpus.manifest["num_videos"] == 3


def test_load_corpus_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does-not-exist")


def test_load_corpus_missing_videos(tmp_path: Path) -> None:
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_corpus(out_dir)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = indexer.write_manifest(tmp_path, {"foo": 1, "bar": [1, 2]})
    data = json.loads(path.read_text())
    assert data["foo"] == 1
    assert data["bar"] == [1, 2]
    assert "built_at" in data  # auto-populated
