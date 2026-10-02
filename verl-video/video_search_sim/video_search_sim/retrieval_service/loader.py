"""Load the on-disk corpus artefacts into memory at service startup.

This module is the inverse of ``video_corpus/indexer.py``: it knows the exact
file layout and reconstructs the in-memory objects the retrievers operate on.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.schemas import VideoRecord

logger = logging.getLogger(__name__)


@dataclass
class Corpus:
    """Everything the FastAPI service needs to answer ``/video_search``.

    This dataclass is instantiated once per process at startup and shared read-only
    across request handlers.
    """

    root: Path
    videos: list[VideoRecord]
    videos_df: pd.DataFrame
    url_to_path: dict[str, str]
    # BM25 — may be None if the corpus was built with ``--skip-bm25``
    bm25_backend: str | None
    bm25_state: Any
    # Dense — may be None if the corpus was built with ``--skip-dense``
    faiss_index: Any
    keyframe_meta: pd.DataFrame | None
    embedding_dim: int
    manifest: dict[str, Any]

    @property
    def num_videos(self) -> int:
        return len(self.videos)

    @property
    def num_keyframes(self) -> int:
        return 0 if self.keyframe_meta is None else int(len(self.keyframe_meta))


def _safe_get(row: pd.Series, key: str, default: Any) -> Any:
    """``pd.Series.get`` friendly accessor.

    Pandas' ``Series.get`` returns numpy arrays for list-typed cells, on which
    ``value or default`` raises "truth value of an array is ambiguous". We
    special-case that here and convert ``NaN`` scalars to ``default`` so
    callers can stay concise.
    """
    if key not in row:
        return default
    v = row[key]
    # numpy arrays / pandas lists: return the raw value, caller must not use ``or``.
    import numpy as _np

    if isinstance(v, _np.ndarray):
        return v
    # Scalar NaN check — a str column may legitimately contain the literal "nan",
    # so we guard with ``pd.isna`` only for non-string values.
    if not isinstance(v, str | bytes) and pd.isna(v):
        return default
    return v


def _read_videos(root: Path) -> tuple[list[VideoRecord], pd.DataFrame]:
    path = root / "videos.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; did you run `vss-build-corpus`?")
    df = pd.read_parquet(path)
    records: list[VideoRecord] = []
    for _, row in df.iterrows():
        scene_raw = _safe_get(row, "scene_splits", [])
        try:
            scene = [tuple(float(x) for x in pair) for pair in list(scene_raw)]
        except (TypeError, ValueError):
            scene = []

        tags_raw = _safe_get(row, "tags", [])
        tags = [str(x) for x in list(tags_raw)] if tags_raw is not None else []

        records.append(
            VideoRecord(
                fake_url=str(row["fake_url"]),
                video_id=str(row["video_id"]),
                local_path=str(row["local_path"]),
                title=str(_safe_get(row, "title", "") or ""),
                description=str(_safe_get(row, "description", "") or ""),
                subtitle=str(_safe_get(row, "subtitle", "") or ""),
                duration=float(_safe_get(row, "duration", 0.0) or 0.0),
                tags=tags,
                scene_splits=scene,
                source_dataset=str(_safe_get(row, "source_dataset", "") or ""),
                source_id=str(_safe_get(row, "source_id", "") or ""),
            )
        )
    return records, df


def _read_bm25(root: Path) -> tuple[str | None, Any]:
    path = root / "bm25" / "index.pkl"
    if not path.is_file():
        logger.info("BM25 index not found at %s; BM25 retrieval will be disabled.", path)
        return None, None
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return payload["backend"], payload["state"]


def _read_dense(root: Path) -> tuple[Any, pd.DataFrame | None, int]:
    emb_path = root / "clip" / "embeddings.npy"
    meta_path = root / "clip" / "keyframe_meta.parquet"
    faiss_path = root / "clip" / "faiss.index"

    if not (emb_path.is_file() and meta_path.is_file() and faiss_path.is_file()):
        logger.info("Dense index not found under %s/clip; dense retrieval will be disabled.", root)
        return None, None, 0

    import faiss  # local import; optional dep at service time

    embeddings = np.load(emb_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path)
    index = faiss.read_index(str(faiss_path))
    return index, meta, int(embeddings.shape[1]) if embeddings.size else 0


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    with path.open("r") as fh:
        return json.load(fh)


def load_corpus(root: str | Path) -> Corpus:
    """Load every artefact under ``root`` into a single ``Corpus`` object."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Corpus dir not found: {root_path}")

    videos, videos_df = _read_videos(root_path)

    url_map_path = root_path / "url_to_path.json"
    if url_map_path.is_file():
        with url_map_path.open("r") as fh:
            url_to_path = json.load(fh)
    else:
        url_to_path = {r.fake_url: r.local_path for r in videos}

    bm25_backend, bm25_state = _read_bm25(root_path)
    faiss_index, keyframe_meta, dim = _read_dense(root_path)
    manifest = _read_manifest(root_path)

    corpus = Corpus(
        root=root_path,
        videos=videos,
        videos_df=videos_df,
        url_to_path=url_to_path,
        bm25_backend=bm25_backend,
        bm25_state=bm25_state,
        faiss_index=faiss_index,
        keyframe_meta=keyframe_meta,
        embedding_dim=dim,
        manifest=manifest,
    )
    logger.info(
        "Loaded corpus from %s: %d videos, %d keyframes, bm25=%s, dense_dim=%d",
        root_path,
        corpus.num_videos,
        corpus.num_keyframes,
        bm25_backend or "disabled",
        dim,
    )
    return corpus
