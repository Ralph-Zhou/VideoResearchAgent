"""Build + persist the BM25 and FAISS Flat indices.

Artefact layout written to ``<output_dir>/``::

    videos.parquet              # one row per video (VideoRecord)
    url_to_path.json            # fake_url -> local_path bridge
    bm25/
        index.pkl               # pickled BM25 state (bm25s or rank_bm25)
        tokenized_ids.npy       # parallel array of video idx per document
    clip/
        embeddings.npy          # (N_frames, D) float32, L2-normalised
        keyframe_meta.parquet   # video_idx, timestamp, vector_row
        faiss.index             # IndexFlatIP over the embeddings
    manifest.json

The *video idx* used throughout is the row index into ``videos.parquet``; the
BM25 tokenized_ids and keyframe_meta both reference this idx so the service
can join across both indices without extra bookkeeping.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import BM25Config
from ..common.schemas import VideoRecord

logger = logging.getLogger(__name__)


# Simple, deterministic tokenizer used for BM25. Matches the tokenizer that
# the online service loads, so train-time and serve-time BM25 stay in sync.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenise ``text`` for BM25 indexing / querying.

    Keep this function deterministic and dependency-free. It is exported from
    the package so the FastAPI service can call exactly the same tokenizer.
    """
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- BM25


@dataclass
class BM25Artifact:
    """In-memory representation of the BM25 index, ready to persist."""

    backend: str
    state: Any  # ``bm25s.BM25`` instance or ``rank_bm25.BM25Okapi``
    tokens_per_doc: list[list[str]]  # kept for debugging + rebuild

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "index.pkl").open("wb") as fh:
            pickle.dump({"backend": self.backend, "state": self.state}, fh)
        np.save(
            root / "doc_lengths.npy",
            np.array([len(t) for t in self.tokens_per_doc], dtype=np.int32),
        )


def build_bm25(docs: list[str], cfg: BM25Config) -> BM25Artifact:
    """Tokenise and index documents.

    ``docs`` is one string per video; for our pipeline we concatenate
    ``title + description + subtitle`` upstream.
    """
    tokens = [tokenize(d, lowercase=cfg.lowercase) for d in docs]

    if cfg.backend == "bm25s":
        try:
            import bm25s  # type: ignore
        except ImportError as e:  # pragma: no cover - declared in pyproject
            raise RuntimeError("`bm25s` not installed; `pip install bm25s`") from e
        retriever = bm25s.BM25(k1=cfg.k1, b=cfg.b)
        retriever.index(tokens)
        return BM25Artifact(backend="bm25s", state=retriever, tokens_per_doc=tokens)

    if cfg.backend == "rank_bm25":
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("`rank_bm25` not installed; `pip install rank_bm25`") from e
        retriever = BM25Okapi(tokens, k1=cfg.k1, b=cfg.b)
        return BM25Artifact(backend="rank_bm25", state=retriever, tokens_per_doc=tokens)

    raise ValueError(f"Unknown BM25 backend: {cfg.backend}")


# --------------------------------------------------------------------------- FAISS


@dataclass
class DenseIndexArtifact:
    """In-memory representation of the dense (keyframe) index."""

    embeddings: np.ndarray  # (N_frames, D) float32, L2-normalised
    keyframe_meta: pd.DataFrame  # columns: video_idx, timestamp, vector_row
    faiss_index: Any  # faiss.IndexFlatIP

    def save(self, root: Path) -> None:
        import faiss  # local import; tests may skip this path

        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "embeddings.npy", self.embeddings)
        self.keyframe_meta.to_parquet(root / "keyframe_meta.parquet", index=False)
        faiss.write_index(self.faiss_index, str(root / "faiss.index"))


def build_dense_index(
    embeddings: np.ndarray,
    keyframe_meta: pd.DataFrame,
) -> DenseIndexArtifact:
    """Build a FAISS ``IndexFlatIP`` over ``embeddings`` (assumed L2-normalised).

    IP on unit vectors == cosine similarity. We use Flat because:

    - 45K videos * ~30 keyframes ~= 1.4M vectors: Flat IP is ~0.5s on CPU for
      a single query, well below the RL rollout tolerance.
    - Flat is exact (no recall loss) and requires zero training.
    - Memory: 1.4M * 768 * 4B ~= 4.3 GB — fits comfortably on a single box.
    """
    import faiss  # local import

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if embeddings.size > 0:
        index.add(embeddings)
    return DenseIndexArtifact(embeddings=embeddings, keyframe_meta=keyframe_meta, faiss_index=index)


# --------------------------------------------------------------------------- persistence


def write_videos_parquet(records: list[VideoRecord], output_dir: Path) -> Path:
    """Persist ``records`` as ``videos.parquet``.

    ``scene_splits`` is stored as a list-of-lists (parquet-friendly); readers
    rebuild tuples if needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        row = r.model_dump()
        row["scene_splits"] = [list(p) for p in r.scene_splits]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = output_dir / "videos.parquet"
    df.to_parquet(out, index=False)
    return out


def write_url_map(records: list[VideoRecord], output_dir: Path) -> Path:
    """Persist the ``fake_url -> local_path`` bridge mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {r.fake_url: r.local_path for r in records}
    out = output_dir / "url_to_path.json"
    with out.open("w") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    return out


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist a small build manifest for provenance."""
    payload = dict(payload)
    payload.setdefault("built_at", datetime.now(timezone.utc).isoformat())
    out = output_dir / "manifest.json"
    with out.open("w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out
