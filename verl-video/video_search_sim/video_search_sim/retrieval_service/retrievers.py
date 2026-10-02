"""Concrete retrievers wrapping the persisted artefacts.

Two retrievers live here. Both share the same simple contract:

    def search(query: str, topk: int) -> list[tuple[video_idx, score]]

The fusion layer consumes only the ordered ``video_idx`` lists and discards
the per-retriever scores (they live on incompatible scales).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..common.config import CLIPConfig
from ..video_corpus.indexer import tokenize
from .loader import Corpus

if TYPE_CHECKING:
    from ..video_corpus.embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- BM25


class BM25Retriever:
    """Thin adapter over the pickled BM25 state."""

    def __init__(self, corpus: Corpus):
        if corpus.bm25_state is None:
            raise RuntimeError("BM25 index not available in this corpus; rebuild without --skip-bm25.")
        self.corpus = corpus
        self.backend = corpus.bm25_backend

    def search(self, query: str, topk: int) -> list[tuple[int, float]]:
        """Return top-k ``(video_idx, bm25_score)`` pairs."""
        tokens = tokenize(query)
        if not tokens:
            return []

        state = self.corpus.bm25_state
        if self.backend == "bm25s":
            # bm25s returns two ``(k,)`` arrays: scores and doc ids.
            # Newer versions take a list of query token lists; older versions take one list.
            try:
                doc_ids, scores = state.retrieve([tokens], k=topk)
                ids, scs = doc_ids[0], scores[0]
            except (TypeError, ValueError):
                # Older API: single token list in
                doc_ids, scores = state.retrieve(tokens, k=topk)  # type: ignore[arg-type]
                ids, scs = doc_ids, scores
            return [(int(i), float(s)) for i, s in zip(ids, scs, strict=False)]

        if self.backend == "rank_bm25":
            all_scores = np.asarray(state.get_scores(tokens))  # (N,)
            if all_scores.size == 0:
                return []
            idx = np.argsort(-all_scores)[:topk]
            return [(int(i), float(all_scores[i])) for i in idx]

        raise ValueError(f"Unknown BM25 backend: {self.backend}")


# --------------------------------------------------------------------------- Dense


class DenseRetriever:
    """FAISS-backed dense retriever over per-keyframe CLIP embeddings.

    The corpus stores one embedding per keyframe; scores are aggregated to the
    video level via ``aggregation`` ('max' or 'mean') before ranking.
    """

    def __init__(
        self,
        corpus: Corpus,
        clip_cfg: CLIPConfig,
        embedder: CLIPEmbedder,
        aggregation: str = "max",
    ):
        if corpus.faiss_index is None or corpus.keyframe_meta is None:
            raise RuntimeError("Dense index not available in this corpus; rebuild without --skip-dense.")
        if aggregation not in {"max", "mean"}:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        self.corpus = corpus
        self.clip_cfg = clip_cfg
        self.embedder = embedder
        self.aggregation = aggregation
        # Precompute keyframe-row -> video_idx as a contiguous numpy array ONCE,
        # so per-query aggregation is fully vectorised. The old per-row
        # ``meta.iloc[row_idx]["video_idx"]`` loop (pool iterations, building a
        # pandas Series each time) was pure-Python and held the GIL, capping
        # single-process throughput regardless of threadpool size.
        self._kf_video_idx = np.ascontiguousarray(
            corpus.keyframe_meta["video_idx"].to_numpy(), dtype=np.int64
        )

    def search(self, query: str, topk: int, candidate_pool: int | None = None) -> list[tuple[int, float]]:
        """Return top-k ``(video_idx, aggregated_score)`` pairs.

        ``candidate_pool`` controls how many keyframes we fetch before
        aggregating to videos. Must be >= topk; larger values cost more FAISS
        search time but yield better recall at the video level when a single
        video dominates a topic.
        """
        # CLIP text encoding: a handful of strings, not worth batching further.
        query_emb = self.embedder.encode_texts([query])  # (1, D)
        if query_emb.shape[0] == 0:
            return []

        pool = max(candidate_pool or 0, topk)
        pool = min(pool, self.corpus.faiss_index.ntotal or 0)
        if pool == 0:
            return []

        # FAISS ``search`` expects float32 and contiguous input.
        query_vec = np.ascontiguousarray(query_emb.astype(np.float32, copy=False))
        sims, rows = self.corpus.faiss_index.search(query_vec, pool)  # (1, pool) each
        sims = sims[0]
        rows = rows[0]

        # Map keyframe rows -> video indices and aggregate to video level, fully
        # vectorised (no Python loop → minimal GIL hold, so concurrent requests
        # actually parallelise across cores).
        valid = rows >= 0  # FAISS pads with -1 when fewer than ``pool`` hits exist
        rows_v = rows[valid]
        if rows_v.size == 0:
            return []
        sims_v = sims[valid].astype(np.float64, copy=False)
        vids = self._kf_video_idx[rows_v]

        # Group by video_idx via sort + segment-reduce.
        order = np.argsort(vids, kind="stable")
        vids_sorted = vids[order]
        sims_sorted = sims_v[order]
        seg_starts = np.concatenate(([0], np.flatnonzero(vids_sorted[1:] != vids_sorted[:-1]) + 1))
        uniq_vids = vids_sorted[seg_starts]

        if self.aggregation == "max":
            agg_scores = np.maximum.reduceat(sims_sorted, seg_starts)
        else:
            seg_sums = np.add.reduceat(sims_sorted, seg_starts)
            seg_counts = np.diff(np.concatenate((seg_starts, [sims_sorted.size])))
            agg_scores = seg_sums / seg_counts

        # Top-k by aggregated score (descending).
        k = min(topk, uniq_vids.size)
        if k < uniq_vids.size:
            cand = np.argpartition(-agg_scores, k - 1)[:k]
        else:
            cand = np.arange(uniq_vids.size)
        cand = cand[np.argsort(-agg_scores[cand])]
        return [(int(uniq_vids[i]), float(agg_scores[i])) for i in cand]
