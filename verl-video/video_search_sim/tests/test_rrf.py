"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from video_search_sim.retrieval_service.fusion import reciprocal_rank_fusion


class TestRRF:
    def test_single_list_preserves_order(self) -> None:
        fused = reciprocal_rank_fusion([[3, 1, 2]], k=60)
        order = [vid for vid, _ in fused]
        assert order == [3, 1, 2]

    def test_multiple_lists_promote_consensus(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [3, 2, 1, 4, 5]
        fused = reciprocal_rank_fusion([a, b], k=60)
        top = fused[0][0]
        # 2 and 3 both appear at rank 2 and 3/1 respectively; by RRF maths,
        # the consensus winners among {1,2,3} should rank above 4 and 5.
        assert top in {1, 2, 3}
        tail_ids = {vid for vid, _ in fused[-2:]}
        assert tail_ids == {4, 5}

    def test_missing_docs_contribute_zero(self) -> None:
        a = [1, 2, 3]
        b = [1]  # 2 and 3 are absent from this list
        fused = reciprocal_rank_fusion([a, b], k=60)
        scores = dict(fused)
        # Doc 1 appears at rank 1 in both -> score = 2 / 61
        assert scores[1] == pytest.approx(2 / 61)
        # Doc 2 only appears at rank 2 in list a -> score = 1 / 62
        assert scores[2] == pytest.approx(1 / 62)

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1]], k=0)
