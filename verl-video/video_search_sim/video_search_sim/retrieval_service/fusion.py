"""Reciprocal Rank Fusion and helpers shared across retrievers.

We deliberately keep this module tiny and dependency-free (pure numpy) so it
can be unit-tested without the full service stack.
"""

from __future__ import annotations

from collections.abc import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of integer video ids.

    Reference: Cormack, Clarke, Büttcher (SIGIR '09).

    Score for document ``d`` is::

        RRF(d) = sum over lists L : 1 / (k + rank_L(d))

    where ``rank_L`` is the 1-based rank of ``d`` in list ``L`` (or infinity
    if ``d`` is absent, contributing 0).

    Args:
        ranked_lists: Each element is a list of video_idx values, best-first.
        k: RRF smoothing constant. 60 is the canonical value from the paper.

    Returns:
        List of ``(video_idx, score)`` tuples, sorted by descending score.
        Videos absent from every input list are not returned.
    """
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")

    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, vid in enumerate(ranked, start=1):
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
