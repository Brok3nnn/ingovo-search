"""Preserve the authoritative ranker order and align displayed percentages."""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.types.errors import RankingGuardrailViolation

CANONICAL_RANKING_VERSION = "authoritative-order-v2"

# R11.6 — a single, greppable event name so a monitoring rule can alert on any
# public ordering-invariant breach (`event="alert.order_invariant_violation"`).
_logger = structlog.get_logger(__name__)


class MatchOrderError(RankingGuardrailViolation):
    """Raised when a public result list violates rank or percentage order."""


def canonical_rank(candidates: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    """Keep candidate IDs in ranker order and align percentages to those ranks.

    The absolute match calculator provides the percentage value distribution.
    Sorting only those values — never the candidates — guarantees that the
    original ranker order remains authoritative while displayed percentages are
    monotonically non-increasing along it.
    """

    aligned_percentages = sorted((candidate.match_percent for candidate in candidates), reverse=True)
    ranked = [
        candidate.model_copy(update={"rank": index, "match_percent": aligned_percentages[index - 1]})
        for index, candidate in enumerate(candidates, start=1)
    ]
    assert_match_order(ranked)
    return ranked


def assert_match_order(candidates: Sequence[ScoredCandidate]) -> None:
    """Reject a non-contiguous rank or increasing displayed percentage."""

    for index, candidate in enumerate(candidates, start=1):
        if candidate.rank != index:
            _logger.error(
                "alert.order_invariant_violation",
                kind="non_contiguous_rank",
                index=index,
                rank=candidate.rank,
                listing_id=candidate.listing_id,
            )
            raise MatchOrderError(f"non-contiguous rank at index {index}: got {candidate.rank}")
        if index > 1 and candidate.match_percent > candidates[index - 2].match_percent:
            previous = candidates[index - 2]
            _logger.error(
                "alert.order_invariant_violation",
                kind="match_percent_increased",
                rank=index,
                previous_percent=previous.match_percent,
                percent=candidate.match_percent,
                listing_id=candidate.listing_id,
            )
            raise MatchOrderError(
                f"match_percent increased at rank {index}: {previous.match_percent} -> {candidate.match_percent}"
            )
