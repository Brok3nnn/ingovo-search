"""Preserve the authoritative ranker order and align displayed percentages."""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.types.errors import RankingGuardrailViolation

CANONICAL_RANKING_VERSION = "authoritative-order-v3-running-minimum"

# A single, greppable event name so a monitoring rule can alert on any public
# ordering-invariant breach (`event="alert.order_invariant_violation"`).
_logger = structlog.get_logger(__name__)


class MatchOrderError(RankingGuardrailViolation):
    """Raised when a public result list violates rank or percentage order."""


def canonical_rank(candidates: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    """Keep candidate IDs in ranker order and make the displayed percentages non-increasing.

    Each candidate arrives with its own absolute match percentage. The displayed
    value is the running minimum down the ranker's order: a listing never shows
    more than its own percentage, and shows less only when a listing ranked above
    it scored lower. The candidates themselves are never reordered.
    """

    ranked: list[ScoredCandidate] = []
    ceiling = 100
    for index, candidate in enumerate(candidates, start=1):
        ceiling = min(ceiling, candidate.match_percent)
        ranked.append(candidate.model_copy(update={"rank": index, "match_percent": ceiling}))
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
