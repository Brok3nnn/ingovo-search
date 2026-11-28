"""Canonical ranking and the ordering-invariant guard."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from ingovo_search.recommender.canonical_ranking import MatchOrderError, assert_match_order, canonical_rank
from ingovo_search.recommender.match_percent import MatchBreakdown
from ingovo_search.recommender.score import ScoreBreakdown, ScoredCandidate


@dataclass(frozen=True)
class _Signals:
    """The ranking signals a test candidate carries besides its match percent."""

    relevance: float = 0.5
    quality: float = 0.0
    freshness: float = 0.0
    explore: float = 0.0


_DEFAULT_SIGNALS = _Signals()


def _candidate(listing_id: str, match_percent: int, signals: _Signals = _DEFAULT_SIGNALS) -> ScoredCandidate:
    breakdown = ScoreBreakdown(
        content=0.0,
        factual=0.0,
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.04 * signals.quality,
        geo=0.0,
        pooled=0.0,
        recency=0.04 * signals.freshness,
        explore=0.02 * signals.explore,
        total=0.04 * signals.quality + 0.04 * signals.freshness + 0.02 * signals.explore,
    )
    match = MatchBreakdown(
        active_weight=1.0,
        weighted_score=signals.relevance,
        hard_filter_cap=100,
        amenity_cap=100,
        match_percent=match_percent,
    )
    return ScoredCandidate(
        listing_id=listing_id,
        score=breakdown.total,
        match_percent=match_percent,
        match_breakdown=match,
        breakdown=breakdown,
    )


def test_a_higher_score_ranked_below_a_lower_one_shows_the_lower_one() -> None:
    ranked = canonical_rank([_candidate("a", 75), _candidate("b", 84)])
    assert [item.listing_id for item in ranked] == ["a", "b"]
    assert [item.match_percent for item in ranked] == [75, 75]
    assert [item.rank for item in ranked] == [1, 2]


def test_no_listing_is_shown_above_its_own_percentage() -> None:
    own = [62, 90, 71, 71, 40, 88]
    ranked = canonical_rank([_candidate(f"l{index}", percent) for index, percent in enumerate(own)])
    assert [item.match_percent for item in ranked] == [62, 62, 62, 62, 40, 40]
    assert all(item.match_percent <= percent for item, percent in zip(ranked, own, strict=True))


def test_ties_preserve_authoritative_input_order() -> None:
    candidates = [
        _candidate("d", 90, _Signals(relevance=0.8, quality=0.9, freshness=0.9)),
        _candidate("c", 90, _Signals(relevance=0.9, quality=0.1, freshness=0.1)),
        _candidate("b", 90, _Signals(relevance=0.9, quality=0.8, freshness=0.1)),
        _candidate("a", 90, _Signals(relevance=0.9, quality=0.8, freshness=0.9)),
    ]
    assert [item.listing_id for item in canonical_rank(candidates)] == ["d", "c", "b", "a"]


def test_repeated_ranking_is_identical() -> None:
    candidates = [_candidate("z", 80), _candidate("a", 80), _candidate("m", 90)]
    first = canonical_rank(candidates)
    second = canonical_rank(candidates)
    assert [(item.listing_id, item.rank) for item in first] == [(item.listing_id, item.rank) for item in second]


def test_explore_never_moves_candidate() -> None:
    low_explore = canonical_rank(
        [_candidate("b", 80, _Signals(explore=0.0)), _candidate("a", 80, _Signals(explore=1.0))]
    )
    high_explore = canonical_rank(
        [_candidate("b", 80, _Signals(explore=1.0)), _candidate("a", 80, _Signals(explore=0.0))]
    )
    assert [item.listing_id for item in low_explore] == ["b", "a"]
    assert [item.listing_id for item in high_explore] == ["b", "a"]


def test_assert_match_order_rejects_increase() -> None:
    invalid = [
        _candidate("a", 75).model_copy(update={"rank": 1}),
        _candidate("b", 84).model_copy(update={"rank": 2}),
    ]
    with pytest.raises(MatchOrderError, match="increased"):
        assert_match_order(invalid)


def test_assert_match_order_rejects_rank_gap() -> None:
    invalid = [_candidate("a", 84).model_copy(update={"rank": 2})]
    with pytest.raises(MatchOrderError, match="non-contiguous"):
        assert_match_order(invalid)


def test_order_invariant_violation_emits_alert_log() -> None:
    # A monitoring rule alerts on event="alert.order_invariant_violation".
    # Patch the module logger directly so the assertion is independent of
    # structlog's global config / first-use caching in the full suite.
    invalid = [
        _candidate("a", 75).model_copy(update={"rank": 1}),
        _candidate("b", 84).model_copy(update={"rank": 2}),
    ]
    with patch("ingovo_search.recommender.canonical_ranking._logger") as mock_logger, pytest.raises(MatchOrderError):
        assert_match_order(invalid)

    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args.args[0] == "alert.order_invariant_violation"
    assert mock_logger.error.call_args.kwargs["kind"] == "match_percent_increased"
