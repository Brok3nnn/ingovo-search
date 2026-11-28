"""Contract tests for geographic eligibility and canonical result ordering."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ingovo_search.recommender.search_contract import (
    MATCH_PERCENT_MAX,
    MATCH_PERCENT_MIN,
    NEARBY_MAX_DISTANCE_KM,
    RankingSignals,
    RerankRelevanceSignal,
    ResultScope,
    SearchResultContract,
    canonical_order_key,
    location_match_score,
)


def test_contract_constants_are_explicit() -> None:
    assert NEARBY_MAX_DISTANCE_KM == 30.0
    assert (MATCH_PERCENT_MIN, MATCH_PERCENT_MAX) == (1, 100)


def test_result_scope_has_all_internal_states() -> None:
    assert {scope.value for scope in ResultScope} == {"exact", "nearby", "out_of_scope"}


def test_exact_location_has_full_score_and_no_distance() -> None:
    assert location_match_score(ResultScope.EXACT, distance_km=None) == 1.0
    result = SearchResultContract(
        listing_id="listing-exact",
        rank=1,
        match_percent=100,
        result_scope=ResultScope.EXACT,
        distance_km=None,
    )
    assert result.distance_km is None


def test_nearby_location_score_decreases_with_distance() -> None:
    close = location_match_score(ResultScope.NEARBY, distance_km=2.0)
    medium = location_match_score(ResultScope.NEARBY, distance_km=15.0)
    edge = location_match_score(ResultScope.NEARBY, distance_km=NEARBY_MAX_DISTANCE_KM)
    assert 1.0 > close > medium > edge >= 0.5


def test_out_of_scope_has_zero_location_score() -> None:
    assert location_match_score(ResultScope.OUT_OF_SCOPE, distance_km=31.0) == 0.0


@pytest.mark.parametrize("distance", [None, -0.1, 30.1])
def test_nearby_public_result_requires_verified_in_range_distance(distance: float | None) -> None:
    with pytest.raises(ValidationError):
        SearchResultContract(
            listing_id="listing-nearby",
            rank=1,
            match_percent=80,
            result_scope=ResultScope.NEARBY,
            distance_km=distance,
        )


def test_out_of_scope_result_cannot_be_public() -> None:
    with pytest.raises(ValidationError):
        SearchResultContract(
            listing_id="listing-far",
            rank=1,
            match_percent=40,
            result_scope=ResultScope.OUT_OF_SCOPE,
            distance_km=None,
        )


@pytest.mark.parametrize("percent", [0, 101, 75.5])
def test_match_percent_is_integer_in_closed_contract_range(percent: Any) -> None:
    with pytest.raises(ValidationError):
        SearchResultContract(
            listing_id="listing-percent",
            rank=1,
            match_percent=percent,
            result_scope=ResultScope.EXACT,
            distance_km=None,
        )


def test_rank_is_one_based() -> None:
    with pytest.raises(ValidationError):
        SearchResultContract(
            listing_id="listing-rank",
            rank=0,
            match_percent=75,
            result_scope=ResultScope.EXACT,
            distance_km=None,
        )


def test_canonical_order_preserves_authoritative_rank_even_if_percent_is_not_aligned_yet() -> None:
    lower_percent_better_otherwise = RankingSignals(
        listing_id="a",
        authoritative_rank=1,
        match_percent=75,
        relevance_score=1.0,
        quality_score=1.0,
        freshness_score=1.0,
    )
    higher_percent = RankingSignals(
        listing_id="z",
        authoritative_rank=2,
        match_percent=84,
        relevance_score=0.0,
        quality_score=0.0,
        freshness_score=0.0,
    )
    ordered = sorted([lower_percent_better_otherwise, higher_percent], key=canonical_order_key)
    assert [item.listing_id for item in ordered] == ["a", "z"]


def test_canonical_order_uses_id_only_for_invalid_duplicate_rank() -> None:
    values = [
        RankingSignals(
            listing_id="b",
            authoritative_rank=1,
            match_percent=90,
            relevance_score=0.8,
            quality_score=0.6,
            freshness_score=0.5,
        ),
        RankingSignals(
            listing_id="a",
            authoritative_rank=1,
            match_percent=80,
            relevance_score=0.1,
            quality_score=0.1,
            freshness_score=0.1,
        ),
    ]
    assert [item.listing_id for item in sorted(values, key=canonical_order_key)] == ["a", "b"]


def test_reranker_contract_exposes_signal_not_independent_rank() -> None:
    signal = RerankRelevanceSignal(listing_id="listing-1", relevance_score=0.8)
    assert signal.relevance_score == 0.8
    assert "rank" not in RerankRelevanceSignal.model_fields
