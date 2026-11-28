"""Shared domain contract for geographic eligibility and canonical ranking.

This module deliberately contains no database or API orchestration: the search
and scoring code share these validated types and constants.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

SEARCH_CONTRACT_VERSION = "search-contract-v3-authoritative-order-administrative-geo-scope"

NEARBY_MAX_DISTANCE_KM = 30.0
"""Maximum verified distance for an automatically offered nearby alternative."""

MATCH_PERCENT_MIN = 1
MATCH_PERCENT_MAX = 100

NEARBY_LOCATION_SCORE_MAX = 0.85
NEARBY_LOCATION_SCORE_MIN = 0.50


class ResultScope(StrEnum):
    """Internal geographic eligibility state for one candidate."""

    EXACT = "exact"
    NEARBY = "nearby"
    OUT_OF_SCOPE = "out_of_scope"


def location_match_score(scope: ResultScope, *, distance_km: float | None) -> float:
    """Return the contract-level location component in the closed ``0..1`` range."""

    if scope is ResultScope.EXACT:
        if distance_km is not None:
            raise ValueError("exact results must not carry distance_km")
        return 1.0
    if scope is ResultScope.OUT_OF_SCOPE:
        return 0.0
    if distance_km is None or distance_km < 0.0 or distance_km > NEARBY_MAX_DISTANCE_KM:
        raise ValueError("nearby results require a verified distance within the configured limit")
    span = NEARBY_LOCATION_SCORE_MAX - NEARBY_LOCATION_SCORE_MIN
    return NEARBY_LOCATION_SCORE_MAX - span * (distance_km / NEARBY_MAX_DISTANCE_KM)


class SearchResultContract(BaseModel):
    """Required public rank and geography fields for a canonical search result."""

    model_config = {"frozen": True}

    listing_id: str = Field(min_length=1)
    rank: int = Field(strict=True, ge=1)
    match_percent: int = Field(strict=True, ge=MATCH_PERCENT_MIN, le=MATCH_PERCENT_MAX)
    result_scope: ResultScope
    distance_km: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_public_scope(self) -> SearchResultContract:
        if self.result_scope is ResultScope.OUT_OF_SCOPE:
            raise ValueError("out_of_scope candidates cannot be exposed as public results")
        if self.result_scope is ResultScope.EXACT:
            if self.distance_km is not None:
                raise ValueError("exact results must have null distance_km")
            return self
        if self.distance_km is None:
            raise ValueError("nearby results require distance_km")
        if self.distance_km > NEARBY_MAX_DISTANCE_KM:
            raise ValueError("nearby result exceeds NEARBY_MAX_DISTANCE_KM")
        return self


class RankingSignals(BaseModel):
    """Diagnostics attached to an already-authoritative ranker position."""

    model_config = {"frozen": True}

    listing_id: str = Field(min_length=1)
    authoritative_rank: int = Field(strict=True, ge=1)
    match_percent: int = Field(strict=True, ge=MATCH_PERCENT_MIN, le=MATCH_PERCENT_MAX)
    relevance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)


def canonical_order_key(signals: RankingSignals) -> tuple[int, str]:
    """Preserve the original ranker position; ID only stabilises invalid rank ties."""

    return (signals.authoritative_rank, signals.listing_id)


class RerankRelevanceSignal(BaseModel):
    """Allowed reranker output: a relevance signal, never an independent final rank."""

    model_config = {"frozen": True}

    listing_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
