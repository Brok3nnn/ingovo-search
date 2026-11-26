"""The §16 hard-filter count and the R11 per-search observability line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.recommender.search_contract import (
    ResultScope,
)
from ingovo_search.search.constants import search_logger


def count_hard_filters(parsed: ParsedQuery) -> int:
    """Number of distinct hard-filter constraints carried by the parsed query.

    Used as a §16 SearchLog field so the weekly report can correlate
    return-count with the strictness of the input filter set. Empty
    lists and `None` count as "no filter".
    """
    hf = parsed.hard_filters
    return sum(
        1
        for value in (
            hf.city,
            hf.transaction_type,
            hf.rooms,
            hf.rooms_min,
            hf.area_m2_min,
            hf.area_m2_max,
            hf.price_huf_min,
            hf.price_huf_max,
            hf.floor_min,
            hf.near,
        )
        if value is not None
    ) + sum(1 for collection in (hf.district_in, hf.property_type_in, hf.property_type_exclude) if collection)


@dataclass(frozen=True, slots=True)
class SearchObservation:
    """What one finished search is measured by (R11.1-R11.5)."""

    results: list[ScoredCandidate]
    total_candidates: int
    relaxation_path: list[str]
    rerank_pre_count: int | None
    rerank_post_count: int | None


def log_search_observability(parsed: ParsedQuery, observation: SearchObservation, *, logger: Any | None = None) -> None:
    """R11.1-R11.5 structured per-search observability for monitoring/alerting."""
    results = observation.results
    total_candidates = observation.total_candidates
    relaxation_path = observation.relaxation_path
    hard = parsed.hard_filters
    scope_counts: dict[str, int] = {"exact": 0, "nearby": 0, "out_of_scope": 0}
    nearby_distances: list[float] = []
    for candidate in results:
        scope = candidate.result_scope.value if candidate.result_scope else ResultScope.EXACT.value
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        if candidate.result_scope is ResultScope.NEARBY and candidate.distance_km is not None:
            nearby_distances.append(round(candidate.distance_km, 2))
    active_logger = logger if logger is not None else search_logger
    active_logger.info(
        "search.observability",
        location={
            "city": hard.city,
            "city_id": hard.city_id,
            "district_in": hard.district_in,
            "near": hard.near is not None,
            "region": parsed.geo_hints.region_name,
        },
        returned_count=len(results),
        sql_candidate_count=total_candidates,
        excluded_count=max(0, total_candidates - len(results)),
        scope_counts=scope_counts,
        nearby_distances_km=sorted(nearby_distances),
        match_order_ok=True,
        rerank_pre_count=observation.rerank_pre_count,
        rerank_post_count=observation.rerank_post_count,
        relaxed=bool(relaxation_path),
        relaxation_path=relaxation_path,
    )
