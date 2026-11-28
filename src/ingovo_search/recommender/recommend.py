"""`recommend` — the DB-backed orchestrator over `compute_score`.

SQL hard filter → vector pre-filter (when the pool exceeds 500) → per-candidate geo +
pooled-signal lookups → score → sort top-N. Integration-tested only; the pure scoring
layer in `ingovo_search.recommender.score` carries the heavy unit-test coverage.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import Property
from ingovo_search.lib.llm import LlmClient
from ingovo_search.lib.vector_db import knn_by_query_embedding
from ingovo_search.recommender.candidate_rows import (
    fetch_embedding_similarities,
    fetch_properties_in_order,
    fetch_property_coords,
    fetch_searchable_knn_ids,
    to_property,
)
from ingovo_search.recommender.geo_score import ListingGeo, geo_score, poi_distances_from_features
from ingovo_search.recommender.parser_schema import Anchors, ParsedQuery
from ingovo_search.recommender.pooled_signal import pooled_signals
from ingovo_search.recommender.region_centroid import ResolvedRegionMatcher, resolve_named_region
from ingovo_search.recommender.score import ScoreBreakdown, ScoreInputs, compute_score
from ingovo_search.recommender.sql_filter import run_hard_filter

# Cap on the candidate pool entering full scoring.
VECTOR_PREFILTER_LIMIT = 500
TOP_N_DEFAULT = 30
_logger = structlog.get_logger("ingovo_search.recommender.recommend")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _is_unmodified_browse_query(query: ParsedQuery) -> bool:
    return (
        query.hard_filters == type(query.hard_filters)()
        and query.anchors == Anchors()
        and not query.soft_preferences
        and not query.soft_prefs_freetext
        and query.geo_hints == type(query.geo_hints)()
        and not query.semantic_query
    )


@dataclass(frozen=True, slots=True)
class _Run:
    """One `recommend` call: the session, the request and the timing ledger every stage writes to."""

    session: AsyncSession
    query: ParsedQuery
    query_emb: Sequence[float]
    top_n: int
    candidate_ids: Sequence[str] | None
    timings_ms: dict[str, int]


@dataclass(slots=True)
class _Pool:
    """The candidate ids a search scores, and how they were narrowed down."""

    candidate_ids: list[str]
    factual_similarity_by_id: dict[str, float]
    used_vector_prefilter: bool = False
    hard_filter_count: int = 0


@dataclass(frozen=True, slots=True)
class _ScoreParams:
    """What `compute_score` reads for every candidate alike."""

    query_emb: Sequence[float]
    freetext_pref_emb: Sequence[float] | None
    now: datetime | None
    weights: dict[str, float] | None


@dataclass(slots=True)
class _Inputs:
    """The batched fetches the scoring loop reads from — one query each, no per-listing round-trip."""

    properties: list[SimpleNamespace]
    coords: dict[str, tuple[float, float]]
    pooled_values: dict[str, float]
    content_similarity_by_id: dict[str, float]
    factual_similarity_by_id: dict[str, float]
    softpref_free_similarity_by_id: dict[str, float]


@dataclass(slots=True)
class _Scoring:
    """The scoring loop's shared inputs and its per-step timing counters."""

    query: ParsedQuery
    inputs: _Inputs
    params: _ScoreParams
    region_matcher: ResolvedRegionMatcher | None
    region_match_ms: int = 0
    geo_score_ms: int = 0
    compute_score_ms: int = 0


async def _initial_pool(run: _Run) -> _Pool:
    """The SQL hard filter, or for an unmodified browse query the factual kNN head."""
    step_started = time.perf_counter()
    if run.candidate_ids is None and _is_unmodified_browse_query(run.query):
        knn_rows = await fetch_searchable_knn_ids(run.session, run.query_emb, kind="factual", limit=run.top_n)
        run.timings_ms["hard_filter"] = 0
        run.timings_ms["vector_prefilter"] = _elapsed_ms(step_started)
        return _Pool(
            candidate_ids=[property_id for property_id, _similarity in knn_rows],
            factual_similarity_by_id={property_id: similarity for property_id, similarity in knn_rows},
            used_vector_prefilter=True,
        )
    run.timings_ms["vector_prefilter"] = 0
    step_started = time.perf_counter()
    ids = (
        await run_hard_filter(run.session, run.query.hard_filters)
        if run.candidate_ids is None
        else list(run.candidate_ids)
    )
    run.timings_ms["hard_filter"] = _elapsed_ms(step_started)
    return _Pool(candidate_ids=ids, factual_similarity_by_id={})


async def _prefilter_large_pool(run: _Run, pool: _Pool) -> None:
    """Above `VECTOR_PREFILTER_LIMIT` only the factual kNN head is scored; a browse query keeps `top_n` of it."""
    if len(pool.candidate_ids) <= VECTOR_PREFILTER_LIMIT:
        run.timings_ms["vector_prefilter"] = 0
        return
    pool.used_vector_prefilter = True
    step_started = time.perf_counter()
    knn_rows = await knn_by_query_embedding(
        run.session,
        run.query_emb,
        kind="factual",
        sim_threshold=-1.0,
        limit=VECTOR_PREFILTER_LIMIT,
        candidate_ids=pool.candidate_ids,
    )
    pool.factual_similarity_by_id = {row.property_id: row.similarity for row in knn_rows}
    pool.candidate_ids = [row.property_id for row in knn_rows]
    if _is_unmodified_browse_query(run.query):
        pool.candidate_ids = pool.candidate_ids[: run.top_n]
    run.timings_ms["vector_prefilter"] = _elapsed_ms(step_started)


async def _fetch_inputs(
    session: AsyncSession, query: ParsedQuery, pool: _Pool, params: _ScoreParams, timings_ms: dict[str, int]
) -> _Inputs:
    """The batched property, coordinate, pooled-signal and similarity fetches."""
    step_started = time.perf_counter()
    properties = await fetch_properties_in_order(session, pool.candidate_ids)
    timings_ms["property_fetch"] = _elapsed_ms(step_started)

    step_started = time.perf_counter()
    coords = await fetch_property_coords(session, [prop.id for prop in properties])
    timings_ms["coord_fetch"] = _elapsed_ms(step_started)

    step_started = time.perf_counter()
    pooled_values = await pooled_signals(session, [prop.id for prop in properties], params.query_emb)
    timings_ms["pooled_signal"] = _elapsed_ms(step_started)

    property_ids = [prop.id for prop in properties]
    step_started = time.perf_counter()
    content_similarity_by_id = await fetch_embedding_similarities(
        session, property_ids, params.query_emb, "descriptive"
    )
    factual_similarity_by_id = pool.factual_similarity_by_id
    if not factual_similarity_by_id:
        factual_similarity_by_id = await fetch_embedding_similarities(
            session, property_ids, params.query_emb, "factual"
        )
    softpref_free_similarity_by_id = (
        await fetch_embedding_similarities(session, property_ids, params.freetext_pref_emb, "descriptive")
        if params.freetext_pref_emb is not None and query.soft_prefs_freetext
        else {}
    )
    timings_ms["embedding_similarity"] = _elapsed_ms(step_started)
    return _Inputs(
        properties=properties,
        coords=coords,
        pooled_values=pooled_values,
        content_similarity_by_id=content_similarity_by_id,
        factual_similarity_by_id=factual_similarity_by_id,
        softpref_free_similarity_by_id=softpref_free_similarity_by_id,
    )


def _score_one(scoring: _Scoring, prop: SimpleNamespace) -> ScoreBreakdown:
    """One candidate's score, in memory; the timing counters accumulate on `scoring`."""
    inputs = scoring.inputs
    params = scoring.params
    prop_lat, prop_lng = inputs.coords.get(prop.id, (None, None))
    named_region: float | None = None
    if scoring.region_matcher is not None:
        step_started = time.perf_counter()
        named_region = scoring.region_matcher.score(
            property_lat=prop_lat,
            property_lng=prop_lng,
            property_district=prop.district,
        )
        scoring.region_match_ms += _elapsed_ms(step_started)
    step_started = time.perf_counter()
    geo_value = geo_score(
        scoring.query,
        ListingGeo(
            lat=prop_lat,
            lng=prop_lng,
            district=prop.district,
            city=prop.city,
            poi_distances_m=poi_distances_from_features(getattr(prop, "features", None)),
        ),
        named_region_score=named_region,
    )
    scoring.geo_score_ms += _elapsed_ms(step_started)
    step_started = time.perf_counter()
    breakdown = compute_score(
        prop,
        scoring.query,
        ScoreInputs(
            query_emb=params.query_emb,
            freetext_pref_emb=params.freetext_pref_emb,
            content_similarity=inputs.content_similarity_by_id.get(prop.id, 0.0),
            factual_similarity=inputs.factual_similarity_by_id.get(prop.id, 0.0),
            softpref_free_similarity=inputs.softpref_free_similarity_by_id.get(prop.id)
            if inputs.softpref_free_similarity_by_id
            else None,
            pooled_signal_value=inputs.pooled_values.get(prop.id, 0.0),
            geo_score_value=geo_value,
            now=params.now,
            weights=params.weights,
        ),
    )
    scoring.compute_score_ms += _elapsed_ms(step_started)
    return breakdown


async def _resolve_region(
    session: AsyncSession, region_name: str | None, *, client: LlmClient
) -> tuple[ResolvedRegionMatcher | None, int]:
    """Resolve the region once per search (one DB fetch + at most one cold-start LLM hint).

    Every candidate is then scored against it in memory — the per-listing
    `get_region_centroid` round-trip is gone. Returns the matcher and its milliseconds.
    """
    step_started = time.perf_counter()
    region_matcher = (
        await resolve_named_region(session, region_name, client=client) if region_name is not None else None
    )
    return region_matcher, _elapsed_ms(step_started)


def _log_recommend_timing(
    timings_ms: dict[str, int], pool: _Pool, *, candidate_count: int, returned_count: int, region_name: str | None
) -> None:
    _logger.info(
        "recommend timing",
        timings_ms=timings_ms,
        hard_filter_count=pool.hard_filter_count,
        candidate_count=candidate_count,
        returned_count=returned_count,
        used_vector_prefilter=pool.used_vector_prefilter,
        region_name=region_name,
    )


async def recommend(  # noqa: PLR0913 -- the orchestrator's keyword surface is the search request itself
    session: AsyncSession,
    query: ParsedQuery,
    *,
    query_emb: Sequence[float],
    freetext_pref_emb: Sequence[float] | None = None,
    client: LlmClient,
    top_n: int = TOP_N_DEFAULT,
    now: datetime | None = None,
    weights: dict[str, float] | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> list[tuple[Property, ScoreBreakdown]]:
    """Rank listings against `query`. Returns up to `top_n` (Property, breakdown).

    Pipeline: SQL hard filter → vector pre-filter (top 500 by factual similarity
    when the SQL pool exceeds 500) → batched property/coord/pooled-signal fetches
    (one query each) → region resolved once → full score in memory → sort by
    total. There is no per-listing DB round-trip in the scoring loop.

    `candidate_ids` lets a caller that already ran the hard filter (e.g. the
    relaxation probe in the search pipeline) pass the matching ids in, skipping a
    redundant hard-filter query. When omitted, the filter runs as usual.
    """
    total_started = time.perf_counter()
    timings_ms: dict[str, int] = {}
    run = _Run(
        session=session,
        query=query,
        query_emb=query_emb,
        top_n=top_n,
        candidate_ids=candidate_ids,
        timings_ms=timings_ms,
    )
    pool = await _initial_pool(run)
    pool.hard_filter_count = len(pool.candidate_ids)
    if not pool.candidate_ids:
        _log_recommend_timing(
            {**timings_ms, "total": _elapsed_ms(total_started)},
            pool,
            candidate_count=0,
            returned_count=0,
            region_name=query.geo_hints.region_name,
        )
        return []
    await _prefilter_large_pool(run, pool)

    params = _ScoreParams(query_emb=query_emb, freetext_pref_emb=freetext_pref_emb, now=now, weights=weights)
    inputs = await _fetch_inputs(session, query, pool, params, timings_ms)

    region_name = query.geo_hints.region_name
    region_matcher, region_resolve_ms = await _resolve_region(session, region_name, client=client)
    scoring = _Scoring(query=query, inputs=inputs, params=params, region_matcher=region_matcher)
    scoring_started = time.perf_counter()
    scored = [(prop, _score_one(scoring, prop)) for prop in inputs.properties]
    timings_ms["scoring_loop"] = _elapsed_ms(scoring_started)
    timings_ms["region_resolve"] = region_resolve_ms
    timings_ms["region_match"] = scoring.region_match_ms
    timings_ms["geo_score"] = scoring.geo_score_ms
    timings_ms["compute_score"] = scoring.compute_score_ms

    step_started = time.perf_counter()
    # Tiebreak on id: equal-scored listings must rank deterministically, or
    # result order shuffles between identical requests (unstable pagination,
    # cache misses, and snapshot-test drift — seen on 2026-07-03 on queries
    # whose candidates all tie).
    scored.sort(key=lambda pair: (-pair[1].total, str(pair[0].id)))
    timings_ms["sort"] = _elapsed_ms(step_started)
    results = [(to_property(prop, inputs.coords.get(prop.id)), breakdown) for prop, breakdown in scored[:top_n]]
    timings_ms["total"] = _elapsed_ms(total_started)
    _log_recommend_timing(
        timings_ms, pool, candidate_count=len(inputs.properties), returned_count=len(results), region_name=region_name
    )
    return results
