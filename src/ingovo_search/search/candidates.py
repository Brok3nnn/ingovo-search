"""Candidates and scoring: the exact filter, then the scope expansions.

Each expansion runs only while the result is still thin (under ``MIN_VISIBLE_RESULT_COUNT``):
relaxed filters first, then the nearby settlements, then the nearest city with inventory
or, when the visitor allowed it, the whole country with the location dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.api.search_events import CandidatesScoredEvent, ScopeExpandedEvent
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.relaxation import iter_relaxations
from ingovo_search.recommender.sql_filter import run_hard_filter
from ingovo_search.search.constants import (
    AUTOMATIC_LOCATION_FALLBACK_LABEL,
    EXPLICIT_LOCATION_EXPANSION_LABEL,
    MIN_VISIBLE_RESULT_COUNT,
)
from ingovo_search.search.run import ScoredRow, SearchRun, elapsed_ms
from ingovo_search.search.scopes import administrative_geo_city_scope, nearby_filter_scope, ordered_distant_city_scope

# How many rows each scoring pass asks the recommender for.
RECOMMEND_TOP_N = 30


async def nearest_available_distant_ids(
    session: AsyncSession,
    hard_filters: HardFilters,
    ordered_cities: Sequence[str],
) -> tuple[list[str], str | None]:
    """Find inventory in the geographically nearest eligible distant city.

    Prefix probes form a monotonic predicate ("does any matching listing exist
    among the N closest cities?"). Exponential search bounds the first hit and
    binary search identifies its city, avoiding one database call per Hungarian
    settlement while still selecting the nearest one deterministically.
    """

    if not ordered_cities:
        return [], None

    cache: dict[int, list[str]] = {}

    async def probe(prefix_size: int) -> list[str]:
        if prefix_size not in cache:
            cache[prefix_size] = await run_hard_filter(
                session,
                hard_filters,
                city_scope=ordered_cities[:prefix_size],
            )
        return cache[prefix_size]

    total = len(ordered_cities)
    empty_prefix = 0
    upper = 1
    while True:
        ids = await probe(upper)
        if ids:
            break
        if upper == total:
            return [], None
        empty_prefix = upper
        upper = min(total, upper * 2)

    lower = empty_prefix + 1
    while lower < upper:
        middle = (lower + upper) // 2
        if await probe(middle):
            upper = middle
        else:
            lower = middle + 1

    nearest_ids = await probe(lower)
    return nearest_ids, ordered_cities[lower - 1]


async def _score(run: SearchRun, parsed: ParsedQuery, candidate_ids: list[str] | None) -> list[ScoredRow]:
    return await recommend(
        run.deps.db_session,
        parsed,
        query_emb=run.query_emb,
        freetext_pref_emb=run.freetext_emb,
        client=run.deps.llm_client,
        top_n=RECOMMEND_TOP_N,
        candidate_ids=candidate_ids,
    )


def _merge(scored: list[ScoredRow], extra: list[ScoredRow]) -> list[ScoredRow]:
    """The rows already scored stay first; the newcomers follow without duplicates."""
    existing_ids = {prop.id for prop, _ in scored}
    return [*scored, *[(prop, breakdown) for prop, breakdown in extra if prop.id not in existing_ids]]


def _without_location(hard_filters: HardFilters) -> HardFilters:
    return hard_filters.model_copy(update={"city": None, "city_id": None, "district_in": []})


async def _emit_scope(
    run: SearchRun, *, reason: str, before_count: int, scope: Literal["relaxed", "nearby", "distant"]
) -> None:
    await run.emit(
        ScopeExpandedEvent(
            request_id=run.progress.request_id,
            reason=reason,
            before_count=before_count,
            after_count=len(run.scored),
            scope=scope,
        )
    )


async def score_initial(run: SearchRun) -> None:
    """The exact filter where it is needed, then the first scoring pass."""
    timer = run.timer
    run.administrative_city_scope = administrative_geo_city_scope(run.parsed)
    if run.administrative_city_scope is not None:
        step_started = timer.now()
        run.initial_candidate_ids = await run_hard_filter(
            run.deps.db_session,
            run.parsed.hard_filters,
            city_scope=run.administrative_city_scope,
        )
        timer.record("administrative_geo_filter", step_started)
    elif run.progress.emitter is not None:
        # The stream publishes `exact_count`, so obtain it from the hard-filter
        # query instead of relabelling the top-N recommender output as an exact
        # inventory count.
        step_started = timer.now()
        run.initial_candidate_ids = await run_hard_filter(run.deps.db_session, run.parsed.hard_filters)
        timer.record("stream_exact_filter", step_started)
    else:
        run.timings["administrative_geo_filter"] = 0

    step_started = timer.now()
    run.scored = await _score(run, run.parsed, run.initial_candidate_ids)
    timer.record("recommend_initial", step_started)
    await run.emit(
        CandidatesScoredEvent(
            request_id=run.progress.request_id,
            candidate_count=len(run.scored),
            exact_count=len(run.initial_candidate_ids or []),
        )
    )


async def relax_filters(run: SearchRun) -> None:
    """Drop constraints one relaxation step at a time until a dozen rows match; the strict rows stay first."""
    if len(run.scored) >= MIN_VISIBLE_RESULT_COUNT:
        return
    strict_scored = run.scored
    relaxed_steps = list(iter_relaxations(run.parsed.hard_filters))
    probe_ids: list[list[str]] = []
    chosen_idx: int | None = None
    for _label, relaxed_hf in relaxed_steps:
        step_started = run.timer.now()
        ids = await run_hard_filter(
            run.deps.db_session,
            relaxed_hf,
            city_scope=run.administrative_city_scope,
        )
        run.relaxation_ms += elapsed_ms(step_started)
        probe_ids.append(ids)
        if len(ids) >= MIN_VISIBLE_RESULT_COUNT:
            chosen_idx = len(probe_ids) - 1
            break
    if chosen_idx is None and probe_ids:
        chosen_idx = max(range(len(probe_ids)), key=lambda i: len(probe_ids[i]))
    if chosen_idx is None:
        return

    chosen_hf = relaxed_steps[chosen_idx][1]
    best_parsed = run.parsed.model_copy(update={"hard_filters": chosen_hf})
    step_started = run.timer.now()
    best_scored = await _score(run, best_parsed, probe_ids[chosen_idx])
    run.relaxation_ms += elapsed_ms(step_started)
    if len(best_scored) <= len(strict_scored):
        return
    before_count = len(strict_scored)
    run.parsed = best_parsed
    run.relaxation_path = [label for label, _ in relaxed_steps[: chosen_idx + 1]]
    run.scored = _merge(strict_scored, best_scored)
    await _emit_scope(run, reason=" · ".join(run.relaxation_path), before_count=before_count, scope="relaxed")


async def expand_nearby(run: SearchRun) -> None:
    """Add the settlements and districts next to the requested one."""
    if len(run.scored) >= MIN_VISIBLE_RESULT_COUNT:
        return
    nearby_scope = nearby_filter_scope(run.original_parsed)
    if nearby_scope is None:
        return
    step_started = run.timer.now()
    nearby_ids = await run_hard_filter(
        run.deps.db_session,
        run.parsed.hard_filters,
        city_scope=nearby_scope.cities,
        district_scope=nearby_scope.districts,
    )
    run.relaxation_ms += elapsed_ms(step_started)
    if not nearby_ids:
        return
    before_count = len(run.scored)
    step_started = run.timer.now()
    nearby_scored = await _score(run, run.parsed, nearby_ids)
    run.relaxation_ms += elapsed_ms(step_started)
    run.scored = _merge(run.scored, nearby_scored)
    if len(run.scored) > before_count:
        await _emit_scope(run, reason="közeli települések", before_count=before_count, scope="nearby")


async def expand_distant_automatically(run: SearchRun) -> None:
    """Without the visitor's say-so, fall back to the nearest settlement that has an eligible listing."""
    if len(run.scored) >= MIN_VISIBLE_RESULT_COUNT or run.original_parsed.allow_distant_location_expansion:
        return
    ordered_distant_cities = ordered_distant_city_scope(run.original_parsed)
    if not ordered_distant_cities:
        return
    # Keep every non-location relaxation already selected above, but replace the
    # requested city/district with progressively larger nearest-city prefixes. The
    # prefix search returns inventory only from the closest settlement that has an
    # eligible listing.
    distant_hard_filters = _without_location(run.parsed.hard_filters)
    step_started = run.timer.now()
    distant_ids, nearest_city = await nearest_available_distant_ids(
        run.deps.db_session,
        distant_hard_filters,
        ordered_distant_cities,
    )
    run.relaxation_ms += elapsed_ms(step_started)
    existing_ids = {prop.id for prop, _ in run.scored}
    distant_ids = [listing_id for listing_id in distant_ids if listing_id not in existing_ids]
    if not distant_ids:
        return
    distant_parsed = run.parsed.model_copy(update={"hard_filters": distant_hard_filters})
    before_count = len(run.scored)
    step_started = run.timer.now()
    distant_scored = await _score(run, distant_parsed, distant_ids)
    run.relaxation_ms += elapsed_ms(step_started)
    run.scored = _merge(run.scored, distant_scored)
    if len(run.scored) > before_count:
        run.automatic_distant_expansion = True
        run.relaxation_path.append(AUTOMATIC_LOCATION_FALLBACK_LABEL)
        await _emit_scope(
            run, reason=nearest_city or AUTOMATIC_LOCATION_FALLBACK_LABEL, before_count=before_count, scope="distant"
        )


async def expand_distant_explicitly(run: SearchRun) -> None:
    """With the visitor's say-so, search the whole country with the location dropped."""
    if len(run.scored) >= MIN_VISIBLE_RESULT_COUNT or not run.original_parsed.allow_distant_location_expansion:
        return
    expanded_hard_filters = _without_location(run.parsed.hard_filters)
    expanded_parsed = run.parsed.model_copy(update={"hard_filters": expanded_hard_filters})
    step_started = run.timer.now()
    expanded_ids = await run_hard_filter(run.deps.db_session, expanded_hard_filters)
    run.relaxation_ms += elapsed_ms(step_started)
    if not expanded_ids:
        return
    before_count = len(run.scored)
    step_started = run.timer.now()
    expanded_scored = await _score(run, expanded_parsed, expanded_ids)
    run.relaxation_ms += elapsed_ms(step_started)
    run.scored = _merge(run.scored, expanded_scored)
    run.parsed = expanded_parsed
    run.relaxation_path.append(EXPLICIT_LOCATION_EXPANSION_LABEL)
    if len(run.scored) > before_count:
        await _emit_scope(run, reason=EXPLICIT_LOCATION_EXPANSION_LABEL, before_count=before_count, scope="distant")


async def expand_scope(run: SearchRun) -> None:
    """The four expansions in order, each only while the result is still thin."""
    await relax_filters(run)
    await expand_nearby(run)
    await expand_distant_automatically(run)
    await expand_distant_explicitly(run)
    run.timings["recommend_relaxation"] = run.relaxation_ms
