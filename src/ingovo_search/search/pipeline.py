"""The search pipeline, one stage per function.

Shared between ``POST /api/search`` and ``POST /api/search/clarify`` so both endpoints
route through identical scoring, caching and guardrails. The only difference is where
the parsed query comes from: the parser in the search path, the session's last parsed
query in the clarify path.
"""

from __future__ import annotations

import hashlib

from sqlalchemy.exc import SQLAlchemyError

from ingovo_search.lib.cache import get_json, set_json
from ingovo_search.lib.llm import set_llm_phase
from ingovo_search.lib.observability import (
    EVENT_CLARIFICATION_SHOWN,
    EVENT_SEARCH_RUN,
    analytics_distinct_id,
    log_search,
    record_latency_ms,
    track_event,
)
from ingovo_search.recommender.canonical_ranking import assert_match_order, canonical_rank
from ingovo_search.recommender.clarification_policy import Clarification, apply_clarification, should_clarify
from ingovo_search.recommender.refinement_suggestions import suggest_refinements
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.search.candidates import expand_scope, score_initial
from ingovo_search.search.constants import narrator_logger, search_logger
from ingovo_search.search.context import build_search_context
from ingovo_search.search.models import SCORE_VERSION, SearchResponse
from ingovo_search.search.observability import SearchObservation, count_hard_filters, log_search_observability
from ingovo_search.search.query_embedding import embed_freetext, embed_query
from ingovo_search.search.refinements import apply_refinements
from ingovo_search.search.reranking import RerankStage, run_rerank
from ingovo_search.search.response_cache import (
    cached_response_is_publicly_visible,
    discard_stale_response_cache_entry,
    response_cache_key,
    response_cache_ttl_seconds,
)
from ingovo_search.search.run import (
    SearchDependencies,
    SearchInput,
    SearchOptions,
    SearchProgress,
    SearchRun,
    SearchTimer,
)
from ingovo_search.search.scoring import attach_hero_narratives, build_scored_candidates


def _track_search_run(run: SearchRun, resp: SearchResponse, *, cache_hit: bool) -> None:
    """Log the `search_run` product event.

    Deliberately property-only, never the query text: the sentence a user types
    can name a street, a workplace or a school run, so it is personal data the
    moment it leaves our own database. The shape of the search (how many filters,
    how many results, was it cached, did it need relaxing) is what the funnel
    needs, and it carries none of that.
    """
    track_event(
        EVENT_SEARCH_RUN,
        distinct_id=analytics_distinct_id(session_id=run.state.session_id),
        properties={
            "hard_filter_count": count_hard_filters(run.parsed),
            "returned_count": len(resp.results),
            "total_candidates": resp.total_candidates,
            "latency_ms": run.elapsed(),
            "cache_hit": cache_hit,
            "signed_in": run.options.user_signed_in,
            "relaxed": resp.relaxed,
            # Only the full pipeline knows how many relaxation steps it took; a
            # cache hit reports the flag alone, hence 0 rather than a guess.
            "relaxation_steps": 0 if cache_hit else len(run.relaxation_path),
            "has_clarification": False if cache_hit else resp.clarification is not None,
            "score_version": SCORE_VERSION,
        },
    )


async def _write_search_log(
    run: SearchRun, *, results: list[ScoredCandidate], total_candidates: int, cache_hit: bool, rerank_outcome: str
) -> None:
    """The SearchLog row is analytics: losing it must not fail a search that was answered."""
    try:
        await log_search(
            run.deps.db_session,
            query_hash=run.parsed_hash,
            parsed_query=run.parsed.model_dump(mode="json"),
            hard_filter_count=count_hard_filters(run.parsed),
            sql_candidate_count=total_candidates,
            returned_count=len(results),
            latency_ms=run.elapsed(),
            score_version=SCORE_VERSION,
            top_result_ids=[r.listing_id for r in results[:10]],
            cache_hit=cache_hit,
            rerank_backend=run.options.rerank_backend,
            rerank_outcome=rerank_outcome,
        )
        await run.deps.db_session.commit()
    except (SQLAlchemyError, OSError):
        # Only the commit touches the database here (`log_search` is a plain
        # `session.add`). SQLAlchemy wraps driver errors raised by a statement,
        # but a commit on a dropped socket or past the command timeout escapes
        # as a raw OSError / TimeoutError, hence the second type.
        await run.deps.db_session.rollback()
        narrator_logger.warning(
            "log_search (cache-hit) failed; search served without its SearchLog row"
            if cache_hit
            else "log_search failed; search served without its SearchLog row",
            extra={"query_hash": run.parsed_hash},
            exc_info=True,
        )


def _begin(
    search_input: SearchInput,
    deps: SearchDependencies,
    options: SearchOptions,
    progress: SearchProgress,
    timer: SearchTimer,
) -> SearchRun:
    """Apply the session's answers and refinements, then fix the query hash and the cache key."""
    state = search_input.state
    parsed = search_input.raw_parsed
    for answer in state.clarification_answers:
        parsed = apply_clarification(parsed, answer)
    if state.refinement_overrides:
        parsed = apply_refinements(parsed, state.refinement_overrides)
    parsed_hash = hashlib.sha256(parsed.model_dump_json().encode("utf-8")).hexdigest()
    return SearchRun(
        state=state,
        parsed=parsed,
        # Snapshot the parsed query BEFORE relaxation may overwrite its filters,
        # so `revert_relaxation` chips can restore the original constraints.
        original_parsed=parsed,
        parsed_hash=parsed_hash,
        cache_key=response_cache_key(parsed_hash, options),
        deps=deps,
        options=options,
        progress=progress,
        timer=timer,
    )


async def _serve_cache_hit(run: SearchRun, cached: SearchResponse) -> None:
    """A visible cached response answers the search; log and track it as a hit."""
    timings = run.timings
    timings["total"] = run.timer.record("total", run.timer.started)
    search_logger.info(
        "search response cache hit",
        cache_key=run.cache_key,
        returned_count=len(cached.results),
    )
    search_logger.info(
        "search pipeline timing",
        timings_ms=timings,
        cache_hit=True,
        returned_count=len(cached.results),
        total_candidates=cached.total_candidates,
        include_narratives=run.options.include_narratives,
        include_rerank=run.options.include_rerank,
        relaxed=cached.relaxed,
    )
    await _write_search_log(
        run,
        results=cached.results,
        total_candidates=cached.total_candidates,
        cache_hit=True,
        rerank_outcome="cache_hit",
    )
    _track_search_run(run, cached, cache_hit=True)


async def _read_response_cache(run: SearchRun) -> SearchResponse | None:
    """The cached response for this query, if there is one and every row is still public."""
    timer = run.timer
    step_started = timer.now()
    cached = await get_json(run.deps.cache, run.cache_key, SearchResponse)
    timer.record("response_cache_read", step_started)
    if cached is not None:
        assert_match_order(cached.results)
        step_started = timer.now()
        cache_rows_visible = await cached_response_is_publicly_visible(run.deps.db_session, cached)
        timer.record("response_cache_visibility_check", step_started)
        if cache_rows_visible:
            await _serve_cache_hit(run, cached)
            return cached
        await discard_stale_response_cache_entry(run.deps.cache, run.cache_key)
        search_logger.warning(
            "search response cache rejected by current public visibility",
            cache_key=run.cache_key,
            returned_count=len(cached.results),
        )
    search_logger.info("search response cache miss", cache_key=run.cache_key)
    return None


async def _announce_parsed(run: SearchRun) -> None:
    """The intent and image events the request parsed before the pipeline, now that it is a miss."""
    if run.progress.intent_event is not None:
        await run.emit(run.progress.intent_event)
    if run.progress.image_event is not None:
        await run.emit(run.progress.image_event)


def _score_candidates(run: SearchRun) -> tuple[list[ScoredCandidate], Clarification | None]:
    """The public candidates with their match percentages, and whether to ask back first."""
    match_parsed = (
        run.original_parsed.model_copy(update={"allow_distant_location_expansion": True})
        if run.automatic_distant_expansion
        else run.original_parsed
    )
    scored_candidates = build_scored_candidates(run.scored, run.parsed, match_parsed)
    return scored_candidates, should_clarify(run.parsed, scored_candidates)


async def _attach_narratives(run: SearchRun, results: list[ScoredCandidate]) -> list[ScoredCandidate]:
    if run.options.include_narratives and results and run.parsed.semantic_query:
        step_started = run.timer.now()
        set_llm_phase(run.deps.llm_client, "narrative")
        results = await attach_hero_narratives(
            results,
            run.scored,
            run.parsed,
            client=run.deps.llm_client,
            cache=run.deps.cache,
            locale=run.options.narrative_locale,
        )
        run.timer.record("narratives", step_started)
    else:
        run.timings["narratives"] = 0
    return results


def _build_response(
    run: SearchRun, results: list[ScoredCandidate], clarification: Clarification | None
) -> SearchResponse:
    results = canonical_rank(results)
    assert_match_order(results)
    suggestions = suggest_refinements(
        original_parsed=run.original_parsed,
        relaxed_parsed=run.parsed,
        results=results,
        relaxation_path=run.relaxation_path,
    )
    resp = SearchResponse(
        results=results,
        clarification=clarification,
        total_candidates=len(run.scored),
        score_version=SCORE_VERSION,
        relaxed=bool(run.relaxation_path),
        relaxation_path=run.relaxation_path,
        refinement_suggestions=suggestions,
        search_context=build_search_context(run.original_parsed, run.relaxation_path),
    )
    assert_match_order(resp.results)
    return resp


async def _write_response_cache(run: SearchRun, resp: SearchResponse, rerank: RerankStage) -> None:
    """Cache the response, capped at the earliest listing expiry; never under a fallback arm."""
    timings = run.timings
    response_cache_ttl = response_cache_ttl_seconds(resp.results, run.scored)
    if rerank.gemini_fallback:
        # Jev is tried again after the cooldown; do not pin Gemini's order under
        # the Jev cache key (this also skips the raw response cache).
        response_cache_ttl = 0
    timings["response_cache_ttl_seconds"] = response_cache_ttl
    if response_cache_ttl > 0:
        step_started = run.timer.now()
        await set_json(run.deps.cache, run.cache_key, resp, ttl_seconds=response_cache_ttl)
        run.timer.record("response_cache_write", step_started)
    else:
        # A hit always rechecks visibility, but avoid creating a cache entry
        # that is already at its listing expiry boundary.
        timings["response_cache_write"] = 0


async def _finish(run: SearchRun, resp: SearchResponse, rerank: RerankStage) -> None:
    """Observability, the response cache, the SearchLog row and the analytics events."""
    results = resp.results
    log_search_observability(
        run.original_parsed,
        SearchObservation(
            results=results,
            total_candidates=len(run.scored),
            relaxation_path=run.relaxation_path,
            rerank_pre_count=rerank.pre_count,
            rerank_post_count=rerank.post_count,
        ),
        logger=search_logger,
    )
    await _write_response_cache(run, resp, rerank)
    timings = run.timings
    timings["total"] = run.timer.record("total", run.timer.started)
    search_logger.info(
        "search pipeline timing",
        timings_ms=timings,
        cache_hit=False,
        returned_count=len(results),
        total_candidates=len(run.scored),
        include_narratives=run.options.include_narratives,
        include_rerank=run.options.include_rerank,
        rerank_backend=run.options.rerank_backend,
        relaxed=bool(run.relaxation_path),
        relaxation_steps=len(run.relaxation_path),
    )
    await _write_search_log(
        run, results=results, total_candidates=len(run.scored), cache_hit=False, rerank_outcome=rerank.outcome
    )
    _track_search_run(run, resp, cache_hit=False)
    if resp.clarification is not None:
        track_event(
            EVENT_CLARIFICATION_SHOWN,
            distinct_id=analytics_distinct_id(session_id=run.state.session_id),
            properties={
                "reason": resp.clarification.reason,
                "option_count": len(resp.clarification.options),
            },
        )


async def run_search_pipeline(
    search_input: SearchInput,
    deps: SearchDependencies,
    options: SearchOptions,
    progress: SearchProgress | None = None,
    timer: SearchTimer | None = None,
) -> SearchResponse:
    """Apply session state to the parsed query, run the pipeline, return the response."""
    run = _begin(search_input, deps, options, progress or SearchProgress(), timer or SearchTimer.start())
    with record_latency_ms() as elapsed:
        run.elapsed = elapsed
        cached = await _read_response_cache(run)
    if cached is not None:
        return cached

    await _announce_parsed(run)
    await embed_query(run)
    await embed_freetext(run)
    await score_initial(run)
    await expand_scope(run)
    scored_candidates, clarification = _score_candidates(run)
    rerank = await run_rerank(run, scored_candidates)
    results = await _attach_narratives(run, rerank.results)
    resp = _build_response(run, results, clarification)
    await _finish(run, resp, rerank)
    return resp
