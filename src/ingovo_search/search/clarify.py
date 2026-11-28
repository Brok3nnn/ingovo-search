"""``POST /api/search/clarify``: the buyer's answer, then the pipeline again.

Clarify never re-parses. The answer is appended to the session state, and the shared
pipeline (``pipeline.run_search_pipeline``) runs again on the parsed query the
previous ``/api/search`` call stored in the session.
"""

from __future__ import annotations

import time

from fastapi import Request

from ingovo_search.api.deps import get_jev_client
from ingovo_search.api.search_session import ClarifyRequest, get_or_create_session_state
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.observability import (
    EVENT_CLARIFICATION_ANSWERED,
    EVENT_CLARIFICATION_SKIPPED,
    analytics_distinct_id,
    track_event,
)
from ingovo_search.lib.session import SessionState
from ingovo_search.search.models import SearchEnvelope, SearchResponse, ok_with_timings
from ingovo_search.search.pipeline import run_search_pipeline
from ingovo_search.search.request import SessionContext
from ingovo_search.search.reranking import serving_rerank_backend
from ingovo_search.search.run import SearchDependencies, SearchInput, SearchOptions, SearchTimer
from ingovo_search.search.shadow import shadow_sampled_for


def _track_answer(payload: ClarifyRequest, state: SessionState) -> None:
    """The answered / skipped half of the clarification funnel."""
    # "skip" is a real option id (clarification_policy.py:456) — it returns the
    # parsed query unchanged. Distinguishing it here is what makes the
    # shown -> answered / skipped funnel measurable.
    answered = payload.answer.option_id != "skip"
    track_event(
        EVENT_CLARIFICATION_ANSWERED if answered else EVENT_CLARIFICATION_SKIPPED,
        distinct_id=analytics_distinct_id(session_id=state.session_id),
        properties={
            "reason": payload.answer.reason,
            "had_free_text": bool(payload.answer.free_text),
        },
    )


def _clarify_options(request: Request, state: SessionState) -> SearchOptions:
    """The pipeline options of the re-run: the reranker serving this session and its shadow sample."""
    settings = get_settings()
    rerank_backend = serving_rerank_backend(
        settings.ai_search_rerank_backend,
        session_id=state.session_id,
        jev_percent=settings.ai_search_rerank_jev_percent,
    )
    return SearchOptions(
        user_signed_in=getattr(request.state, "user_id", None) is not None,
        rerank_rank_only=settings.ai_search_rerank_rank_only,
        rerank_timeout_ms=settings.ai_search_rerank_timeout_ms,
        rerank_backend=rerank_backend,
        conditional_rerank=settings.ai_search_conditional_rerank,
        jev_client=get_jev_client() if rerank_backend == "jev" else None,
        jev_timeout_ms=settings.ai_search_rerank_jev_timeout_ms,
        jev_cooldown_s=settings.ai_search_rerank_jev_cooldown_s,
        shadow_jev_client=get_jev_client() if shadow_sampled_for(settings, session_id=state.session_id) else None,
    )


async def clarify_search(
    payload: ClarifyRequest,
    session_ctx: SessionContext,
    deps: SearchDependencies,
) -> SearchEnvelope[SearchResponse] | None:
    """Append the answer to the session state and re-run the pipeline.

    ``None`` when the session holds no parsed query yet: clarify only makes sense
    after a prior ``/api/search`` call, and the route answers that with a 400.
    """
    request_started = time.perf_counter()
    timings_ms: dict[str, int] = {"parser": 0}
    state = await get_or_create_session_state(
        session_ctx.request, session_ctx.response, session_ctx.store, session_ctx.secret, session_ctx.cookie_secure
    )
    if not state.recent_parsed_queries:
        return None

    next_state = state.model_copy(update={"clarification_answers": [*state.clarification_answers, payload.answer]})
    state = await session_ctx.store.update(next_state)
    _track_answer(payload, state)

    raw_parsed = state.recent_parsed_queries[-1]
    resp = await run_search_pipeline(
        SearchInput(state, raw_parsed),
        deps,
        _clarify_options(session_ctx.request, state),
        timer=SearchTimer(timings_ms, request_started),
    )
    return ok_with_timings(resp, timings_ms)
