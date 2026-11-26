"""One ``POST /api/search`` request around the pipeline.

Session and reset, the raw-query response cache, the speculative embedding read, the
parser with its timeout, the attached photos, the progress events, the budget guardrail,
then ``run_search_pipeline`` and the raw cache write.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ingovo_search.api.deps import get_jev_client
from ingovo_search.api.search_events import ImageInterpretedEvent, IntentParsedEvent, SearchProgressEmitter
from ingovo_search.api.search_session import (
    accumulate_request_into_session,
    get_or_create_session_state,
    record_parsed_query,
)
from ingovo_search.lib.budget_guard import current_budget_policy
from ingovo_search.lib.cache import get_json, normalise_query, raw_search_response_cache_key, set_json
from ingovo_search.lib.env import Settings, get_settings
from ingovo_search.lib.jev import JevClient
from ingovo_search.lib.llm import CountingLlmClient, set_llm_phase
from ingovo_search.lib.llm_models import TIER_L_RERANK_FALLBACK, TIER_S_PARSER_FALLBACK
from ingovo_search.lib.session import SessionState, SessionStore
from ingovo_search.recommender.canonical_ranking import assert_match_order
from ingovo_search.recommender.image_query import ImageQuery, analyze_images
from ingovo_search.recommender.parser import CacheParserCache, ParserSettings, fallback_parse_query, parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.search_preview import build_search_preview
from ingovo_search.search.constants import (
    RAW_SEARCH_RESPONSE_CACHE_VERSION,
    RAW_SEARCH_RESPONSE_RANK_ONLY_CACHE_VERSION,
    RERANK_CANDIDATE_COUNT,
    search_logger,
)
from ingovo_search.search.context import build_search_context
from ingovo_search.search.embedding_cache import (
    SpeculativeCachedEmbedding,
    discard_speculative_embedding,
    raw_response_cache_allowed,
    read_cached_embedding,
    speculative_embedding_allowed,
)
from ingovo_search.search.images import SearchImages, await_image_analysis, cancel_image_task
from ingovo_search.search.models import (
    ImageContextSummary,
    SearchEnvelope,
    SearchRequest,
    SearchResponse,
    ok_with_timings,
)
from ingovo_search.search.pipeline import run_search_pipeline
from ingovo_search.search.reranking import serving_rerank_backend
from ingovo_search.search.response_cache import (
    cached_response_is_publicly_visible,
    discard_stale_response_cache_entry,
    rerank_cache_suffix,
)
from ingovo_search.search.run import (
    RerankBackend,
    SearchDependencies,
    SearchInput,
    SearchOptions,
    SearchProgress,
    SearchTimer,
    elapsed_ms,
)
from ingovo_search.search.shadow import shadow_sampled_for


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What reading and writing the session cookie takes."""

    request: Request
    response: Response
    store: SessionStore
    secret: bytes
    cookie_secure: bool


@dataclass(frozen=True, slots=True)
class StreamHandoff:
    """The streaming route's state: it already created the session and listens for progress."""

    initial_state: SessionState | None = None
    progress: SearchProgressEmitter | None = None
    request_id: str | None = None


@dataclass(slots=True)
class _RequestRun:
    """One request in flight, between the stages of ``execute_search_request``."""

    payload: SearchRequest
    images: SearchImages
    session_ctx: SessionContext
    deps: SearchDependencies
    stream: StreamHandoff
    settings: Settings
    started: float
    timings: dict[str, int]
    state: SessionState
    counting_client: CountingLlmClient
    user_signed_in: bool
    rerank_backend: RerankBackend
    rerank_rank_only: bool
    shadow_jev_client: JevClient | None
    image_client: CountingLlmClient
    image_task: asyncio.Task[ImageQuery] | None
    raw_cache_key: str | None = None
    speculative: SpeculativeCachedEmbedding | None = None
    raw_parsed: ParsedQuery | None = None
    image_summary: ImageContextSummary | None = None

    @property
    def progress_request_id(self) -> str:
        return self.stream.request_id or ""

    @property
    def parsed(self) -> ParsedQuery:
        if self.raw_parsed is None:
            raise RuntimeError("the parser has not run yet")
        return self.raw_parsed


async def _open(
    payload: SearchRequest,
    images: SearchImages,
    session_ctx: SessionContext,
    deps: SearchDependencies,
    stream: StreamHandoff,
) -> _RequestRun:
    """The session (reset when asked), the serving reranker, and the vision call already on its way."""
    started = time.perf_counter()
    state = stream.initial_state or await get_or_create_session_state(
        session_ctx.request, session_ctx.response, session_ctx.store, session_ctx.secret, session_ctx.cookie_secure
    )
    if payload.reset_session:
        state = state.model_copy(
            update={
                "session_signals": [],
                "clarification_answers": [],
                "refinement_overrides": None,
                "recent_parsed_queries": [],
            }
        )
        state = await session_ctx.store.update(state)
    state = await accumulate_request_into_session(state, payload, session_ctx.store)

    # Count every Gemini call this search makes, by phase, for §16 cost auditing.
    counting_client = CountingLlmClient(deps.llm_client)
    user_signed_in = getattr(session_ctx.request.state, "user_id", None) is not None
    settings = get_settings()
    rerank_backend = serving_rerank_backend(
        settings.ai_search_rerank_backend,
        session_id=state.session_id,
        jev_percent=settings.ai_search_rerank_jev_percent,
    )
    shadow_jev_client = get_jev_client() if shadow_sampled_for(settings, session_id=state.session_id) else None

    # Start reading the photos before anything else: the vision call is the
    # slowest step in an image search, and running it alongside the parser (and
    # the cache reads below) keeps its cost off the critical path.
    image_client = CountingLlmClient(deps.llm_client)
    set_llm_phase(image_client, "image_query")
    image_task: asyncio.Task[ImageQuery] | None = (
        asyncio.create_task(analyze_images(images.files, client=image_client, cache=deps.cache, locale=images.locale))
        if images.files
        else None
    )
    return _RequestRun(
        payload=payload,
        images=images,
        session_ctx=session_ctx,
        deps=deps,
        stream=stream,
        settings=settings,
        started=started,
        timings={},
        state=state,
        counting_client=counting_client,
        user_signed_in=user_signed_in,
        rerank_backend=rerank_backend,
        rerank_rank_only=settings.ai_search_rerank_rank_only,
        shadow_jev_client=shadow_jev_client,
        image_client=image_client,
        image_task=image_task,
    )


async def _serve_raw_cache(run: _RequestRun) -> SearchResponse | None:
    """The response cached under the raw sentence, when there is one and its rows are still public."""
    timings = run.timings
    if not raw_response_cache_allowed(run.payload, has_images=bool(run.images.files)):
        timings["raw_response_cache_read"] = 0
        return None
    run.raw_cache_key = raw_search_response_cache_key(
        run.payload.query or "",
        user_signed_in=run.user_signed_in,
        response_version=(
            RAW_SEARCH_RESPONSE_RANK_ONLY_CACHE_VERSION if run.rerank_rank_only else RAW_SEARCH_RESPONSE_CACHE_VERSION
        )
        + rerank_cache_suffix(run.rerank_backend),
    )
    step_started = time.perf_counter()
    cached_resp = await get_json(run.deps.cache, run.raw_cache_key, SearchResponse)
    timings["raw_response_cache_read"] = elapsed_ms(step_started)
    if cached_resp is not None and cached_resp.clarification is None:
        assert_match_order(cached_resp.results)
        step_started = time.perf_counter()
        cache_rows_visible = await cached_response_is_publicly_visible(run.deps.db_session, cached_resp)
        timings["raw_response_cache_visibility_check"] = elapsed_ms(step_started)
        if cache_rows_visible:
            timings["total"] = elapsed_ms(run.started)
            search_logger.info(
                "raw search response cache hit",
                extra={
                    "cache_key": run.raw_cache_key,
                    "returned_count": len(cached_resp.results),
                    "timings_ms": timings,
                },
            )
            return cached_resp

        await discard_stale_response_cache_entry(run.deps.cache, run.raw_cache_key)
        search_logger.warning(
            "raw search response cache rejected by current public visibility",
            extra={"cache_key": run.raw_cache_key, "returned_count": len(cached_resp.results)},
        )
    if cached_resp is not None:
        search_logger.info(
            "raw search response cache ignored because cached response needs clarification",
            extra={"cache_key": run.raw_cache_key},
        )
    else:
        search_logger.info("raw search response cache miss", extra={"cache_key": run.raw_cache_key})
    return None


def _start_speculative_embedding(run: _RequestRun) -> None:
    """Read the raw sentence's cached embedding while the parser works; used if the parser keeps the sentence."""
    if not speculative_embedding_allowed(run.payload):
        return
    run.speculative = SpeculativeCachedEmbedding(
        normalised_text=normalise_query(run.payload.query or ""),
        task=asyncio.create_task(
            read_cached_embedding(
                text_value=run.payload.query or "",
                input_type="search_query",
                cache=run.deps.cache,
            )
        ),
    )


async def _parse(run: _RequestRun) -> None:
    """The parser with its timeout and deterministic fallback; an image-only search skips it."""
    step_started = time.perf_counter()
    set_llm_phase(run.counting_client, "parser")
    query_text = (run.payload.query or "").strip()
    timeout_ms = run.settings.ai_search_parser_timeout_ms
    if not query_text and run.image_task is not None:
        # Image-only search: there is no sentence to parse, so skip the parser
        # call entirely and let the merge below supply the whole query.
        run.raw_parsed = ParsedQuery()
        run.timings["parser_timeout"] = 0
    else:
        try:
            parser_call = parse_query(
                run.payload.query or "",
                client=run.counting_client,
                cache=CacheParserCache(run.deps.cache),
                session=run.deps.db_session,
                settings=ParserSettings(locale=run.payload.query_locale or run.payload.narrative_locale),
            )
            if timeout_ms > 0:
                run.raw_parsed = await asyncio.wait_for(parser_call, timeout=timeout_ms / 1000)
            else:
                run.raw_parsed = await parser_call
            run.timings["parser_timeout"] = 0
        except TimeoutError:
            run.raw_parsed = fallback_parse_query(run.payload.query or "")
            run.timings["parser_timeout"] = 1
            search_logger.warning(
                "parser timed out; using deterministic fallback",
                extra={"timeout_ms": timeout_ms},
            )
        except Exception:
            await discard_speculative_embedding(run.speculative)
            await cancel_image_task(run.image_task)
            raise
    run.timings["parser"] = elapsed_ms(step_started)


async def _interpret_images(run: _RequestRun) -> None:
    """Fold the photos into the parsed query as soft signal; the typed sentence always wins."""
    # Attached photos are soft signal folded in *after* parsing, so the typed
    # sentence always wins: `merge_image_query` only adds soft preferences,
    # freetext and semantic text, never a price / location / room constraint.
    run.timings["image_query"] = 0
    if run.image_task is None:
        return
    step_started = time.perf_counter()
    run.raw_parsed, run.image_summary = await await_image_analysis(
        run.image_task,
        parsed=run.parsed,
        images=run.images,
        timeout_ms=run.settings.ai_search_image_timeout_ms,
        request_id=run.progress_request_id,
    )
    run.timings["image_query"] = elapsed_ms(step_started)
    run.timings["image_llm_calls"] = run.image_client.total_calls


def _progress(run: _RequestRun) -> SearchProgress:
    """What the stream hears first: how the sentence was understood, and what the photos said."""
    preview = build_search_preview(
        run.payload.query or "",
        run.images.locale,
        run.parsed,
        run.progress_request_id or "search",
    )
    intent_event = IntentParsedEvent(
        request_id=run.progress_request_id,
        summary=preview.understood_summary,
        confidence=preview.confidence,
        constraints=build_search_context(run.parsed, []),
        needs_clarification=bool(preview.clarification) or preview.confidence < 0.52,
    )
    image_event = (
        ImageInterpretedEvent(
            request_id=run.progress_request_id,
            applied=run.image_summary.applied,
            reason=run.image_summary.reason,
            labels=run.image_summary.chips,
            summary=run.image_summary.summary,
        )
        if run.image_summary is not None
        else None
    )
    return SearchProgress(
        emitter=run.stream.progress,
        request_id=run.progress_request_id,
        intent_event=intent_event,
        image_event=image_event,
    )


async def _options(run: _RequestRun) -> SearchOptions:
    """The pipeline options: the request's wishes, bounded by the §16 budget guardrail."""
    payload = run.payload
    settings = run.settings
    # §16 budget guardrail — when the day's spend is high, drop optional LLM work
    # (narratives first, then rerank) so search stays up and costs stay bounded.
    budget_policy = await current_budget_policy(run.deps.db_session, budget_usd=settings.llm_daily_budget_usd)
    include_narratives = payload.include_narratives and budget_policy.allow_narratives
    include_rerank = payload.include_rerank and budget_policy.allow_rerank
    if budget_policy.throttled and (payload.include_narratives or payload.include_rerank):
        search_logger.info(
            "search llm throttled by budget",
            extra={
                "spend_fraction": round(budget_policy.spend_fraction, 3),
                "allow_narratives": budget_policy.allow_narratives,
                "allow_rerank": budget_policy.allow_rerank,
            },
        )
    return SearchOptions(
        user_signed_in=run.user_signed_in,
        include_narratives=include_narratives,
        narrative_locale=payload.narrative_locale,
        include_rerank=include_rerank,
        rerank_candidate_count=(
            payload.rerank_candidate_count if payload.rerank_candidate_count is not None else RERANK_CANDIDATE_COUNT
        ),
        rerank_rank_only=run.rerank_rank_only,
        rerank_timeout_ms=settings.ai_search_rerank_timeout_ms,
        rerank_backend=run.rerank_backend,
        conditional_rerank=settings.ai_search_conditional_rerank,
        jev_client=get_jev_client() if run.rerank_backend == "jev" else None,
        jev_timeout_ms=settings.ai_search_rerank_jev_timeout_ms,
        jev_cooldown_s=settings.ai_search_rerank_jev_cooldown_s,
        shadow_jev_client=run.shadow_jev_client,
        speculative_query_embedding=run.speculative,
    )


async def _write_raw_cache(run: _RequestRun, resp: SearchResponse, *, include_rerank: bool) -> None:
    """Cache the finished response under the raw sentence too, with the TTL the pipeline computed."""
    raw_response_cache_ttl = run.timings.get("response_cache_ttl_seconds")
    if (
        run.raw_cache_key is not None
        and include_rerank
        and resp.clarification is None
        and isinstance(raw_response_cache_ttl, int)
        and raw_response_cache_ttl > 0
    ):
        step_started = time.perf_counter()
        await set_json(run.deps.cache, run.raw_cache_key, resp, ttl_seconds=raw_response_cache_ttl)
        run.timings["raw_response_cache_write"] = elapsed_ms(step_started)
        search_logger.info(
            "raw search response cache set",
            extra={
                "cache_key": run.raw_cache_key,
                "returned_count": len(resp.results),
                "ttl_seconds": raw_response_cache_ttl,
            },
        )
    elif run.raw_cache_key is not None:
        # If the parsed-query cache supplied the result it did not carry the
        # source row expiry metadata needed to cap this second cache. Skip this
        # optional write; hit-time DB visibility checks remain mandatory.
        run.timings["raw_response_cache_write"] = 0


def _log_llm_calls(counting_client: CountingLlmClient) -> None:
    by_phase = counting_client.calls_by_phase
    fallback_calls = sum(
        counting_client.calls_by_model.get(model, 0) for model in (TIER_S_PARSER_FALLBACK, TIER_L_RERANK_FALLBACK)
    )
    search_logger.info(
        "search llm calls",
        extra={
            "llm_calls_total": counting_client.total_calls,
            "parser_calls": by_phase.get("parser", 0),
            "rerank_calls": by_phase.get("rerank", 0),
            "narrative_calls": by_phase.get("narrative", 0),
            "fallback_model_calls": fallback_calls,
            "calls_by_model": counting_client.calls_by_model,
        },
    )


async def execute_search_request(
    payload: SearchRequest,
    images: SearchImages,
    session_ctx: SessionContext,
    deps: SearchDependencies,
    stream: StreamHandoff | None = None,
) -> SearchEnvelope[SearchResponse] | JSONResponse:
    """Shared body of both search routes — the attached photos are the only difference."""
    run = await _open(payload, images, session_ctx, deps, stream or StreamHandoff())
    cached = await _serve_raw_cache(run)
    if cached is not None:
        return ok_with_timings(cached, run.timings, request_id=run.stream.request_id)

    _start_speculative_embedding(run)
    await _parse(run)
    await _interpret_images(run)
    progress = _progress(run)
    run.state = await record_parsed_query(run.state, run.parsed, session_ctx.store)
    options = await _options(run)
    resp = await run_search_pipeline(
        SearchInput(run.state, run.parsed),
        replace(deps, llm_client=run.counting_client),
        options,
        progress,
        SearchTimer(run.timings, run.started),
    )
    # Attached *after* the pipeline so the cached response body stays free of
    # per-request image state — the same parsed query must reuse the same cache
    # entry whether or not it arrived with a photo.
    if run.image_summary is not None:
        resp = resp.model_copy(update={"image_context": run.image_summary})
    await _write_raw_cache(run, resp, include_rerank=options.include_rerank)
    _log_llm_calls(run.counting_client)
    return ok_with_timings(resp, run.timings, request_id=run.stream.request_id)
