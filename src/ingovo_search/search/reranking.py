"""The LLM rerank of the scored head: Gemini, Jev with a Gemini fallback, or none.

Which reranker serves is decided per request (`serving_rerank_backend`); the stage keeps
the deterministic order whenever the rerank is off, pointless, or out of time.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Literal

from ingovo_search.api.search_events import RerankCompletedEvent, RerankSkippedEvent, RerankStartedEvent
from ingovo_search.lib.llm import set_llm_phase
from ingovo_search.recommender.ranking_guardrails import (
    assert_hard_filter_preserved,
    assert_rerank_candidate_set_preserved,
    enforce_bounded_rerank,
)
from ingovo_search.recommender.rerank import (
    CacheRerankCache,
    RerankDeps,
    RerankOptions,
    RerankResult,
    rerank_and_explain,
)
from ingovo_search.recommender.rerank_jev import JEV_LIMIT_STATUSES, JevRerankOptions, jev_rerank
from ingovo_search.recommender.rerank_prompt import RerankCandidate
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.search.constants import search_logger
from ingovo_search.search.run import JEV_TIMEOUT_GRACE_MS, RerankBackend, ScoredRow, SearchRun, served_outcome
from ingovo_search.search.scoring import listing_rerank_summary, should_rerank
from ingovo_search.search.shadow import ShadowSample, record_rerank_shadow, spawn_rerank_shadow
from ingovo_search.types.errors import JevError

type SkipReason = Literal["deterministic", "disabled", "no_semantic_query"]


def serving_rerank_backend(configured: str, *, session_id: str, jev_percent: int) -> RerankBackend:
    """Which reranker serves this request.

    `shadow` still serves Gemini. `canary` buckets by session, so a visitor keeps
    one arm across searches. Anything unrecognised serves Gemini, today's behaviour.
    """
    if configured == "jev":
        return "jev"
    if configured == "off":
        return "off"
    if configured == "canary":
        bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "jev" if bucket < jev_percent else "gemini"
    return "gemini"


class JevCooldown:
    """Jev first, Gemini as the fallback (decision of 2026-09-28).

    After Jev fails with anything but a timeout, the Gemini rerank serves until
    `until` (a monotonic instant), so a rate-limited or broken Jev is not called on
    every search. Process-local on purpose: each worker finds out on its own first failure.
    """

    def __init__(self) -> None:
        self.until = 0.0

    def active(self) -> bool:
        return time.monotonic() < self.until

    def start(self, seconds: float) -> None:
        self.until = time.monotonic() + max(0.0, seconds)


jev_cooldown = JevCooldown()


def _log_jev_unavailable(exc: JevError, *, cooldown_s: float) -> None:
    extra = {"status": exc.status, "reason": exc.reason, "cooldown_s": cooldown_s}
    if exc.status in JEV_LIMIT_STATUSES:
        search_logger.warning("jev rerank limit reached; the Gemini rerank serves during the cooldown", **extra)
    elif exc.status in (401, 403, 422):
        # A rejected key or a request Jev cannot parse will not heal by itself.
        search_logger.error("jev rerank rejected the request; the Gemini rerank serves during the cooldown", **extra)
    else:
        search_logger.warning("jev rerank unavailable; the Gemini rerank serves during the cooldown", **extra)


@dataclass(slots=True)
class RerankStage:
    """What the rerank stage hands on: the results and the numbers the search log wants."""

    results: list[ScoredCandidate]
    pre_count: int | None = None
    post_count: int | None = None
    outcome: str = "skipped:deterministic"
    gemini_fallback: bool = False


@dataclass(slots=True)
class _RerankCall:
    """One reranker call as it unfolds; read after a timeout too, so it is mutated in place."""

    result: RerankResult | None = None
    # True when `jev` mode had to hand the rerank to Gemini (no key, cooldown, or
    # a Jev failure other than a timeout).
    gemini_fallback: bool = False
    timeout_ms: int = 0


def _skip_reason(run: SearchRun) -> SkipReason | None:
    options = run.options
    if not options.include_rerank or options.rerank_backend == "off":
        return "disabled"
    if not run.scored:
        return "deterministic"
    if not run.parsed.semantic_query:
        return "no_semantic_query"
    if options.conditional_rerank and not should_rerank(run.parsed, run.scored):
        search_logger.info(
            "rerank skipped — deterministic order trusted",
            candidate_count=len(run.scored),
            semantic_query=run.parsed.semantic_query,
        )
        return "deterministic"
    return None


def _deterministic_scores(rerank_head: list[ScoredRow]) -> dict[str, float]:
    return {prop.id: breakdown.total for prop, breakdown in rerank_head}


async def _call_reranker(
    run: SearchRun, call: _RerankCall, candidates: list[RerankCandidate], rerank_head: list[ScoredRow]
) -> None:
    """Jev first when it serves, Gemini otherwise; a timeout propagates to the caller."""
    options = run.options
    use_gemini = options.rerank_backend != "jev"
    call.timeout_ms = options.rerank_timeout_ms
    if not use_gemini:
        # Jev first; Gemini when Jev has no key, is cooling down after a
        # failure, or fails now with anything but a timeout. A Jev timeout
        # has spent the time budget, so it keeps the deterministic order.
        if options.jev_client is None or jev_cooldown.active():
            use_gemini = True
        else:
            call.timeout_ms = options.jev_timeout_ms + JEV_TIMEOUT_GRACE_MS
            try:
                call.result = await asyncio.wait_for(
                    jev_rerank(
                        run.parsed,
                        candidates,
                        RerankDeps(options.jev_client, CacheRerankCache(run.deps.cache), run.deps.db_session),
                        JevRerankOptions(
                            deterministic_scores=_deterministic_scores(rerank_head),
                            timeout_s=options.jev_timeout_ms / 1000,
                            raise_on_unavailable=True,
                        ),
                    ),
                    timeout=call.timeout_ms / 1000,
                )
            except JevError as exc:
                jev_cooldown.start(options.jev_cooldown_s)
                _log_jev_unavailable(exc, cooldown_s=options.jev_cooldown_s)
                use_gemini = True
        call.gemini_fallback = use_gemini
    if use_gemini:
        call.timeout_ms = options.rerank_timeout_ms
        gemini_call = rerank_and_explain(
            run.parsed,
            candidates,
            RerankDeps(run.deps.llm_client, CacheRerankCache(run.deps.cache), run.deps.db_session),
            RerankOptions(
                session_signals=[s.signal_type for s in run.state.session_signals],
                rank_only=options.rerank_rank_only,
                locale=options.narrative_locale,
            ),
        )
        if call.timeout_ms > 0:
            call.result = await asyncio.wait_for(gemini_call, timeout=call.timeout_ms / 1000)
        else:
            call.result = await gemini_call


async def _report_timeout(run: SearchRun, call: _RerankCall, *, candidate_count: int) -> None:
    search_logger.warning(
        "rerank timed out; falling back to deterministic order",
        timeout_ms=call.timeout_ms,
        rerank_backend=run.options.rerank_backend,
        gemini_fallback=call.gemini_fallback,
        candidate_count=candidate_count,
        rerank_candidate_count=run.options.rerank_candidate_count,
    )
    await run.emit(
        RerankSkippedEvent(request_id=run.progress.request_id, candidate_count=candidate_count, reason="timeout")
    )


def _apply_rerank_result(
    run: SearchRun,
    scored_candidates: list[ScoredCandidate],
    head_and_tail: tuple[list[ScoredRow], list[ScoredRow]],
    result: RerankResult,
) -> tuple[list[ScoredCandidate], int]:
    """The reranked head (guardrails applied) in front of the untouched tail; returns the post-rerank count."""
    rerank_head, rerank_tail = head_and_tail
    head_ids = [prop.id for prop, _ in rerank_head]
    raw_post_order = [item.listing_id for item in result.items]
    assert_hard_filter_preserved(hard_filter_ids=head_ids, post_rerank_ids=raw_post_order)
    post_order = list(dict.fromkeys(raw_post_order))
    post_count = len(post_order)
    post_ids = set(post_order)
    post_order.extend(listing_id for listing_id in head_ids if listing_id not in post_ids)
    assert_rerank_candidate_set_preserved(pre_rerank_ids=head_ids, post_rerank_ids=post_order)
    bounded_head_order = enforce_bounded_rerank(head_ids, post_order)
    final_order = bounded_head_order + [prop.id for prop, _ in rerank_tail]

    rationale_by_id: dict[str, str] = {item.listing_id: item.rationale_hu for item in result.items}
    rerank_relevance_by_id: dict[str, float] = {}
    if not result.fallback:
        denominator = max(1, len(bounded_head_order) - 1)
        rerank_relevance_by_id = {
            listing_id: max(0.0, 1.0 - index / denominator) for index, listing_id in enumerate(bounded_head_order)
        }
    scored_by_id: dict[str, ScoredCandidate] = {}
    for candidate in scored_candidates:
        if candidate.listing_id in rationale_by_id:
            scored_by_id[candidate.listing_id] = candidate.model_copy(
                update={
                    "rationale": rationale_by_id[candidate.listing_id],
                    "rerank_relevance": rerank_relevance_by_id.get(candidate.listing_id),
                }
            )
        else:
            scored_by_id[candidate.listing_id] = candidate

    results = [scored_by_id[pid] for pid in final_order if pid in scored_by_id]
    if not results and scored_candidates:
        search_logger.warning(
            "rerank returned empty usable order; falling back to deterministic order",
            candidate_count=len(rerank_head),
            rerank_candidate_count=run.options.rerank_candidate_count,
            scored_count=len(scored_candidates),
        )
        results = scored_candidates
    return results, post_count


def _spawn_shadow(run: SearchRun, call: _RerankCall, candidates: list[RerankCandidate], head: list[ScoredRow]) -> None:
    if run.options.shadow_jev_client is None:
        return
    served_order = (
        [item.listing_id for item in call.result.items] if call.result is not None else [prop.id for prop, _ in head]
    )
    sample = ShadowSample(
        parsed=run.parsed,
        candidates=candidates,
        deterministic_scores=_deterministic_scores(head),
        query_hash=run.parsed_hash,
        served_backend=run.options.rerank_backend,
        served_order=served_order,
        served_outcome=served_outcome(call.result),
        served_latency_ms=run.timings["rerank"],
    )
    spawn_rerank_shadow(
        record_rerank_shadow(run.options.shadow_jev_client, sample, timeout_s=run.options.jev_timeout_ms / 1000)
    )


async def run_rerank(run: SearchRun, scored_candidates: list[ScoredCandidate]) -> RerankStage:
    """Rerank the head of the scored list with the serving backend, or keep the deterministic order."""
    stage = RerankStage(results=scored_candidates)
    skip_reason = _skip_reason(run)
    if skip_reason is not None:
        stage.outcome = f"skipped:{skip_reason}"
        run.timings["rerank"] = 0
        run.timings["rerank_timeout"] = 0
        await run.emit(
            RerankSkippedEvent(request_id=run.progress.request_id, candidate_count=len(run.scored), reason=skip_reason)
        )
        return stage

    step_started = run.timer.now()
    set_llm_phase(run.deps.llm_client, "rerank")
    count = run.options.rerank_candidate_count
    rerank_head, rerank_tail = run.scored[:count], run.scored[count:]
    candidates = [
        RerankCandidate(listing_id=prop.id, short_summary=listing_rerank_summary(prop)) for prop, _ in rerank_head
    ]
    stage.pre_count = len(candidates)
    await run.emit(RerankStartedEvent(request_id=run.progress.request_id, candidate_count=stage.pre_count))
    call = _RerankCall()
    try:
        await _call_reranker(run, call, candidates, rerank_head)
        run.timings["rerank_timeout"] = 0
    except TimeoutError:
        call.result = None
        run.timings["rerank_timeout"] = 1
        await _report_timeout(run, call, candidate_count=stage.pre_count)
    run.timer.record("rerank", step_started)
    run.timings["rerank_gemini_fallback"] = int(call.gemini_fallback)
    stage.gemini_fallback = call.gemini_fallback
    stage.outcome = served_outcome(call.result)
    if call.gemini_fallback and stage.outcome == "success":
        stage.outcome = "gemini_fallback"
    _spawn_shadow(run, call, candidates, rerank_head)
    if call.result is not None:
        stage.results, stage.post_count = _apply_rerank_result(
            run, scored_candidates, (rerank_head, rerank_tail), call.result
        )
        await run.emit(RerankCompletedEvent(request_id=run.progress.request_id, candidate_count=stage.post_count))
    return stage
