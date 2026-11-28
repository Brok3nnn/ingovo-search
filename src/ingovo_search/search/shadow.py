"""The shadow rerank: Jev runs on the served head after the response went out.

``shadow`` mode serves Gemini and runs the Jev rerank on the same head in the
background; the response never waits for it. One ``rerank_shadow_comparison`` row per
sampled search records both orders for a later comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Question

from ingovo_search.db.models import RerankShadowComparison
from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.jev import JevClient, JevResponse, JevState
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank import RerankDeps, RerankResult
from ingovo_search.recommender.rerank_jev import JevRerankOptions, jev_failure_outcome, jev_rerank
from ingovo_search.recommender.rerank_jev_prompt import DEFAULT_JEV_RERANK_VARIANT, build_jev_rerank_request
from ingovo_search.recommender.rerank_prompt import RerankCandidate
from ingovo_search.search.constants import search_logger
from ingovo_search.search.run import JEV_TIMEOUT_GRACE_MS, served_outcome
from ingovo_search.types.errors import JevError


def shadow_sampled(session_id: str, sample: float) -> bool:
    """Whether this session's searches get a shadow rerank (`AI_SEARCH_RERANK_SHADOW_SAMPLE`).

    Bucketed by session, with its own salt, so a visitor is sampled consistently
    and the shadow buckets are independent of the canary buckets.
    """
    if sample <= 0:
        return False
    if sample >= 1:
        return True
    bucket = int(hashlib.sha256(f"{session_id}:shadow".encode()).hexdigest()[:8], 16) % 10_000
    return bucket < sample * 10_000


def shadow_sampled_for(settings: Any, *, session_id: str) -> bool:
    """Whether this search gets a shadow rerank: `shadow` mode, and the session is in the sample."""
    if getattr(settings, "ai_search_rerank_backend", "gemini") != "shadow":
        return False
    return shadow_sampled(session_id, float(getattr(settings, "ai_search_rerank_shadow_sample", 1.0)))


# Fire-and-forget tasks must be referenced until they finish, or the loop may
# garbage-collect them mid-flight. `drain_rerank_shadow_tasks` waits for them
# (tests, shutdown).
_rerank_shadow_tasks: set[asyncio.Task[None]] = set()


def spawn_rerank_shadow(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(coroutine)
    _rerank_shadow_tasks.add(task)
    task.add_done_callback(_rerank_shadow_tasks.discard)
    return task


async def drain_rerank_shadow_tasks() -> None:
    """Wait for every in-flight shadow rerank (never raises)."""
    if _rerank_shadow_tasks:
        await asyncio.gather(*list(_rerank_shadow_tasks), return_exceptions=True)


class _RecordingJevClient:
    """Delegates to the real client and keeps the response (or the error) for the shadow row."""

    def __init__(self, inner: JevClient) -> None:
        self._inner = inner
        self.response: JevResponse | None = None
        self.error: JevError | None = None

    async def system_one(
        self, state: JevState, questions: Mapping[str, Question], *, model: str, timeout_s: float
    ) -> JevResponse:
        try:
            self.response = await self._inner.system_one(state, questions, model=model, timeout_s=timeout_s)
        except JevError as exc:
            self.error = exc
            raise
        return self.response


@dataclass(frozen=True, slots=True)
class ShadowSample:
    """The served rerank the shadow is compared against."""

    parsed: ParsedQuery
    candidates: Sequence[RerankCandidate]
    deterministic_scores: Mapping[str, float]
    query_hash: str
    served_backend: str
    served_order: Sequence[str]
    served_outcome: str
    served_latency_ms: int


async def _shadow_rerank(
    recorder: _RecordingJevClient, sample: ShadowSample, *, session: AsyncSession, timeout_s: float
) -> RerankResult | None:
    try:
        result = await asyncio.wait_for(
            jev_rerank(
                sample.parsed,
                sample.candidates,
                RerankDeps(recorder, session=session),
                JevRerankOptions(deterministic_scores=sample.deterministic_scores, timeout_s=timeout_s),
            ),
            timeout=timeout_s + JEV_TIMEOUT_GRACE_MS / 1000,
        )
    except TimeoutError:
        return None
    return result


def _jev_outcome(recorder: _RecordingJevClient, result: RerankResult | None) -> str:
    if recorder.error is not None:
        return jev_failure_outcome(recorder.error)
    return served_outcome(result) if result is None or not result.fallback else "parse_error"


def _jev_scores(response: JevResponse | None, sample: ShadowSample) -> dict[str, float] | None:
    if response is None:
        return None
    request = build_jev_rerank_request(sample.parsed, sample.candidates, DEFAULT_JEV_RERANK_VARIANT)
    raw = (
        {key: answer.score / request.max_level for key, answer in response.scores.items()}
        if DEFAULT_JEV_RERANK_VARIANT.primitive == "score"
        else dict(response.nouls)
    )
    return {
        request.key_to_listing_id[key]: round(min(1.0, max(0.0, value)), 4)
        for key, value in raw.items()
        if key in request.key_to_listing_id
    }


def _comparison_row(
    sample: ShadowSample, recorder: _RecordingJevClient, result: RerankResult | None, *, jev_latency_ms: int
) -> RerankShadowComparison:
    response = recorder.response
    served_gemini = sample.served_backend == "gemini"
    return RerankShadowComparison(
        query_hash=sample.query_hash,
        semantic_query=sample.parsed.semantic_query,
        candidate_ids=[candidate.listing_id for candidate in sample.candidates],
        served_backend=sample.served_backend,
        gemini_order=list(sample.served_order) if served_gemini else None,
        # Only an order Jev actually produced; a fallback or fast path carries the deterministic one.
        jev_order=(
            [item.listing_id for item in result.items]
            if result is not None and not result.fallback and not result.fast_path
            else None
        ),
        jev_scores=_jev_scores(response, sample),
        gemini_latency_ms=sample.served_latency_ms if served_gemini else None,
        jev_latency_ms=jev_latency_ms,
        jev_input_tokens=response.input_tokens if response is not None else None,
        gemini_outcome=sample.served_outcome if served_gemini else None,
        jev_outcome=_jev_outcome(recorder, result),
        jev_model=response.model if response is not None else None,
        prompt_version=DEFAULT_JEV_RERANK_VARIANT.prompt_version,
    )


async def record_rerank_shadow(jev_client: JevClient, sample: ShadowSample, *, timeout_s: float) -> None:
    """Run the Jev rerank on the served head and store both orders; swallows every failure."""
    recorder = _RecordingJevClient(jev_client)
    started = time.perf_counter()
    try:
        async with get_async_session_factory()() as session:
            result = await _shadow_rerank(recorder, sample, session=session, timeout_s=timeout_s)
            jev_latency_ms = int((time.perf_counter() - started) * 1000)
            session.add(_comparison_row(sample, recorder, result, jev_latency_ms=jev_latency_ms))
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        # Deliberate fail-soft boundary: the shadow runs as a detached background
        # task on its own session, after the response has been served. Nothing
        # it raises (a failed commit, a malformed Jev payload, a guardrail
        # assertion) may reach the request or a sibling task, and there is no
        # caller left to re-raise to — so every failure is swallowed here, but
        # with its traceback and the query hash so it shows up in the log.
        search_logger.warning(
            "rerank shadow failed; comparison row not written",
            error=f"{type(exc).__name__}: {exc}"[:300],
            query_hash=sample.query_hash,
            candidate_count=len(sample.candidates),
            served_backend=sample.served_backend,
            exc_info=exc,
        )
