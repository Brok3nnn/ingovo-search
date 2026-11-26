"""Jev rerank — reorder the rerank head by TypeSafe Jev fit scores.

The Jev counterpart of `rerank_and_explain`. It returns the same `RerankResult`, so the search pipeline's
guardrails run unchanged, but works differently:

- one Jev request scores every candidate independently, and the order is
  computed here (`final = alpha * fit + (1 - alpha) * deterministic`), so the
  model can neither drop nor invent a listing id;
- `rationale_hu` is always empty: Jev returns no text, and the UI does not show
  the rerank rationale anyway (by design);
- the `EnrichmentLog` row carries the provider-reported token counts.

Any Jev failure returns the deterministic order with `fallback=True`, exactly
like a Gemini transport error. With `JevRerankOptions.raise_on_unavailable` every
failure except a timeout propagates as `JevError` instead, so the search pipeline can
hand the rerank to Gemini (decision of 2026-09-28: Jev first, Gemini as the
fallback when Jev cannot serve).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import structlog

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.cost_tracker import cost_usd_from_tokens
from ingovo_search.lib.jev import JevClient, JevResponse
from ingovo_search.lib.llm_models import TIER_L_RERANK_JEV
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank import LOW_MATCH_THRESHOLD, RERANK_TOP_N, RerankDeps, RerankItem, RerankResult
from ingovo_search.recommender.rerank_jev_prompt import (
    DEFAULT_JEV_RERANK_VARIANT,
    JevRerankRequest,
    JevRerankVariant,
    build_jev_rerank_request,
)
from ingovo_search.recommender.rerank_prompt import RerankCandidate
from ingovo_search.types.errors import JevError

_logger = structlog.get_logger("ingovo_search.recommender.rerank_jev")

# Fixed by the offline evaluation (2026-09-28) on the 60-query offline set: alpha 1.0
# (pure Jev order) beat 0.7 and 0.5 for every variant.
DEFAULT_ALPHA = 1.0
# On the 0..1 fit scale; for the 4-level Score this is 1.5 of 3. Below it a
# candidate counts as a weak match, the Jev equivalent of Gemini's
# `low_confidence`. Calibration against the judge's grades: 0.5 has the best
# balanced accuracy (0.62; the curve is flat from 0.2). Nothing reads
# `low_confidence` / `low_match` downstream yet, so it has no visible effect.
LOW_FIT_THRESHOLD = 0.5
# Measured 2026-09-28: 290-410 ms for 12 real candidates in one request.
DEFAULT_JEV_TIMEOUT_S = 1.2
# "Out of capacity for us": rate limit (429) and TypeSafe overload (529), the
# only limit statuses the API documents (docs.typesafe.ai/api, 2026-09-28).
JEV_LIMIT_STATUSES = frozenset({429, 529})


def jev_failure_outcome(exc: JevError) -> str:
    """The `EnrichmentLog` / shadow-row outcome for a failed Jev call."""
    if exc.reason == "timeout":
        return "timeout"
    if exc.status in JEV_LIMIT_STATUSES:
        return "rate_limited"
    return "transport_error"


@dataclass(frozen=True, slots=True)
class JevRerankOptions:
    """Everything `jev_rerank` can be told besides what to rank and who ranks it.

    `deterministic_scores` maps listing id → the pipeline's score
    (`ScoreBreakdown.total`); it is only read when `alpha < 1`, and the incoming
    rank stands in for it when missing. `raise_on_unavailable` is for a caller that
    has another reranker to fall back on (see `jev_rerank`).
    """

    deterministic_scores: Mapping[str, float] | None = None
    model: str = TIER_L_RERANK_JEV
    variant: JevRerankVariant = DEFAULT_JEV_RERANK_VARIANT
    alpha: float = DEFAULT_ALPHA
    low_fit_threshold: float = LOW_FIT_THRESHOLD
    top_n: int = RERANK_TOP_N
    timeout_s: float = DEFAULT_JEV_TIMEOUT_S
    raise_on_unavailable: bool = False


DEFAULT_JEV_RERANK_OPTIONS: Final = JevRerankOptions()


def jev_rerank_cache_key(
    query: ParsedQuery,
    candidates: Sequence[RerankCandidate],
    deterministic: Sequence[float],
    options: JevRerankOptions,
) -> str:
    """Hash of everything that can change the result.

    The deterministic scores only matter when they are blended in (alpha < 1).
    """
    blended = ",".join(f"{value:.6f}" for value in deterministic) if options.alpha < 1 else ""
    payload = "\x00".join(
        [
            options.model,
            options.variant.prompt_version,
            f"{options.alpha:.3f}",
            f"{options.low_fit_threshold:.3f}",
            query.model_dump_json(),
            ",".join(candidate.listing_id for candidate in candidates),
            blended,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_fit(
    candidates: Sequence[RerankCandidate], deterministic_scores: Mapping[str, float] | None
) -> list[float]:
    """Each candidate's deterministic standing on 0..1 (best = 1).

    Min-max normalised pipeline scores when every candidate has one; otherwise
    the incoming rank, which is the deterministic order.
    """
    count = len(candidates)
    if deterministic_scores is not None:
        values: list[float] = []
        for candidate in candidates:
            value = deterministic_scores.get(candidate.listing_id)
            if value is None:
                break
            values.append(value)
        else:
            low, high = min(values, default=0.0), max(values, default=0.0)
            if high > low:
                return [(value - low) / (high - low) for value in values]
            return [1.0] * count
    return [1.0 - index / max(1, count - 1) for index in range(count)]


def _fits_by_key(request: JevRerankRequest, response: JevResponse, variant: JevRerankVariant) -> dict[str, float]:
    """Jev answers normalised to 0..1, for the candidate keys we actually asked about."""
    if variant.primitive == "score":
        raw = {key: answer.score / request.max_level for key, answer in response.scores.items()}
    else:
        raw = dict(response.nouls)
    return {key: min(1.0, max(0.0, value)) for key, value in raw.items() if key in request.key_to_listing_id}


def _final_order(
    keys: Sequence[str], fits: Mapping[str, float], deterministic: Sequence[float], alpha: float
) -> list[int]:
    """Candidate indices in their new order.

    Only answered candidates move, and only among the slots answered candidates
    held: a candidate Jev skipped keeps its deterministic position. Ties keep
    the deterministic order.
    """
    answered = [index for index, key in enumerate(keys) if key in fits]
    ranked = sorted(
        answered,
        key=lambda index: (-(alpha * fits[keys[index]] + (1 - alpha) * deterministic[index]), index),
    )
    order = list(range(len(keys)))
    for slot, index in zip(answered, ranked, strict=True):
        order[slot] = index
    return order


def _deterministic_result(candidates: Sequence[RerankCandidate], *, fast_path: bool) -> RerankResult:
    items = [RerankItem(listing_id=candidate.listing_id, rank=index + 1) for index, candidate in enumerate(candidates)]
    return RerankResult(items=items, low_match=False, fast_path=fast_path, fallback=not fast_path)


@dataclass(frozen=True, slots=True)
class _JevRun:
    """One `jev_rerank` call past the cache: the candidates, who answers, and the knobs its two stages read."""

    candidates: Sequence[RerankCandidate]
    backend: RerankDeps[JevClient]
    options: JevRerankOptions

    def log(self, started: float, outcome: str, response: JevResponse | None = None) -> None:
        """One `EnrichmentLog` row for the call: the served model and its reported tokens once Jev answered."""
        session = self.backend.session
        if session is None:
            return
        model = response.model if response is not None else self.options.model
        input_tokens = response.input_tokens if response is not None else 0
        output_tokens = response.output_tokens if response is not None else 0
        session.add(
            EnrichmentLog(
                property_id=None,
                model_used=model,
                prompt_version=self.options.variant.prompt_version,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd_from_tokens(model, input_tokens=input_tokens, output_tokens=output_tokens),
                latency_ms=int((time.monotonic() - started) * 1000),
                outcome=outcome,
            )
        )

    async def fits(self, request: JevRerankRequest) -> dict[str, float] | None:
        """The Jev fit per candidate key, or None when the deterministic order has to stand."""
        started = time.monotonic()
        try:
            response = await self.backend.client.system_one(
                request.state, request.questions, model=self.options.model, timeout_s=self.options.timeout_s
            )
        except JevError as exc:
            self.log(started, jev_failure_outcome(exc))
            if self.options.raise_on_unavailable and exc.reason != "timeout":
                raise
            _logger.warning(
                "jev rerank failed; using deterministic order",
                reason=exc.reason,
                status=exc.status,
                candidate_count=len(self.candidates),
            )
            return None

        fits = _fits_by_key(request, response, self.options.variant)
        self.log(started, "success" if fits else "parse_error", response)
        if not fits:
            _logger.warning("jev rerank returned no usable answers", candidate_count=len(self.candidates))
            return None
        if missing := len(self.candidates) - len(fits):
            _logger.warning("jev rerank skipped candidates", missing=missing, candidate_count=len(self.candidates))
        return fits

    def ranked(
        self, request: JevRerankRequest, fits: Mapping[str, float], deterministic: Sequence[float]
    ) -> RerankResult:
        """The candidates in their blended order, weak fits flagged."""
        keys = list(request.key_to_listing_id)
        order = _final_order(keys, fits, deterministic, self.options.alpha)
        items = [
            RerankItem(
                listing_id=self.candidates[index].listing_id,
                rank=position,
                low_confidence=keys[index] in fits and fits[keys[index]] < self.options.low_fit_threshold,
            )
            for position, index in enumerate(order, start=1)
        ]
        low_confidence_count = sum(1 for item in items[: self.options.top_n] if item.low_confidence)
        return RerankResult(items=items, low_match=low_confidence_count >= LOW_MATCH_THRESHOLD, fast_path=False)


async def jev_rerank(
    query: ParsedQuery,
    candidates: Sequence[RerankCandidate],
    backend: RerankDeps[JevClient],
    options: JevRerankOptions = DEFAULT_JEV_RERANK_OPTIONS,
) -> RerankResult:
    """Rerank `candidates` (given in deterministic order) by Jev fit scores.

    `options.deterministic_scores` maps listing id → the pipeline's score
    (`ScoreBreakdown.total`); it is only used when `alpha < 1`, and the incoming
    rank stands in for it when missing. `low_match` flips when at least
    `LOW_MATCH_THRESHOLD` of the top `top_n` score below `low_fit_threshold`.

    `options.raise_on_unavailable` re-raises every `JevError` except a timeout
    (limit, overload, key, outage) after logging it, for a caller that has
    another reranker to fall back on; a timeout has spent the time budget, so it
    still returns the deterministic order.
    """
    if not (query.semantic_query and query.semantic_query.strip()):
        return _deterministic_result(candidates, fast_path=True)

    deterministic = _deterministic_fit(candidates, options.deterministic_scores)
    key = jev_rerank_cache_key(query, candidates, deterministic, options)
    cache = backend.cache
    if cache is not None and (cached := await cache.get(key)) is not None:
        return cached

    request = build_jev_rerank_request(query, candidates, options.variant)
    run = _JevRun(candidates=candidates, backend=backend, options=options)
    fits = await run.fits(request)
    if fits is None:
        return _deterministic_result(candidates, fast_path=False)
    result = run.ranked(request, fits, deterministic)

    if cache is not None:
        await cache.set(key, result)
    return result
