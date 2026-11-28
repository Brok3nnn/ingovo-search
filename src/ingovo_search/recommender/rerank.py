"""LLM rerank — Tier-L reorder of the top-30 + HU rationale generation.

`rerank_and_explain` takes the head of the scored list plus user / session context
and either returns the fast-path top-10 (when there is no semantic query
to rerank), or calls the Tier-L LLM, validates the structured response, caches
the result, and surfaces a `low_match` flag when too many returned items are
flagged low-confidence.

Two patterns mirror the parser deliberately: a cache behind a `RerankCache`
Protocol, and one correction retry on schema-validation failure with one
`EnrichmentLog` row per LLM attempt. The service uses `CacheRerankCache` over the
shared Redis-backed cache; tests can still use `InMemoryRerankCache`.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cache import Cache, get_json, set_json
from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.lib.llm_models import TIER_L_RERANK
from ingovo_search.lib.llm_usage import reset_last_llm_usage
from ingovo_search.recommender.llm_attempt_log import LlmAttempt, log_llm_attempt
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank_prompt import (
    DEFAULT_RERANK_LOCALE,
    RERANK_CORRECTION_SYSTEM_PROMPT,
    RERANK_PROMPT_VERSION,
    RERANK_RANK_ONLY_CORRECTION_SYSTEM_PROMPT,
    RERANK_RANK_ONLY_PROMPT_VERSION,
    RERANK_RANK_ONLY_SYSTEM_PROMPT,
    RERANK_SYSTEM_PROMPT,
    RerankCandidate,
    localized_rerank_prompt,
    render_correction_user_prompt,
    render_rerank_user_prompt,
)
from ingovo_search.types.errors import LlmError

DEFAULT_CACHE_TTL_SECONDS = 60 * 60
RERANK_TOP_N = 10
_logger = structlog.get_logger("ingovo_search.recommender.rerank")
# `RerankResult.low_match` flips when this many of the returned top items
# carry `low_confidence=True`.
LOW_MATCH_THRESHOLD = 7


class RerankItem(BaseModel):
    """One reranked listing slot with its LLM-written HU rationale."""

    model_config = {"frozen": True}

    listing_id: str
    rank: int = Field(ge=1)
    rationale_hu: str = ""
    low_confidence: bool = False


def _empty_rerank_items() -> list[RerankItem]:
    """Typed empty-list factory for the rerank-items field defaults."""
    return []


class RerankResponse(BaseModel):
    """Raw structured output from the Tier-L LLM call."""

    model_config = {"frozen": True}

    items: list[RerankItem] = Field(default_factory=_empty_rerank_items)


class RerankRankOnlyItem(BaseModel):
    """One reranked slot in rank-only mode — order only, no rationale."""

    model_config = {"frozen": True}

    listing_id: str
    rank: int = Field(ge=1)
    low_confidence: bool = False


def _empty_rank_only_items() -> list[RerankRankOnlyItem]:
    return []


class RerankRankOnlyResponse(BaseModel):
    """Raw structured output from the Tier-L LLM call in rank-only mode.

    Mirrors `RerankResponse` minus `rationale_hu`, so the LLM emits far fewer
    output tokens. Normalised back into `RerankItem` (with an empty rationale)
    before it leaves `rerank_and_explain`, so callers see one result shape.
    """

    model_config = {"frozen": True}

    items: list[RerankRankOnlyItem] = Field(default_factory=_empty_rank_only_items)


class RerankResult(BaseModel):
    """Final rerank result returned to the API layer."""

    model_config = {"frozen": True}

    items: list[RerankItem] = Field(default_factory=_empty_rerank_items)
    low_match: bool = False
    fast_path: bool = False
    fallback: bool = False


@runtime_checkable
class RerankCache(Protocol):
    """Cache contract for `rerank_and_explain`."""

    async def get(self, key: str) -> RerankResult | None: ...
    async def set(self, key: str, value: RerankResult) -> None: ...


class InMemoryRerankCache:
    """Process-local TTL cache. Entries expire `ttl_seconds` after `set`."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, RerankResult]] = {}

    async def get(self, key: str) -> RerankResult | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: RerankResult) -> None:
        self._store[key] = (self._clock() + self._ttl, value)


class CacheRerankCache:
    """`RerankCache` adapter backed by the shared async cache."""

    def __init__(self, cache: Cache, *, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(key: str) -> str:
        return f"rerank:{key}"

    async def get(self, key: str) -> RerankResult | None:
        return await get_json(self._cache, self._key(key), RerankResult)

    async def set(self, key: str, value: RerankResult) -> None:
        await set_json(self._cache, self._key(key), value, ttl_seconds=self._ttl_seconds)


@dataclass(frozen=True, slots=True)
class RerankDeps[ClientT]:
    """Who answers a rerank and where its traces go — shared by the Gemini and the Jev rerank.

    `cache` holds finished results; `session` receives one `EnrichmentLog` row per
    model attempt. Both are optional: without them the call simply leaves no trace.
    """

    client: ClientT
    cache: RerankCache | None = None
    session: AsyncSession | None = None


@dataclass(frozen=True, slots=True)
class RerankOptions:
    """Everything `rerank_and_explain` can be told besides what to rank and who ranks it.

    `user_prefs` / `session_signals` personalise the prompt; the rest pick the model,
    the prompt, how many items keep their LLM order and the language of the rationale.
    """

    user_prefs: Sequence[str] | None = None
    session_signals: Sequence[str] | None = None
    model: str = TIER_L_RERANK
    prompt_version: str = RERANK_PROMPT_VERSION
    top_n: int = RERANK_TOP_N
    rank_only: bool = False
    locale: str = DEFAULT_RERANK_LOCALE


DEFAULT_RERANK_OPTIONS: Final = RerankOptions()


def rerank_cache_key(
    query: ParsedQuery,
    candidate_ids: Sequence[str],
    model: str,
    prompt_version: str,
    locale: str = DEFAULT_RERANK_LOCALE,
) -> str:
    """Cache key: hash of model + prompt version + canonical query + candidate id list.

    Pydantic v2 serialises fields in declaration order, so `model_dump_json`
    is deterministic enough for cache keying without an extra canonical-JSON
    pass.
    """
    canonical_query = query.model_dump_json()
    payload = f"{model}\x00{prompt_version}\x00{locale}\x00{canonical_query}\x00{','.join(candidate_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_semantic_query(query: ParsedQuery) -> bool:
    return bool(query.semantic_query and query.semantic_query.strip())


def _fast_path_result(candidates: Sequence[RerankCandidate]) -> RerankResult:
    """Fast path: preserve every candidate, no LLM call."""
    items = [RerankItem(listing_id=candidate.listing_id, rank=index + 1) for index, candidate in enumerate(candidates)]
    return RerankResult(items=items, low_match=False, fast_path=True)


def _fallback_result(candidates: Sequence[RerankCandidate]) -> RerankResult:
    items = [RerankItem(listing_id=candidate.listing_id, rank=index + 1) for index, candidate in enumerate(candidates)]
    return RerankResult(items=items, low_match=False, fast_path=False, fallback=True)


def _normalise_items(
    parsed_items: Sequence[RerankItem],
    candidates: Sequence[RerankCandidate],
    *,
    top_n: int,
) -> tuple[list[RerankItem], int, int]:
    """Reject unknowns, dedupe stably and append every omitted candidate."""

    candidate_ids = [candidate.listing_id for candidate in candidates]
    allowed = set(candidate_ids)
    ordered = sorted(enumerate(parsed_items), key=lambda pair: (pair[1].rank, pair[0]))
    selected: list[RerankItem] = []
    seen: set[str] = set()
    unknown_count = 0
    duplicate_count = 0
    for _, item in ordered:
        if item.listing_id not in allowed:
            unknown_count += 1
            continue
        if item.listing_id in seen:
            duplicate_count += 1
            continue
        if len(selected) >= max(0, top_n):
            continue
        selected.append(item)
        seen.add(item.listing_id)

    for candidate_id in candidate_ids:
        if candidate_id in seen:
            continue
        selected.append(RerankItem(listing_id=candidate_id, rank=len(selected) + 1))
        seen.add(candidate_id)

    normalised = [item.model_copy(update={"rank": index}) for index, item in enumerate(selected, start=1)]
    return normalised, unknown_count, duplicate_count


def _parse_rerank_items(raw: str, *, rank_only: bool) -> list[RerankItem]:
    """Validate the LLM output against the active schema and normalise to `RerankItem`.

    Rank-only responses carry no rationale, so they parse against the smaller
    `RerankRankOnlyResponse` and are widened to `RerankItem` with an empty
    `rationale_hu`. Raises `ValidationError` on schema mismatch (caller retries).
    """
    if rank_only:
        parsed = RerankRankOnlyResponse.model_validate_json(raw)
        return [
            RerankItem(listing_id=item.listing_id, rank=item.rank, low_confidence=item.low_confidence)
            for item in parsed.items
        ]
    return list(RerankResponse.model_validate_json(raw).items)


@dataclass(frozen=True, slots=True)
class _RerankMode:
    """Rationale mode or rank-only mode: the prompt version, the system prompts, the output schema."""

    version: str
    system_initial: str
    system_correction: str
    response_schema: type[BaseModel]
    rank_only: bool


def _rerank_mode(*, rank_only: bool, locale: str, prompt_version: str) -> _RerankMode:
    # Rank-only is a distinct mode with its own prompt + output schema, so it
    # keys the cache and the EnrichmentLog under a dedicated version — never
    # mixing with rationale-mode results. Rank-only mode emits no rationale, so
    # its prompt needs no language directive.
    return _RerankMode(
        version=RERANK_RANK_ONLY_PROMPT_VERSION if rank_only else prompt_version,
        system_initial=(
            RERANK_RANK_ONLY_SYSTEM_PROMPT if rank_only else localized_rerank_prompt(RERANK_SYSTEM_PROMPT, locale)
        ),
        system_correction=(
            RERANK_RANK_ONLY_CORRECTION_SYSTEM_PROMPT
            if rank_only
            else localized_rerank_prompt(RERANK_CORRECTION_SYSTEM_PROMPT, locale)
        ),
        response_schema=RerankRankOnlyResponse if rank_only else RerankResponse,
        rank_only=rank_only,
    )


@dataclass(frozen=True, slots=True)
class _ModelCall:
    """What one rerank call needs besides the query and the candidates."""

    client: LlmClient
    session: AsyncSession | None
    model: str
    mode: _RerankMode
    user_prefs: Sequence[str] | None
    session_signals: Sequence[str] | None


@dataclass(frozen=True, slots=True)
class _ModelAnswer:
    items: list[RerankItem] | None
    transport_error: bool = False
    last_error: ValidationError | None = None


def _log_model_attempt(call: _ModelCall, req: LlmRequest, *, started: float, outcome: str, raw: str) -> None:
    if call.session is None:
        return
    log_llm_attempt(call.session, LlmAttempt(req, call.mode.version, started), outcome=outcome, raw=raw)


async def _ask_model(call: _ModelCall, query: ParsedQuery, candidates: Sequence[RerankCandidate]) -> _ModelAnswer:
    """The LLM call with one correction retry on a schema failure, one `EnrichmentLog` row per attempt."""
    last_error: ValidationError | None = None
    for attempt in (1, 2):
        is_correction = attempt > 1
        user_prompt = (render_correction_user_prompt if is_correction else render_rerank_user_prompt)(
            query, candidates, user_prefs=call.user_prefs, session_signals=call.session_signals
        )
        req = LlmRequest(
            model=call.model,
            system=call.mode.system_correction if is_correction else call.mode.system_initial,
            user=user_prompt,
            response_schema=call.mode.response_schema,
        )
        started = time.monotonic()
        try:
            reset_last_llm_usage()
            raw = await call.client.complete(req)
        except LlmError:
            _log_model_attempt(call, req, started=started, outcome="transport_error", raw="")
            return _ModelAnswer(None, transport_error=True)
        try:
            items = _parse_rerank_items(raw, rank_only=call.mode.rank_only)
        except ValidationError as exc:
            last_error = exc
            _log_model_attempt(call, req, started=started, outcome="parse_error", raw=raw)
            continue
        _log_model_attempt(call, req, started=started, outcome="success", raw=raw)
        return _ModelAnswer(items)
    return _ModelAnswer(None, last_error=last_error)


async def _cached_rerank(
    cache: RerankCache, key: str, candidates: Sequence[RerankCandidate], *, top_n: int
) -> RerankResult | None:
    cached = await cache.get(key)
    if cached is None:
        _logger.info("rerank cache miss", cache_key=key, candidate_count=len(candidates))
        return None
    _logger.info("rerank cache hit", cache_key=key, candidate_count=len(candidates))
    items, _, _ = _normalise_items(cached.items, candidates, top_n=top_n)
    return cached.model_copy(update={"items": items})


def _result_from_items(
    parsed_items: list[RerankItem], candidates: Sequence[RerankCandidate], *, top_n: int
) -> RerankResult:
    """The model's order, with the ids it mangled dropped, and the low-match flag."""
    # The LLM occasionally echoes a mangled listing id (a dropped or transposed
    # UUID character) — an id we never sent as a candidate. Such an id must not
    # reach the hard-filter guardrail in the search pipeline: that guardrail
    # is a safety net for genuine ranking bugs, not LLM typos, and tripping it
    # would 500 the whole search. Drop unknown ids here and keep the valid order;
    # any candidate the LLM thereby omits falls back to its deterministic slot.
    top_items, unknown_count, duplicate_count = _normalise_items(parsed_items, candidates, top_n=top_n)
    if unknown_count or duplicate_count:
        _logger.warning(
            "rerank normalised invalid listing ids",
            unknown_count=unknown_count,
            duplicate_count=duplicate_count,
            candidate_count=len(candidates),
        )
    low_confidence_count = sum(1 for item in top_items if item.low_confidence)
    return RerankResult(items=top_items, low_match=low_confidence_count >= LOW_MATCH_THRESHOLD, fast_path=False)


async def rerank_and_explain(
    query: ParsedQuery,
    candidates: Sequence[RerankCandidate],
    backend: RerankDeps[LlmClient],
    options: RerankOptions = DEFAULT_RERANK_OPTIONS,
) -> RerankResult:
    """Rerank the top-30 by Tier-L LLM, attach HU rationales, surface low-confidence.

    Returns the fast-path result without an LLM call when `query.semantic_query`
    is empty ("filter-only fast path"). Otherwise: cache lookup → LLM call
    (one correction retry on schema failure, one `EnrichmentLog` row per attempt)
    → flip `low_match=True` when ≥ `LOW_MATCH_THRESHOLD` of the top items carry
    `low_confidence=True` → cache write → return.

    With `options.rank_only` the LLM returns only the order — rank +
    low_confidence, no per-item rationale — which cuts the rerank output-token
    count and latency. The ranking signal is unchanged; the returned
    `RerankItem`s simply carry an empty `rationale_hu`. A dedicated prompt
    version keeps the rerank cache from mixing the two modes.
    """
    if not _has_semantic_query(query):
        return _fast_path_result(candidates)

    mode = _rerank_mode(rank_only=options.rank_only, locale=options.locale, prompt_version=options.prompt_version)
    candidate_ids = [candidate.listing_id for candidate in candidates]
    key = rerank_cache_key(
        query,
        candidate_ids,
        options.model,
        mode.version,
        DEFAULT_RERANK_LOCALE if options.rank_only else options.locale,
    )
    cache = backend.cache
    if cache is not None:
        cached = await _cached_rerank(cache, key, candidates, top_n=options.top_n)
        if cached is not None:
            return cached

    call = _ModelCall(backend.client, backend.session, options.model, mode, options.user_prefs, options.session_signals)
    answer = await _ask_model(call, query, candidates)
    if answer.items is None:
        if not answer.transport_error:
            _logger.warning(
                "rerank schema validation failed; using deterministic fallback",
                candidate_count=len(candidate_ids),
                error=str(answer.last_error),
            )
        return _fallback_result(candidates)

    result = _result_from_items(answer.items, candidates, top_n=options.top_n)
    if cache is not None:
        await cache.set(key, result)
        _logger.info("rerank cache set", cache_key=key, candidate_count=len(candidate_ids))
    return result
