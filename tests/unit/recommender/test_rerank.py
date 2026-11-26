"""Unit tests for §11 rerank — `rerank_and_explain`, cache, fast path.

Covers acceptance criteria 1-4 from the design spec §11:
1. Mock Tier-L returns 10 ranked items with rationales; `rerank_and_explain`
   returns them in rank order.
2. Same query twice within 10 min → second call hits the cache.
3. Mock returns `low_confidence=True` for 7/10 → `RerankResult.low_match=True`.
4. Empty NL query → fast path; no LLM call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.llm import LlmClient, LlmRequest, MockLlmClient, fixture_key
from ingovo_search.lib.llm_models import TIER_L_RERANK
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.rerank import (
    CacheRerankCache,
    InMemoryRerankCache,
    RerankDeps,
    RerankItem,
    RerankOptions,
    RerankResult,
    rerank_and_explain,
    rerank_cache_key,
)
from ingovo_search.recommender.rerank_prompt import (
    RERANK_PROMPT_VERSION,
    RERANK_RANK_ONLY_PROMPT_VERSION,
    RERANK_SYSTEM_PROMPT,
    RerankCandidate,
    localized_rerank_prompt,
    render_correction_user_prompt,
    render_rerank_user_prompt,
)


def _query(text: str = "kétszobás panel a XII. kerületben") -> ParsedQuery:
    return ParsedQuery(hard_filters=HardFilters(), semantic_query=text)


def _candidates(n: int = 10) -> list[RerankCandidate]:
    return [RerankCandidate(listing_id=f"L{i:02d}", short_summary=f"Listing summary {i}") for i in range(n)]


def _initial_key(query: ParsedQuery, candidates: Sequence[RerankCandidate]) -> str:
    prompt = render_rerank_user_prompt(query, candidates)
    return fixture_key(LlmRequest(model=TIER_L_RERANK, system="", user=prompt))


def _correction_key(query: ParsedQuery, candidates: Sequence[RerankCandidate]) -> str:
    prompt = render_correction_user_prompt(query, candidates)
    return fixture_key(LlmRequest(model=TIER_L_RERANK, system="", user=prompt))


def _valid_response(
    candidates: Sequence[RerankCandidate],
    *,
    low_confidence_count: int = 0,
    rank_order: Sequence[int] | None = None,
) -> str:
    """Build a `RerankResponse` JSON over the first 10 `candidates`.

    `rank_order` lets a test scramble the rank-assignment order — useful for
    confirming `rerank_and_explain` sorts by `rank` before returning.
    """
    selected = list(candidates[:10])
    ranks = list(rank_order) if rank_order is not None else list(range(1, len(selected) + 1))
    items: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        items.append(
            {
                "listing_id": candidate.listing_id,
                "rank": ranks[index],
                "rationale_hu": f"Ez egy szép lakás: {candidate.listing_id}",
                "low_confidence": index < low_confidence_count,
            }
        )
    return json.dumps({"items": items})


class _RecordingSession:
    """Captures `session.add(...)` calls without touching a database."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, entity: Any) -> None:
        self.added.append(entity)


class _CountingLlmClient:
    """Wraps `MockLlmClient` to count `complete()` invocations."""

    def __init__(self, fixtures: dict[str, str] | None = None) -> None:
        self._inner = MockLlmClient(fixtures)
        self.calls: int = 0

    async def complete(self, req: LlmRequest) -> str:
        self.calls += 1
        return await self._inner.complete(req)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _fake_session() -> AsyncSession:
    return cast(AsyncSession, _RecordingSession())


def _enrichment_logs(session: AsyncSession) -> list[EnrichmentLog]:
    return [entity for entity in cast(_RecordingSession, session).added if isinstance(entity, EnrichmentLog)]


# --- Acceptance 1 ------------------------------------------------------------


async def test_rerank_returns_items_sorted_by_rank() -> None:
    query = _query()
    candidates = _candidates(10)
    # LLM returns ranks scrambled; rerank_and_explain must re-sort by rank.
    scrambled = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates, rank_order=scrambled)})
    session = _fake_session()

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), session=session))

    assert isinstance(result, RerankResult)
    assert result.fast_path is False
    assert result.low_match is False
    assert [item.rank for item in result.items] == list(range(1, 11))
    # Rank 1 was assigned to the LAST input candidate (L09); confirm it's first.
    assert result.items[0].listing_id == "L09"
    assert all(item.rationale_hu for item in result.items)
    assert client.calls == 1

    logs = _enrichment_logs(session)
    assert len(logs) == 1
    assert logs[0].outcome == "success"
    assert logs[0].model_used == TIER_L_RERANK
    assert logs[0].prompt_version == RERANK_PROMPT_VERSION


async def test_rerank_top_n_limits_llm_order_but_preserves_all_candidates() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)), RerankOptions(top_n=5))

    assert len(result.items) == 10
    assert [item.rank for item in result.items] == list(range(1, 11))
    assert {item.listing_id for item in result.items} == {candidate.listing_id for candidate in candidates}


async def test_rerank_drops_listing_ids_not_in_candidate_set() -> None:
    """A mangled/hallucinated id from the LLM is dropped, not surfaced.

    The Tier-L LLM occasionally echoes a listing id with a dropped UUID
    character. That id was never a candidate, so `rerank_and_explain` must
    discard it — otherwise the §14a hard-filter guardrail downstream raises and
    500s the whole search. The remaining valid ids keep their rank order.
    """
    query = _query()
    candidates = _candidates(10)
    # Corrupt the rank-1 item's id (drop a character) — exactly the failure the
    # production guardrail caught (`...ba103ebcc0f3` echoed as `...ba103bcc0f3`).
    response = json.loads(_valid_response(candidates))
    response["items"][0]["listing_id"] = "L0"  # not in {L00..L09}
    client = _CountingLlmClient({_initial_key(query, candidates): json.dumps(response)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    candidate_ids = {candidate.listing_id for candidate in candidates}
    returned_ids = [item.listing_id for item in result.items]
    assert "L0" not in returned_ids
    assert all(listing_id in candidate_ids for listing_id in returned_ids)
    assert len(returned_ids) == 10
    assert set(returned_ids) == candidate_ids
    assert returned_ids[-1] == "L00"  # the omitted real candidate is deterministically reinserted


@pytest.mark.parametrize("candidate_count", [12, 20])
async def test_partial_llm_response_preserves_complete_candidate_set(candidate_count: int) -> None:
    query = _query()
    candidates = _candidates(candidate_count)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert len(result.items) == candidate_count
    assert {item.listing_id for item in result.items} == {candidate.listing_id for candidate in candidates}


async def test_empty_llm_items_return_full_deterministic_set() -> None:
    query = _query("üres rerank")
    candidates = _candidates(12)
    client = _CountingLlmClient({_initial_key(query, candidates): json.dumps({"items": []})})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert [item.listing_id for item in result.items] == [candidate.listing_id for candidate in candidates]


async def test_duplicate_ids_are_stably_deduplicated_and_missing_candidate_reinserted() -> None:
    query = _query("duplikált rerank")
    candidates = _candidates(3)
    response = json.loads(_valid_response(candidates))
    response["items"] = [response["items"][1], response["items"][1], response["items"][0]]
    client = _CountingLlmClient({_initial_key(query, candidates): json.dumps(response)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert [item.listing_id for item in result.items] == ["L00", "L01", "L02"]
    assert len({item.listing_id for item in result.items}) == 3


def test_rerank_prompt_contains_structured_query_contract() -> None:
    query = ParsedQuery(
        semantic_query="lakás",
        hard_filters=HardFilters(city="Budapest", rooms_min=2, price_huf_max=80_000_000),
    )
    prompt = render_rerank_user_prompt(query, _candidates(1))
    assert "Structured query JSON:" in prompt
    assert '"city":"Budapest"' in prompt
    assert '"rooms_min":2' in prompt
    assert '"price_huf_max":80000000' in prompt


# --- Acceptance 2 ------------------------------------------------------------


async def test_same_query_twice_within_window_hits_cache() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates)})
    cache = InMemoryRerankCache()

    first = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))
    second = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))

    assert first == second
    assert client.calls == 1


async def test_shared_cache_adapter_survives_rerank_cache_reinstantiation() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates)})
    backend = MemoryCache()

    first = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), cache=CacheRerankCache(backend))
    )
    second = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), cache=CacheRerankCache(backend))
    )

    assert first == second
    assert client.calls == 1


async def test_expired_cache_entry_triggers_new_llm_call() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates)})
    clock = _Clock()
    cache = InMemoryRerankCache(ttl_seconds=600.0, clock=clock)

    await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))
    clock.now = 601.0
    await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))

    assert client.calls == 2


# --- Acceptance 3 ------------------------------------------------------------


async def test_low_match_flips_true_when_seven_of_ten_are_low_confidence() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates, low_confidence_count=7)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert result.low_match is True


async def test_low_match_stays_false_with_six_low_confidence_items() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient({_initial_key(query, candidates): _valid_response(candidates, low_confidence_count=6)})

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert result.low_match is False


# --- Acceptance 4 (fast path) -----------------------------------------------


async def test_empty_semantic_query_takes_fast_path_without_llm_call() -> None:
    query = _query(text="")
    candidates = _candidates(10)
    client = _CountingLlmClient({})  # no fixtures → any LLM call would raise

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client)))

    assert client.calls == 0
    assert result.fast_path is True
    assert result.low_match is False
    assert [item.listing_id for item in result.items] == [c.listing_id for c in candidates[:10]]
    assert [item.rank for item in result.items] == list(range(1, 11))
    # Fast path does not invent rationales.
    assert all(item.rationale_hu == "" for item in result.items)


async def test_whitespace_only_semantic_query_takes_fast_path() -> None:
    query = _query(text="   ")
    client = _CountingLlmClient({})
    result = await rerank_and_explain(query, _candidates(3), RerankDeps(cast(LlmClient, client)))
    assert result.fast_path is True
    assert client.calls == 0


# --- Schema retry + transport handling --------------------------------------


async def test_persistent_malformed_json_returns_full_fallback() -> None:
    query = _query("hibás lekérdezés")
    candidates = _candidates(3)
    client = _CountingLlmClient(
        {
            _initial_key(query, candidates): "not json {",
            _correction_key(query, candidates): "still not json",
        }
    )
    session = _fake_session()

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), session=session))

    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "parse_error"]
    assert result.fallback is True
    assert [item.listing_id for item in result.items] == [candidate.listing_id for candidate in candidates]


async def test_wrong_schema_then_valid_on_retry() -> None:
    query = _query("rossz séma majd jó")
    candidates = _candidates(3)
    client = _CountingLlmClient(
        {
            _initial_key(query, candidates): json.dumps({"items": "not-a-list"}),
            _correction_key(query, candidates): _valid_response(candidates),
        }
    )
    session = _fake_session()

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), session=session))

    assert isinstance(result, RerankResult)
    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "success"]


async def test_transport_error_returns_full_fallback_and_logs() -> None:
    query = _query("transport hiba")
    candidates = _candidates(3)
    client = _CountingLlmClient({})  # no fixtures → MockLlmClient raises LlmError
    session = _fake_session()

    result = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), session=session))

    logs = _enrichment_logs(session)
    assert len(logs) == 1
    assert logs[0].outcome == "transport_error"
    assert result.fallback is True
    assert [item.listing_id for item in result.items] == [candidate.listing_id for candidate in candidates]


# --- Cache key + schema --------------------------------------------------


def test_cache_key_changes_with_prompt_version() -> None:
    query = _query()
    ids = [c.listing_id for c in _candidates(5)]
    key_a = rerank_cache_key(query, ids, TIER_L_RERANK, "rerank/v0-skeleton")
    key_b = rerank_cache_key(query, ids, TIER_L_RERANK, "rerank/v1")
    assert key_a != key_b


def test_cache_key_separates_rationale_locales() -> None:
    """`rationale_hu` is written per language, so the entries must not collide."""
    query = _query()
    ids = [c.listing_id for c in _candidates(5)]
    key_hu = rerank_cache_key(query, ids, TIER_L_RERANK, RERANK_PROMPT_VERSION, "hu")
    key_de = rerank_cache_key(query, ids, TIER_L_RERANK, RERANK_PROMPT_VERSION, "de")
    assert key_hu != key_de


def test_localized_rerank_prompt_pins_the_output_language() -> None:
    directive = localized_rerank_prompt(RERANK_SYSTEM_PROMPT, "pl")
    assert RERANK_SYSTEM_PROMPT in directive
    # Leading and trailing, so the Hungarian prompt body cannot outweigh it.
    assert directive.startswith("KIMENETI NYELV:")
    assert directive.count("LENGYEL") == 2


@pytest.mark.parametrize("locale", ["hu", "klingon", ""])
def test_localized_rerank_prompt_falls_back_to_hungarian(locale: str) -> None:
    assert localized_rerank_prompt(RERANK_SYSTEM_PROMPT, locale) == RERANK_SYSTEM_PROMPT


def test_cache_key_changes_with_candidate_set() -> None:
    query = _query()
    key_a = rerank_cache_key(query, ["L00", "L01"], TIER_L_RERANK, RERANK_PROMPT_VERSION)
    key_b = rerank_cache_key(query, ["L00", "L02"], TIER_L_RERANK, RERANK_PROMPT_VERSION)
    assert key_a != key_b


def test_rerank_result_is_frozen() -> None:
    result = RerankResult(items=[RerankItem(listing_id="L00", rank=1)], low_match=False)
    with pytest.raises(ValidationError):
        result.low_match = True  # type: ignore[misc]


def test_rerank_item_rejects_zero_rank() -> None:
    with pytest.raises(ValidationError):
        RerankItem(listing_id="L00", rank=0)


# --- §11 Phase 5: rank-only mode --------------------------------------------


def _rank_only_response(
    candidates: Sequence[RerankCandidate],
    *,
    low_confidence_count: int = 0,
    rank_order: Sequence[int] | None = None,
) -> str:
    """Build a `RerankRankOnlyResponse` JSON (no rationale) over the first 10 candidates."""
    selected = list(candidates[:10])
    ranks = list(rank_order) if rank_order is not None else list(range(1, len(selected) + 1))
    items = [
        {"listing_id": candidate.listing_id, "rank": ranks[index], "low_confidence": index < low_confidence_count}
        for index, candidate in enumerate(selected)
    ]
    return json.dumps({"items": items})


async def test_rank_only_returns_order_without_rationale() -> None:
    query = _query()
    candidates = _candidates(10)
    scrambled = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    client = _CountingLlmClient(
        {_initial_key(query, candidates): _rank_only_response(candidates, rank_order=scrambled)}
    )
    session = _fake_session()

    result = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), session=session), RerankOptions(rank_only=True)
    )

    assert result.fast_path is False
    assert [item.rank for item in result.items] == list(range(1, 11))
    assert result.items[0].listing_id == "L09"  # rank 1 went to the last input
    # Rank-only mode carries no rationale — it is deferred to the on-demand drawer.
    assert all(item.rationale_hu == "" for item in result.items)
    assert client.calls == 1

    logs = _enrichment_logs(session)
    assert len(logs) == 1
    assert logs[0].outcome == "success"
    assert logs[0].prompt_version == RERANK_RANK_ONLY_PROMPT_VERSION


async def test_rank_only_low_match_flips_with_seven_low_confidence() -> None:
    query = _query()
    candidates = _candidates(10)
    client = _CountingLlmClient(
        {_initial_key(query, candidates): _rank_only_response(candidates, low_confidence_count=7)}
    )

    result = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client)), RerankOptions(rank_only=True)
    )

    assert result.low_match is True


async def test_rank_only_and_rationale_modes_do_not_share_cache_entry() -> None:
    """Cache must never serve a rank-only result where a rationale result is
    expected (or vice versa) — they key under distinct prompt versions."""
    query = _query()
    candidates = _candidates(10)
    # A single rank-only-shaped fixture parses under both schemas (rationale_hu
    # defaults to ""), so the same MockLlmClient serves both modes.
    client = _CountingLlmClient({_initial_key(query, candidates): _rank_only_response(candidates)})
    cache = InMemoryRerankCache()

    rationale = await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))
    rank_only = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), cache=cache), RerankOptions(rank_only=True)
    )

    # Distinct versions → distinct cache keys → the second mode is a cache miss
    # and triggers its own LLM call (not served the first mode's cached entry).
    assert client.calls == 2
    # Re-running each mode now hits its own cached entry (no further LLM calls).
    await rerank_and_explain(query, candidates, RerankDeps(cast(LlmClient, client), cache=cache))
    await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), cache=cache), RerankOptions(rank_only=True)
    )
    assert client.calls == 2
    assert [i.listing_id for i in rationale.items] == [i.listing_id for i in rank_only.items]


async def test_rank_only_retries_then_succeeds_on_schema_error() -> None:
    query = _query("rank only retry")
    candidates = _candidates(3)
    client = _CountingLlmClient(
        {
            _initial_key(query, candidates): json.dumps({"items": "not-a-list"}),
            _correction_key(query, candidates): _rank_only_response(candidates),
        }
    )
    session = _fake_session()

    result = await rerank_and_explain(
        query, candidates, RerankDeps(cast(LlmClient, client), session=session), RerankOptions(rank_only=True)
    )

    assert isinstance(result, RerankResult)
    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "success"]
    assert all(log.prompt_version == RERANK_RANK_ONLY_PROMPT_VERSION for log in _enrichment_logs(session))
