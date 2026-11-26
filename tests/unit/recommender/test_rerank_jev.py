"""`jev_rerank` — ordering, blending, fallbacks, low_match, cache and cost logging."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.jev import FakeJevClient
from ingovo_search.lib.llm_models import TIER_L_RERANK_JEV
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.rerank import InMemoryRerankCache, RerankDeps, RerankResult
from ingovo_search.recommender.rerank_jev import JevRerankOptions, jev_rerank
from ingovo_search.recommender.rerank_jev_prompt import JEV_RERANK_PROMPT_VERSION, NOUL_HU
from ingovo_search.recommender.rerank_prompt import RerankCandidate
from ingovo_search.types.errors import JevError


class _RecordingSession:
    """Captures `session.add(...)` calls without touching a database."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, entity: Any) -> None:
        self.added.append(entity)


def _session() -> AsyncSession:
    return cast(AsyncSession, _RecordingSession())


def _logs(session: AsyncSession) -> list[EnrichmentLog]:
    return [entity for entity in cast(_RecordingSession, session).added if isinstance(entity, EnrichmentLog)]


def _query(text: str = "csendes, erkélyes lakás Budán") -> ParsedQuery:
    return ParsedQuery(hard_filters=HardFilters(), semantic_query=text)


def _candidates(n: int) -> list[RerankCandidate]:
    return [RerankCandidate(listing_id=f"L{index}", short_summary=f"Lakás {index}") for index in range(n)]


def _ids(result: RerankResult) -> list[str]:
    return [item.listing_id for item in result.items]


async def test_orders_by_jev_fit_when_alpha_is_one() -> None:
    fake = FakeJevClient(scores={"c0": 0.5, "c1": 3.0, "c2": 1.8, "c3": 2.4})

    result = await jev_rerank(_query(), _candidates(4), RerankDeps(fake))

    assert _ids(result) == ["L1", "L3", "L2", "L0"]
    assert [item.rank for item in result.items] == [1, 2, 3, 4]
    assert not result.fallback
    assert not result.fast_path


async def test_alpha_blends_in_the_deterministic_score() -> None:
    # Jev fit (score/3): L0 0.2, L1 0.6, L2 0.4. Deterministic, min-max: L0 1.0, L1 0.0, L2 0.5.
    # alpha 0.5 → L0 0.60, L1 0.30, L2 0.45.
    fake = FakeJevClient(scores={"c0": 0.6, "c1": 1.8, "c2": 1.2})

    result = await jev_rerank(
        _query(),
        _candidates(3),
        RerankDeps(fake),
        JevRerankOptions(deterministic_scores={"L0": 0.9, "L1": 0.5, "L2": 0.7}, alpha=0.5),
    )

    assert _ids(result) == ["L0", "L2", "L1"]


async def test_alpha_falls_back_to_incoming_rank_without_deterministic_scores() -> None:
    # Rank standing: L0 1.0, L1 0.5, L2 0.0; equal Jev fit → the deterministic order wins.
    fake = FakeJevClient(scores={"c0": 1.5, "c1": 1.5, "c2": 1.5})

    result = await jev_rerank(_query(), _candidates(3), RerankDeps(fake), JevRerankOptions(alpha=0.7))

    assert _ids(result) == ["L0", "L1", "L2"]


async def test_ties_keep_the_deterministic_order() -> None:
    fake = FakeJevClient(scores={"c0": 2.0, "c1": 3.0, "c2": 2.0, "c3": 2.0})

    result = await jev_rerank(_query(), _candidates(4), RerankDeps(fake))

    assert _ids(result) == ["L1", "L0", "L2", "L3"]


async def test_every_candidate_appears_exactly_once() -> None:
    candidates = _candidates(12)
    fake = FakeJevClient(scores={f"c{index}": (index * 7) % 4 for index in range(12)})

    result = await jev_rerank(_query(), candidates, RerankDeps(fake))

    assert sorted(_ids(result)) == sorted(c.listing_id for c in candidates)
    assert len(result.items) == 12


async def test_no_semantic_query_takes_the_fast_path_without_a_call() -> None:
    fake = FakeJevClient(scores={"c0": 3.0})

    result = await jev_rerank(_query("  "), _candidates(3), RerankDeps(fake))

    assert result.fast_path
    assert _ids(result) == ["L0", "L1", "L2"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("reason", "status", "outcome"),
    [
        ("http", 529, "rate_limited"),
        ("http", 429, "rate_limited"),
        ("http", 503, "transport_error"),
        ("connection", None, "transport_error"),
        ("timeout", None, "timeout"),
    ],
)
async def test_jev_error_returns_the_deterministic_order_and_logs_it(
    reason: str, status: int | None, outcome: str
) -> None:
    session = _session()
    fake = FakeJevClient(error=JevError("boom", reason=reason, status=status))

    result = await jev_rerank(_query(), _candidates(3), RerankDeps(fake, session=session))

    assert result.fallback
    assert _ids(result) == ["L0", "L1", "L2"]
    [log] = _logs(session)
    assert log.outcome == outcome
    assert log.model_used == TIER_L_RERANK_JEV
    assert log.cost_usd == 0.0


async def test_a_skipped_candidate_keeps_its_deterministic_slot() -> None:
    # c1 has no answer: L1 stays at position 2, the others reorder around it.
    fake = FakeJevClient(scores={"c0": 0.5, "c2": 2.5, "c3": 1.5})

    result = await jev_rerank(_query(), _candidates(4), RerankDeps(fake))

    assert _ids(result) == ["L2", "L1", "L3", "L0"]
    assert not result.items[1].low_confidence


async def test_no_usable_answers_falls_back_and_logs_a_parse_error() -> None:
    session = _session()
    fake = FakeJevClient(scores={})

    result = await jev_rerank(_query(), _candidates(3), RerankDeps(fake, session=session))

    assert result.fallback
    assert [log.outcome for log in _logs(session)] == ["parse_error"]


@pytest.mark.parametrize(("weak", "low_match"), [(7, True), (6, False)])
async def test_low_match_flips_at_seven_weak_results_in_the_top_ten(weak: int, low_match: bool) -> None:
    # Weak = fit below 0.5 (score < 1.5). Strong ones rank first, weak ones fill the rest of the top 10.
    scores = {f"c{index}": 3.0 for index in range(10 - weak)}
    scores.update({f"c{index}": 1.0 for index in range(10 - weak, 12)})
    fake = FakeJevClient(scores=scores)

    result = await jev_rerank(_query(), _candidates(12), RerankDeps(fake))

    assert sum(item.low_confidence for item in result.items[:10]) == weak
    assert result.low_match is low_match


async def test_cache_hit_skips_the_call_and_returns_the_same_result() -> None:
    cache = InMemoryRerankCache()
    fake = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    first = await jev_rerank(_query(), _candidates(2), RerankDeps(fake, cache=cache))
    second = await jev_rerank(_query(), _candidates(2), RerankDeps(fake, cache=cache))

    assert second == first
    assert len(fake.calls) == 1


async def test_fallback_results_are_not_cached() -> None:
    cache = InMemoryRerankCache()
    failing = FakeJevClient(error=JevError("busy", reason="http", status=529))
    healthy = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    await jev_rerank(_query(), _candidates(2), RerankDeps(failing, cache=cache))
    result = await jev_rerank(_query(), _candidates(2), RerankDeps(healthy, cache=cache))

    assert _ids(result) == ["L1", "L0"]
    assert len(healthy.calls) == 1


async def test_log_uses_reported_tokens_and_the_served_model() -> None:
    session = _session()
    fake = FakeJevClient(scores={"c0": 2.0, "c1": 1.0}, input_tokens=4255, model="jev-1.13.0")

    await jev_rerank(_query(), _candidates(2), RerankDeps(fake, session=session))

    [log] = _logs(session)
    assert log.outcome == "success"
    assert log.model_used == "jev-1.13.0"
    assert log.prompt_version == JEV_RERANK_PROMPT_VERSION
    assert log.input_tokens == 4255
    assert log.output_tokens == 0
    assert log.cost_usd == pytest.approx(4255 * 0.042 / 1_000_000)  # pyright: ignore[reportUnknownMemberType]


async def test_rationale_is_always_empty() -> None:
    fake = FakeJevClient(scores={"c0": 2.0, "c1": 1.0, "c2": 3.0})

    result = await jev_rerank(_query(), _candidates(3), RerankDeps(fake))

    assert all(item.rationale_hu == "" for item in result.items)


async def test_noul_variant_orders_by_probability() -> None:
    fake = FakeJevClient(nouls={"c0": 0.2, "c1": 0.9, "c2": 0.6})

    result = await jev_rerank(_query(), _candidates(3), RerankDeps(fake), JevRerankOptions(variant=NOUL_HU))

    assert _ids(result) == ["L1", "L2", "L0"]
    assert [item.low_confidence for item in result.items] == [False, False, True]


async def test_passes_model_timeout_and_request_to_the_client() -> None:
    fake = FakeJevClient(scores={"c0": 1.0})

    await jev_rerank(_query(), _candidates(1), RerankDeps(fake), JevRerankOptions(timeout_s=0.8))

    [call] = fake.calls
    assert call.model == TIER_L_RERANK_JEV
    assert call.timeout_s == 0.8
    assert list(call.questions) == ["c0"]
    assert "jelöltek" in call.state


@pytest.mark.parametrize(
    ("reason", "status", "outcome"),
    [("http", 429, "rate_limited"), ("http", 401, "transport_error"), ("connection", None, "transport_error")],
)
async def test_raise_on_unavailable_hands_the_failure_to_the_caller_after_logging_it(
    reason: str, status: int | None, outcome: str
) -> None:
    session = _session()
    fake = FakeJevClient(error=JevError("no capacity", reason=reason, status=status))

    with pytest.raises(JevError) as raised:
        await jev_rerank(
            _query(), _candidates(3), RerankDeps(fake, session=session), JevRerankOptions(raise_on_unavailable=True)
        )

    assert raised.value.status == status
    [log] = _logs(session)
    assert log.outcome == outcome


async def test_raise_on_unavailable_still_absorbs_a_timeout() -> None:
    session = _session()
    fake = FakeJevClient(error=JevError("slow", reason="timeout"))

    result = await jev_rerank(
        _query(), _candidates(3), RerankDeps(fake, session=session), JevRerankOptions(raise_on_unavailable=True)
    )

    assert result.fallback
    assert _ids(result) == ["L0", "L1", "L2"]
    [log] = _logs(session)
    assert log.outcome == "timeout"
