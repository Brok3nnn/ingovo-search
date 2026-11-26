"""Real token usage and the cost rows built from it.

Covers the chain end to end without a provider or a database: `usage_metadata` →
`LlmUsage` → price → `EnrichmentLog` row, the context variable that hands a call's
usage to the attempt loggers, and the best-effort background writer.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib import llm_cost_log
from ingovo_search.lib.cost_tracker import (
    PRICING_USD_PER_M_TOKENS,
    cache_storage_cost_usd,
    call_cost,
    cost_usd_from_tokens,
    cost_usd_from_usage,
    estimate_cost_usd,
)
from ingovo_search.lib.llm import GoogleAiClient, LlmClient, LlmImage, LlmRequest
from ingovo_search.lib.llm_cost_log import drain_usage_writes, llm_call_row, record_llm_calls, record_llm_calls_later
from ingovo_search.lib.llm_usage import (
    LlmUsage,
    last_llm_usage,
    reset_last_llm_usage,
    set_last_llm_usage,
    usage_from_metadata,
)
from ingovo_search.recommender.image_query import analyze_images
from ingovo_search.recommender.parser import parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15)


class _Usage:
    """The attribute shape of google-genai's `GenerateContentResponseUsageMetadata`."""

    def __init__(
        self,
        prompt: object = None,
        cached: object = None,
        candidates: object = None,
        thoughts: object = None,
    ) -> None:
        self.prompt_token_count: object = prompt
        self.cached_content_token_count: object = cached
        self.candidates_token_count: object = candidates
        self.thoughts_token_count: object = thoughts


# ── usage_from_metadata ────────────────────────────────────────────────────


def test_usage_is_read_from_the_provider_fields() -> None:
    usage = usage_from_metadata(_Usage(prompt=6000, cached=4000, candidates=300, thoughts=50), model="m")

    assert usage == LlmUsage(model="m", prompt_tokens=6000, cached_tokens=4000, output_tokens=300, thinking_tokens=50)
    assert usage is not None
    assert usage.uncached_prompt_tokens == 2000
    assert usage.billed_output_tokens == 350


def test_no_metadata_or_all_zero_counts_is_no_usage() -> None:
    assert usage_from_metadata(None) is None
    assert usage_from_metadata(_Usage()) is None
    assert usage_from_metadata(_Usage(prompt=0, candidates=0)) is None


def test_a_cached_count_above_the_prompt_is_clamped() -> None:
    usage = usage_from_metadata(_Usage(prompt=100, cached=500, candidates=1))

    assert usage is not None
    assert usage.cached_tokens == 100
    assert usage.uncached_prompt_tokens == 0


def test_non_integer_counts_read_as_zero() -> None:
    odd = _Usage(prompt=10, cached="7", candidates=5, thoughts=True)

    usage = usage_from_metadata(odd)

    assert usage is not None
    assert (usage.thinking_tokens, usage.cached_tokens) == (0, 0)


def test_the_last_usage_is_task_local_and_resettable() -> None:
    reset_last_llm_usage()
    assert last_llm_usage() is None

    set_last_llm_usage(LlmUsage(model="m", prompt_tokens=1))
    assert last_llm_usage() == LlmUsage(model="m", prompt_tokens=1)

    reset_last_llm_usage()
    assert last_llm_usage() is None


# ── pricing ────────────────────────────────────────────────────────────────


def test_gemini_3_1_flash_lite_carries_its_own_published_price() -> None:
    """It used to carry the retired 2.0-flash's price, a third of the real one."""
    assert PRICING_USD_PER_M_TOKENS["gemini-3.1-flash-lite"] == (0.25, 1.50)


def test_usage_without_cache_costs_the_same_as_plain_tokens() -> None:
    usage = LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=6000, output_tokens=300)

    assert _close(
        cost_usd_from_usage(usage), cost_usd_from_tokens("gemini-2.5-flash-lite", input_tokens=6000, output_tokens=300)
    )


def test_cached_tokens_are_billed_at_the_cached_rate() -> None:
    all_cached = LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=10_000, cached_tokens=10_000)
    none_cached = LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=10_000)

    assert _close(cost_usd_from_usage(all_cached), 10_000 * 0.01 / 1_000_000)
    assert _close(cost_usd_from_usage(none_cached), 10_000 * 0.10 / 1_000_000)


def test_thinking_tokens_are_billed_as_output() -> None:
    usage = LlmUsage(model="gemini-2.5-flash", prompt_tokens=0, output_tokens=100, thinking_tokens=900)

    assert _close(cost_usd_from_usage(usage), 1000 * 2.50 / 1_000_000)


def test_an_unknown_model_is_priced_as_the_fallback() -> None:
    usage = LlmUsage(model="", prompt_tokens=1000, output_tokens=100)

    assert _close(
        cost_usd_from_usage(usage, fallback_model="gemini-2.5-flash"),
        cost_usd_from_tokens("gemini-2.5-flash", input_tokens=1000, output_tokens=100),
    )


def test_explicit_cache_storage_is_priced_per_token_hour() -> None:
    # 6 000 tokens for ten minutes at $1 / 1M token-hours.
    assert _close(cache_storage_cost_usd("gemini-2.5-flash-lite", tokens=6000, seconds=600), 0.001)
    assert cache_storage_cost_usd("jev-1.13.0", tokens=6000, seconds=600) == 0.0


def test_call_cost_prefers_reported_usage_and_names_the_answering_model() -> None:
    usage = LlmUsage(model="gemini-2.5-flash", prompt_tokens=4000, cached_tokens=1000, output_tokens=200)

    cost = call_cost("gemini-2.5-flash-lite", usage=usage, prompt_text="x" * 400, completion_text="y" * 40)

    assert cost.model == "gemini-2.5-flash"
    assert (cost.input_tokens, cost.output_tokens, cost.cached_input_tokens) == (4000, 200, 1000)
    assert _close(cost.cost_usd, cost_usd_from_usage(usage))


def test_call_cost_falls_back_to_the_text_estimate() -> None:
    cost = call_cost("gemini-2.5-flash-lite", usage=None, prompt_text="x" * 400, completion_text="y" * 40)

    assert cost.model == "gemini-2.5-flash-lite"
    assert (cost.input_tokens, cost.output_tokens, cost.cached_input_tokens) == (100, 10, None)
    assert _close(
        cost.cost_usd, estimate_cost_usd("gemini-2.5-flash-lite", prompt_text="x" * 400, completion_text="y" * 40)
    )


# ── the client hands its usage to the caller ─────────────────


class _Response:
    def __init__(self, text: str, usage: _Usage | None) -> None:
        self.text = text
        self.usage_metadata = usage


class _SyncModels:
    def __init__(self, fail: set[str], usage: _Usage | None) -> None:
        self._fail = fail
        self._usage = usage

    def generate_content(self, *, model: str, contents: object, config: object = None) -> _Response:
        _ = contents, config
        if model in self._fail:
            raise RuntimeError(f"{model} unavailable")
        return _Response('{"ok": true}', self._usage)


class _SyncClient:
    def __init__(self, models: _SyncModels) -> None:
        self.models = models


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, *, fail: set[str], usage: _Usage | None) -> None:
    client = _SyncClient(_SyncModels(fail, usage))

    def _factory(*, api_key: str) -> _SyncClient:
        _ = api_key
        return client

    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr("ingovo_search.lib.llm._load_client_factory", lambda: _factory)


async def test_a_completed_call_leaves_its_usage_with_the_model_that_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_sdk(monkeypatch, fail={"gemini-2.5-flash-lite"}, usage=_Usage(prompt=900, cached=100, candidates=40))

    await GoogleAiClient(api_key="k").complete(LlmRequest(model="gemini-2.5-flash-lite", system="s", user="u"))

    usage = last_llm_usage()
    assert usage is not None
    assert usage.model == "gemini-2.5-flash", "after a fallback the row must be priced as the model that answered"
    assert (usage.prompt_tokens, usage.cached_tokens, usage.output_tokens) == (900, 100, 40)
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


async def test_a_failed_call_leaves_no_usage_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    set_last_llm_usage(LlmUsage(model="stale", prompt_tokens=1))
    _fake_sdk(
        monkeypatch,
        fail={"gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3.1-flash-lite"},
        usage=_Usage(prompt=1),
    )

    with pytest.raises(RuntimeError):
        await GoogleAiClient(api_key="k").complete(LlmRequest(model="gemini-2.5-flash-lite", system="s", user="u"))

    assert last_llm_usage() is None
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


class _UsageReportingClient:
    """An `LlmClient` that reports usage the way `GoogleAiClient` does."""

    def __init__(self, raw: str, usage: LlmUsage) -> None:
        self._raw = raw
        self._usage = usage

    async def complete(self, req: LlmRequest) -> str:
        _ = req
        set_last_llm_usage(self._usage)
        return self._raw


class _SessionStub:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, instance: object) -> None:
        self.added.append(instance)


async def test_the_parser_row_carries_the_reported_tokens() -> None:
    usage = LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=3100, cached_tokens=2048, output_tokens=120)
    client: LlmClient = _UsageReportingClient(ParsedQuery(semantic_query="csendes").model_dump_json(), usage)
    session = _SessionStub()

    await parse_query(
        "csendes, napos otthon a hegyekben, ahol reggel madárcsicsergés van",
        client=client,
        session=session,  # pyright: ignore[reportArgumentType]
    )

    rows = [row for row in session.added if isinstance(row, EnrichmentLog)]
    assert len(rows) == 1
    assert (rows[0].input_tokens, rows[0].output_tokens, rows[0].cached_input_tokens) == (3100, 120, 2048)
    assert _close(rows[0].cost_usd, cost_usd_from_usage(usage))


async def test_the_image_analysis_row_is_written_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[EnrichmentLog] = []
    monkeypatch.setattr("ingovo_search.recommender.image_query.record_llm_calls_later", captured.extend)
    usage = LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=1800, output_tokens=60)

    await analyze_images([LlmImage(mime_type="image/jpeg", data=b"jpeg")], client=_UsageReportingClient("{}", usage))

    assert len(captured) == 1
    assert (captured[0].prompt_version, captured[0].input_tokens, captured[0].outcome) == (
        "image_query/v1.0",
        1800,
        "success",
    )


# ── the writer ─────────────────────────────────────────────────────────────


def _row() -> EnrichmentLog:
    return llm_call_row(
        usage=LlmUsage(model="gemini-2.5-flash-lite", prompt_tokens=100, output_tokens=10),
        requested_model="gemini-2.5-flash-lite",
        prompt_version="image_query/v3",
        latency_ms=-5,
        outcome="success",
        property_id="p1",
    )


def test_a_row_carries_the_reported_numbers() -> None:
    row = _row()

    assert (row.model_used, row.input_tokens, row.output_tokens, row.latency_ms) == (
        "gemini-2.5-flash-lite",
        100,
        10,
        0,
    )
    assert (row.prompt_version, row.outcome, row.property_id) == ("image_query/v3", "success", "p1")


async def test_the_writer_is_off_in_tests_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[list[EnrichmentLog]] = []

    async def _fake_write(rows: list[EnrichmentLog]) -> None:
        written.append(rows)

    monkeypatch.setattr(llm_cost_log, "_write", _fake_write)

    await record_llm_calls([_row()])

    assert written == [], "the conftest keeps AI_USAGE_LOG off so no test reaches the real database"


async def test_the_writer_persists_when_switched_on(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[list[EnrichmentLog]] = []

    async def _fake_write(rows: list[EnrichmentLog]) -> None:
        written.append(rows)

    monkeypatch.setenv("AI_USAGE_LOG", "on")
    monkeypatch.setattr(llm_cost_log, "_write", _fake_write)

    await record_llm_calls([_row(), _row()])
    record_llm_calls_later([_row()])
    await drain_usage_writes()

    assert [len(batch) for batch in written] == [2, 1]


async def test_a_failing_or_slow_write_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _broken(rows: list[EnrichmentLog]) -> None:
        _ = rows
        raise RuntimeError("pool wedged")

    async def _slow(rows: list[EnrichmentLog]) -> None:
        _ = rows
        await asyncio.sleep(1)

    monkeypatch.setenv("AI_USAGE_LOG", "on")
    monkeypatch.setattr(llm_cost_log, "_write", _broken)
    await record_llm_calls([_row()])

    monkeypatch.setattr(llm_cost_log, "RECORD_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(llm_cost_log, "_write", _slow)
    await record_llm_calls([_row()])
