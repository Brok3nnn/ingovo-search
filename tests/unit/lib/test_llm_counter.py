"""Unit tests for the §16 per-request LLM call counter."""

from __future__ import annotations

from typing import cast

from ingovo_search.lib.llm import CountingLlmClient, LlmClient, LlmRequest, set_llm_phase


class _StubClient:
    """Minimal LlmClient that echoes a fixed string and records nothing."""

    async def complete(self, _req: LlmRequest) -> str:
        return "ok"


def _req(model: str) -> LlmRequest:
    return LlmRequest(model=model, system="sys", user="usr")


async def test_counts_calls_by_phase_and_model() -> None:
    counter = CountingLlmClient(cast(LlmClient, _StubClient()))

    set_llm_phase(counter, "parser")
    await counter.complete(_req("gemini-2.5-flash-lite"))

    set_llm_phase(counter, "rerank")
    await counter.complete(_req("gemini-2.5-flash-lite"))
    await counter.complete(_req("gemini-2.5-flash"))  # a fallback-tier call

    assert counter.calls_by_phase == {"parser": 1, "rerank": 2}
    assert counter.calls_by_model == {"gemini-2.5-flash-lite": 2, "gemini-2.5-flash": 1}
    assert counter.total_calls == 3


async def test_calls_default_to_other_phase() -> None:
    counter = CountingLlmClient(cast(LlmClient, _StubClient()))
    await counter.complete(_req("gemini-2.5-flash-lite"))
    assert counter.calls_by_phase == {"other": 1}


def test_set_llm_phase_is_noop_for_plain_client() -> None:
    # Must not raise when the client is not a CountingLlmClient.
    set_llm_phase(cast(LlmClient, _StubClient()), "parser")
