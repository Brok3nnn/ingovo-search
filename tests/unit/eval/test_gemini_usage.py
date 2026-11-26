"""`UsageRecordingGeminiClient` — billed tokens next to the log's estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, override

import pytest
from pydantic import BaseModel

from ingovo_search.eval.gemini_usage import GeminiUsage, GenAiClient, UsageRecordingGeminiClient
from ingovo_search.lib.cost_tracker import estimate_token_count
from ingovo_search.lib.llm import LlmImage, LlmRequest
from ingovo_search.types.errors import LlmError


class _Schema(BaseModel):
    ok: bool


@dataclass
class _FakeUsage:
    prompt_token_count: int | None
    candidates_token_count: int | None
    thoughts_token_count: int | None = None


@dataclass
class _FakeResponse:
    text: str | None
    usage_metadata: _FakeUsage | None


def _no_calls() -> list[dict[str, Any]]:
    return []


@dataclass
class _FakeModels:
    text: str | None
    usage: _FakeUsage | None
    calls: list[dict[str, Any]] = field(default_factory=_no_calls)

    def generate_content(self, *, model: str, contents: str, config: dict[str, object] | None = None) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(text=self.text, usage_metadata=self.usage)


@dataclass
class _FakeSdk:
    models: _FakeModels


def _sdk(text: str | None, usage: _FakeUsage | None) -> GenAiClient:
    return _FakeSdk(_FakeModels(text, usage))


def _usage(prompt: int | None, candidates: int | None, thoughts: int | None = None) -> _FakeUsage:
    return _FakeUsage(prompt_token_count=prompt, candidates_token_count=candidates, thoughts_token_count=thoughts)


async def test_records_billed_tokens_and_the_estimate_side_by_side() -> None:
    client = UsageRecordingGeminiClient(_sdk('{"ok": true}', _usage(1200, 300, 40)))
    req = LlmRequest(model="gemini-2.5-flash-lite", system="rendszer" * 50, user="kérdés" * 80, response_schema=_Schema)

    text = await client.complete(req)

    assert text == '{"ok": true}'
    [record] = client.records
    assert (record.prompt_tokens, record.candidates_tokens, record.thoughts_tokens) == (1200, 300, 40)
    assert record.output_tokens == 340
    assert record.estimated_input_tokens == estimate_token_count(req.system) + estimate_token_count(req.user)
    assert record.estimated_output_tokens == estimate_token_count('{"ok": true}')
    assert math.isclose(record.cost_usd, (1200 * 0.10 + 340 * 0.40) / 1_000_000)
    assert record.latency_ms >= 0


async def test_shapes_the_request_like_the_production_client() -> None:
    sdk = _FakeSdk(_FakeModels("x", _usage(1, 1)))
    client = UsageRecordingGeminiClient(sdk)

    await client.complete(LlmRequest(model="m", system="s", user="u", response_schema=_Schema, temperature=0.3))

    [call] = sdk.models.calls
    assert call["model"] == "m"
    assert call["contents"] == "u"
    assert call["config"] == {
        "system_instruction": "s",
        "temperature": 0.3,
        "response_mime_type": "application/json",
        "response_schema": _Schema,
    }


async def test_missing_usage_metadata_records_zero_tokens() -> None:
    client = UsageRecordingGeminiClient(_sdk("ok", None))

    await client.complete(LlmRequest(model="m", system="s", user="u"))

    [record] = client.records
    assert (record.prompt_tokens, record.output_tokens) == (0, 0)
    assert record.cost_usd == 0.0


async def test_empty_completion_is_an_llm_error_and_not_recorded() -> None:
    client = UsageRecordingGeminiClient(_sdk("", _usage(10, 0)))

    with pytest.raises(LlmError, match="empty"):
        await client.complete(LlmRequest(model="m", system="s", user="u"))
    assert client.records == []


async def test_sdk_failures_surface_as_llm_error() -> None:
    class _Boom(_FakeModels):
        @override
        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object] | None = None
        ) -> _FakeResponse:
            raise RuntimeError("503 overloaded")

    client = UsageRecordingGeminiClient(_FakeSdk(_Boom("x", None)))

    with pytest.raises(LlmError, match="503"):
        await client.complete(LlmRequest(model="m", system="s", user="u"))


async def test_images_are_refused() -> None:
    client = UsageRecordingGeminiClient(_sdk("ok", _usage(1, 1)))
    req = LlmRequest(model="m", system="s", user="u", images=[LlmImage(mime_type="image/png", data=b"x")])

    with pytest.raises(LlmError, match="text-only"):
        await client.complete(req)


def test_estimated_cost_uses_the_same_price_table() -> None:
    usage = GeminiUsage(
        model="gemini-2.5-flash-lite",
        prompt_tokens=2000,
        candidates_tokens=100,
        thoughts_tokens=0,
        latency_ms=1.0,
        estimated_input_tokens=1000,
        estimated_output_tokens=50,
    )

    assert math.isclose(usage.estimated_cost_usd, (1000 * 0.10 + 50 * 0.40) / 1_000_000)
    assert usage.cost_usd > usage.estimated_cost_usd


async def test_extra_config_is_merged_into_every_request() -> None:
    sdk = _FakeSdk(_FakeModels("x", _usage(1, 1)))
    client = UsageRecordingGeminiClient(sdk, extra_config={"thinking_config": {"thinking_budget": 0}})

    await client.complete(LlmRequest(model="m", system="s", user="u"))

    [call] = sdk.models.calls
    assert call["config"] == {"system_instruction": "s", "temperature": 0.0, "thinking_config": {"thinking_budget": 0}}
