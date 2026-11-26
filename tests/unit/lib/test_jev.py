"""TypeSafeJevClient over a mock HTTP transport, FakeJevClient, and Jev pricing."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx2
import pytest
from typesafe_sdk import Noul, Score

from ingovo_search.lib.cost_tracker import cost_usd_from_tokens
from ingovo_search.lib.jev import FakeJevClient, JevResponse, JevScore, TypeSafeJevClient
from ingovo_search.types.errors import JevError, LlmError

_API_KEY = "apikey_unit_test_not_a_real_secret_0123456789"
_MODEL = "jev-1.13.0"

_OK_BODY: dict[str, object] = {
    "model": _MODEL,
    "answers": {
        "c0": {
            "type": "score",
            "score": 2.5,
            "confidence": 0.8,
            "legend": {"0": "nem", "1": "részben", "2": "jól", "3": "teljesen"},
            "probabilities": {"0": 0.0, "1": 0.1, "2": 0.3, "3": 0.6},
        },
        "is_quiet": {"type": "noul", "noul": 0.7},
    },
    "usage": {"input_tokens": 4255, "output_tokens": 174},
}

_QUESTIONS = {
    "c0": Score(instructions="Mennyire felel meg a `jelöltek.c0`?", criteria=["nem", "részben", "jól", "teljesen"]),
    "is_quiet": Noul(instructions="Csendes-e a `jelöltek.c0`?"),
}
_STATE = {"keresés": {"szöveg": "csendes lakás Budán"}, "jelöltek": {"c0": "Budai 2 szobás lakás"}}


class _Recorder:
    """Mock transport handler that replays a scripted list of responses."""

    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(handler: Callable[[httpx2.Request], httpx2.Response]) -> TypeSafeJevClient:
    return TypeSafeJevClient(api_key=_API_KEY, transport=httpx2.MockTransport(handler))


async def _ask(client: TypeSafeJevClient) -> JevResponse:
    return await client.system_one(_STATE, _QUESTIONS, model=_MODEL, timeout_s=1.2)


async def test_success_parses_scores_nouls_usage_and_model() -> None:
    recorder = _Recorder(httpx2.Response(200, json=_OK_BODY))

    response = await _ask(_client(recorder))

    assert response == JevResponse(
        model=_MODEL,
        scores={"c0": JevScore(score=2.5, confidence=0.8)},
        nouls={"is_quiet": 0.7},
        input_tokens=4255,
        output_tokens=174,
    )
    [request] = recorder.requests
    assert request.url.path == "/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    body = json.loads(request.content)
    assert body["model"] == _MODEL
    assert body["state"] == _STATE
    assert set(body["questions"]) == {"c0", "is_quiet"}


async def test_unauthorized_is_not_retried() -> None:
    recorder = _Recorder(httpx2.Response(401, json={"error": "invalid api key"}))

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.reason == "http"
    assert caught.value.status == 401
    assert len(recorder.requests) == 1


async def test_unprocessable_request_is_not_retried() -> None:
    recorder = _Recorder(httpx2.Response(422, json={"detail": "questions: field required"}))

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.status == 422
    assert len(recorder.requests) == 1


async def test_rate_limit_is_retried_once_then_succeeds() -> None:
    recorder = _Recorder(httpx2.Response(429, json={"error": "slow down"}), httpx2.Response(200, json=_OK_BODY))

    response = await _ask(_client(recorder))

    assert response.scores["c0"].score == 2.5
    assert len(recorder.requests) == 2


@pytest.mark.parametrize("status", [429, 529])
async def test_repeated_rate_limit_or_overload_gives_up_after_one_retry(status: int) -> None:
    recorder = _Recorder(
        httpx2.Response(status, json={"error": "busy"}), httpx2.Response(status, json={"error": "busy"})
    )

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.status == status
    assert len(recorder.requests) == 2


async def test_timeout_is_not_retried() -> None:
    recorder = _Recorder(httpx2.ReadTimeout("read timed out"))

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.reason == "timeout"
    assert caught.value.status is None
    assert len(recorder.requests) == 1


async def test_connection_failure_maps_to_connection_reason() -> None:
    recorder = _Recorder(httpx2.ConnectError("refused"), httpx2.ConnectError("refused"))

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.reason == "connection"


async def test_malformed_success_body_maps_to_invalid_response() -> None:
    recorder = _Recorder(httpx2.Response(200, json={"answers": {}}))

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert caught.value.reason == "invalid_response"


async def test_jev_error_is_an_llm_error() -> None:
    """Existing LLM error handling (envelope, fallbacks) must catch Jev failures too."""
    recorder = _Recorder(httpx2.Response(401, json={"error": "invalid api key"}))

    with pytest.raises(LlmError):
        await _ask(_client(recorder))


async def test_api_key_never_reaches_logs_or_errors(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _Recorder(httpx2.Response(401, json={"error": "invalid api key"}))
    caplog.set_level(logging.DEBUG)

    with pytest.raises(JevError) as caught:
        await _ask(_client(recorder))

    assert _API_KEY not in str(caught.value)
    assert _API_KEY not in repr(caught.value)
    assert all(_API_KEY not in record.getMessage() for record in caplog.records)


async def test_fake_answers_preset_questions_and_records_the_call() -> None:
    fake = FakeJevClient(scores={"c0": 2.0}, nouls={"is_quiet": 0.4}, input_tokens=321)

    response = await fake.system_one(_STATE, _QUESTIONS, model=_MODEL, timeout_s=1.2)

    assert response.scores == {"c0": JevScore(score=2.0, confidence=0.9)}
    assert response.nouls == {"is_quiet": 0.4}
    assert response.input_tokens == 321
    [call] = fake.calls
    assert call.state == _STATE
    assert call.questions is _QUESTIONS
    assert call.timeout_s == 1.2


async def test_fake_leaves_out_questions_without_a_preset_answer() -> None:
    fake = FakeJevClient(scores={})

    response = await fake.system_one(_STATE, _QUESTIONS, model=_MODEL, timeout_s=1.2)

    assert response.scores == {}
    assert response.nouls == {}


async def test_fake_raises_the_configured_error_after_recording() -> None:
    fake = FakeJevClient(error=JevError("boom", reason="timeout"))

    with pytest.raises(JevError):
        await fake.system_one(_STATE, _QUESTIONS, model=_MODEL, timeout_s=1.2)
    assert len(fake.calls) == 1


def test_jev_cost_bills_input_tokens_only() -> None:
    # The 2026-09-28 baseline: 12 real candidates, 4255 input + 174 output tokens.
    assert cost_usd_from_tokens(_MODEL, input_tokens=4255, output_tokens=174) == pytest.approx(0.00017871)  # pyright: ignore[reportUnknownMemberType]
    assert cost_usd_from_tokens(_MODEL, input_tokens=0, output_tokens=10_000) == 0.0


def test_unknown_model_cost_falls_back_to_cheapest_gemini_tier() -> None:
    assert cost_usd_from_tokens("jev-9.9.9", input_tokens=1_000_000, output_tokens=0) == pytest.approx(0.10)  # pyright: ignore[reportUnknownMemberType]
