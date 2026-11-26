"""MockLlmClient behavior, multimodal request shaping, and model rotation."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest

from ingovo_search.lib.llm import (
    DEFAULT_FALLBACK_CHAIN,
    GoogleAiClient,
    LlmClient,
    LlmImage,
    LlmRequest,
    MockLlmClient,
    _request_contents,
    fixture_key,
)
from ingovo_search.types.errors import LlmError


async def test_mock_returns_registered_fixture() -> None:
    client = MockLlmClient()
    req = LlmRequest(model="gemini-2.5-flash-lite", system="sys", user="hello")
    client.add(fixture_key(req), "world")

    response = await client.complete(req)
    assert response == "world"


async def test_mock_raises_on_unknown_request() -> None:
    client = MockLlmClient()
    req = LlmRequest(model="gemini-2.5-flash-lite", system="sys", user="unknown")
    with pytest.raises(LlmError, match="no fixture"):
        await client.complete(req)


def test_mock_satisfies_llm_client_protocol() -> None:
    client: LlmClient = MockLlmClient()
    # Protocol conformance: type system would catch this at static-check time;
    # at runtime, MockLlmClient is decorated as runtime_checkable in llm.py.
    assert isinstance(client, LlmClient)


def test_fixture_key_is_deterministic() -> None:
    req_a = LlmRequest(model="m", system="s", user="user prompt text")
    req_b = LlmRequest(model="m", system="other-system", user="user prompt text")
    # Key derives from model + user prompt (first 64 chars), not system.
    assert fixture_key(req_a) == fixture_key(req_b)


async def test_google_client_reuses_sdk_client_per_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class _FakeResponse:
        text = '{"ok": true}'

    class _FakeModels:
        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object] | None = None
        ) -> _FakeResponse:
            _ = model
            _ = contents
            _ = config
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    def _fake_factory(*, api_key: str) -> _FakeClient:
        created.append(api_key)
        return _FakeClient()

    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr("ingovo_search.lib.llm._load_client_factory", lambda: _fake_factory)

    req = LlmRequest(model="gemini-2.5-flash-lite", system="sys", user="hello")
    assert await GoogleAiClient(api_key="same-key").complete(req) == '{"ok": true}'
    assert await GoogleAiClient(api_key="same-key").complete(req) == '{"ok": true}'

    assert created == ["same-key"]
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


# --- multimodal requests + model rotation ---------------------------------


def test_fixture_key_separates_text_only_from_image_requests() -> None:
    text_only = LlmRequest(model="m", system="s", user="same prompt")
    with_image = LlmRequest(
        model="m", system="s", user="same prompt", images=[LlmImage(mime_type="image/png", data=b"x")]
    )
    assert fixture_key(text_only) != fixture_key(with_image)


def test_images_become_inline_parts_ahead_of_the_instruction() -> None:
    req = LlmRequest(
        model="m",
        system="s",
        user="analyse these",
        images=[
            LlmImage(mime_type="image/png", data=b"first"),
            LlmImage(mime_type="image/jpeg", data=b"second"),
        ],
    )
    contents = _request_contents(req)

    assert isinstance(contents, list)
    assert contents[0] == {"inline_data": {"mime_type": "image/png", "data": b"first"}}
    assert contents[1] == {"inline_data": {"mime_type": "image/jpeg", "data": b"second"}}
    assert contents[2] == {"text": "analyse these"}


def test_a_text_only_request_keeps_the_plain_string_form() -> None:
    assert _request_contents(LlmRequest(model="m", system="s", user="hello")) == "hello"


def _rotating_client(monkeypatch: pytest.MonkeyPatch, *, succeed_on: str, attempted: list[str]) -> None:
    """Fake SDK client that fails every model except `succeed_on`."""

    class _FakeResponse:
        text = '{"ok": true}'

    class _FakeModels:
        def generate_content(
            self, *, model: str, contents: str | list[object], config: dict[str, object] | None = None
        ) -> _FakeResponse:
            _ = contents, config
            attempted.append(model)
            if model != succeed_on:
                raise RuntimeError(f"{model} unavailable")
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]

    def _factory(*, api_key: str) -> _FakeClient:
        _ = api_key
        return _FakeClient()

    monkeypatch.setattr("ingovo_search.lib.llm._load_client_factory", lambda: _factory)


async def test_a_failing_model_rotates_to_the_next_one(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []
    _rotating_client(monkeypatch, succeed_on="gemini-2.5-flash", attempted=attempted)

    req = LlmRequest(model="gemini-2.5-flash-lite", system="s", user="hello")
    assert await GoogleAiClient(api_key="k").complete(req) == '{"ok": true}'

    assert attempted == ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


async def test_an_image_request_rotates_only_across_its_own_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multimodal call must never retry on a model that cannot read images."""
    attempted: list[str] = []
    _rotating_client(monkeypatch, succeed_on="vision-c", attempted=attempted)

    req = LlmRequest(
        model="vision-a",
        system="s",
        user="hello",
        images=[LlmImage(mime_type="image/png", data=b"x")],
        fallback_models=["vision-b", "vision-c"],
    )
    assert await GoogleAiClient(api_key="k").complete(req) == '{"ok": true}'

    assert attempted == ["vision-a", "vision-b", "vision-c"]
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


async def test_the_primary_is_never_retried_inside_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []
    _rotating_client(monkeypatch, succeed_on="gemini-3.1-flash-lite", attempted=attempted)

    # The primary also appears in DEFAULT_FALLBACK_CHAIN; it must be tried once.
    req = LlmRequest(model="gemini-2.5-flash-lite", system="s", user="hello")
    await GoogleAiClient(api_key="k").complete(req)

    assert attempted == list(DEFAULT_FALLBACK_CHAIN)
    assert attempted.count("gemini-2.5-flash-lite") == 1
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]


async def test_the_last_failure_surfaces_when_every_model_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []
    _rotating_client(monkeypatch, succeed_on="nothing-succeeds", attempted=attempted)

    req = LlmRequest(model="gemini-2.5-flash-lite", system="s", user="hello")
    with pytest.raises(RuntimeError, match="unavailable"):
        await GoogleAiClient(api_key="k").complete(req)

    assert attempted == list(DEFAULT_FALLBACK_CHAIN)
    GoogleAiClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]
