"""LlmClient Protocol, GoogleAiClient, and MockLlmClient."""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import ClassVar, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm_usage import LlmUsage, reset_last_llm_usage, set_last_llm_usage, usage_from_metadata
from ingovo_search.types.errors import LlmError


class LlmImage(BaseModel):
    """One inline image part of a multimodal request.

    Bytes are sent inline (never a URL) so nothing has to be published to
    reach the model. Callers are responsible for size + MIME validation at
    the trust boundary — see `api/search_image.py`.
    """

    model_config = {"frozen": True}

    mime_type: str
    data: bytes


def _empty_image_list() -> list[LlmImage]:
    return []


def _empty_str_list() -> list[str]:
    return []


# Tried in order after a request's own model fails. `gemini-2.0-flash` used to
# sit at the end and was removed on 2026-07-26: the endpoint now answers 404
# ("no longer available"), so every exhausted retry burned a round trip on a
# model that could never succeed. Verified live against the API on that date —
# all three below accept text and images.
DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
)


class LlmRequest(BaseModel):
    """A single LLM completion request."""

    model: str
    system: str
    user: str
    response_schema: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic schema class when structured output is required.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. Defaults to 0.0 for deterministic structured "
            "extraction (parser/rerank); raise per-call for creative generation."
        ),
    )
    images: list[LlmImage] = Field(
        default_factory=_empty_image_list,
        description="Inline image parts. Empty for the text-only tiers (parser / rerank / narrator).",
    )
    fallback_models: list[str] = Field(
        default_factory=_empty_str_list,
        description=(
            "Models to try, in order, after `model` fails. Empty uses "
            "`DEFAULT_FALLBACK_CHAIN`. Set it when a call needs a specific "
            "capability — image understanding, for instance — so the retry "
            "never lands on a model that cannot serve the request at all."
        ),
    )


@runtime_checkable
class LlmClient(Protocol):
    """All LLM calls go through this Protocol. Implementations: GoogleAiClient (prod), MockLlmClient (tests)."""

    async def complete(self, req: LlmRequest) -> str: ...


class CountingLlmClient:
    """Wraps an `LlmClient` to tally calls per phase + per model for one request.

    Observability: every Gemini call in a search goes through `complete`,
    so wrapping the per-request client here yields an exact, auditable count of
    parser / rerank / narrative / fallback calls. The orchestrator sets `phase`
    around each pipeline step via `set_llm_phase`; calls outside a known step
    fall under "other".
    """

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner
        self.phase = "other"
        self.calls_by_phase: dict[str, int] = {}
        self.calls_by_model: dict[str, int] = {}

    async def complete(self, req: LlmRequest) -> str:
        self.calls_by_phase[self.phase] = self.calls_by_phase.get(self.phase, 0) + 1
        self.calls_by_model[req.model] = self.calls_by_model.get(req.model, 0) + 1
        return await self._inner.complete(req)

    @property
    def total_calls(self) -> int:
        return sum(self.calls_by_phase.values())


def set_llm_phase(client: LlmClient, phase: str) -> None:
    """Set the accounting phase if `client` is a `CountingLlmClient`; else no-op."""
    if isinstance(client, CountingLlmClient):
        client.phase = phase


class _GenerateResponse(Protocol):
    text: str | None
    usage_metadata: object | None


class _ModelsClient(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str | list[object],
        config: dict[str, object] | None = None,
    ) -> _GenerateResponse: ...


class _GenAiClient(Protocol):
    models: _ModelsClient


class _ClientFactory(Protocol):
    def __call__(self, *, api_key: str) -> _GenAiClient: ...


class _GenAiModule(Protocol):
    Client: _ClientFactory


def _load_client_factory() -> _ClientFactory:
    try:
        genai_module = cast(_GenAiModule, import_module("google.genai"))
    except ImportError as exc:
        raise LlmError("google-genai is not installed") from exc

    return genai_module.Client


def _request_contents(req: LlmRequest) -> str | list[object]:
    """Text-only requests keep the plain-string form; images become inline parts.

    The image parts go first so the model reads the pictures before the
    instruction that refers to them, which is what the Gemini docs recommend
    for single-turn image understanding.
    """
    if not req.images:
        return req.user
    parts: list[object] = [{"inline_data": {"mime_type": image.mime_type, "data": image.data}} for image in req.images]
    parts.append({"text": req.user})
    return parts


def _generate_content(
    client: _GenAiClient, req: LlmRequest, config: dict[str, object] | None
) -> tuple[str, LlmUsage | None]:
    """The completion text and what the provider billed for it.

    Runs in a worker thread (`asyncio.to_thread`), which gets a *copy* of the
    context, so the usage is returned here and recorded by the coroutine.
    """
    response = client.models.generate_content(
        model=req.model,
        contents=_request_contents(req),
        config=config,
    )
    if not isinstance(response.text, str) or response.text == "":
        raise LlmError("GoogleAiClient: empty completion response")
    return response.text, usage_from_metadata(getattr(response, "usage_metadata", None), model=req.model)


class GoogleAiClient:
    """Production Google AI Studio client.

    Migration target: this same class with Vertex AI mode once production
    residency/security controls require it.
    """

    _clients_by_api_key: ClassVar[dict[str, _GenAiClient]] = {}

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @classmethod
    def _client_for_api_key(cls, api_key: str) -> _GenAiClient:
        client = cls._clients_by_api_key.get(api_key)
        if client is None:
            client = _load_client_factory()(api_key=api_key)
            cls._clients_by_api_key[api_key] = client
        return client

    async def complete(self, req: LlmRequest) -> str:
        api_key = self._api_key or get_settings().gemini_api_key
        if not api_key:
            raise LlmError("GEMINI_API_KEY is not set")
        config: dict[str, object] = {
            "system_instruction": req.system,
            "temperature": req.temperature,
        }
        if req.response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = req.response_schema

        client = self._client_for_api_key(api_key)

        # The request's own chain wins when it declares one — a multimodal call
        # must not retry on a model that cannot read images. Otherwise the
        # shared default applies. Either way the primary is tried first and
        # never repeated further down.
        fallback_chain = req.fallback_models or list(DEFAULT_FALLBACK_CHAIN)
        models_to_try = [req.model] + [model for model in fallback_chain if model != req.model]

        import structlog

        logger = structlog.get_logger()

        # A failed call must not leave the previous call's usage behind for the
        # caller's cost row to pick up.
        reset_last_llm_usage()
        last_exc = None
        for attempt_model in models_to_try:
            try:
                current_req = req if attempt_model == req.model else req.model_copy(update={"model": attempt_model})
                text, usage = await asyncio.to_thread(_generate_content, client, current_req, config)
                set_last_llm_usage(usage)
                return text
            except Exception as exc:  # noqa: BLE001
                # Not a swallow: every failure is logged here with its traceback and
                # the last one is raised below as an `LlmError`. The breadth only decides which
                # failures earn a try on the next model, and the SDK's surface —
                # APIError (quota, a retired model id), httpx transport errors, the
                # ValueErrors of response parsing, our own LlmError on an empty
                # completion — is wider than any list that would stay true across
                # SDK upgrades.
                last_exc = exc
                await logger.awarning(
                    "llm_model_failed",
                    model=attempt_model,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    has_images=bool(req.images),
                    remaining=models_to_try[models_to_try.index(attempt_model) + 1 :],
                    exc_info=exc,
                )

        # Every model in the chain failed. Raise the project's own error, chained to the
        # provider's, so the callers that degrade on `LlmError` (the parser's rule-based
        # fallback, the rerank's deterministic order) treat an outage like any model failure.
        raise LlmError(f"every model failed for {req.model}: {type(last_exc).__name__}") from last_exc


def fixture_key(req: LlmRequest) -> str:
    """Deterministic key used by MockLlmClient — model id plus the first 64 chars of the user prompt.

    Multimodal requests append the attached image count so a text-only and an
    image-carrying call on the same prompt cannot collide in a fixture map.
    Text-only keys are unchanged.
    """
    base = f"{req.model}::{req.user[:64]}"
    return f"{base}::img{len(req.images)}" if req.images else base


class MockLlmClient:
    """Fixture-keyed mock for unit tests. Add expected (request → response) pairs via `add()`."""

    def __init__(self, fixtures: dict[str, str] | None = None) -> None:
        self._fixtures: dict[str, str] = dict(fixtures or {})

    def add(self, key: str, response: str) -> None:
        """Register a response for a given fixture key (use `fixture_key(req)` or the same convention)."""
        self._fixtures[key] = response

    async def complete(self, req: LlmRequest) -> str:
        key = fixture_key(req)
        if key not in self._fixtures:
            raise LlmError(f"MockLlmClient: no fixture for key '{key}'")
        return self._fixtures[key]
