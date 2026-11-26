"""Gemini calls that keep the provider-reported token counts.

`GoogleAiClient` returns only the completion text, so the `EnrichmentLog` rows
it feeds carry a character-based token *estimate* (`estimate_token_count`).
This client makes the same request and records the response's
`usage_metadata`, so the offline evaluation can report what Gemini actually
bills next to what the log would have estimated.

Deliberately without the production fallback chain: a measurement wants the
named model or a clean failure, never a silent switch to another model.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from ingovo_search.lib.cost_tracker import cost_usd_from_tokens, estimate_token_count
from ingovo_search.lib.llm import LlmRequest
from ingovo_search.types.errors import LlmError


@dataclass(frozen=True)
class GeminiUsage:
    """One call's billed tokens next to the estimate the production log would carry."""

    model: str
    prompt_tokens: int
    candidates_tokens: int
    thoughts_tokens: int
    latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int

    @property
    def output_tokens(self) -> int:
        """Thinking tokens are billed as output, so they count here."""
        return self.candidates_tokens + self.thoughts_tokens

    @property
    def cost_usd(self) -> float:
        return cost_usd_from_tokens(self.model, input_tokens=self.prompt_tokens, output_tokens=self.output_tokens)

    @property
    def estimated_cost_usd(self) -> float:
        return cost_usd_from_tokens(
            self.model, input_tokens=self.estimated_input_tokens, output_tokens=self.estimated_output_tokens
        )


# Read-only members: a Protocol attribute is invariant, a property is covariant,
# and the test fakes carry their own concrete types.
class _UsageMetadata(Protocol):
    @property
    def prompt_token_count(self) -> int | None: ...
    @property
    def candidates_token_count(self) -> int | None: ...
    @property
    def thoughts_token_count(self) -> int | None: ...


class _GenerateResponse(Protocol):
    @property
    def text(self) -> str | None: ...
    @property
    def usage_metadata(self) -> _UsageMetadata | None: ...


class _Models(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: dict[str, object] | None = None,
    ) -> _GenerateResponse: ...


class GenAiClient(Protocol):
    """The slice of `google.genai.Client` this module uses."""

    @property
    def models(self) -> _Models: ...


class _ClientFactory(Protocol):
    def __call__(self, *, api_key: str) -> GenAiClient: ...


class _GenAiModule(Protocol):
    Client: _ClientFactory


def _sdk_client(api_key: str) -> GenAiClient:
    try:
        genai_module = cast(_GenAiModule, import_module("google.genai"))
    except ImportError as exc:
        raise LlmError("google-genai is not installed") from exc
    return genai_module.Client(api_key=api_key)


class UsageRecordingGeminiClient:
    """`LlmClient` over the Gemini SDK that records every call's `usage_metadata`.

    The request is shaped exactly as `GoogleAiClient.complete` shapes it (system
    instruction, temperature, JSON schema), so a measured call is the production
    call. Text only: the rerank sends no images.
    """

    def __init__(self, sdk_client: GenAiClient, *, extra_config: Mapping[str, object] | None = None) -> None:
        self._sdk = sdk_client
        # Merged into every request's config. Empty for a production-faithful
        # measurement; the judge uses it to switch Gemini's thinking off.
        self.extra_config: dict[str, object] = dict(extra_config or {})
        self.records: list[GeminiUsage] = []

    @classmethod
    def from_api_key(
        cls, api_key: str, *, extra_config: Mapping[str, object] | None = None
    ) -> UsageRecordingGeminiClient:
        return cls(_sdk_client(api_key), extra_config=extra_config)

    async def complete(self, req: LlmRequest) -> str:
        if req.images:
            raise LlmError("UsageRecordingGeminiClient is text-only")
        config: dict[str, object] = {"system_instruction": req.system, "temperature": req.temperature}
        if req.response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = req.response_schema
        config.update(self.extra_config)

        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._sdk.models.generate_content, model=req.model, contents=req.user, config=config
            )
        except Exception as exc:
            raise LlmError(f"gemini call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if not isinstance(response.text, str) or response.text == "":
            raise LlmError("UsageRecordingGeminiClient: empty completion response")

        usage = response.usage_metadata
        self.records.append(
            GeminiUsage(
                model=req.model,
                prompt_tokens=(usage.prompt_token_count or 0) if usage is not None else 0,
                candidates_tokens=(usage.candidates_token_count or 0) if usage is not None else 0,
                thoughts_tokens=(usage.thoughts_token_count or 0) if usage is not None else 0,
                latency_ms=latency_ms,
                estimated_input_tokens=estimate_token_count(req.system) + estimate_token_count(req.user),
                estimated_output_tokens=estimate_token_count(response.text),
            )
        )
        return response.text
