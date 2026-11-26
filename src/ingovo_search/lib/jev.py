"""JevClient Protocol, TypeSafeJevClient, and FakeJevClient.

Jev (TypeSafe's System One model) answers typed questions about a JSON state
with probabilities instead of text. The rerank asks one Score question per
candidate in a single request. Every
Jev call goes through `JevClient`; `TypeSafeJevClient` wraps the official
`typesafe-sdk` async client, `FakeJevClient` serves unit tests.

Every SDK failure surfaces as `JevError` so callers handle one exception type
and fall back to the deterministic order, exactly as they do for Gemini.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import httpx2
from pydantic import BaseModel
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)

from ingovo_search.types.errors import JevError

# The SDK types `state` with a recursive `TypeAliasType` that pyright strict
# reads as partially unknown. The boundary therefore uses this plain recursive
# alias, and the SDK client is reached through `_SystemOneClient` — the same
# approach `lib/llm.py` takes with the Gemini SDK.
type JevJson = str | int | float | bool | Sequence[JevJson] | Mapping[str, JevJson] | None
type JevState = Mapping[str, JevJson]

# Statuses worth one retry: rate limit (429), TypeSafe overload (529) and
# transient gateway errors. Everything else (401, 422, …) is a bug or a config
# problem that a retry cannot fix.
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


class JevScore(BaseModel):
    """One Score answer: the probability-weighted level and how concentrated it is."""

    model_config = {"frozen": True}

    score: float
    confidence: float


class JevResponse(BaseModel):
    """What the rerank needs from one Jev call, decoupled from the SDK types."""

    model_config = {"frozen": True}

    model: str
    scores: dict[str, JevScore]
    nouls: dict[str, float]
    input_tokens: int
    output_tokens: int


@runtime_checkable
class JevClient(Protocol):
    """All Jev calls go through this Protocol. Implementations: TypeSafeJevClient (prod), FakeJevClient (tests)."""

    async def system_one(
        self,
        state: JevState,
        questions: Mapping[str, Question],
        *,
        model: str,
        timeout_s: float,
    ) -> JevResponse: ...


class _SystemOneClient(Protocol):
    """The slice of `AsyncTypeSafeClient` this module uses."""

    async def system_one(
        self,
        state: JevState,
        questions: Mapping[str, Question],
        *,
        model: str,
        timeout: float,  # noqa: ASYNC109 - mirrors the SDK's own keyword; the SDK enforces it.
        retry: RetryPolicy,
    ) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


def _retry_policy(timeout_s: float) -> RetryPolicy:
    # One retry at most, and only if it still fits the caller's budget: the
    # search pipeline has its own ceiling and falls back to the deterministic
    # order, so a slow retry is worth less than a fast fallback. A timed-out
    # attempt has already spent the budget, so timeouts are never retried.
    return RetryPolicy(
        max_retries=1,
        backoff_initial=0.1,
        backoff_max=0.2,
        http_statuses=set(_RETRY_STATUSES),
        api_timeout_error=False,
        timeout=timeout_s,
    )


def _to_jev_response(response: SystemOneResponse) -> JevResponse:
    return JevResponse(
        model=response.model,
        scores={
            key: JevScore(score=answer.score, confidence=answer.confidence) for key, answer in response.scores.items()
        },
        nouls={key: answer.noul for key, answer in response.nouls.items()},
        input_tokens=response.usage.input_tokens or 0,
        output_tokens=response.usage.output_tokens or 0,
    )


class TypeSafeJevClient:
    """Production client over `typesafe_sdk.AsyncTypeSafeClient`.

    The SDK client (and its connection pool) is created on first use, inside the
    running event loop, and reused for the life of the process. A malformed key
    therefore surfaces as a `JevError` on the first call instead of failing
    dependency injection.
    """

    def __init__(self, api_key: str, *, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        self._api_key = api_key
        self._transport = transport
        self._client: _SystemOneClient | None = None

    def _sdk_client(self) -> _SystemOneClient:
        if self._client is None:
            self._client = cast(
                "_SystemOneClient", AsyncTypeSafeClient(api_key=self._api_key, transport=self._transport)
            )
        return self._client

    async def system_one(
        self,
        state: JevState,
        questions: Mapping[str, Question],
        *,
        model: str,
        timeout_s: float,
    ) -> JevResponse:
        try:
            response = await self._sdk_client().system_one(
                state,
                questions,
                model=model,
                timeout=timeout_s,
                retry=_retry_policy(timeout_s),
            )
        except TypeSafeAPITimeoutError as exc:
            raise JevError(f"jev request timed out after {timeout_s:.2f}s", reason="timeout") from exc
        except TypeSafeAPIConnectionError as exc:
            raise JevError(f"jev connection failed: {exc}", reason="connection") from exc
        except TypeSafeAPIResponseValidationError as exc:
            raise JevError(f"jev returned an invalid response: {exc}", reason="invalid_response") from exc
        except TypeSafeAPIError as exc:
            raise JevError(f"jev request failed: {exc}", reason="http", status=exc.status) from exc
        except TypeSafeError as exc:
            raise JevError(f"jev client error: {exc}", reason="client") from exc
        return _to_jev_response(response)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass(frozen=True)
class FakeJevCall:
    """One recorded `FakeJevClient.system_one` call, kept exactly as passed."""

    state: JevState
    questions: Mapping[str, Question]
    model: str
    timeout_s: float


def _question_type(question: Question) -> str:
    return question.type if isinstance(question, BaseModel) else question["type"]


class FakeJevClient:
    """Preset-answer stand-in for unit tests; records every call.

    A question whose id has no preset answer is left out of the response, which
    is how tests exercise the "Jev skipped a candidate" path.
    """

    def __init__(
        self,
        *,
        scores: Mapping[str, float] | None = None,
        nouls: Mapping[str, float] | None = None,
        model: str = "jev-1.13.0",
        input_tokens: int = 1000,
        delay_s: float = 0.0,
        error: JevError | None = None,
    ) -> None:
        self._scores = dict(scores or {})
        self._nouls = dict(nouls or {})
        self._model = model
        self._input_tokens = input_tokens
        self._delay_s = delay_s
        self._error = error
        self.calls: list[FakeJevCall] = []

    async def system_one(
        self,
        state: JevState,
        questions: Mapping[str, Question],
        *,
        model: str,
        timeout_s: float,
    ) -> JevResponse:
        self.calls.append(FakeJevCall(state=state, questions=questions, model=model, timeout_s=timeout_s))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        return JevResponse(
            model=self._model,
            scores={
                key: JevScore(score=self._scores[key], confidence=0.9)
                for key, question in questions.items()
                if _question_type(question) == "score" and key in self._scores
            },
            nouls={
                key: self._nouls[key]
                for key, question in questions.items()
                if _question_type(question) == "noul" and key in self._nouls
            },
            input_tokens=self._input_tokens,
            output_tokens=0,
        )
