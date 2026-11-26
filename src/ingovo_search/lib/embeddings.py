"""EmbeddingClient Protocol, CohereEmbeddingClient (prod), MockEmbeddingClient (tests).

`input_type` distinguishes "search_document" (corpus side) from "search_query"
(runtime query side) — Cohere's multilingual model uses different prefixes
internally and the two are not interchangeable. The Protocol pins this so the
recommender path can't accidentally swap them.

The mock implementation is deterministic and L2-normalized so unit tests can
assert exact vector equality and downstream cosine math behaves as expected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib import import_module
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

from ingovo_search.lib.embedding_models import EMBED_MULTILINGUAL
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.retry import async_with_retry
from ingovo_search.types.errors import LlmError

EmbeddingInputType = Literal["search_document", "search_query"]
EMBEDDING_DIM: int = 1024
COHERE_BATCH_LIMIT: int = 96


@runtime_checkable
class EmbeddingClient(Protocol):
    """All embedding calls go through this Protocol.

    Implementations: `CohereEmbeddingClient` (production) and
    `MockEmbeddingClient` (tests, deterministic + L2-normalized).
    """

    async def embed(self, text: str, input_type: EmbeddingInputType) -> list[float]: ...

    async def embed_batch(
        self,
        texts: Sequence[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]: ...


class _CohereEmbedFloats(Protocol):
    float: list[list[float]]


class _CohereEmbedResponse(Protocol):
    embeddings: _CohereEmbedFloats


class _AsyncCohereClient(Protocol):
    async def embed(
        self,
        *,
        texts: Sequence[str],
        model: str,
        input_type: str,
        embedding_types: Sequence[str],
    ) -> _CohereEmbedResponse: ...


class _AsyncCohereFactory(Protocol):
    def __call__(self, *, api_key: str) -> _AsyncCohereClient: ...


class _CohereModule(Protocol):
    AsyncClient: _AsyncCohereFactory


def _load_async_factory() -> _AsyncCohereFactory:
    try:
        cohere_module = cast(_CohereModule, import_module("cohere"))
    except ImportError as exc:
        raise LlmError("cohere SDK is not installed") from exc
    return cohere_module.AsyncClient


class CohereEmbeddingClient:
    """Production Cohere client. Retries transient failures via `async_with_retry`."""

    _clients_by_api_key: ClassVar[dict[str, _AsyncCohereClient]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = EMBED_MULTILINGUAL,
        retries: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._retries = retries
        self._backoff_s = backoff_s

    @classmethod
    def _client_for_api_key(cls, api_key: str) -> _AsyncCohereClient:
        client = cls._clients_by_api_key.get(api_key)
        if client is None:
            client = _load_async_factory()(api_key=api_key)
            cls._clients_by_api_key[api_key] = client
        return client

    async def embed(self, text: str, input_type: EmbeddingInputType) -> list[float]:
        batch = await self.embed_batch([text], input_type)
        return batch[0]

    async def embed_batch(
        self,
        texts: Sequence[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > COHERE_BATCH_LIMIT:
            raise LlmError(f"batch size {len(texts)} exceeds Cohere limit {COHERE_BATCH_LIMIT}")

        api_key = self._api_key or get_settings().cohere_api_key
        if not api_key:
            raise LlmError("COHERE_API_KEY is not set")
        client = self._client_for_api_key(api_key)

        import structlog

        logger = structlog.get_logger()

        async def _call() -> list[list[float]]:
            response = await client.embed(
                texts=list(texts),
                model=self._model,
                input_type=input_type,
                embedding_types=["float"],
            )
            return [list(vec) for vec in response.embeddings.float]

        try:
            return await async_with_retry(
                _call,
                retries=self._retries,
                backoff_s=self._backoff_s,
                retryable_exceptions=(Exception,),
            )
        except Exception as exc:
            await logger.aerror(
                "cohere_embed_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                model=self._model,
                retries=self._retries,
            )
            raise LlmError(f"Cohere embed failed after {self._retries} attempts: {type(exc).__name__}: {exc}") from exc


class MockEmbeddingClient:
    """Deterministic + L2-normalized mock for unit + scale tests.

    The same `(text, input_type)` pair always returns the same vector; different
    `input_type`s for the same text return different vectors (Cohere's contract).
    """

    def __init__(self, model: str = EMBED_MULTILINGUAL) -> None:
        self._model = model
        self.calls: int = 0
        self.batch_calls: int = 0

    async def embed(self, text: str, input_type: EmbeddingInputType) -> list[float]:
        self.calls += 1
        return _deterministic_vector(text, input_type)

    async def embed_batch(
        self,
        texts: Sequence[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        self.batch_calls += 1
        self.calls += len(texts)
        return [_deterministic_vector(text, input_type) for text in texts]


def _deterministic_vector(text: str, input_type: str) -> list[float]:
    """Return a deterministic 1024-dim L2-normalized vector seeded by `(text, input_type)`.

    Implementation: chained SHA-256 expansion of the seed into 1024 * 4 bytes,
    interpreted as uint32 then mapped to `[-1, 1]`, then L2 normalized.
    """
    seed = hashlib.sha256(f"{input_type}::{text}".encode()).digest()
    floats: list[float] = []
    counter = 0
    while len(floats) < EMBEDDING_DIM:
        chunk = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        for offset in range(0, 32, 4):
            value = int.from_bytes(chunk[offset : offset + 4], "big")
            mapped = (value / 0xFFFFFFFF) * 2.0 - 1.0
            floats.append(mapped)
            if len(floats) == EMBEDDING_DIM:
                break
        counter += 1

    norm = sum(component * component for component in floats) ** 0.5
    if norm == 0.0:
        return floats
    return [component / norm for component in floats]
