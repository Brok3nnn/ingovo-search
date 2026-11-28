"""Unit tests for `MockEmbeddingClient` — Protocol conformance, determinism, L2 norm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import pytest

from ingovo_search.lib.embeddings import (
    EMBEDDING_DIM,
    CohereEmbeddingClient,
    EmbeddingClient,
    MockEmbeddingClient,
)


async def test_embed_returns_1024_dim_vector() -> None:
    client = MockEmbeddingClient()
    vec = await client.embed("hello", "search_document")
    assert len(vec) == EMBEDDING_DIM


async def test_embed_same_text_same_input_type_is_deterministic() -> None:
    client = MockEmbeddingClient()
    first = await client.embed("Lakás Budapesten", "search_document")
    second = await client.embed("Lakás Budapesten", "search_document")
    assert first == second


async def test_embed_different_input_types_yield_different_vectors() -> None:
    client = MockEmbeddingClient()
    as_document = await client.embed("Lakás Budapesten", "search_document")
    as_query = await client.embed("Lakás Budapesten", "search_query")
    assert as_document != as_query


async def test_embed_returns_l2_normalized_vector() -> None:
    client = MockEmbeddingClient()
    vec = await client.embed("normalize me", "search_document")
    norm = math.sqrt(sum(component * component for component in vec))
    assert abs(norm - 1.0) < 1e-6


async def test_embed_batch_matches_individual_embed() -> None:
    client = MockEmbeddingClient()
    texts = ["alpha", "beta", "gamma"]
    batch = await client.embed_batch(texts, "search_document")
    individual = [await client.embed(text, "search_document") for text in texts]
    assert batch == individual


async def test_embed_batch_increments_call_counters() -> None:
    client = MockEmbeddingClient()
    await client.embed_batch(["one", "two", "three"], "search_document")
    assert client.batch_calls == 1
    assert client.calls == 3


def test_mock_satisfies_embedding_client_protocol() -> None:
    client: EmbeddingClient = MockEmbeddingClient()
    assert isinstance(client, EmbeddingClient)


async def test_cohere_client_reuses_sdk_client_per_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class _FakeEmbeddings:
        float: ClassVar[list[list[float]]] = [[0.1] * EMBEDDING_DIM]

    class _FakeResponse:
        embeddings = _FakeEmbeddings()

    class _FakeClient:
        async def embed(
            self,
            *,
            texts: Sequence[str],
            model: str,
            input_type: str,
            embedding_types: Sequence[str],
        ) -> _FakeResponse:
            _ = texts
            _ = model
            _ = input_type
            _ = embedding_types
            return _FakeResponse()

    def _fake_factory(*, api_key: str) -> _FakeClient:
        created.append(api_key)
        return _FakeClient()

    CohereEmbeddingClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr("ingovo_search.lib.embeddings._load_async_factory", lambda: _fake_factory)

    assert await CohereEmbeddingClient(api_key="same-key").embed("hello", "search_query") == [0.1] * EMBEDDING_DIM
    assert await CohereEmbeddingClient(api_key="same-key").embed("hello", "search_query") == [0.1] * EMBEDDING_DIM

    assert created == ["same-key"]
    CohereEmbeddingClient._clients_by_api_key.clear()  # pyright: ignore[reportPrivateUsage]
