"""Unit tests for `InMemoryEmbeddingCache` and cache-key derivation."""

from __future__ import annotations

from ingovo_search.lib.embedding_cache import (
    EmbeddingCacheBackend,
    InMemoryEmbeddingCache,
    cache_key_for,
    text_hash,
)


def test_text_hash_is_deterministic() -> None:
    assert text_hash("hello") == text_hash("hello")
    assert text_hash("hello") != text_hash("hello!")


def test_cache_key_includes_model_and_input_type() -> None:
    """Model swap or input-type swap must invalidate the cache."""
    key_doc = cache_key_for("text", "embed-multilingual-v3.0", "search_document")
    key_query = cache_key_for("text", "embed-multilingual-v3.0", "search_query")
    key_alt_model = cache_key_for("text", "bge-m3", "search_document")

    assert key_doc != key_query
    assert key_doc != key_alt_model


async def test_in_memory_cache_round_trips_a_vector() -> None:
    cache = InMemoryEmbeddingCache()
    key = cache_key_for("text", "embed-multilingual-v3.0", "search_document")
    vec = [0.1, 0.2, 0.3]

    assert await cache.get(key) is None

    await cache.set(
        key,
        text_value="text",
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding=vec,
    )

    assert await cache.get(key) == vec


async def test_in_memory_cache_returns_copies() -> None:
    """Mutating the retrieved list must not affect the cache."""
    cache = InMemoryEmbeddingCache()
    key = cache_key_for("text", "embed-multilingual-v3.0", "search_document")
    await cache.set(
        key,
        text_value="text",
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding=[1.0, 2.0],
    )

    fetched = await cache.get(key)
    assert fetched is not None
    fetched.append(99.0)

    refetched = await cache.get(key)
    assert refetched == [1.0, 2.0]


def test_in_memory_cache_satisfies_protocol() -> None:
    cache: EmbeddingCacheBackend = InMemoryEmbeddingCache()
    assert isinstance(cache, EmbeddingCacheBackend)
