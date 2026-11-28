"""Cache backends for embeddings.

Cache key encodes `(model, input_type, sha256(text))` so the cache invalidates
naturally on model swap or input-type change. Two backends ship by default:

- `InMemoryEmbeddingCache` for unit tests.
- `PostgresEmbeddingCache` for production, backed by `embedding_cache`
  (see `db/schema.sql`). Stored as `vector(1024)` so retrieval returns the exact
  bytes that pgvector would use in a similarity query.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.vector_db import VECTOR_DIM, vector_literal


def text_hash(value: str) -> str:
    """SHA-256 hex digest of `value` (utf-8 encoded)."""
    return hashlib.sha256(value.encode()).hexdigest()


def cache_key_for(text_value: str, model: str, input_type: str) -> str:
    return f"{model}:{input_type}:{text_hash(text_value)}"


@runtime_checkable
class EmbeddingCacheBackend(Protocol):
    async def get(self, key: str) -> list[float] | None: ...

    async def set(
        self,
        key: str,
        *,
        text_value: str,
        model: str,
        input_type: str,
        embedding: Sequence[float],
    ) -> None: ...


class InMemoryEmbeddingCache:
    """Process-local cache for unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    async def get(self, key: str) -> list[float] | None:
        cached = self._store.get(key)
        if cached is None:
            return None
        return list(cached)

    async def set(
        self,
        key: str,
        *,
        text_value: str,
        model: str,
        input_type: str,
        embedding: Sequence[float],
    ) -> None:
        _ = text_value
        _ = model
        _ = input_type
        self._store[key] = list(embedding)


class PostgresEmbeddingCache:
    """Backs embedding lookups with the `embedding_cache` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, key: str) -> list[float] | None:
        result = await self._session.execute(
            text("SELECT embedding::text AS embedding_text FROM embedding_cache WHERE cache_key = :key"),
            {"key": key},
        )
        row = result.one_or_none()
        if row is None:
            return None
        return _parse_vector_literal(row.embedding_text)

    async def set(
        self,
        key: str,
        *,
        text_value: str,
        model: str,
        input_type: str,
        embedding: Sequence[float],
    ) -> None:
        if len(embedding) != VECTOR_DIM:
            raise ValueError(f"expected {VECTOR_DIM}-dim vector, got {len(embedding)}")
        await self._session.execute(
            text(
                """
                INSERT INTO embedding_cache (
                    cache_key, text_hash, model_used, input_type, embedding
                )
                VALUES (
                    :key, :text_hash, :model, :input_type, CAST(:embedding AS vector)
                )
                ON CONFLICT (cache_key) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_used = EXCLUDED.model_used,
                    updated_at = now()
                """
            ),
            {
                "key": key,
                "text_hash": text_hash(text_value),
                "model": model,
                "input_type": input_type,
                "embedding": vector_literal(embedding),
            },
        )


def _parse_vector_literal(literal: str) -> list[float]:
    """Parse `'[1.0,2.0,...]'` back into `list[float]`."""
    inner = literal.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    if not inner:
        return []
    return [float(component) for component in inner.split(",")]
