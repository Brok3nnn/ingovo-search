"""Shared caching layer.

Single `Cache` Protocol with two implementations:

- `MemoryCache` — process-local TTL cache for tests and dev.
- `RedisCache` — wraps an Upstash-compatible `redis.asyncio.Redis` client
  for production time-windowed caches (parser, search, rerank).

The listing-embedding cache has its own backends (`lib/embedding_cache.py`). The
key-prefix helpers below give callers a single source of truth for every cache
key, regardless of which backend stores the entry.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Cache(Protocol):
    """Bytes-level async cache contract. Backends must round-trip exactly."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


class MemoryCache:
    """Process-local TTL cache. Entries expire `ttl_seconds` after `set`.

    Not safe across processes; intended for unit tests and single-worker dev.
    `clock` is injectable so TTL expiry is testable without sleeping.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._store: dict[str, tuple[float | None, bytes]] = {}

    async def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        expires_at = self._clock() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (expires_at, bytes(value))

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class _RedisLike(Protocol):
    """Subset of `redis.asyncio.Redis` that `RedisCache` consumes.

    `redis.asyncio.Redis` satisfies this Protocol structurally; tests can
    pass any object that exposes the same async surface.
    """

    def get(self, name: str) -> Awaitable[Any]: ...

    def set(
        self,
        name: str,
        value: bytes,
        ex: int | None = ...,
    ) -> Awaitable[Any]: ...

    def delete(self, *names: str) -> Awaitable[Any]: ...


class RedisCache:
    """Cache backed by an Upstash-compatible `redis.asyncio.Redis` client.

    The client must be configured with `decode_responses=False` (the default)
    so `get` returns `bytes | None` rather than `str | None`. TTL is forwarded
    via Redis `SET key value EX seconds`; omitting `ttl_seconds` stores the
    entry persistently (Redis defaults to no expiry).
    """

    def __init__(self, client: _RedisLike) -> None:
        self._client = client

    async def get(self, key: str) -> bytes | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return raw.encode("utf-8")
        raise TypeError(f"unexpected Redis value type: {type(raw)!r}")

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is None:
            await self._client.set(key, value)
            return
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        await self._client.set(key, value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)


async def get_json[T: BaseModel](cache: Cache, key: str, model: type[T]) -> T | None:
    """Typed read-through helper: deserialize the stored JSON bytes into `model`.

    Returns `None` on cache miss. Raises `pydantic.ValidationError` if the
    stored payload no longer matches `model` — bump the cache-key prefix
    (e.g. via `PROMPT_VERSION`) when changing the model shape.
    """
    raw = await cache.get(key)
    if raw is None:
        return None
    return model.model_validate_json(raw)


async def set_json(
    cache: Cache,
    key: str,
    value: BaseModel,
    *,
    ttl_seconds: int | None = None,
) -> None:
    """Typed write helper: serialize `value` to compact JSON bytes."""
    payload = value.model_dump_json().encode("utf-8")
    await cache.set(key, payload, ttl_seconds=ttl_seconds)


# --- Cache-key builders ----------------------------------------------------
#
# Each helper returns a fully-qualified key string so callers never assemble
# prefixes by hand.


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_WHITESPACE_RE = re.compile(r"\s+")


def normalise_query(text: str) -> str:
    """Whitespace-collapsed, casefolded query — stable across trivial variations."""
    return _WHITESPACE_RE.sub(" ", text.strip()).casefold()


def parser_cache_key(prompt_version: str, query: str) -> str:
    """`parse:v{PROMPT_VERSION}:{sha256(normalised_query)}` — 10-min TTL."""
    return f"parse:v{prompt_version}:{_sha256(normalise_query(query))}"


def search_cache_key(parsed_query_hash: str, *, user_signed_in: bool) -> str:
    """`search:{parsed_query_hash}:{0|1}` — 10-min TTL."""
    flag = "1" if user_signed_in else "0"
    return f"search:{parsed_query_hash}:{flag}"


def raw_search_response_cache_key(query: str, *, user_signed_in: bool, response_version: str) -> str:
    """Final search-response cache keyed before the LLM parser runs.

    This is intentionally narrower than `search_cache_key`: callers must only
    use it for clean, session-independent searches where the raw query fully
    determines the final response. `response_version` lets the API layer fold in
    score/prompt/shape changes without making this low-level helper import
    search-specific constants.
    """
    flag = "1" if user_signed_in else "0"
    payload = f"{response_version}\x00{flag}\x00{normalise_query(query)}"
    return f"search:raw:{_sha256(payload)}"


def rerank_cache_key(query_hash: str, top_ids: Sequence[str]) -> str:
    """`rerank:{query_hash}:{sha256(sorted_ids)}` — 10-min TTL.

    Candidate ids are sorted before hashing so the key is stable under
    permutations of the top-N input order.
    """
    sorted_ids = sorted(top_ids)
    return f"rerank:{query_hash}:{_sha256(','.join(sorted_ids))}"


def enrichment_cache_key(description: str, prompt_version: str) -> str:
    """`enrich:v{PROMPT_VERSION}:{sha256(description)}` — persistent.

    `prompt_version` is folded in so a prompt rewrite invalidates the cache
    without touching the underlying table.
    """
    return f"enrich:v{prompt_version}:{_sha256(description)}"


def embedding_cache_key(content: str, *, model: str, input_type: str) -> str:
    """`embed:{model}:{input_type}:{sha256(content)}` — persistent.

    Mirrors `lib.embedding_cache.cache_key_for` so both layers agree on the
    canonical key. The prefix `embed:` lets a future caller distinguish
    it from other namespaces if Redis is ever used for embeddings.
    """
    return f"embed:{model}:{input_type}:{_sha256(content)}"


__all__ = [
    "Cache",
    "MemoryCache",
    "RedisCache",
    "embedding_cache_key",
    "enrichment_cache_key",
    "get_json",
    "normalise_query",
    "parser_cache_key",
    "raw_search_response_cache_key",
    "rerank_cache_key",
    "search_cache_key",
    "set_json",
]
