"""Unit tests for the §13 shared caching layer.

The four §13 acceptance criteria are exercised against both backends via a
single parametrised conformance suite:

1. `set("k", b"...", 60)` then `get("k")` returns the bytes.
2. After TTL, `get("k")` returns `None`.
3. `delete("k")` then `get("k")` returns `None`.
4. `MemoryCache` and `RedisCache` pass the same suite.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ingovo_search.lib.cache import (
    Cache,
    MemoryCache,
    RedisCache,
    embedding_cache_key,
    enrichment_cache_key,
    geocode_cache_key,
    get_json,
    normalise_address,
    normalise_query,
    parser_cache_key,
    raw_search_response_cache_key,
    rerank_cache_key,
    search_cache_key,
    set_json,
)


class _FakeClock:
    """Manually-advanced clock for deterministic TTL tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeRedis:
    """In-memory `redis.asyncio.Redis` stand-in covering get / set / delete + TTL.

    Mirrors `redis-py` semantics: `set(name, value, ex=seconds)` for time-windowed
    entries, `delete(*names)` returning the count removed, `bytes | None` from
    `get`. The fake honours a `Callable[[], float]` clock so the cache's TTL is
    testable without sleeping.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._store: dict[str, tuple[float | None, bytes]] = {}

    async def get(self, name: str) -> bytes | None:
        entry = self._store.get(name)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._store[name]
            return None
        return value

    async def set(self, name: str, value: bytes, ex: int | None = None) -> bool:
        if ex is None:
            self._store[name] = (None, bytes(value))
        else:
            self._store[name] = (self._clock() + ex, bytes(value))
        return True

    async def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            if name in self._store:
                del self._store[name]
                removed += 1
        return removed


CacheFactory = Callable[[Callable[[], float]], Cache]


def _memory_factory(clock: Callable[[], float]) -> Cache:
    return MemoryCache(clock=clock)


def _redis_factory(clock: Callable[[], float]) -> Cache:
    return RedisCache(_FakeRedis(clock))


CACHE_FACTORIES: list[tuple[str, CacheFactory]] = [
    ("memory", _memory_factory),
    ("redis", _redis_factory),
]


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_set_then_get_returns_bytes(name: str, factory: CacheFactory) -> None:
    """§13 acceptance 1 — round-trip across both backends."""
    _ = name
    cache = factory(_FakeClock())
    await cache.set("k", b'{"hello": "world"}', ttl_seconds=60)

    assert await cache.get("k") == b'{"hello": "world"}'


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_get_after_ttl_returns_none(name: str, factory: CacheFactory) -> None:
    """§13 acceptance 2 — entries expire after their TTL."""
    _ = name
    clock = _FakeClock()
    cache = factory(clock)
    await cache.set("k", b"v", ttl_seconds=60)

    clock.advance(60)
    assert await cache.get("k") is None


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_get_before_ttl_returns_bytes(name: str, factory: CacheFactory) -> None:
    """Entry must remain visible until the TTL window closes."""
    _ = name
    clock = _FakeClock()
    cache = factory(clock)
    await cache.set("k", b"v", ttl_seconds=60)

    clock.advance(59.999)
    assert await cache.get("k") == b"v"


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_delete_evicts_entry(name: str, factory: CacheFactory) -> None:
    """§13 acceptance 3 — delete is immediate."""
    _ = name
    cache = factory(_FakeClock())
    await cache.set("k", b"v", ttl_seconds=60)

    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_get_unknown_key_returns_none(name: str, factory: CacheFactory) -> None:
    """Miss must be `None`, never raise."""
    _ = name
    cache = factory(_FakeClock())
    assert await cache.get("never-set") is None


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_delete_missing_key_is_no_op(name: str, factory: CacheFactory) -> None:
    """Deleting a missing key must not raise."""
    _ = name
    cache = factory(_FakeClock())
    await cache.delete("never-set")


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_set_without_ttl_persists_across_clock_advance(name: str, factory: CacheFactory) -> None:
    """`ttl_seconds=None` stores persistently — Redis SET without EX, Memory keeps None."""
    _ = name
    clock = _FakeClock()
    cache = factory(clock)
    await cache.set("k", b"forever")

    clock.advance(10_000)
    assert await cache.get("k") == b"forever"


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_set_with_non_positive_ttl_raises(name: str, factory: CacheFactory) -> None:
    """Reject `ttl_seconds <= 0` rather than silently caching forever."""
    _ = name
    cache = factory(_FakeClock())

    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        await cache.set("k", b"v", ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        await cache.set("k", b"v", ttl_seconds=-5)


@pytest.mark.parametrize(("name", "factory"), CACHE_FACTORIES)
async def test_overwrites_replace_previous_value(name: str, factory: CacheFactory) -> None:
    _ = name
    cache = factory(_FakeClock())
    await cache.set("k", b"first")
    await cache.set("k", b"second")

    assert await cache.get("k") == b"second"


def test_cache_protocol_runtime_check() -> None:
    """Both backends satisfy the `Cache` Protocol structurally."""
    assert isinstance(MemoryCache(), Cache)
    assert isinstance(RedisCache(_FakeRedis(_FakeClock())), Cache)


# --- JSON helpers -----------------------------------------------------------


class _DummyPayload(BaseModel):
    name: str
    count: int


async def test_set_json_then_get_json_round_trips() -> None:
    cache = MemoryCache()
    payload = _DummyPayload(name="alpha", count=3)
    await set_json(cache, "k", payload, ttl_seconds=60)

    fetched = await get_json(cache, "k", _DummyPayload)
    assert fetched == payload


async def test_get_json_returns_none_on_miss() -> None:
    cache = MemoryCache()
    assert await get_json(cache, "absent", _DummyPayload) is None


async def test_get_json_validates_payload_shape() -> None:
    """Storing arbitrary bytes under a typed key must raise on read."""
    cache = MemoryCache()
    await cache.set("k", b'{"unexpected": true}', ttl_seconds=60)

    with pytest.raises(ValidationError):
        await get_json(cache, "k", _DummyPayload)


# --- Redis client surface ---------------------------------------------------


async def test_redis_cache_decodes_string_responses() -> None:
    """Clients with `decode_responses=True` return `str`; we encode to bytes."""

    class _StringRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}

        async def get(self, name: str) -> str | None:
            return self._store.get(name)

        async def set(self, name: str, value: bytes, ex: int | None = None) -> bool:
            _ = ex
            self._store[name] = value.decode("utf-8")
            return True

        async def delete(self, *names: str) -> int:
            for n in names:
                self._store.pop(n, None)
            return len(names)

    cache = RedisCache(_StringRedis())
    await cache.set("k", b"hello")
    assert await cache.get("k") == b"hello"


async def test_redis_cache_rejects_unexpected_response_type() -> None:
    """An exotic Redis response (e.g. an int) is a configuration bug — surface it."""

    class _IntRedis:
        def get(self, name: str) -> Awaitable[Any]:
            _ = name

            async def _inner() -> int:
                return 42

            return _inner()

        def set(self, name: str, value: bytes, ex: int | None = None) -> Awaitable[Any]:
            _ = name
            _ = value
            _ = ex

            async def _inner() -> bool:
                return True

            return _inner()

        def delete(self, *names: str) -> Awaitable[Any]:
            _ = names

            async def _inner() -> int:
                return 0

            return _inner()

    cache = RedisCache(_IntRedis())
    with pytest.raises(TypeError):
        await cache.get("k")


async def test_redis_cache_forwards_ttl_via_ex_kwarg() -> None:
    """Confirm the wrapper actually calls `SET key value EX ttl`, not just sets."""
    captured: dict[str, Any] = {}

    class _SpyRedis:
        async def get(self, name: str) -> bytes | None:
            _ = name
            return None

        async def set(self, name: str, value: bytes, ex: int | None = None) -> bool:
            captured["name"] = name
            captured["value"] = value
            captured["ex"] = ex
            return True

        async def delete(self, *names: str) -> int:
            return len(names)

    cache = RedisCache(_SpyRedis())
    await cache.set("k", b"v", ttl_seconds=42)

    assert captured == {"name": "k", "value": b"v", "ex": 42}


async def test_redis_cache_omits_ex_when_persistent() -> None:
    """`ttl_seconds=None` must not pass `ex` — Redis defaults to no expiry."""
    captured: dict[str, Any] = {}

    class _SpyRedis:
        async def get(self, name: str) -> bytes | None:
            _ = name
            return None

        async def set(self, name: str, value: bytes, ex: int | None = None) -> bool:
            captured["ex"] = ex
            _ = name
            _ = value
            return True

        async def delete(self, *names: str) -> int:
            return len(names)

    cache = RedisCache(_SpyRedis())
    await cache.set("k", b"v")
    assert captured["ex"] is None


# --- Key builders -----------------------------------------------------------


def test_normalise_query_collapses_whitespace_and_case() -> None:
    assert normalise_query("  Buda  Pest  ") == "buda pest"
    assert normalise_query("BUDA\tPEST\n") == "buda pest"


def test_normalise_address_lowercases_and_collapses() -> None:
    assert normalise_address("  Andrássy út  10. ") == "andrássy út 10."


def test_parser_cache_key_includes_prompt_version_and_normalises_query() -> None:
    k1 = parser_cache_key("parser/v0", "Olcsó panel Pest 13")
    k2 = parser_cache_key("parser/v0", "olcsó   panel   pest   13")
    k3 = parser_cache_key("parser/v1", "Olcsó panel Pest 13")

    assert k1.startswith("parse:vparser/v0:")
    assert k1 == k2  # normalisation collapses whitespace + case
    assert k1 != k3  # version bump invalidates


def test_search_cache_key_distinguishes_signed_in_flag() -> None:
    anon = search_cache_key("hash123", user_signed_in=False)
    signed = search_cache_key("hash123", user_signed_in=True)
    assert anon == "search:hash123:0"
    assert signed == "search:hash123:1"
    assert anon != signed


def test_raw_search_response_cache_key_normalises_and_versions() -> None:
    k1 = raw_search_response_cache_key(" Budai   lakás ", user_signed_in=False, response_version="response-v1")
    k2 = raw_search_response_cache_key("budai lakás", user_signed_in=False, response_version="response-v1")
    k3 = raw_search_response_cache_key("budai lakás", user_signed_in=True, response_version="response-v1")
    k4 = raw_search_response_cache_key("budai lakás", user_signed_in=False, response_version="response-v2")

    assert k1.startswith("search:raw:")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_rerank_cache_key_is_stable_under_id_permutation() -> None:
    k1 = rerank_cache_key("qhash", ["c", "a", "b"])
    k2 = rerank_cache_key("qhash", ["a", "b", "c"])
    k3 = rerank_cache_key("qhash", ["a", "b", "d"])

    assert k1.startswith("rerank:qhash:")
    assert k1 == k2  # order-independent
    assert k1 != k3  # different candidate set → different key


def test_enrichment_cache_key_bumps_with_prompt_version() -> None:
    k_old = enrichment_cache_key("egy nagy panel", "text_enrich/v0")
    k_new = enrichment_cache_key("egy nagy panel", "text_enrich/v1")
    assert k_old.startswith("enrich:vtext_enrich/v0:")
    assert k_old != k_new


def test_embedding_cache_key_distinguishes_model_and_input_type() -> None:
    base = embedding_cache_key("text", model="embed-multilingual-v3.0", input_type="search_document")
    other_model = embedding_cache_key("text", model="bge-m3", input_type="search_document")
    other_input = embedding_cache_key("text", model="embed-multilingual-v3.0", input_type="search_query")

    assert base.startswith("embed:embed-multilingual-v3.0:search_document:")
    assert base != other_model
    assert base != other_input


def test_geocode_cache_key_uses_normalised_address() -> None:
    key = geocode_cache_key("  Andrássy út  10.  ")
    assert key == "geocode:andrássy út 10."
