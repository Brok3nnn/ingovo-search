"""FastAPI dependency providers: the model clients, the cache and the session store.

Every provider is overridable through `app.dependency_overrides`, which is how the
tests swap in `MockLlmClient`, `MockEmbeddingClient` and a `MemoryCache` without
Redis, Postgres or network access.

`get_settings()` is `@lru_cache`d so per-request resolution is cheap. The Redis
client is a process-wide singleton: `redis-py` reuses its connection pool across
awaits, so one client per process is right. With `REDIS_URL` empty, one
process-local `MemoryCache` stands in (fine for a single development worker).
"""

from __future__ import annotations

from dataclasses import dataclass

import redis.asyncio as redis_async

from ingovo_search.db.session import get_session as get_db_session
from ingovo_search.lib.cache import Cache, MemoryCache, RedisCache
from ingovo_search.lib.embeddings import CohereEmbeddingClient, EmbeddingClient
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.jev import JevClient, TypeSafeJevClient
from ingovo_search.lib.llm import GoogleAiClient, LlmClient
from ingovo_search.lib.session import SessionStore


@dataclass(slots=True)
class ProcessClients:
    """The process-wide clients, each built on first use and then reused.

    Each owns a connection pool (or, for the memory cache, the data) every request
    shares, so there is one per process; tests reset a slot to ``None`` to force a rebuild.
    """

    redis: redis_async.Redis | None = None
    memory_cache: MemoryCache | None = None
    jev: TypeSafeJevClient | None = None


process_clients = ProcessClients()


def redis_configured() -> bool:
    return get_settings().redis_url is not None


def get_redis_client() -> redis_async.Redis:
    """Module-level singleton — connection pool is shared across requests."""
    redis_url = get_settings().redis_url
    if redis_url is None:
        raise RuntimeError("REDIS_URL is not set")
    if process_clients.redis is None:
        process_clients.redis = redis_async.from_url(redis_url, decode_responses=False)
    return process_clients.redis


def get_session_cache() -> Cache:
    """The shared cache behind the session store and the search caches."""
    if not redis_configured():
        if process_clients.memory_cache is None:
            process_clients.memory_cache = MemoryCache()
        return process_clients.memory_cache
    return RedisCache(get_redis_client())


def get_session_secret() -> bytes:
    """HMAC secret for `sign_session_id` / `verify_signed_session_id`."""
    return get_settings().session_signing_key.encode("utf-8")


def get_session_store() -> SessionStore:
    """The anonymous-session store over the shared cache; tests override it."""
    return SessionStore(get_session_cache())


def get_session_cookie_secure() -> bool:
    """`Secure` flag on the session cookie — on in production, off in development and tests.

    httpx (and real browsers) refuse to send `Secure` cookies over plain HTTP,
    so development and tests run with this off; production uses HTTPS only.
    """
    return get_settings().env == "production"


def get_llm_client() -> LlmClient:
    """Production LLM client (Gemini)."""
    return GoogleAiClient(api_key=get_settings().gemini_api_key)


def get_embedding_client() -> EmbeddingClient:
    """Production embedding client (Cohere)."""
    return CohereEmbeddingClient(api_key=get_settings().cohere_api_key)


def get_jev_client() -> JevClient | None:
    """Process-wide Jev client, or None when TYPESAFE_API_KEY is unset.

    One instance per process, like the Redis client: it owns an HTTP connection
    pool that every search reuses.
    """
    api_key = get_settings().typesafe_api_key
    if api_key is None:
        return None
    if process_clients.jev is None:
        process_clients.jev = TypeSafeJevClient(api_key=api_key)
    return process_clients.jev


__all__ = [
    "ProcessClients",
    "get_db_session",
    "get_embedding_client",
    "get_jev_client",
    "get_llm_client",
    "get_redis_client",
    "get_session_cache",
    "get_session_cookie_secure",
    "get_session_secret",
    "get_session_store",
    "process_clients",
    "redis_configured",
]
