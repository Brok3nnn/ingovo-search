"""Per-client sliding-window rate limiting backed by Redis sorted sets."""

from __future__ import annotations

import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ingovo_search.api.deps import get_redis_client
from ingovo_search.lib.env import get_settings

_FALLBACK_IP = "127.0.0.1"


def resolve_client_ip(request: Request) -> str:
    """Resolve the address to bill this request against.

    `X-Forwarded-For` is client-writable, so it is only consulted when
    `RATE_LIMIT_TRUSTED_PROXY_HOPS` says how many entries our own infrastructure
    appended. Each trusted hop appends the peer *it* saw, so with N trusted hops
    the real client is the Nth entry from the right — every entry to its left was
    supplied by the caller and must never be trusted. With the default of 0 hops
    the header is ignored entirely and the socket peer is billed.

    A header with fewer entries than the configured hop count means the request
    did not traverse the expected proxy chain (direct hit on the origin, or a
    truncated header), so we fall back to the socket peer rather than trusting a
    forgeable entry.
    """
    peer = request.client.host if request.client else _FALLBACK_IP

    hops = get_settings().rate_limit_trusted_proxy_hops
    if hops <= 0:
        return peer

    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return peer

    parts = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(parts) < hops:
        return peer

    return parts[-hops]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed sliding window rate limiter.

    Limits requests by IP address using a sorted set in Redis to track request
    timestamps. `path_prefix` scopes the limit to a single endpoint family —
    multiple middleware instances can stack with distinct prefixes (e.g. a
    permissive global cap plus a tighter `/api/search` cap), as long as each
    instance uses a distinct `key_prefix` so they do not share a counter.

    `enabled=False` (no Redis configured, or `ENV=test`) lets every request through.
    """

    def __init__(
        self,
        app: Any,
        max_requests: int = 200,
        window_seconds: int = 60,
        *,
        path_prefix: str | None = None,
        key_prefix: str = "rate_limit",
        enabled: bool = True,
    ):
        super().__init__(app)
        self.enabled = enabled
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.path_prefix = path_prefix
        self.key_prefix = key_prefix

    async def dispatch(self, request: Request, call_next: Any):
        # Health checks are never limited.
        if not self.enabled or request.url.path == "/healthz":
            return await call_next(request)

        # If this middleware instance is scoped to a path family, fall through
        # for everything outside that scope so the global instance can handle it.
        if self.path_prefix is not None and not request.url.path.startswith(self.path_prefix):
            return await call_next(request)

        client_ip = resolve_client_ip(request)

        redis = get_redis_client()
        key = f"{self.key_prefix}:{client_ip}"
        now = time.time()

        async with redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, now - self.window_seconds)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds)
            results = await pipe.execute()

        request_count = results[2]

        if request_count > self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": (
                            f"Too many requests: at most {self.max_requests} per {self.window_seconds}s "
                            "on this endpoint. Try again later."
                        ),
                    }
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)
