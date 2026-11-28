"""Unit tests for RateLimitMiddleware path-scope + key-isolation behavior."""

from __future__ import annotations

from typing import Any

import pytest

from ingovo_search.api.main import create_app
from ingovo_search.api.middleware import rate_limit as rate_limit_module
from ingovo_search.api.middleware.rate_limit import RateLimitMiddleware, resolve_client_ip

_PEER = "10.0.0.1"


def _request_with_forwarded(forwarded: str | None, *, peer: str = _PEER) -> Any:
    """Stub `starlette.Request` exposing only what `resolve_client_ip` reads."""

    class _Client:
        host = peer

    class _Request:
        def __init__(self) -> None:
            self.client = _Client()
            self.headers: dict[str, str] = {} if forwarded is None else {"x-forwarded-for": forwarded}

    return _Request()


def _with_hops(monkeypatch: pytest.MonkeyPatch, hops: int) -> None:
    class _FakeSettings:
        rate_limit_trusted_proxy_hops = hops

    monkeypatch.setattr(rate_limit_module, "get_settings", _FakeSettings)


def test_zero_hops_ignores_a_spoofed_forwarded_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config: XFF is never consulted, so it cannot mint new buckets."""
    _with_hops(monkeypatch, 0)

    assert resolve_client_ip(_request_with_forwarded("1.2.3.4")) == _PEER


def test_one_hop_reads_the_rightmost_entry_not_the_spoofable_left(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behind one LB the real peer is the entry our LB appended — the last one.

    Everything left of it is caller-supplied. Reading `parts[0]` (the pre-fix
    behaviour) would bill the attacker-chosen `9.9.9.9` and hand out a fresh
    rate-limit bucket on every request.
    """
    _with_hops(monkeypatch, 1)

    assert resolve_client_ip(_request_with_forwarded("9.9.9.9, 203.0.113.7")) == "203.0.113.7"


def test_two_hops_skips_the_cdn_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """CDN + LB: the LB appended the CDN's address, the CDN appended the client."""
    _with_hops(monkeypatch, 2)

    assert resolve_client_ip(_request_with_forwarded("9.9.9.9, 203.0.113.7, 198.51.100.2")) == "203.0.113.7"


def test_spoofed_header_cannot_shift_the_billed_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Padding the header with extra left-hand entries must not change the result."""
    _with_hops(monkeypatch, 1)

    padded = resolve_client_ip(_request_with_forwarded("a, b, c, d, e, 203.0.113.7"))
    plain = resolve_client_ip(_request_with_forwarded("203.0.113.7"))

    assert padded == plain == "203.0.113.7"


def test_shorter_chain_than_configured_falls_back_to_the_socket_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct hit on the origin must not let a 1-entry header impersonate."""
    _with_hops(monkeypatch, 2)

    assert resolve_client_ip(_request_with_forwarded("9.9.9.9")) == _PEER


def test_missing_header_falls_back_to_the_socket_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_hops(monkeypatch, 1)

    assert resolve_client_ip(_request_with_forwarded(None)) == _PEER


def test_blank_entries_are_discarded_before_counting_hops(monkeypatch: pytest.MonkeyPatch) -> None:
    """`"a, , 203.0.113.7"` must not let an empty slot shift the hop index."""
    _with_hops(monkeypatch, 1)

    assert resolve_client_ip(_request_with_forwarded("9.9.9.9, , 203.0.113.7")) == "203.0.113.7"


def _request_at_path(path: str) -> Any:
    """Stub `starlette.Request` that exposes only what `dispatch` reads."""

    class _Url:
        def __init__(self, p: str) -> None:
            self.path = p

    class _Client:
        host = "10.0.0.1"

    class _Headers(dict[str, str]):
        pass

    class _Request:
        def __init__(self, p: str) -> None:
            self.url = _Url(p)
            self.client = _Client()
            self.headers = _Headers()

    return _Request(path)


@pytest.mark.asyncio
async def test_path_prefix_lets_unrelated_paths_pass_through() -> None:
    """A scoped middleware does NOT rate-limit paths outside its prefix —
    the next-middleware-up handles them (or they fall through to the app)."""

    middleware = RateLimitMiddleware(
        app=None,
        max_requests=1,
        window_seconds=60,
        path_prefix="/api/search",
        key_prefix="rate_limit_search",
    )

    calls: list[str] = []

    async def _next(_req: Any) -> str:
        calls.append("passed")
        return "OK"

    # The path filter lets the request through before Redis is ever touched.
    result = await middleware.dispatch(_request_at_path("/api/properties"), _next)  # pyright: ignore[reportArgumentType]
    assert result == "OK"
    assert calls == ["passed"]


def _registered_rate_limiters() -> list[dict[str, Any]]:
    """The RateLimitMiddleware instances `create_app` actually installs.

    Read off the real app rather than reconstructed here: the thing worth guarding
    is the wiring, and a test that restates the constants would pass no matter what
    main.py does with them.
    """
    return [
        middleware.kwargs
        for middleware in create_app().user_middleware
        if middleware.cls is RateLimitMiddleware  # pyright: ignore[reportUnknownMemberType]
    ]


def test_create_app_installs_a_tight_search_limit_and_a_global_one() -> None:
    """The wiring, read off the real app: 20/min on `/api/search*`, 200/min everywhere."""
    limits = {
        (kwargs.get("path_prefix"), kwargs["max_requests"], kwargs["window_seconds"], kwargs.get("key_prefix"))
        for kwargs in _registered_rate_limiters()
    }

    assert limits == {
        ("/api/search", 20, 60, "rate_limit_search"),
        (None, 200, 60, None),
    }


def test_limiters_are_off_under_env_test() -> None:
    """The conftest pins ENV=test, so the installed limiters must not touch Redis."""
    assert all(kwargs["enabled"] is False for kwargs in _registered_rate_limiters())


def test_middleware_uses_distinct_key_prefixes() -> None:
    """Two instances must NOT share a Redis sorted set — verify constructor."""
    global_mw = RateLimitMiddleware(app=None, max_requests=200, window_seconds=60)
    search_mw = RateLimitMiddleware(
        app=None,
        max_requests=20,
        window_seconds=60,
        path_prefix="/api/search",
        key_prefix="rate_limit_search",
    )

    assert global_mw.key_prefix == "rate_limit"
    assert search_mw.key_prefix == "rate_limit_search"
    assert global_mw.path_prefix is None
    assert search_mw.path_prefix == "/api/search"
