# pyright: reportUnusedFunction=false
"""Unit tests for the session-cookie read + write helpers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from ingovo_search.api.cookies import (
    SESSION_COOKIE_NAME,
    clear_session_cookie,
    read_session_id,
    set_session_cookie,
)
from ingovo_search.lib.session import (
    SESSION_TTL_SECONDS,
    sign_session_id,
    verify_signed_session_id,
)

SECRET = b"x" * 32


def _stub_request_with_cookies(cookies: dict[str, str]) -> Request:
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    scope: dict[str, Any] = {
        "type": "http",
        "headers": [(b"cookie", cookie_header.encode())] if cookie_header else [],
        "method": "GET",
        "path": "/",
    }
    return Request(scope)


# --- read_session_id -----------------------------------------------------


def test_read_session_id_returns_none_when_no_cookie() -> None:
    request = _stub_request_with_cookies({})
    assert read_session_id(request, secret=SECRET) is None


def test_read_session_id_returns_id_for_valid_cookie() -> None:
    signed = sign_session_id("session-abc", secret=SECRET)
    request = _stub_request_with_cookies({SESSION_COOKIE_NAME: signed})
    assert read_session_id(request, secret=SECRET) == "session-abc"


def test_read_session_id_returns_none_for_tampered_cookie() -> None:
    signed = sign_session_id("session-abc", secret=SECRET)
    head, _, sig = signed.partition(".")
    tampered = f"{head}-x.{sig}"
    request = _stub_request_with_cookies({SESSION_COOKIE_NAME: tampered})
    assert read_session_id(request, secret=SECRET) is None


def test_read_session_id_returns_none_for_wrong_secret() -> None:
    signed = sign_session_id("session-abc", secret=SECRET)
    request = _stub_request_with_cookies({SESSION_COOKIE_NAME: signed})
    assert read_session_id(request, secret=b"y" * 32) is None


def test_read_session_id_returns_none_for_malformed_value() -> None:
    request = _stub_request_with_cookies({SESSION_COOKIE_NAME: "no-dot-here"})
    assert read_session_id(request, secret=SECRET) is None


# --- set_session_cookie (via a tiny FastAPI app) -------------------------


def _app_setting_cookie(session_id: str, *, secure: bool = True) -> FastAPI:
    app = FastAPI()

    @app.get("/login")
    def _login() -> Response:
        response = Response()
        set_session_cookie(response, session_id, secret=SECRET, secure=secure)
        return response

    return app


def test_set_session_cookie_stamps_signed_value() -> None:
    response = TestClient(_app_setting_cookie("session-123")).get("/login")
    cookie = response.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None
    # The cookie value is the signed form; verify via the helper.
    assert verify_signed_session_id(cookie, secret=SECRET) == "session-123"


def test_set_session_cookie_uses_httponly_lax_secure_defaults() -> None:
    response = TestClient(_app_setting_cookie("session-123")).get("/login")
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "secure" in header
    assert f"max-age={SESSION_TTL_SECONDS}" in header
    assert "path=/" in header


def test_set_session_cookie_can_drop_secure_for_local_dev() -> None:
    response = TestClient(_app_setting_cookie("session-123", secure=False)).get("/login")
    header = response.headers["set-cookie"].lower()
    assert "secure" not in header
    assert "httponly" in header


def test_set_then_read_round_trip_through_test_client() -> None:
    """A round-trip request → response → request validates the full path."""
    app = FastAPI()

    @app.get("/issue")
    def _issue() -> Response:
        response = Response(content="ok")
        set_session_cookie(response, "rt-session", secret=SECRET, secure=False)
        return response

    @app.get("/echo")
    def _echo(request: Request) -> dict[str, str | None]:
        sid = read_session_id(request, secret=SECRET)
        return {"session_id": sid}

    client = TestClient(app)
    client.get("/issue")
    body = client.get("/echo").json()
    assert body == {"session_id": "rt-session"}


# --- clear_session_cookie -------------------------------------------------


def test_clear_session_cookie_emits_delete_directive() -> None:
    app = FastAPI()

    @app.get("/logout")
    def _logout() -> Response:
        response = Response()
        clear_session_cookie(response, secure=False)
        return response

    response = TestClient(app).get("/logout")
    header = response.headers["set-cookie"].lower()
    assert SESSION_COOKIE_NAME in header
    # Delete cookies use max-age=0 or an expires-in-past directive.
    assert "max-age=0" in header or "expires" in header


def test_session_cookie_name_constant_is_stable() -> None:
    """Renaming the cookie would break every existing browser session — pin it."""
    assert SESSION_COOKIE_NAME == "ingovo_sid"
