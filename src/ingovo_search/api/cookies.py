"""Read + write the signed anonymous-session cookie.

Sits between Starlette `Request` / `Response` and the
`sign_session_id` / `verify_signed_session_id` helpers. Two operations:

- `read_session_id(request, *, secret)` — pulls the cookie, verifies the
  HMAC, returns the bare session id (or `None` if missing / tampered /
  malformed).
- `set_session_cookie(response, session_id, *, secret)` — re-signs the
  id and writes the cookie with `HttpOnly`, `SameSite=Lax`, `Secure`
  (controllable per env), and `Max-Age=SESSION_TTL_SECONDS`.

`SESSION_COOKIE_NAME = "ingovo_sid"` is the agreed-upon name; routers
+ tests + frontend code all read it from this constant.

The buyer-search router uses these together with the `SessionStore`:
on the first request it sees a missing cookie, calls `store.create()`,
and stamps the cookie via `set_session_cookie`; subsequent requests
read the cookie, call `store.get(session_id)`, and operate on the
existing state.
"""

from __future__ import annotations

from fastapi import Request, Response

from ingovo_search.lib.session import (
    SESSION_TTL_SECONDS,
    sign_session_id,
    verify_signed_session_id,
)

SESSION_COOKIE_NAME: str = "ingovo_sid"
"""Cookie name agreed with the frontend; never spell this string elsewhere."""


def read_session_id(request: Request, *, secret: bytes) -> str | None:
    """Read + verify the session id from the request's cookies.

    Returns the unwrapped session id, or `None` if no cookie is set,
    the value is malformed, or the HMAC doesn't verify. Routers should
    treat `None` as "first visit" and mint a new state.
    """
    signed = request.cookies.get(SESSION_COOKIE_NAME)
    if signed is None:
        return None
    return verify_signed_session_id(signed, secret=secret)


def set_session_cookie(
    response: Response,
    session_id: str,
    *,
    secret: bytes,
    secure: bool = True,
) -> None:
    """Sign `session_id` and stamp the cookie on the response.

    `secure=True` is the default (production); dev/test pass
    `secure=False` so the cookie is sent over plain HTTP. The cookie
    is `HttpOnly` and `SameSite=Lax`. `Max-Age = SESSION_TTL_SECONDS`
    aligns with the session store's rolling TTL — the browser drops the
    cookie once the server-side state has expired.
    """
    response.set_cookie(
        SESSION_COOKIE_NAME,
        sign_session_id(session_id, secret=secret),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def clear_session_cookie(response: Response, *, secure: bool = True) -> None:
    """Used by the "start fresh" route to drop the cookie."""
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=secure,
    )


__all__ = [
    "SESSION_COOKIE_NAME",
    "clear_session_cookie",
    "read_session_id",
    "set_session_cookie",
]
