"""§14 step 10 — the `{"ok": bool, "data": ..., "error": ...}` response envelope.

`Envelope[T]` is the shape every successful route returns. `ErrorBody`
carries a code + message + optional details on the failure path. Three
helpers:

- `ok(data)` — wrap a successful payload.
- `err(code, message, *, details=None)` — wrap an error payload.
- `install_exception_handlers(app)` — register FastAPI handlers that
  translate domain exceptions into the right envelope + status code:

  | Exception                    | Status | Code in body              |
  |------------------------------|--------|---------------------------|
  | `ParseError`                 | 400    | `"invalid_input"`         |
  | `RankingGuardrailViolation`  | 500    | `"ranking_guardrail"`     |
  | `DbError`                    | 500    | `"db_error"`              |
  | `LlmBudgetExceededError`     | 402    | `"llm_budget_exceeded"`   |
  | `LlmError`                   | 502    | `"llm_error"`             |
  | `IngovoError` (catch-all)    | 500    | `"internal_error"`        |
  | `HTTPException` (FastAPI)    | as-is  | `"http_<status>"`         |

  Note `ParseError` is a subclass of `LlmError`, and
  `LlmBudgetExceededError` is too — order matters and is enforced by
  the handler registration sequence.

  The `HTTPException` handler exists so router-side raises (a 404, a
  framework 405) get envelope-wrapped without per-call-site code.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ingovo_search.types.errors import (
    DbError,
    IngovoError,
    LlmBudgetExceededError,
    LlmError,
    ParseError,
    RankingGuardrailViolation,
)


class ErrorBody(BaseModel):
    """Structured error payload — `code` is machine-readable, `message` user-facing."""

    model_config = {"frozen": True}

    code: str
    message: str
    details: dict[str, Any] | None = None


class Envelope[T](BaseModel):
    """`{"ok": True, "data": ...}` on success, `{"ok": False, "error": {...}}` on failure.

    Exactly one of `data` / `error` is populated. The model isn't frozen
    because FastAPI mutates response models during serialisation.
    """

    ok: bool
    data: T | None = None
    error: ErrorBody | None = None


def ok[T](data: T) -> Envelope[T]:
    """Wrap a successful payload in the envelope."""
    return Envelope(ok=True, data=data)


def err(code: str, message: str, *, details: dict[str, Any] | None = None) -> Envelope[None]:
    """Wrap an error payload in the envelope."""
    return Envelope(ok=False, error=ErrorBody(code=code, message=message, details=details))


def _error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a JSONResponse with the structured error envelope."""
    return JSONResponse(
        status_code=status_code,
        content=err(code, message, details=details).model_dump(mode="json"),
    )


# --- Per-exception handlers ------------------------------------------------


async def _parse_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(400, code="invalid_input", message=str(exc))


async def _ranking_guardrail_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, code="ranking_guardrail", message=str(exc))


async def _db_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, code="db_error", message=str(exc))


async def _budget_exceeded_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(402, code="llm_budget_exceeded", message=str(exc))


async def _llm_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(502, code="llm_error", message=str(exc))


async def _ingovo_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, code="internal_error", message=str(exc))


async def _http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Wrap router-side `HTTPException` raises in the envelope shape.

    FastAPI's default returns `{"detail": ...}`; we return the structured
    envelope so the frontend can dispatch on `error.code` everywhere.
    """
    if not isinstance(exc, StarletteHTTPException):
        # Registered for `HTTPException` only, so Starlette never routes anything else here.
        raise TypeError(f"the HTTPException handler got {type(exc).__name__}")
    detail = str(exc.detail)
    return _error_response(exc.status_code, code=f"http_{exc.status_code}", message=detail)


def install_exception_handlers(app: FastAPI) -> None:
    """Register the §14 step-10 exception → envelope mappings on `app`.

    Order matters: more-specific subclasses first, the `IngovoError`
    base last. FastAPI's `add_exception_handler` does its own most-
    specific-match dispatch on subclasses anyway, but ordering reads
    cleaner alongside the docstring table.

    `HTTPException` is registered last so it catches any
    `HTTPException` (or subclass) raised by routes or dependencies.
    """
    app.add_exception_handler(ParseError, _parse_error_handler)
    app.add_exception_handler(RankingGuardrailViolation, _ranking_guardrail_handler)
    app.add_exception_handler(DbError, _db_error_handler)
    app.add_exception_handler(LlmBudgetExceededError, _budget_exceeded_handler)
    app.add_exception_handler(LlmError, _llm_error_handler)
    app.add_exception_handler(IngovoError, _ingovo_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)


__all__ = [
    "Envelope",
    "ErrorBody",
    "err",
    "install_exception_handlers",
    "ok",
]
