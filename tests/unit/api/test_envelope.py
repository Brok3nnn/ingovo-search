# pyright: reportUnusedFunction=false
"""Unit tests for the response envelope + exception handlers.

Routes are wired against a tiny in-test FastAPI app so the handler
matrix can be exercised end-to-end via `TestClient` without any
real router code.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ingovo_search.api.envelope import (
    Envelope,
    ErrorBody,
    err,
    install_exception_handlers,
    ok,
)
from ingovo_search.types.errors import (
    DbError,
    IngovoError,
    LlmBudgetExceededError,
    LlmError,
    ParseError,
    RankingGuardrailViolation,
)

# --- Envelope helpers -----------------------------------------------------


class _Payload(BaseModel):
    value: int


def test_ok_sets_ok_true_and_data() -> None:
    env = ok(_Payload(value=42))
    assert env.ok is True
    assert env.data == _Payload(value=42)
    assert env.error is None


def test_err_sets_ok_false_and_error() -> None:
    env = err("invalid_input", "bad payload")
    assert env.ok is False
    assert env.error == ErrorBody(code="invalid_input", message="bad payload", details=None)
    assert env.data is None


def test_err_carries_optional_details() -> None:
    env = err("invalid_input", "bad payload", details={"field": "query"})
    assert env.error is not None
    assert env.error.details == {"field": "query"}


def test_envelope_serializes_to_expected_keys() -> None:
    payload = ok({"hello": "world"}).model_dump()
    assert set(payload.keys()) == {"ok", "data", "error"}
    assert payload["ok"] is True
    assert payload["data"] == {"hello": "world"}
    assert payload["error"] is None


def test_error_envelope_serializes_to_expected_keys() -> None:
    payload = err("x", "y", details={"k": "v"}).model_dump()
    assert payload == {"ok": False, "data": None, "error": {"code": "x", "message": "y", "details": {"k": "v"}}}


def test_error_body_is_frozen() -> None:
    body = ErrorBody(code="x", message="y")
    with pytest.raises(ValueError, match="frozen"):
        body.code = "z"  # type: ignore[misc]


# --- Exception handler matrix ---------------------------------------------


@pytest.fixture
def app_with_handlers() -> FastAPI:
    """Build a tiny app whose routes each raise one of the exceptions."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/raise/parse")
    def _raise_parse() -> None:
        raise ParseError("bad parser output")

    @app.get("/raise/guardrail")
    def _raise_guardrail() -> None:
        raise RankingGuardrailViolation("paid placement detected")

    @app.get("/raise/db")
    def _raise_db() -> None:
        raise DbError("connection lost")

    @app.get("/raise/budget")
    def _raise_budget() -> None:
        raise LlmBudgetExceededError("daily cap hit")

    @app.get("/raise/llm")
    def _raise_llm() -> None:
        raise LlmError("upstream 502")

    @app.get("/raise/base")
    def _raise_base() -> None:
        raise IngovoError("something broke")

    return app


@pytest.fixture
def client(app_with_handlers: FastAPI) -> TestClient:
    return TestClient(app_with_handlers)


def test_parse_error_maps_to_400(client: TestClient) -> None:
    response = client.get("/raise/parse")
    assert response.status_code == 400
    body = response.json()
    assert body == {
        "ok": False,
        "data": None,
        "error": {"code": "invalid_input", "message": "bad parser output", "details": None},
    }


def test_ranking_guardrail_maps_to_500(client: TestClient) -> None:
    response = client.get("/raise/guardrail")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ranking_guardrail"


def test_db_error_maps_to_500(client: TestClient) -> None:
    response = client.get("/raise/db")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "db_error"


def test_budget_exceeded_maps_to_402(client: TestClient) -> None:
    response = client.get("/raise/budget")
    assert response.status_code == 402
    assert response.json()["error"]["code"] == "llm_budget_exceeded"


def test_llm_error_maps_to_502(client: TestClient) -> None:
    response = client.get("/raise/llm")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_error"


def test_base_ingovo_error_maps_to_500(client: TestClient) -> None:
    response = client.get("/raise/base")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


def test_parse_error_takes_precedence_over_llm_error(client: TestClient) -> None:
    """`ParseError` is a subclass of `LlmError` — handler must match the subclass first."""
    response = client.get("/raise/parse")
    assert response.json()["error"]["code"] == "invalid_input"


def test_budget_exceeded_takes_precedence_over_llm_error(client: TestClient) -> None:
    """`LlmBudgetExceededError` is also a subclass of `LlmError`."""
    response = client.get("/raise/budget")
    assert response.json()["error"]["code"] == "llm_budget_exceeded"


def test_http_exception_maps_to_the_envelope_with_its_status() -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/forbidden")
    def _forbidden() -> None:
        raise HTTPException(status_code=403, detail="no agency")

    response = TestClient(app).get("/forbidden")
    assert response.status_code == 403
    assert response.json()["error"] == {"code": "http_403", "message": "no agency", "details": None}


async def test_the_http_exception_handler_refuses_any_other_exception() -> None:
    """Starlette routes only `HTTPException` here; anything else is a wiring bug, not a 500 envelope."""
    app = FastAPI()
    install_exception_handlers(app)
    handled = app.exception_handlers[StarletteHTTPException](Request({"type": "http"}), ValueError("boom"))
    assert inspect.isawaitable(handled)
    with pytest.raises(TypeError, match="ValueError"):
        await handled


# --- Smoke: install_exception_handlers does not break an OK path ---------


def test_install_exception_handlers_does_not_alter_success_response() -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/ok")
    def _ok() -> dict[str, str]:
        return {"hello": "world"}

    response = TestClient(app).get("/ok")
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}


def test_envelope_generic_typed_data_round_trips() -> None:
    env = Envelope[_Payload](ok=True, data=_Payload(value=7), error=None)
    restored = Envelope[_Payload].model_validate(env.model_dump())
    assert restored.data == _Payload(value=7)
