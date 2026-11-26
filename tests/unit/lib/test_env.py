"""Settings: defaults that boot against the local stack, validation, and URL normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from ingovo_search.lib.env import DEFAULT_DATABASE_URL, Settings, sqlalchemy_asyncpg_url

_ALL_KEYS = (
    "DATABASE_URL",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "REDIS_URL",
    "SESSION_SIGNING_KEY",
    "ENV",
    "LLM_DAILY_BUDGET_USD",
    "TYPESAFE_API_KEY",
    "AI_SEARCH_RERANK_BACKEND",
    "AI_SEARCH_RERANK_JEV_PERCENT",
    "AI_SEARCH_RERANK_JEV_TIMEOUT_MS",
    "AI_SEARCH_RERANK_SHADOW_SAMPLE",
    "AI_SEARCH_RERANK_JEV_COOLDOWN_S",
)


@pytest.fixture
def empty_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No variables set, and an empty working directory so no `.env` file is read."""
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def _settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue] pydantic-settings populates fields from env.


@pytest.mark.usefixtures("empty_env")
def test_settings_boot_with_no_configuration() -> None:
    settings = _settings()

    assert settings.env == "development"
    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.gemini_api_key is None
    assert settings.cohere_api_key is None
    assert settings.llm_daily_budget_usd == 25.0
    assert settings.ai_search_rerank_backend == "gemini"


@pytest.mark.usefixtures("empty_env")
def test_an_unset_signing_key_is_replaced_by_an_ephemeral_one_outside_production() -> None:
    first = _settings().session_signing_key
    second = _settings().session_signing_key

    assert len(first) >= 32
    assert first != second


@pytest.mark.usefixtures("empty_env")
def test_production_requires_a_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    with pytest.raises(PydanticValidationError, match="SESSION_SIGNING_KEY"):
        _settings()


@pytest.mark.usefixtures("empty_env")
def test_a_short_signing_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_SIGNING_KEY", "too-short")
    with pytest.raises(PydanticValidationError, match="32 characters"):
        _settings()


@pytest.mark.usefixtures("empty_env")
@pytest.mark.parametrize("key", ["GEMINI_API_KEY", "COHERE_API_KEY", "TYPESAFE_API_KEY", "REDIS_URL"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_value_means_unset(monkeypatch: pytest.MonkeyPatch, key: str, blank: str) -> None:
    # docker compose passes `KEY=` for a variable the host never set.
    monkeypatch.setenv(key, blank)
    settings = _settings()
    assert getattr(settings, key.lower()) is None


def test_sqlalchemy_asyncpg_url_removes_pooler_flags() -> None:
    url = sqlalchemy_asyncpg_url(
        "postgresql+asyncpg://db.example.test:6543/postgres?pgbouncer=true&connection_limit=1&sslmode=require"
    )

    assert "pgbouncer" not in url
    assert "connection_limit" not in url
    assert "sslmode=require" in url


def test_sqlalchemy_asyncpg_url_adds_the_async_driver() -> None:
    assert sqlalchemy_asyncpg_url("postgresql://localhost:5432/ingovo") == "postgresql+asyncpg://localhost:5432/ingovo"


def test_cors_origins_drop_the_wildcard_and_duplicates() -> None:
    settings = Settings.model_validate(
        {"ALLOWED_ORIGINS": "https://a.example, *, https://a.example,https://b.example", "ENV": "test"}
    )
    assert settings.cors_allow_origins == ["https://a.example", "https://b.example"]


@pytest.mark.usefixtures("empty_env")
@pytest.mark.parametrize("backend", ["shadow", "canary", "jev"])
def test_jev_backends_boot_without_a_key_and_leave_the_rerank_to_gemini(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    # A missing key must never stop the API: the search serves the Gemini rerank.
    monkeypatch.setenv("AI_SEARCH_RERANK_BACKEND", backend)
    settings = _settings()
    assert settings.ai_search_rerank_backend == backend
    assert settings.typesafe_api_key is None


@pytest.mark.usefixtures("empty_env")
def test_jev_settings_defaults() -> None:
    settings = _settings()
    assert settings.ai_search_rerank_jev_percent == 0
    assert settings.ai_search_rerank_jev_timeout_ms == 1200
    assert settings.ai_search_rerank_shadow_sample == 1.0
    assert settings.ai_search_rerank_jev_cooldown_s == 60.0


@pytest.mark.usefixtures("empty_env")
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AI_SEARCH_RERANK_BACKEND", "unknown"),
        ("AI_SEARCH_RERANK_JEV_PERCENT", "101"),
        ("AI_SEARCH_RERANK_JEV_PERCENT", "-1"),
        ("AI_SEARCH_RERANK_JEV_TIMEOUT_MS", "0"),
        ("AI_SEARCH_RERANK_SHADOW_SAMPLE", "1.5"),
        ("AI_SEARCH_RERANK_JEV_COOLDOWN_S", "-1"),
    ],
)
def test_out_of_range_values_are_rejected(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(PydanticValidationError):
        _settings()
