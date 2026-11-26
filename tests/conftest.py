"""Shared pytest fixtures.

Unit tests never call a real model and never write to a real database: the model
clients are `MockLlmClient` / `MockEmbeddingClient`, and the database is either an
in-memory stub or a recorded fake session. Tests marked `integration` need a local
Postgres (and sometimes Redis) and are deselected by default.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm import MockLlmClient


class AsyncDbSessionStub:
    """Small no-DB session double for unit tests that only need to observe calls."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.executed: list[tuple[object, object | None]] = []
        self.committed = False
        self.rolled_back = False

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object, params: object | None = None) -> list[object]:
        self.executed.append((statement, params))
        return []

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the settings every test runs with, whatever the developer's shell or `.env` says.

    Environment variables outrank the `.env` file, so these keep real keys and the
    optional Jev reranker out of the suite: the Gemini rerank path is the default, the
    model keys are blank (so an accidental real client fails fast with "key is not set"
    instead of reaching the network), Redis is off (in-process memory cache) and
    `ENV=test` disables the rate limiter. `get_settings()` is cached, so the cache is
    cleared on both sides of every test.
    """
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("AI_SEARCH_RERANK_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("COHERE_API_KEY", "")
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("SESSION_SIGNING_KEY", "test-session-signing-key-0123456789abcdef")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_usage_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the background LLM cost rows (`lib/llm_cost_log.py`) out of any database.

    Tests that check those rows switch the writer back on and patch it.
    """
    monkeypatch.setenv("AI_USAGE_LOG", "off")


@pytest.fixture
def async_db_session() -> AsyncDbSessionStub:
    return AsyncDbSessionStub()


@pytest.fixture
def mock_llm() -> MockLlmClient:
    """Empty MockLlmClient — tests add fixtures via `.add()`."""
    return MockLlmClient()


def pytest_addoption(parser: pytest.Parser) -> None:
    """`--update-golden` rewrites the golden snapshots under tests/golden/snapshots."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite the golden snapshots instead of comparing against them",
    )
