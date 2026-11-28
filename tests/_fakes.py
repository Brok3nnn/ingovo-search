"""Test doubles shared across the API and golden tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def fake_db_session() -> AsyncMock:
    """A fresh `AsyncSession` double for one request.

    Its async methods (`execute`, `commit`, `rollback`) are `AsyncMock`s. `add` is
    synchronous on a real session, so here it is a plain `MagicMock`; a bare
    `AsyncMock()` would turn every `session.add(row)` into a coroutine that is
    never awaited. FastAPI resolves a dependency override's own signature, so this
    parameterless factory can be passed as the override itself.
    """
    return AsyncMock(add=MagicMock())
