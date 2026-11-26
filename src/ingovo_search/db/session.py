"""Async SQLAlchemy session factory. Used as a FastAPI dependency once routes exist (§14)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        sqlalchemy_asyncpg_url(settings.database_url),
        echo=False,
        future=True,
        # Hosted connection poolers (pgbouncer and the like) drop idle server connections,
        # so a pooled connection can be dead by the next request. `pool_pre_ping`
        # validates (and transparently reconnects) before use; `pool_recycle`
        # discards connections older than the pooler's idle timeout. Without
        # these, the first query after an idle gap fails with
        # "connection was closed in the middle of operation" -> surfaced as the
        # generic "hard filter query failed" 500.
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args=asyncpg_connect_args(),
    )


@lru_cache(maxsize=1)
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an async session and rolls back on exception."""
    async with get_async_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def asyncpg_connect_args() -> dict[str, Any]:
    return {
        "prepared_statement_cache_size": 0,
        "prepared_statement_name_func": lambda: f"__ingovo_{uuid4().hex}__",
        "statement_cache_size": 0,
    }
