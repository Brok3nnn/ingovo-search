"""The app lifespan: it warns about a Jev mode without a key and drains shadow reranks on shutdown."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from ingovo_search.api.main import lifespan


async def test_shutdown_drains_in_flight_shadow_reranks() -> None:
    settings = MagicMock(env="test", ai_search_rerank_backend="gemini")
    with (
        patch("ingovo_search.api.main.get_settings", return_value=settings),
        patch("ingovo_search.api.main.configure_logging"),
        patch("ingovo_search.api.main.drain_rerank_shadow_tasks", new_callable=AsyncMock) as drain,
    ):
        async with lifespan(FastAPI()):
            drain.assert_not_awaited()
        drain.assert_awaited_once()


async def test_startup_warns_when_a_jev_mode_has_no_key() -> None:
    settings = MagicMock(
        env="test",
        uses_ephemeral_session_key=False,
        ai_search_rerank_backend="jev",
        typesafe_api_key=None,
    )
    logger = MagicMock()
    with (
        patch("ingovo_search.api.main.get_settings", return_value=settings),
        patch("ingovo_search.api.main.configure_logging"),
        patch("ingovo_search.api.main.drain_rerank_shadow_tasks", new_callable=AsyncMock),
        patch("ingovo_search.api.main.structlog.get_logger", return_value=logger),
    ):
        async with lifespan(FastAPI()):
            pass

    logger.warning.assert_called_once()
    assert "TYPESAFE_API_KEY" in logger.warning.call_args.args[0]
