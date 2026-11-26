"""FastAPI app factory: the search routes plus CORS, rate limits and request ids.

Run it with `uv run ingovo-search-api` (development server with reload) or
`uvicorn ingovo_search.api.main:create_app --factory`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ingovo_search.api.deps import redis_configured
from ingovo_search.api.envelope import install_exception_handlers
from ingovo_search.api.middleware.rate_limit import RateLimitMiddleware
from ingovo_search.api.middleware.request_id import RequestIdMiddleware
from ingovo_search.api.refinement import router as refinement_router
from ingovo_search.api.search import router as search_router
from ingovo_search.api.search_preview import router as search_preview_router
from ingovo_search.api.search_probe import router as search_probe_router
from ingovo_search.api.search_stream import router as search_stream_router
from ingovo_search.lib.env import Settings, get_settings
from ingovo_search.lib.observability import configure_logging
from ingovo_search.search.shadow import drain_rerank_shadow_tasks

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    configure_logging(env=settings.env)
    logger = structlog.get_logger("ingovo_search.api.main")
    if settings.uses_ephemeral_session_key:
        logger.warning("SESSION_SIGNING_KEY is unset; sessions use a per-process key")
    if settings.ai_search_rerank_backend in ("jev", "canary", "shadow") and settings.typesafe_api_key is None:
        # Not fatal: the Gemini rerank serves instead.
        logger.warning(
            "jev rerank configured without TYPESAFE_API_KEY; the Gemini rerank serves",
            rerank_backend=settings.ai_search_rerank_backend,
        )
    yield
    # Let in-flight shadow reranks (AI_SEARCH_RERANK_BACKEND=shadow) write their
    # comparison row; each is bounded by the Jev timeout, so this is short.
    await drain_rerank_shadow_tasks()


async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# In registration order: the order of `app.routes` and of the OpenAPI paths.
_ROUTERS: tuple[APIRouter, ...] = (
    search_router,
    search_stream_router,
    search_preview_router,
    search_probe_router,
    refinement_router,
)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ingovo-search", version=APP_VERSION, lifespan=lifespan)
    _add_cors(app, settings)
    install_exception_handlers(app)
    _add_rate_limits(app, settings)
    # Added LAST so it is the OUTERMOST middleware (Starlette runs the last-added
    # first): the correlation id is bound before the rate limiter logs anything.
    app.add_middleware(RequestIdMiddleware)

    app.add_api_route("/healthz", healthz, methods=["GET"])
    for router in _ROUTERS:
        app.include_router(router)
    return app


def _add_cors(app: FastAPI, settings: Settings) -> None:
    """Explicit origin allowlist (ALLOWED_ORIGINS); local addresses are allowed in development."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?" if settings.env == "development" else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _add_rate_limits(app: FastAPI, settings: Settings) -> None:
    """Two sliding-window limiters: a tight one on the LLM-backed search routes, a permissive global one.

    The search cap is 20 requests per minute per client: one search every ~3 s, well
    above human pace. Anything beyond is bot-like and would burn model quota. Each
    limiter has its own `key_prefix`, so they do not share a counter. Both need Redis
    and are off when it is not configured or under `ENV=test`.
    """
    enabled = redis_configured() and settings.env != "test"
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=20,
        window_seconds=60,
        path_prefix="/api/search",
        key_prefix="rate_limit_search",
        enabled=enabled,
    )
    app.add_middleware(RateLimitMiddleware, max_requests=200, window_seconds=60, enabled=enabled)


def run() -> None:
    """`uv run ingovo-search-api` — the development server with autoreload."""
    uvicorn.run("ingovo_search.api.main:create_app", factory=True, host="127.0.0.1", port=8000, reload=True)
