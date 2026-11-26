"""Settings, read from the environment (and an optional `.env` file) via pydantic-settings.

Every variable has a default that works against the local `docker compose` stack, so the
API, the CLI and the seed command boot without any configuration. The two model keys are
optional at boot and checked when a model is actually called: the deterministic parser
fast path and the offline tests never need them.
"""

from __future__ import annotations

import secrets
import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

ASYNC_PG_UNSUPPORTED_QUERY_KEYS = frozenset({"connection_limit", "pgbouncer"})

# Matches docker-compose.yml: Postgres on localhost:5432, database and role `ingovo`.
DEFAULT_DATABASE_URL = "postgresql+asyncpg://ingovo@localhost:5432/ingovo"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class Settings(BaseSettings):
    """All environment variables the search service reads. Loaded once per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    _ephemeral_session_key: bool = PrivateAttr(default=False)

    # Database (Postgres with PostGIS and pgvector)
    database_url: str = Field(default=DEFAULT_DATABASE_URL, alias="DATABASE_URL", min_length=1)

    # Model providers. Optional at boot; a call without its key raises `LlmError`.
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")

    # Cache and session store. Empty means an in-process memory cache (single worker only).
    redis_url: str | None = Field(default=DEFAULT_REDIS_URL, alias="REDIS_URL")

    # HMAC secret for the anonymous session cookie. Required in production; outside
    # production a random per-process key is generated when it is unset.
    session_signing_key: str = Field(default="", alias="SESSION_SIGNING_KEY")

    # Budget: optional LLM work (narratives, then the rerank) is switched off as the
    # day's logged spend approaches this many US dollars. 0 disables the guard.
    llm_daily_budget_usd: float = Field(default=25.0, alias="LLM_DAILY_BUDGET_USD")

    # Skip the LLM rerank when the deterministic order is already clearly good
    # (simple query, clear top). Disable to force the rerank on every semantic search.
    ai_search_conditional_rerank: bool = Field(default=True, alias="AI_SEARCH_CONDITIONAL_RERANK")
    # Rank-only rerank: the same ordering signal without per-listing rationales,
    # which costs far fewer output tokens.
    ai_search_rerank_rank_only: bool = Field(default=False, alias="AI_SEARCH_RERANK_RANK_ONLY")
    # Safety net against a hung parser call, not a latency knob: a healthy LLM parse
    # finishes in about 1.5 s, and the fast path plus the cache make simple and repeated
    # queries nearly free. On a timeout the deterministic fallback parse is used.
    ai_search_parser_timeout_ms: int = Field(default=2500, alias="AI_SEARCH_PARSER_TIMEOUT_MS")
    # Ceiling on the vision call that reads attached photos. It runs concurrently with
    # the parser, so it bounds the slower of the two rather than adding to it.
    ai_search_image_timeout_ms: int = Field(default=15_000, alias="AI_SEARCH_IMAGE_TIMEOUT_MS")
    # Ceiling on the Gemini rerank. A rerank of 12 candidates with a rationale each
    # measured about 3.1 s in the original deployment; on a timeout the
    # deterministic order is served.
    ai_search_rerank_timeout_ms: int = Field(default=3500, alias="AI_SEARCH_RERANK_TIMEOUT_MS")

    # Optional second reranker: TypeSafe's Jev model (`lib/jev.py`). `gemini` serves the
    # Gemini rerank only; `jev` tries Jev first and falls back to Gemini whenever Jev
    # cannot serve (no key, rate limit, outage); `shadow` serves Gemini and runs Jev
    # alongside for comparison; `canary` serves Jev to AI_SEARCH_RERANK_JEV_PERCENT % of
    # sessions; `off` skips the rerank entirely.
    typesafe_api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY", min_length=1)
    ai_search_rerank_backend: Literal["gemini", "shadow", "canary", "jev", "off"] = Field(
        default="gemini", alias="AI_SEARCH_RERANK_BACKEND"
    )
    ai_search_rerank_jev_percent: int = Field(default=0, alias="AI_SEARCH_RERANK_JEV_PERCENT", ge=0, le=100)
    ai_search_rerank_jev_timeout_ms: int = Field(default=1200, alias="AI_SEARCH_RERANK_JEV_TIMEOUT_MS", gt=0)
    ai_search_rerank_shadow_sample: float = Field(default=1.0, alias="AI_SEARCH_RERANK_SHADOW_SAMPLE", ge=0.0, le=1.0)
    # After Jev fails with anything but a timeout, Gemini serves for this long
    # before Jev is tried again, so a rate-limited Jev is not hit on every search.
    ai_search_rerank_jev_cooldown_s: float = Field(default=60.0, alias="AI_SEARCH_RERANK_JEV_COOLDOWN_S", ge=0.0)

    # Ceiling on the hero-narrative phase. It fans out to a few concurrent calls, so
    # this bounds the slowest one; on a timeout each row keeps a heuristic narrative.
    ai_search_narrator_timeout_ms: int = Field(default=6000, alias="AI_SEARCH_NARRATOR_TIMEOUT_MS")

    # Mode
    env: Literal["development", "production", "test"] = Field(default="development", alias="ENV")

    # How many trusted reverse proxies sit in front of the app. `X-Forwarded-For` is
    # client-writable, so with N trusted proxies the real peer is the Nth entry from the
    # right. 0 ignores the header and rate-limits the socket peer.
    rate_limit_trusted_proxy_hops: int = Field(default=0, alias="RATE_LIMIT_TRUSTED_PROXY_HOPS", ge=0)

    # CORS: comma-separated allowlist of browser origins. A bare "*" is dropped,
    # because a wildcard combined with credentials is invalid per the CORS spec.
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="ALLOWED_ORIGINS",
    )

    @field_validator("typesafe_api_key", "gemini_api_key", "cohere_api_key", "redis_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """`KEY=` (what docker compose passes for an unset variable) means no value."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _session_key(self) -> Settings:
        """Require a real signing key in production; generate an ephemeral one elsewhere."""
        if len(self.session_signing_key) >= 32:
            return self
        if self.env == "production":
            raise ValueError("SESSION_SIGNING_KEY must be at least 32 characters in production")
        if self.session_signing_key:
            raise ValueError("SESSION_SIGNING_KEY must be at least 32 characters")
        # Sessions then last for one process only, which is fine for development.
        self.session_signing_key = secrets.token_urlsafe(48)
        self._ephemeral_session_key = True
        return self

    @property
    def uses_ephemeral_session_key(self) -> bool:
        """True when SESSION_SIGNING_KEY was unset and a per-process key was generated."""
        return self._ephemeral_session_key

    @property
    def cors_allow_origins(self) -> list[str]:
        """Parsed, de-duplicated CORS origin allowlist (the bare '*' is never returned)."""
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip() and o.strip() != "*"]
        return list(dict.fromkeys(origins))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the singleton Settings. Raises Pydantic ValidationError on malformed env."""
    return Settings()  # pyright: ignore[reportCallIssue] pydantic-settings populates fields from env.


def sqlalchemy_asyncpg_url(database_url: str) -> str:
    """Normalise a Postgres URL for SQLAlchemy's asyncpg driver.

    A plain `postgresql://` URL gets the `+asyncpg` driver; query options asyncpg
    does not understand (pooler hints such as `pgbouncer=true`) are dropped.
    """
    url = make_url(database_url)
    if url.drivername in {"postgresql", "postgres"}:
        url = url.set(drivername="postgresql+asyncpg")
    if url.drivername != "postgresql+asyncpg":
        return database_url

    query = {key: value for key, value in url.query.items() if key not in ASYNC_PG_UNSUPPORTED_QUERY_KEYS}
    return url.set(query=query).render_as_string(hide_password=False)


def check() -> None:
    """CLI entrypoint: validates env vars and prints OK or fails loudly with exit code 1."""
    try:
        settings = get_settings()
    except PydanticValidationError as exc:
        print(f"env: FAIL — {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    print(f"env: OK ({settings.env})")  # noqa: T201
    sys.exit(0)


if __name__ == "__main__":
    check()
