"""§14 step 3 — anonymous session state.

Each browser carries a signed cookie holding an opaque session id; server-side
state keyed by that id lives in Redis (via the §13 `Cache`) with a 7-day
rolling TTL. State accumulates:

- recent `ParsedQuery` values (last `MAX_RECENT_QUERIES`, FIFO),
- `RefinementOverrides` from the latest /api/search/clarify call,
- in-session feedback signals (`SessionSignal` — `view_short` / `view_long` /
  `save` / `favorite` / `hide` / `thumb_up` / `thumb_down` / `contact` /
  `viewing_booked`, vocabulary mirrors `recommender.pooled_signal`),
- previously-answered `ClarificationAnswer` items,
- `w_*` overrides from the trade-off-button affordance.

A signed-out user's state is *not* mixed with another anonymous user's: the
cookie signature is the only token of trust. On account creation the routers
promote the state into a persistent `UserPreferences` row (§14e) and call
`clear`. An explicit "start fresh" UI action also calls `clear`.

The whole module is pure: no FastAPI imports, no database imports, just the
§13 `Cache` Protocol and Pydantic. Routers wire `SessionStore` against the
production Redis cache; tests pass `MemoryCache`.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

from ingovo_search.lib.cache import Cache
from ingovo_search.recommender.clarification_policy import ClarificationAnswer
from ingovo_search.recommender.parser_schema import (
    Anchors,
    GeoHints,
    HardFilters,
    ParsedQuery,
    SoftPrefTag,
    public_anchors_json,
    public_hard_filters_json,
)

# --- Constants -------------------------------------------------------------

SESSION_TTL_SECONDS: int = 7 * 24 * 60 * 60
"""7-day rolling TTL — re-written on every `update` so an active session stays alive."""

MAX_RECENT_QUERIES: int = 20
"""FIFO cap on `SessionState.recent_parsed_queries`."""

MAX_SESSION_SIGNALS: int = 200
"""FIFO cap on `SessionState.session_signals`."""

MAX_CLARIFICATION_ANSWERS: int = 50
"""FIFO cap on `SessionState.clarification_answers`."""

SESSION_ID_LENGTH_BYTES: int = 32
"""`secrets.token_urlsafe(SESSION_ID_LENGTH_BYTES)` → 43 url-safe chars."""

CACHE_KEY_PREFIX: str = "session:"

# Signal-type vocab mirrors `recommender.pooled_signal.SIGNAL_WEIGHTS` keys.
# `refinement_*` events never join the pool and never land here either; they
# travel via `RefinementOverrides` instead.
SignalType = Literal[
    "thumb_up",
    "thumb_down",
    "favorite",
    "save",
    "contact",
    "viewing_booked",
    "view_long",
    "view_short",
    "hide",
]


# --- Schema ----------------------------------------------------------------


class SessionSignal(BaseModel):
    """One in-session feedback event on a listing."""

    model_config = {"frozen": True}

    listing_id: str
    signal_type: SignalType
    timestamp: datetime


class RefinementOverrides(BaseModel):
    """Per-search refinement layer — the §14 step-6 trade-off-button affordance.

    Mirrors the design spec §14 step 6's `RefinementOverrides`. Each field
    is optional: a `None` means "leave the parsed-query field as parsed", a
    value means "replace it for this search". The four parser-schema models
    are already structurally `Partial`-compatible (every field defaults to
    `None`), so we reuse them rather than introducing parallel `*_Partial`
    types.
    """

    model_config = {"frozen": True}

    hard_filters: HardFilters | None = None
    anchors: Anchors | None = None
    soft_preferences: list[SoftPrefTag] | None = None
    soft_prefs_freetext: list[str] | None = None
    geo_hints: GeoHints | None = None

    @field_serializer("hard_filters", when_used="json")
    def serialize_hard_filters(self, value: HardFilters | None) -> dict[str, Any] | None:
        """Do not round an exact HUF override before the browser can replay it."""

        return public_hard_filters_json(value) if value is not None else None

    @field_serializer("anchors", when_used="json")
    def serialize_anchors(self, value: Anchors | None) -> dict[str, Any] | None:
        """Do not round an exact HUF anchor override before the browser can replay it."""

        return public_anchors_json(value) if value is not None else None


def _empty_query_list() -> list[ParsedQuery]:
    return []


def _empty_signal_list() -> list[SessionSignal]:
    return []


def _empty_answer_list() -> list[ClarificationAnswer]:
    return []


def _empty_w_overrides() -> dict[str, float]:
    return {}


class SessionState(BaseModel):
    """Server-side anonymous-session state, JSON-serialized into Redis."""

    model_config = {"frozen": True}

    session_id: str
    created_at: datetime
    updated_at: datetime
    recent_parsed_queries: list[ParsedQuery] = Field(default_factory=_empty_query_list)
    refinement_overrides: RefinementOverrides | None = None
    session_signals: list[SessionSignal] = Field(default_factory=_empty_signal_list)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=_empty_answer_list)
    w_overrides: dict[str, float] = Field(default_factory=_empty_w_overrides)


# --- Cookie signing --------------------------------------------------------


def new_session_id() -> str:
    """Cryptographically random url-safe session id."""
    return secrets.token_urlsafe(SESSION_ID_LENGTH_BYTES)


def sign_session_id(session_id: str, *, secret: bytes) -> str:
    """Produce `f"{session_id}.{hex_hmac_sha256}"` cookie payload."""
    if not session_id:
        raise ValueError("session_id must be non-empty")
    if len(secret) < 32:
        raise ValueError("secret must be at least 32 bytes")
    mac = hmac.new(secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{session_id}.{mac}"


def verify_signed_session_id(signed: str, *, secret: bytes) -> str | None:
    """Return the session id if the HMAC verifies; `None` on tamper / malformed.

    Uses `hmac.compare_digest` so verification is timing-attack resistant.
    """
    if not signed or "." not in signed:
        return None
    session_id, _, sig = signed.partition(".")
    if not session_id or not sig:
        return None
    expected = hmac.new(secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return session_id


# --- Store -----------------------------------------------------------------


def _cache_key(session_id: str) -> str:
    return f"{CACHE_KEY_PREFIX}{session_id}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SessionStore:
    """Get / update / clear / create against a §13 `Cache`.

    Every `update` rewrites the entry with a fresh TTL — the spec's "7-day
    rolling" semantic. `get` does not refresh on its own; reads cost one
    Redis op and don't extend lifetime. Sessions die after 7 idle days
    even if the browser keeps GET-polling, which is the security-side
    desired behaviour. Active users naturally generate writes (search,
    feedback) so their state stays warm.
    """

    def __init__(self, cache: Cache, *, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._cache = cache
        self._ttl = ttl_seconds

    async def get(self, session_id: str) -> SessionState | None:
        raw = await self._cache.get(_cache_key(session_id))
        if raw is None:
            return None
        return SessionState.model_validate_json(raw)

    async def update(self, state: SessionState) -> SessionState:
        """Bump `updated_at`, enforce FIFO caps, write with rolling TTL."""
        bounded = _enforce_caps(state).model_copy(update={"updated_at": _utcnow()})
        payload = bounded.model_dump_json().encode("utf-8")
        await self._cache.set(_cache_key(bounded.session_id), payload, ttl_seconds=self._ttl)
        return bounded

    async def clear(self, session_id: str) -> None:
        """Evict the entry — used by "start fresh" and post-account-creation handoff."""
        await self._cache.delete(_cache_key(session_id))

    async def create(self) -> SessionState:
        """Mint a fresh session and persist its initial state."""
        now = _utcnow()
        state = SessionState(session_id=new_session_id(), created_at=now, updated_at=now)
        payload = state.model_dump_json().encode("utf-8")
        await self._cache.set(_cache_key(state.session_id), payload, ttl_seconds=self._ttl)
        return state


def _enforce_caps(state: SessionState) -> SessionState:
    """Trim oldest entries so FIFO caps hold."""
    needs_trim = (
        len(state.session_signals) > MAX_SESSION_SIGNALS
        or len(state.recent_parsed_queries) > MAX_RECENT_QUERIES
        or len(state.clarification_answers) > MAX_CLARIFICATION_ANSWERS
    )
    if not needs_trim:
        return state
    return state.model_copy(
        update={
            "session_signals": list(state.session_signals[-MAX_SESSION_SIGNALS:]),
            "recent_parsed_queries": list(state.recent_parsed_queries[-MAX_RECENT_QUERIES:]),
            "clarification_answers": list(state.clarification_answers[-MAX_CLARIFICATION_ANSWERS:]),
        }
    )


__all__ = [
    "CACHE_KEY_PREFIX",
    "MAX_CLARIFICATION_ANSWERS",
    "MAX_RECENT_QUERIES",
    "MAX_SESSION_SIGNALS",
    "SESSION_ID_LENGTH_BYTES",
    "SESSION_TTL_SECONDS",
    "RefinementOverrides",
    "SessionSignal",
    "SessionState",
    "SessionStore",
    "SignalType",
    "new_session_id",
    "sign_session_id",
    "verify_signed_session_id",
]
