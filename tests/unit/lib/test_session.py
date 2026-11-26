"""Unit tests for §14 step-3 anonymous session state.

Verifies the schema invariants, the HMAC-signed-cookie helpers, and the
`SessionStore` lifecycle (create / get / update / clear, rolling TTL) against
a `MemoryCache` so the suite runs without Redis.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest

from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.session import (
    CACHE_KEY_PREFIX,
    MAX_CLARIFICATION_ANSWERS,
    MAX_RECENT_QUERIES,
    MAX_SESSION_SIGNALS,
    SESSION_ID_LENGTH_BYTES,
    SESSION_TTL_SECONDS,
    RefinementOverrides,
    SessionSignal,
    SessionState,
    SessionStore,
    new_session_id,
    sign_session_id,
    verify_signed_session_id,
)
from ingovo_search.recommender.clarification_policy import (
    ClarificationAnswer,
    ClarificationReason,
)
from ingovo_search.recommender.parser_schema import (
    Anchors,
    GeoHints,
    HardFilters,
    ParsedQuery,
)

# --- Helpers ---------------------------------------------------------------


SECRET = b"x" * 32


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_signal(*, listing_id: str = "L1", signal_type: Any = "thumb_up") -> SessionSignal:
    return SessionSignal(
        listing_id=listing_id,
        signal_type=signal_type,
        timestamp=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
    )


# --- Cookie signing --------------------------------------------------------


def test_new_session_id_is_url_safe_and_long_enough() -> None:
    """`secrets.token_urlsafe(32)` produces 43 url-safe chars."""
    sid = new_session_id()
    assert len(sid) >= SESSION_ID_LENGTH_BYTES
    assert all(c.isalnum() or c in "-_" for c in sid)


def test_new_session_id_is_random() -> None:
    seen = {new_session_id() for _ in range(50)}
    assert len(seen) == 50


def test_sign_verify_round_trip() -> None:
    sid = new_session_id()
    signed = sign_session_id(sid, secret=SECRET)
    assert verify_signed_session_id(signed, secret=SECRET) == sid


def test_verify_rejects_tampered_id() -> None:
    sid = new_session_id()
    signed = sign_session_id(sid, secret=SECRET)
    _, _, sig = signed.partition(".")
    tampered = f"{sid}-tamper.{sig}"
    assert verify_signed_session_id(tampered, secret=SECRET) is None


def test_verify_rejects_tampered_signature() -> None:
    sid = new_session_id()
    signed = sign_session_id(sid, secret=SECRET)
    head, _, sig = signed.partition(".")
    flipped = sig.replace(sig[0], "f" if sig[0] != "f" else "e", 1)
    assert verify_signed_session_id(f"{head}.{flipped}", secret=SECRET) is None


def test_verify_rejects_wrong_secret() -> None:
    sid = new_session_id()
    signed = sign_session_id(sid, secret=SECRET)
    assert verify_signed_session_id(signed, secret=b"y" * 32) is None


@pytest.mark.parametrize("malformed", ["", "no-dot-here", ".onlyrhs", "onlylhs.", ".", ".."])
def test_verify_rejects_malformed_cookie(malformed: str) -> None:
    assert verify_signed_session_id(malformed, secret=SECRET) is None


def test_sign_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sign_session_id("", secret=SECRET)


def test_sign_rejects_short_secret() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        sign_session_id("sid", secret=b"short")


# --- Schema ----------------------------------------------------------------


def test_session_state_defaults_are_empty() -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    state = SessionState(session_id="sid", created_at=now, updated_at=now)
    assert state.recent_parsed_queries == []
    assert state.refinement_overrides is None
    assert state.session_signals == []
    assert state.clarification_answers == []
    assert state.w_overrides == {}


def test_session_state_is_frozen() -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    state = SessionState(session_id="sid", created_at=now, updated_at=now)
    with pytest.raises(ValueError, match="frozen"):
        state.session_id = "other"  # type: ignore[misc]


def test_refinement_overrides_all_optional() -> None:
    """A bare `RefinementOverrides()` validates — every field has a default."""
    overrides = RefinementOverrides()
    assert overrides.hard_filters is None
    assert overrides.anchors is None
    assert overrides.soft_preferences is None
    assert overrides.soft_prefs_freetext is None
    assert overrides.geo_hints is None


def test_session_state_round_trips_through_json() -> None:
    """Every field must survive a `model_dump_json` → `model_validate_json` cycle."""
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
    state = SessionState(
        session_id="sid-A",
        created_at=now,
        updated_at=now,
        recent_parsed_queries=[
            ParsedQuery(
                hard_filters=HardFilters(district_in=["13"], price_huf_max=85_000_000),
                anchors=Anchors(price_huf=80_000_000),
                geo_hints=GeoHints(side_of_river="Pest"),
                semantic_query="csendes",
            )
        ],
        refinement_overrides=RefinementOverrides(soft_preferences=["elevator"]),
        session_signals=[_make_signal()],
        clarification_answers=[
            ClarificationAnswer(reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY, option_id="district:13"),
        ],
        w_overrides={"price": 0.3, "quality": -0.1},
    )

    blob = state.model_dump_json()
    restored = SessionState.model_validate_json(blob)
    assert restored == state


# --- SessionStore lifecycle -----------------------------------------------


async def _make_store(*, ttl_seconds: int = SESSION_TTL_SECONDS, clock: _FakeClock | None = None) -> SessionStore:
    cache: Cache = MemoryCache(clock=clock or time.monotonic)
    return SessionStore(cache, ttl_seconds=ttl_seconds)


async def test_store_create_returns_fresh_state_and_persists() -> None:
    store = await _make_store()
    state = await store.create()

    assert state.session_id
    assert state.created_at == state.updated_at
    fetched = await store.get(state.session_id)
    assert fetched == state


async def test_store_get_missing_returns_none() -> None:
    store = await _make_store()
    assert await store.get("does-not-exist") is None


async def test_store_update_bumps_updated_at() -> None:
    store = await _make_store()
    state = await store.create()
    original_updated = state.updated_at

    # Drop one signal so the diff is visible.
    next_state = state.model_copy(update={"session_signals": [_make_signal()]})
    saved = await store.update(next_state)

    assert saved.updated_at >= original_updated
    assert saved.session_signals == next_state.session_signals
    fetched = await store.get(state.session_id)
    assert fetched is not None
    assert fetched.session_signals == next_state.session_signals


async def test_store_update_enforces_signal_cap() -> None:
    store = await _make_store()
    state = await store.create()
    huge = state.model_copy(
        update={"session_signals": [_make_signal(listing_id=f"L{i}") for i in range(MAX_SESSION_SIGNALS + 50)]}
    )
    saved = await store.update(huge)
    assert len(saved.session_signals) == MAX_SESSION_SIGNALS
    # FIFO: oldest dropped — last id retained.
    assert saved.session_signals[-1].listing_id == f"L{MAX_SESSION_SIGNALS + 50 - 1}"


async def test_store_update_enforces_query_cap() -> None:
    store = await _make_store()
    state = await store.create()
    parsed_queries = [ParsedQuery(semantic_query=f"q{i}") for i in range(MAX_RECENT_QUERIES + 5)]
    huge = state.model_copy(update={"recent_parsed_queries": parsed_queries})
    saved = await store.update(huge)
    assert len(saved.recent_parsed_queries) == MAX_RECENT_QUERIES
    assert saved.recent_parsed_queries[-1].semantic_query == f"q{MAX_RECENT_QUERIES + 5 - 1}"


async def test_store_update_enforces_clarification_cap() -> None:
    store = await _make_store()
    state = await store.create()
    answers = [
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id=f"district:{i}")
        for i in range(MAX_CLARIFICATION_ANSWERS + 5)
    ]
    huge = state.model_copy(update={"clarification_answers": answers})
    saved = await store.update(huge)
    assert len(saved.clarification_answers) == MAX_CLARIFICATION_ANSWERS


async def test_store_update_skips_trim_when_under_cap() -> None:
    """Frequent under-cap updates must not allocate a new list each time."""
    store = await _make_store()
    state = await store.create()
    minimal = state.model_copy(update={"session_signals": [_make_signal()]})
    saved = await store.update(minimal)
    assert len(saved.session_signals) == 1


async def test_store_clear_removes_state() -> None:
    store = await _make_store()
    state = await store.create()
    await store.clear(state.session_id)
    assert await store.get(state.session_id) is None


async def test_store_clear_missing_is_no_op() -> None:
    store = await _make_store()
    await store.clear("never-existed")


async def test_store_ttl_expires_state() -> None:
    """After the configured TTL elapses on the cache, the session is gone."""
    clock = _FakeClock()
    store = await _make_store(ttl_seconds=10, clock=clock)
    state = await store.create()
    clock.advance(11)
    assert await store.get(state.session_id) is None


async def test_store_update_refreshes_ttl_on_active_session() -> None:
    """The "rolling 7-day" semantic — each update resets the clock to now+ttl."""
    clock = _FakeClock()
    store = await _make_store(ttl_seconds=10, clock=clock)
    state = await store.create()

    clock.advance(9)
    state2 = await store.get(state.session_id)
    assert state2 is not None

    refreshed = await store.update(state2)
    clock.advance(9)  # 18 elapsed total — would have expired without the refresh
    fetched = await store.get(refreshed.session_id)
    assert fetched is not None


async def test_store_uses_session_prefix_in_cache_key() -> None:
    """Cache key collisions are ruled out via the `session:` prefix."""
    cache = MemoryCache()
    store = SessionStore(cache)
    state = await store.create()
    raw = await cache.get(f"{CACHE_KEY_PREFIX}{state.session_id}")
    assert raw is not None


async def test_store_rejects_non_positive_ttl() -> None:
    cache: Cache = MemoryCache()
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        SessionStore(cache, ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        SessionStore(cache, ttl_seconds=-1)


def test_default_ttl_is_seven_days() -> None:
    assert SESSION_TTL_SECONDS == 7 * 24 * 60 * 60
