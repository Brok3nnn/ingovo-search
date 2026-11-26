"""§14e-5 — `POST /api/refinement`.

Persists results-page refinement state (sliders / chips / disambiguation
picks per the design spec §6.7) into the anonymous session so the
next `/search` call re-scores against the user's edits without forcing
them to re-type the query.

Refinement is read on every `/search` via `state.refinement_overrides`;
this endpoint is just the write side. `w_overrides` lets the frontend
nudge individual score weights when the trade-off buttons fire.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ingovo_search.api.envelope import Envelope, ok
from ingovo_search.api.route_deps import SessionContextDep
from ingovo_search.api.search_session import get_or_create_session_state
from ingovo_search.lib.observability import (
    EVENT_REFINEMENT_CHIP_REMOVE,
    EVENT_REFINEMENT_DISAMBIGUATE,
    EVENT_REFINEMENT_TRADEOFF,
    analytics_distinct_id,
    track_event,
)
from ingovo_search.lib.session import RefinementOverrides

router = APIRouter(prefix="/api", tags=["refinement"])


class RefinementRequest(BaseModel):
    """`POST /api/refinement` body — at least one of the two override blocks."""

    model_config = {"frozen": True}

    refinement_overrides: RefinementOverrides | None = None
    w_overrides: dict[str, float] | None = None


@router.post("/refinement", response_model=Envelope[dict[str, str]])
async def refinement(
    payload: RefinementRequest,
    session_ctx: SessionContextDep,
) -> Envelope[dict[str, str]]:
    """Update refinement overrides and weight overrides in the session."""
    state = await get_or_create_session_state(
        session_ctx.request, session_ctx.response, session_ctx.store, session_ctx.secret, session_ctx.cookie_secure
    )

    updates: dict[str, Any] = {}
    if payload.refinement_overrides is not None:
        updates["refinement_overrides"] = payload.refinement_overrides
    if payload.w_overrides is not None:
        updates["w_overrides"] = payload.w_overrides

    if updates:
        next_state = state.model_copy(update=updates)
        await session_ctx.store.update(next_state)
        _track_refinement(
            distinct_id=analytics_distinct_id(session_id=state.session_id),
            previous=state.refinement_overrides,
            payload=payload,
        )

    return ok({"status": "updated"})


_OVERRIDE_FIELDS = ("hard_filters", "anchors", "soft_preferences", "soft_prefs_freetext", "geo_hints")


def _track_refinement(
    *,
    distinct_id: str,
    previous: RefinementOverrides | None,
    payload: RefinementRequest,
) -> None:
    """Emit the §16 refinement events for one write.

    This endpoint takes a full override block, not an action, so which of the
    three events fired has to be *inferred* from the payload shape:

    - `w_overrides` present        -> the trade-off buttons moved a score weight.
    - `geo_hints` present          -> a disambiguation pick (the §6.7 "which
                                      rózsadomb did you mean" affordance).
    - a field that was set is now  -> the user removed that chip.
      absent

    One write can legitimately be more than one of these, so they are not
    exclusive. If the frontend ever sends an explicit action type, drop this
    inference and read it directly — that would be strictly more accurate.
    """
    incoming = payload.refinement_overrides

    if payload.w_overrides is not None:
        track_event(
            EVENT_REFINEMENT_TRADEOFF,
            distinct_id=distinct_id,
            properties={"weights_changed": sorted(payload.w_overrides)},
        )

    if incoming is None:
        return

    if incoming.geo_hints is not None:
        track_event(EVENT_REFINEMENT_DISAMBIGUATE, distinct_id=distinct_id, properties={"field": "geo_hints"})

    if previous is None:
        return

    removed = [
        field for field in _OVERRIDE_FIELDS if getattr(previous, field) is not None and getattr(incoming, field) is None
    ]
    if removed:
        track_event(EVENT_REFINEMENT_CHIP_REMOVE, distinct_id=distinct_id, properties={"removed_fields": removed})
