from __future__ import annotations

from fastapi import Request, Response
from pydantic import BaseModel

from ingovo_search.api.cookies import read_session_id, set_session_cookie
from ingovo_search.lib.session import SessionState, SessionStore
from ingovo_search.recommender.clarification_policy import ClarificationAnswer
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.search.models import SearchRequest


async def get_or_create_session_state(
    request: Request,
    response: Response,
    store: SessionStore,
    secret: bytes,
    cookie_secure: bool,
) -> SessionState:
    """Read the cookie, fetch state; on miss create a fresh state + stamp the cookie."""
    sid = read_session_id(request, secret=secret)
    if sid is not None:
        state = await store.get(sid)
        if state is not None:
            return state
    state = await store.create()
    set_session_cookie(response, state.session_id, secret=secret, secure=cookie_secure)
    return state


async def accumulate_request_into_session(
    state: SessionState,
    payload: SearchRequest,
    store: SessionStore,
) -> SessionState:
    """Append the request's signals + clarifications + refinement into session state."""
    updated_signals = list(state.session_signals) + list(payload.session_signals)
    updated_answers = list(state.clarification_answers) + list(payload.prior_clarifications)
    next_state = state.model_copy(
        update={
            "session_signals": updated_signals,
            "clarification_answers": updated_answers,
            "refinement_overrides": payload.refinement_overrides or state.refinement_overrides,
        }
    )
    return await store.update(next_state)


async def record_parsed_query(state: SessionState, parsed: ParsedQuery, store: SessionStore) -> SessionState:
    """Append the freshly-parsed query to `recent_parsed_queries` (FIFO-capped in store.update)."""
    next_state = state.model_copy(update={"recent_parsed_queries": [*state.recent_parsed_queries, parsed]})
    return await store.update(next_state)


class ClarifyRequest(BaseModel):
    """`POST /api/search/clarify` body — the buyer's answer to one clarification."""

    model_config = {"frozen": True}

    answer: ClarificationAnswer
