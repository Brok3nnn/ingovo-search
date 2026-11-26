"""The parameter objects routes take as one FastAPI dependency each.

The search and refinement routes would otherwise list the session cookie's five
collaborators and the search pipeline's four one by one. Each group is a single
dependency here, so a handler stays within a handful of parameters and hands the
object on as it is. The underlying providers stay the keys of
``app.dependency_overrides``, so tests override them individually.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Response

from ingovo_search.api.deps import (
    get_embedding_client,
    get_llm_client,
    get_session_cache,
    get_session_cookie_secure,
    get_session_secret,
    get_session_store,
)
from ingovo_search.db.deps import DbSession
from ingovo_search.lib.cache import Cache
from ingovo_search.lib.embeddings import EmbeddingClient
from ingovo_search.lib.llm import LlmClient
from ingovo_search.lib.session import SessionStore
from ingovo_search.search.request import SessionContext
from ingovo_search.search.run import SearchDependencies


async def session_context(
    request: Request,
    response: Response,
    store: Annotated[SessionStore, Depends(get_session_store)],
    secret: Annotated[bytes, Depends(get_session_secret)],
    cookie_secure: Annotated[bool, Depends(get_session_cookie_secure)],
) -> SessionContext:
    """What reading and writing the session cookie takes."""
    return SessionContext(request, response, store, secret, cookie_secure)


async def search_dependencies(
    db_session: DbSession,
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    cache: Annotated[Cache, Depends(get_session_cache)],
) -> SearchDependencies:
    """The per-request collaborators every search stage shares."""
    return SearchDependencies(db_session, llm_client, embedding_client, cache)


SessionContextDep = Annotated[SessionContext, Depends(session_context)]
SearchDependenciesDep = Annotated[SearchDependencies, Depends(search_dependencies)]

__all__ = [
    "SearchDependenciesDep",
    "SessionContextDep",
    "search_dependencies",
    "session_context",
]
