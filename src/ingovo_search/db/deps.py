"""The database session as a FastAPI dependency.

The HTTP layer (`ingovo_search.api`) does not import SQLAlchemy: a handler declares
``db_session: DbSession`` and hands the session to a service, which owns every
query. The dependency key is still `ingovo_search.db.session.get_session`, so the
tests' ``app.dependency_overrides[get_db_session]`` keeps working unchanged.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.session import get_session

DbSession = Annotated[AsyncSession, Depends(get_session)]

__all__ = ["DbSession"]
