"""`ingovo_search.api` is the HTTP layer only.

Two rules over the whole `api/` package, read with `ast` by `tests/_api_layer.py`:

1. No module imports SQLAlchemy. A handler takes ``db_session: DbSession``
   (`ingovo_search.db.deps`) and passes it to the search layer; queries,
   transactions and `SQLAlchemyError` handling live in `ingovo_search.search`,
   `ingovo_search.recommender` or `ingovo_search.db`.
2. A route handler is at most 30 lines, from its ``def`` line to its last
   line (decorators excluded). A handler is a function decorated with
   ``<router>.<verb>(...)`` or registered with ``<app>.add_api_route(path, handler)``.

Both rules hold with no exceptions. Fix the code; there is no list to add to.
"""

from __future__ import annotations

import ast

from tests._api_layer import (
    HANDLER_LINE_LIMIT,
    all_handlers,
    handlers,
    imports_sqlalchemy,
    long_handlers,
    sqlalchemy_importers,
)


def test_the_scan_sees_the_routes() -> None:
    """A broken detector would pass both rules vacuously."""
    found = all_handlers()
    assert len(found) >= 9
    assert any(handler.key == "api/main.py::healthz" for handler in found)


def test_the_detector_recognises_offenders() -> None:
    tree = ast.parse(
        "from sqlalchemy.ext.asyncio import AsyncSession\n"
        "@router.get('/x')\n"
        "async def long_one() -> None:\n" + "    pass\n" * HANDLER_LINE_LIMIT + "def registered() -> None:\n    pass\n"
        "app.add_api_route('/y', registered)\n"
    )
    assert imports_sqlalchemy(tree)
    keys = {handler.key: handler.lines for handler in handlers("api/fake.py", tree)}
    assert keys == {"api/fake.py::long_one": HANDLER_LINE_LIMIT + 1, "api/fake.py::registered": 2}


def test_no_sqlalchemy_import_in_api() -> None:
    offenders = sorted(sqlalchemy_importers())
    assert not offenders, (
        f"api/ modules importing SQLAlchemy: {offenders}. Take `db_session: DbSession` from "
        "ingovo_search.db.deps and move the queries into ingovo_search.search."
    )


def test_no_long_route_handler() -> None:
    offenders = sorted(long_handlers())
    assert not offenders, (
        f"route handlers over {HANDLER_LINE_LIMIT} lines: {offenders}. Keep the handler to request "
        "parsing and the response; move the orchestration into ingovo_search.search."
    )
