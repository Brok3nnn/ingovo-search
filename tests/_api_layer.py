"""The two api-layer rules, read from the source with `ast`.

`tests/test_api_layer_boundary.py` enforces them, with no exceptions:

1. No module under `src/ingovo_search/api` imports SQLAlchemy (any import statement).
2. A route handler is at most `HANDLER_LINE_LIMIT` lines from its ``def`` line to its
   last line (decorators excluded). A handler is a function decorated with
   ``<router>.<verb>(...)`` or registered with ``<app>.add_api_route(path, handler)``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ingovo_search"
API = SRC / "api"
HANDLER_LINE_LIMIT = 30
ROUTE_VERBS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "api_route", "websocket", "route"})


@dataclass(frozen=True)
class Handler:
    key: str
    lines: int


def api_modules() -> list[tuple[str, ast.Module]]:
    """Every module of the api package as (path relative to src/ingovo_search, tree)."""
    modules: list[tuple[str, ast.Module]] = []
    for path in sorted(API.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC).as_posix()
        modules.append((rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return modules


def _is_sqlalchemy(name: str | None) -> bool:
    return name is not None and (name == "sqlalchemy" or name.startswith("sqlalchemy."))


def imports_sqlalchemy(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_is_sqlalchemy(alias.name) for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_sqlalchemy(node.module):
            return True
    return False


def _is_route_decorator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr in ROUTE_VERBS
    )


def _registered_endpoints(tree: ast.Module) -> set[str]:
    """Names passed as the endpoint of ``<app>.add_api_route(path, endpoint, ...)``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_api_route":
            continue
        endpoint = node.args[1] if len(node.args) > 1 else None
        for keyword in node.keywords:
            if keyword.arg == "endpoint":
                endpoint = keyword.value
        if isinstance(endpoint, ast.Name):
            names.add(endpoint.id)
    return names


def handlers(rel: str, tree: ast.Module) -> list[Handler]:
    registered = _registered_endpoints(tree)
    found: list[Handler] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.end_lineno is None:
            continue
        if node.name in registered or any(_is_route_decorator(d) for d in node.decorator_list):
            found.append(Handler(f"{rel}::{node.name}", node.end_lineno - node.lineno + 1))
    return found


def all_handlers() -> list[Handler]:
    return [handler for rel, tree in api_modules() for handler in handlers(rel, tree)]


def sqlalchemy_importers() -> set[str]:
    return {rel for rel, tree in api_modules() if imports_sqlalchemy(tree)}


def long_handlers() -> set[str]:
    return {handler.key for handler in all_handlers() if handler.lines > HANDLER_LINE_LIMIT}
