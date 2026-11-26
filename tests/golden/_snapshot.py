"""Golden-snapshot helper.

A snapshot is a JSON file under ``tests/golden/snapshots``. A test computes the
current value, and either compares it to the file (the default — any difference is a
failure that names the first diverging key) or rewrites the file when pytest runs with
``--update-golden``. Rewriting is deliberate and explicit: a refactor is only safe when
the snapshot did not have to change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _diff_dict(expected: dict[str, Any], actual: dict[str, Any], path: str) -> str | None:
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            return f"{path}.{key}: new key"
        if key not in actual:
            return f"{path}.{key}: missing key"
        found = _first_difference(expected[key], actual[key], f"{path}.{key}")
        if found:
            return found
    return None


def _diff_list(expected: list[Any], actual: list[Any], path: str) -> str | None:
    if len(expected) != len(actual):
        return f"{path}: length {len(expected)} -> {len(actual)}"
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        found = _first_difference(left, right, f"{path}[{index}]")
        if found:
            return found
    return None


def _first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    """The first diverging key path between two JSON values, or ``None`` when equal."""
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} -> {type(actual).__name__}"
    if isinstance(expected, dict):
        return _diff_dict(cast(dict[str, Any], expected), cast(dict[str, Any], actual), path)
    if isinstance(expected, list):
        return _diff_list(cast(list[Any], expected), cast(list[Any], actual), path)
    return None if expected == actual else f"{path}: {expected!r} -> {actual!r}"


def check_snapshot(request: pytest.FixtureRequest, name: str, actual: Any) -> None:
    """Compare ``actual`` with ``snapshots/<name>.json`` or rewrite it under ``--update-golden``."""
    path = SNAPSHOT_DIR / f"{name}.json"
    serialised = json.loads(json.dumps(actual, ensure_ascii=False, sort_keys=False))
    if request.config.getoption("--update-golden"):
        path.write_text(json.dumps(serialised, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        return
    if not path.exists():
        pytest.fail(f"golden snapshot {path.name} is missing — run pytest with --update-golden once to create it")
    expected = json.loads(path.read_text(encoding="utf-8"))
    found = _first_difference(expected, serialised)
    assert found is None, (
        f"golden {path.name} differs at {found} (run with --update-golden only if the change is intended)"
    )
