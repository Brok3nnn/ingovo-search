"""`get_jev_client` — one process-wide client, none without a key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ingovo_search.api import deps
from ingovo_search.lib.jev import TypeSafeJevClient


def _use_key(monkeypatch: pytest.MonkeyPatch, api_key: str | None) -> None:
    monkeypatch.setattr(deps, "get_settings", lambda: SimpleNamespace(typesafe_api_key=api_key))
    monkeypatch.setattr(deps.process_clients, "jev", None)


def test_no_key_means_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_key(monkeypatch, None)
    assert deps.get_jev_client() is None


def test_client_is_built_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_key(monkeypatch, "apikey_dummy")

    first = deps.get_jev_client()
    second = deps.get_jev_client()

    assert isinstance(first, TypeSafeJevClient)
    assert second is first
