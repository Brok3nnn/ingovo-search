"""Unit tests for the rerank candidate summary (the trimmed payload sent to the reranker)."""

from __future__ import annotations

from types import SimpleNamespace

from ingovo_search.search.constants import RERANK_DESCRIPTION_MAX_CHARS
from ingovo_search.search.scoring import listing_rerank_summary


def _prop(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "id": "p1",
        "title": "Napfényes otthon",
        "transaction_type": "sale",
        "property_type": "Lakás",
        "city": "Budapest",
        "district": "XI",
        "price": 75_000_000,
        "area_m2": 62,
        "rooms": "3",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_summary_keeps_all_structured_facts() -> None:
    summary = listing_rerank_summary(_prop())
    assert "Cím: Napfényes otthon" in summary
    assert "Tranzakció: eladó" in summary
    assert "Típus: Lakás" in summary
    assert "Hely: Budapest, XI" in summary
    assert "Ár: 75 000 000 Ft" in summary
    assert "Méret: 62 m2" in summary
    assert "Szobák: 3" in summary


def test_long_description_is_trimmed_to_cap() -> None:
    long_desc = "A" * 600
    summary = listing_rerank_summary(_prop(description=long_desc))
    label, _, body = summary.rpartition("Leírás: ")
    assert label  # the description section is present
    assert len(body) == RERANK_DESCRIPTION_MAX_CHARS
    assert "A" * 600 not in summary  # full description never reaches the prompt


def test_summary_has_no_newlines() -> None:
    summary = listing_rerank_summary(_prop(description="első sor\nmásodik sor"))
    assert "\n" not in summary


def test_short_summary_preferred_over_raw_description() -> None:
    summary = listing_rerank_summary(_prop(short_summary="Tömör összefoglaló.", description="X" * 600))
    assert "Leírás: Tömör összefoglaló." in summary
    assert "X" * 600 not in summary
