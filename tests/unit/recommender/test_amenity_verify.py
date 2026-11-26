"""Unit tests for lexical amenity verification (honest match%)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ingovo_search.recommender.amenity_verify import (
    amenity_match_ratio,
    listing_text,
    requested_amenities,
)
from ingovo_search.recommender.parser_schema import ParsedQuery


def _q(semantic: str = "", freetext: list[str] | None = None) -> ParsedQuery:
    return ParsedQuery(semantic_query=semantic, soft_prefs_freetext=freetext or [])


def _listing(**fields: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "title": "",
        "description": "",
        "cleaned_description": None,
        "one_paragraph_summary": None,
        "short_summary": None,
        "features": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


def test_requested_amenities_detects_pool_from_semantic_and_freetext() -> None:
    assert requested_amenities(_q("úszómedencés villa")) == {"medence"}
    assert requested_amenities(_q(freetext=["úszómedence"])) == {"medence"}
    assert requested_amenities(_q("kandallós nappali panorámával")) == {"kandalló", "panoráma"}


def test_requested_amenities_empty_when_no_concrete_amenity() -> None:
    assert requested_amenities(_q("kétszobás lakás budán")) == set()
    assert requested_amenities(_q("")) == set()


def test_amenity_match_ratio_is_one_when_nothing_requested() -> None:
    # Ordinary searches must be completely untouched.
    assert amenity_match_ratio(_q("olcsó lakás"), _listing(description="bármi")) == 1.0


def test_amenity_match_ratio_zero_when_requested_pool_absent() -> None:
    q = _q("úszómedencés villa", freetext=["úszómedence"])
    listing = _listing(description="Tágas villa nagy kerttel és kandallóval.")
    assert amenity_match_ratio(q, listing) == 0.0


def test_amenity_match_ratio_one_when_pool_present_in_text() -> None:
    q = _q("úszómedencés villa")
    listing = _listing(description="Luxusvilla saját úszómedencével és szaunával.")
    assert amenity_match_ratio(q, listing) == 1.0


def test_amenity_match_ratio_partial_for_two_requested_one_present() -> None:
    q = _q("kandallós villa medencével")
    listing = _listing(description="Villa kandallóval, de medence nincs.")
    # "medence" appears literally ("medence nincs") → lexical check is presence,
    # not negation-aware; documented limitation. Both requested are present here.
    assert amenity_match_ratio(q, listing) == 1.0
    # A listing that truly lacks the pool:
    listing2 = _listing(description="Hangulatos villa kandallóval és nagy terasszal.")
    assert amenity_match_ratio(q, listing2) == 0.5


def test_listing_text_reads_summary_and_features() -> None:
    listing = _listing(
        title="Villa",
        description=None,
        one_paragraph_summary="Csendes utcában, panorámával.",
        features={"extra": "medence a kertben"},
    )
    text = listing_text(listing)
    assert "panorámával" in text
    assert "medence a kertben" in text


def test_amenity_match_ratio_verifies_via_features_field() -> None:
    q = _q("medencés ház")
    listing = _listing(description="Szép ház.", features=["úszómedence", "garázs"])
    assert amenity_match_ratio(q, listing) == 1.0


def test_structured_has_pool_is_authoritative_over_text() -> None:
    q = _q("úszómedencés villa")
    # Column True with no text mention → present (column wins).
    assert amenity_match_ratio(q, _listing(description="Szép villa.", has_pool=True)) == 1.0
    # Column False even though the text mentions a pool (e.g. "medence a közelben")
    # → absent (column wins over a misleading text match).
    assert amenity_match_ratio(q, _listing(description="A közelben van egy medence.", has_pool=False)) == 0.0


def test_text_fallback_when_has_pool_is_null() -> None:
    q = _q("úszómedencés villa")
    assert amenity_match_ratio(q, _listing(description="Saját úszómedencével.", has_pool=None)) == 1.0
    assert amenity_match_ratio(q, _listing(description="Kerttel.", has_pool=None)) == 0.0
