"""Unit tests for `build_factual_text` + `build_descriptive_text`."""

from __future__ import annotations

from ingovo_search.db.models import Property
from ingovo_search.ingestion.build_factual_text import build_descriptive_text, build_factual_text


def _make_property(**overrides: object) -> Property:
    defaults: dict[str, object] = {
        "id": "prop-1",
        "title": "Test listing",
        "city": "Budapest",
        "status": "active",
        "images": [],
        "view_count": 0,
        "save_count": 0,
        "contact_count": 0,
        "photo_count": 0,
        "transaction_type": "sale",
    }
    defaults.update(overrides)
    return Property(**defaults)  # type: ignore[arg-type]


def test_build_factual_text_full_listing_matches_snapshot() -> None:
    """Deterministic structured-fields snapshot."""
    prop = _make_property(
        rooms="3",
        property_type="tégla lakás",
        district="5",
        city="Budapest",
        area_m2=72.0,
        floor=2,
        total_floors=4,
        year_built=1928,
        renovation_state="fully_renovated",
        heating="gáz",
        orientation="S",
        has_balcony=True,
        elevator=True,
        has_parking=True,
        has_terrace=False,
        has_basement=True,
        air_conditioning=False,
    )

    rendered = build_factual_text(prop)

    assert rendered == (
        "3-szobás tégla lakás, "
        "a 5. kerületben, Budapest, "
        "72 m², "
        "2/4. emelet, "
        "1928-ben épült, "
        "felújított, "
        "gáz fűtés, "
        "déli tájolású, "
        "erkély, lift, parkoló, pince"
    )


def test_build_factual_text_omits_missing_fields() -> None:
    prop = _make_property(city="Pécs", area_m2=45.5)
    rendered = build_factual_text(prop)
    assert rendered == "Pécs, 45.5 m²"


def test_build_factual_text_is_deterministic() -> None:
    prop = _make_property(
        rooms="2",
        property_type="panel",
        district="13",
        area_m2=55.0,
        has_balcony=True,
        elevator=True,
    )
    assert build_factual_text(prop) == build_factual_text(prop)


def test_build_descriptive_text_prefers_cleaned_and_summary() -> None:
    prop = _make_property(
        description="Original raw text.",
        cleaned_description="Clean version.",
        one_paragraph_summary="Summary paragraph.",
    )
    rendered = build_descriptive_text(prop)
    assert rendered == "Clean version.\n\nSummary paragraph."


def test_build_descriptive_text_falls_back_to_raw_description() -> None:
    prop = _make_property(description="Raw only.", cleaned_description=None, one_paragraph_summary=None)
    rendered = build_descriptive_text(prop)
    assert rendered == "Raw only."


def test_build_descriptive_text_empty_when_no_text() -> None:
    prop = _make_property()
    assert build_descriptive_text(prop) == ""
