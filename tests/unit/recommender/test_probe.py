"""Unit tests for the homepage index probe's pure layer.

The DB-backed orchestration (`build_probe`, the COUNT / percentile / sample SQL)
is integration-tested; here we pin the deterministic ranking + preview mapping
that decide *which* real listings surface and *what* they read. Fixtures build
real `Property` instances — the scorer's listing Protocol is structural, so no DB
is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ingovo_search.db.models import Property
from ingovo_search.recommender.parser_schema import Anchors, HardFilters, ParsedQuery
from ingovo_search.recommender.probe import (
    PriceStats,
    ProbePreviewListing,
    SearchProbe,
    preview_from,
    score_and_rank,
    widen_candidates,
)

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)


def _make_property(**overrides: Any) -> Property:
    defaults: dict[str, Any] = {
        "id": "p-1",
        "title": "Test",
        "city": "Budapest",
        "status": "active",
        "images": [],
        "view_count": 0,
        "save_count": 0,
        "contact_count": 0,
        "photo_count": 0,
        "created_at": NOW,
    }
    defaults.update(overrides)
    return Property(**defaults)


def _query(**overrides: Any) -> ParsedQuery:
    return ParsedQuery(
        hard_filters=overrides.get("hard_filters", HardFilters()),
        anchors=overrides.get("anchors", Anchors()),
        soft_preferences=overrides.get("soft_preferences", []),
    )


def test_score_and_rank_prefers_listing_near_price_anchor() -> None:
    query = _query(anchors=Anchors(price_huf=60_000_000))
    on_anchor = _make_property(id="on", price_huf=60_000_000)
    far = _make_property(id="far", price_huf=180_000_000)

    ranked = score_and_rank([far, on_anchor], query, now=NOW)

    assert [row.id for row, _ in ranked] == ["on", "far"]
    assert ranked[0][1].total >= ranked[1][1].total


def test_score_and_rank_prefers_matching_soft_preference() -> None:
    query = _query(soft_preferences=["balcony"])
    with_balcony = _make_property(id="balcony", has_balcony=True)
    without = _make_property(id="plain", has_balcony=False)

    ranked = score_and_rank([without, with_balcony], query, now=NOW)

    assert ranked[0][0].id == "balcony"


def test_score_and_rank_is_deterministic_without_exploration_noise() -> None:
    query = _query(anchors=Anchors(price_huf=50_000_000))
    rows = [_make_property(id=f"p-{i}", price_huf=50_000_000 + i) for i in range(5)]

    first = [row.id for row, _ in score_and_rank(list(rows), query, now=NOW)]
    second = [row.id for row, _ in score_and_rank(list(rows), query, now=NOW)]

    assert first == second


def test_preview_from_maps_card_fields_and_first_image() -> None:
    query = _query(anchors=Anchors(price_huf=60_000_000))
    listing = _make_property(
        id="p-9",
        title="Napfényes lakás",
        price_huf=61_000_000,
        area_m2=72.0,
        rooms="3",
        city="Budapest",
        district="XIII",
        property_type="Lakás",
        transaction_type="sale",
        images=["https://cdn/first.jpg", "https://cdn/second.jpg"],
    )

    (_, breakdown), *_ = score_and_rank([listing], query, now=NOW)
    preview = preview_from(listing, breakdown, query)

    assert isinstance(preview, ProbePreviewListing)
    assert preview.listing_id == "p-9"
    assert preview.title == "Napfényes lakás"
    assert preview.price == 61_000_000
    assert preview.area_m2 == 72.0
    assert preview.rooms == "3"
    assert preview.district == "XIII"
    assert preview.image == "https://cdn/first.jpg"
    assert 1 <= preview.match_percent <= 100


def test_preview_from_exposes_an_unsafe_canonical_bigint_without_float_rounding() -> None:
    query = _query(anchors=Anchors(price_huf=9_007_199_254_740_993))
    listing = _make_property(
        id="p-precise",
        price_huf=9_007_199_254_740_993,
        currency="HUF",
    )

    (_, breakdown), *_ = score_and_rank([listing], query, now=NOW)
    preview = preview_from(listing, breakdown, query)

    assert preview.price_huf == "9007199254740993"
    assert preview.price is None


def test_price_stats_serializes_unsafe_huf_values_as_decimal_strings() -> None:
    stats = PriceStats(segment_count=1, min=9_007_199_254_740_993, median=9_007_199_254_740_993)

    encoded = stats.model_dump(mode="json")

    assert encoded["min"] == "9007199254740993"
    assert encoded["median"] == "9007199254740993"


def test_probe_json_keeps_bigint_huf_intent_fields_as_decimal_strings() -> None:
    query = _query(
        hard_filters=HardFilters(price_huf_min=9_007_199_254_740_992, price_huf_max=9_007_199_254_740_993),
        anchors=Anchors(price_huf=9_007_199_254_740_994),
    )
    probe = SearchProbe(
        request_id="probe-precise",
        intent_hash="hash",
        parsed_query=query,
        understood_summary="",
        exact_count=0,
        price_stats=PriceStats(segment_count=0),
    )

    encoded = probe.model_dump(mode="json")["parsed_query"]
    assert encoded["hard_filters"]["price_huf_min"] == "9007199254740992"
    assert encoded["hard_filters"]["price_huf_max"] == "9007199254740993"
    assert encoded["anchors"]["price_huf"] == "9007199254740994"


def test_widen_candidates_uses_integer_only_huf_math_above_number_precision() -> None:
    budget = 9_007_199_250_000_001

    candidates = widen_candidates(HardFilters(price_huf_max=budget), None, "hu")

    assert candidates[0][2].price_huf_max == 10_808_639_105_000_000


def test_preview_from_handles_missing_image() -> None:
    query = _query()
    listing = _make_property(id="no-img", images=[])

    (_, breakdown), *_ = score_and_rank([listing], query, now=NOW)
    preview = preview_from(listing, breakdown, query)

    assert preview.image is None
