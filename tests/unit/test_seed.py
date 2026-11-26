"""The seed command's pure parts: row mapping, stable ids, and the shipped data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ingovo_search.seed import (
    DEFAULT_LISTINGS_PATH,
    DEFAULT_POIS_PATH,
    SeedSummary,
    listing_params,
    load_json_list,
    poi_params,
    schema_sql,
    stable_id,
)


def _listing(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_id": "syn-00000",
        "source_fingerprint": "sha256:abc",
        "title": "2 szobás · panel · Budapest",
        "description": "2 szoba, 61 m², Budapest, 13. kerület.",
        "price": 41_521_746,
        "transaction_type": "sale",
        "city": "Budapest",
        "district": "13",
        "address": "Example utca 1",
        "property_type": "panel",
        "rooms": 2,
        "area": 61.1,
        "lat": 47.52,
        "lng": 19.06,
        "features": {"synthetic": {"tier": "skeletal"}},
    }
    row.update(overrides)
    return row


def test_a_listing_maps_to_the_upsert_parameters() -> None:
    params = listing_params(_listing())

    assert params is not None
    assert params["id"] == stable_id("synthetic", "syn-00000")
    assert params["price_huf"] == 41_521_746
    assert params["rooms"] == "2"
    assert params["area_m2"] == 61.1
    assert (params["lat"], params["lng"]) == (47.52, 19.06)
    assert json.loads(str(params["features"])) == {"synthetic": {"tier": "skeletal"}}


def test_stable_ids_are_deterministic_and_source_scoped() -> None:
    assert stable_id("synthetic", "x") == stable_id("synthetic", "x")
    assert stable_id("synthetic", "x") != stable_id("osm", "x")


@pytest.mark.parametrize(
    "overrides",
    [{"title": ""}, {"price": -1}, {"transaction_type": "lease"}, {"city": ""}],
)
def test_an_invalid_listing_is_skipped(overrides: dict[str, Any]) -> None:
    assert listing_params(_listing(**overrides)) is None


def test_out_of_range_coordinates_become_null() -> None:
    params = listing_params(_listing(lat=123.0, lng=True))
    assert params is not None
    assert (params["lat"], params["lng"]) == (None, None)


def test_a_zero_price_is_stored_as_unknown() -> None:
    params = listing_params(_listing(price=0))
    assert params is not None
    assert params["price_huf"] is None


def test_poi_mapping_and_rejection() -> None:
    params = poi_params({"category": "metro", "name": "Example", "lat": 47.5, "lng": 19.0, "source_id": "node/1"})
    assert params is not None
    assert params["source"] == "osm"
    assert poi_params({"category": "metro", "lat": 47.5, "source_id": "node/1"}) is None


def test_every_shipped_listing_is_valid() -> None:
    rows = load_json_list(DEFAULT_LISTINGS_PATH)
    params = [listing_params(row) for row in rows]

    assert len(rows) == 1800
    assert all(p is not None for p in params)
    assert len({p["id"] for p in params if p is not None}) == len(rows)


def test_every_shipped_poi_is_usable() -> None:
    rows = load_json_list(DEFAULT_POIS_PATH)
    assert rows
    assert all(poi_params(row) is not None for row in rows)


def test_load_json_list_rejects_a_non_list(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(TypeError, match="JSON list"):
        load_json_list(path)


def test_the_schema_ships_with_the_package() -> None:
    assert "CREATE TABLE IF NOT EXISTS properties" in schema_sql()


def test_the_summary_says_why_embeddings_were_skipped() -> None:
    summary = SeedSummary(
        listings=10, skipped=0, pois=5, poi_distances=7, embedded=0, embeddings_skipped_reason="no key"
    )
    assert "embeddings skipped:" in summary.format()
    assert "no key" in summary.format()
