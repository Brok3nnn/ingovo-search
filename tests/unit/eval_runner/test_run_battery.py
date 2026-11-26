"""Unit tests for the Phase 1 battery runner's pure pieces.

Loaded via importlib like the other eval scripts, which are standalone files
rather than an installed package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "eval" / "run_battery.py"

_spec = importlib.util.spec_from_file_location("run_battery", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

build_query = _module.build_query
hard_filter_violations = _module.hard_filter_violations
diff_snapshots = _module.diff_snapshots


class _Listing:
    def __init__(self, **attrs: object) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


def test_expected_block_builds_a_parsed_query() -> None:
    parsed = build_query(
        {
            "hard_filters": {"transaction_type": "sale", "city": "Szeged", "price_huf_max": 50_000_000},
            "poi_near": [{"category": "school", "max_walk_minutes": None}],
        }
    )

    assert parsed.hard_filters.city == "Szeged"
    assert parsed.poi_near[0].category == "school"


def test_conforming_listing_has_no_violations() -> None:
    parsed = build_query(
        {
            "hard_filters": {
                "transaction_type": "sale",
                "property_type_in": ["Lakás"],
                "city": "Budapest",
                "district_in": ["13"],
                "price_huf_max": 60_000_000,
                "area_m2_min": 40,
                "rooms_min": 2,
            }
        }
    )
    listing = _Listing(
        transaction_type="sale",
        property_type="Lakás",
        city="Budapest",
        district="13",
        price=55_000_000,
        area_m2=62.0,
        rooms="3",
    )

    assert hard_filter_violations(parsed.hard_filters, listing) == []


def test_each_violated_bound_is_reported() -> None:
    parsed = build_query({"hard_filters": {"price_huf_max": 50_000_000, "area_m2_min": 80, "rooms_min": 3}})
    listing = _Listing(price=70_000_000, area_m2=50.0, rooms="2")

    violations = hard_filter_violations(parsed.hard_filters, listing)

    assert len(violations) == 3


def test_null_categorical_stays_eligible_but_null_numeric_violates() -> None:
    """Mirrors sql_filter's missing-data policy."""
    parsed = build_query({"hard_filters": {"city": "Pécs", "property_type_in": ["Ház"], "price_huf_max": 50_000_000}})
    listing = _Listing(city=None, property_type=None, price=None, area_m2=None, rooms=None)

    violations = hard_filter_violations(parsed.hard_filters, listing)

    assert violations == ["price=None > max 50000000"]


def test_roman_district_variant_is_accepted() -> None:
    parsed = build_query({"hard_filters": {"district_in": ["13"]}})
    listing = _Listing(district="XIII. kerület", price=None, area_m2=None, rooms=None)

    violations = [v for v in hard_filter_violations(parsed.hard_filters, listing) if "district" in v]

    assert violations == []


def test_snapshot_diff_reports_drift_new_and_removed() -> None:
    golden = {"q1": ["a", "b"], "q_gone": ["x"]}
    current = {"q1": ["b", "a"], "q_new": ["c"]}

    drift = diff_snapshots(golden, current)

    assert any(line.startswith("DRIFT 'q1'") for line in drift)
    assert any("NEW query" in line and "q_new" in line for line in drift)
    assert any("REMOVED query" in line and "q_gone" in line for line in drift)


def test_snapshot_diff_empty_on_match() -> None:
    assert diff_snapshots({"q": ["a"]}, {"q": ["a"]}) == []
