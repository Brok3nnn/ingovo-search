"""Unit tests for `eval/run_parser_eval.py::check_dict_subset` (§17.1).

The eval scripts are standalone files, not an installed package, so the test
loads `eval/run_parser_eval.py` directly via `importlib.util`. Pure-function
tests only — the async `run_eval` orchestrator and the `GoogleAiClient` path
run live in CI's parser-eval job when a `GEMINI_API_KEY` secret exists.

The test directory is `eval_runner/` (not `eval/`) so its package name does not
collide with the repository's `eval/` directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "eval" / "run_parser_eval.py"

_spec = importlib.util.spec_from_file_location("run_parser_eval", _MODULE_PATH)
assert _spec is not None, f"could not build spec for {_MODULE_PATH}"
assert _spec.loader is not None, f"spec has no loader for {_MODULE_PATH}"
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

check_dict_subset = _module.check_dict_subset
_load_gemini_api_key = _module._load_gemini_api_key


def test_full_primitive_match_returns_no_missing() -> None:
    expected: dict[str, Any] = {"transaction_type": "sale", "rooms_min": 3}
    actual: dict[str, Any] = {"transaction_type": "sale", "rooms_min": 3, "city": "Budapest"}

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 2
    assert total == 2
    assert missing == []


def test_missing_primitive_appears_in_missing_path() -> None:
    expected: dict[str, Any] = {"transaction_type": "sale", "rooms_min": 3}
    actual: dict[str, Any] = {"transaction_type": "sale"}

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 1
    assert total == 2
    assert len(missing) == 1
    assert "rooms_min" in missing[0]


def test_primitive_type_mismatch_counted_as_miss() -> None:
    expected: dict[str, Any] = {"price_huf_max": 50000000}
    actual: dict[str, Any] = {"price_huf_max": "50000000"}  # string instead of int

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 0
    assert total == 1
    assert "price_huf_max" in missing[0]


def test_nested_dict_subset_descends_recursively() -> None:
    expected: dict[str, Any] = {
        "hard_filters": {"transaction_type": "sale", "rooms_min": 3},
        "soft_preferences": ["balcony", "elevator"],
    }
    actual: dict[str, Any] = {
        "hard_filters": {"transaction_type": "sale", "rooms_min": 3, "city": "Budapest"},
        "soft_preferences": ["balcony", "elevator", "natural_light"],
    }

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 4  # 2 nested primitives + 2 list elements
    assert total == 4
    assert missing == []


def test_nested_missing_path_carries_parent_key() -> None:
    expected: dict[str, Any] = {"hard_filters": {"rooms_min": 3, "city": "Pécs"}}
    actual: dict[str, Any] = {"hard_filters": {"rooms_min": 3}}  # city missing

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 1
    assert total == 2
    assert any("hard_filters." in m and "city" in m for m in missing)


def test_list_subset_counts_each_element() -> None:
    expected: dict[str, Any] = {"soft_preferences": ["balcony", "elevator", "garden"]}
    actual: dict[str, Any] = {"soft_preferences": ["balcony", "elevator"]}  # garden missing

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 2
    assert total == 3
    assert any("garden" in m for m in missing)


def test_list_subset_empty_actual_counts_all_as_missing() -> None:
    expected: dict[str, Any] = {"soft_preferences": ["balcony", "elevator"]}
    actual: dict[str, Any] = {}

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 0
    assert total == 2
    assert len(missing) == 2


def test_extra_actual_keys_do_not_inflate_total() -> None:
    """The subset semantic: actual may carry extras; only expected keys count."""
    expected: dict[str, Any] = {"transaction_type": "sale"}
    actual: dict[str, Any] = {
        "transaction_type": "sale",
        "city": "Budapest",
        "district_in": ["5"],
        "rooms_min": 3,
    }

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 1
    assert total == 1
    assert missing == []


def test_real_starter_query_shape_matches() -> None:
    """Mirrors the first entry in `eval/golden/eval_set.starter.jsonl` — sanity check
    that the subset checker handles the real fixture shape."""
    expected: dict[str, Any] = {
        "hard_filters": {
            "transaction_type": "sale",
            "property_type_in": ["Lakás"],
            "city": "Budapest",
            "district_in": ["13"],
        },
        "soft_preferences": ["terrace"],
    }
    actual: dict[str, Any] = {
        "hard_filters": {
            "transaction_type": "sale",
            "property_type_in": ["Lakás"],
            "city": "Budapest",
            "district_in": ["13"],
            "rooms_min": None,
        },
        "soft_preferences": ["terrace"],
        "geo_hints": {},
    }

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 5  # 4 hard_filter primitives + 1 list element
    assert total == 5
    assert missing == []


def test_load_gemini_api_key_rejects_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        _load_gemini_api_key()

    assert exc_info.value.code == 1
    assert "GEMINI_API_KEY is missing or empty" in capsys.readouterr().err


def test_load_gemini_api_key_strips_env_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "  test-key  ")

    assert _load_gemini_api_key() == "test-key"


# --- §17.1 per-category recall + hallucinated-key gate ---

find_hallucinated_keys = _module.find_hallucinated_keys
gate_verdict = _module.gate_verdict


def test_hallucinated_key_detected_in_hard_filters() -> None:
    expected = {"hard_filters": {"city": "Budapest"}}
    parsed = {
        "hard_filters": {"city": "Budapest", "price_huf_max": 50_000_000},
        "anchors": {"price_huf": None},
    }

    assert find_hallucinated_keys(expected, parsed) == [("hard_filters.price_huf_max", 50_000_000)]


def test_hallucinated_key_detected_in_anchors() -> None:
    expected = {"hard_filters": {"city": "Szeged"}}
    parsed = {
        "hard_filters": {"city": "Szeged"},
        "anchors": {"price_huf": 80_000_000, "area_m2": None},
    }

    assert find_hallucinated_keys(expected, parsed) == [("anchors.price_huf", 80_000_000)]


def test_schema_defaults_are_not_hallucinations() -> None:
    """None / empty-list / empty-string defaults must not count as invented keys."""
    expected: dict[str, Any] = {"hard_filters": {}}
    parsed: dict[str, Any] = {
        "hard_filters": {
            "city": None,
            "district_in": [],
            "property_type_in": [],
            "near": None,
        },
        "anchors": {"price_huf": None, "area_m2": None, "rooms": None},
    }

    assert find_hallucinated_keys(expected, parsed) == []


def test_expected_keys_are_not_hallucinations() -> None:
    expected = {"hard_filters": {"city": "Budapest", "district_in": ["13"]}}
    parsed = {"hard_filters": {"city": "Budapest", "district_in": ["13"]}}

    assert find_hallucinated_keys(expected, parsed) == []


def test_soft_preferences_never_count_as_hallucinated() -> None:
    """Extra soft tags reorder results; only hard_filters/anchors exclude listings."""
    expected = {"hard_filters": {"city": "Pécs"}}
    parsed = {
        "hard_filters": {"city": "Pécs"},
        "soft_preferences": ["balcony", "sunny"],
        "soft_prefs_freetext": ["csendes utca"],
    }

    assert find_hallucinated_keys(expected, parsed) == []


def test_gate_uses_hard_constraint_category_when_present() -> None:
    passed, reasons = gate_verdict(
        0.99,  # global recall high
        {"hard_constraint": 0.80, "soft_pref": 1.0},  # gated category low
        0,
    )

    assert not passed
    assert any("hard_constraint recall" in r for r in reasons)


def test_gate_falls_back_to_global_recall_without_categories() -> None:
    passed_low, reasons_low = gate_verdict(0.80, {}, 0)
    passed_high, reasons_high = gate_verdict(0.90, {}, 0)

    assert not passed_low
    assert any("global recall" in r for r in reasons_low)
    assert passed_high
    assert reasons_high == []


def test_gate_fails_on_any_adversarial_hallucination() -> None:
    passed, reasons = gate_verdict(1.0, {"hard_constraint": 1.0}, 1)

    assert not passed
    assert any("hallucinated key" in r for r in reasons)


def test_gate_reports_both_failures_together() -> None:
    passed, reasons = gate_verdict(0.5, {}, 3)

    assert not passed
    assert len(reasons) == 2


def test_derived_and_defaulted_keys_are_exempt() -> None:
    """city_id (settlement-registry derivation) and transaction_type (defaulting
    convention) are set by the parser on nearly every query — verified on the
    2026-07-03 live run (18 occurrences each, all benign). They must not trip
    the adversarial gate."""
    expected = {"hard_filters": {"city": "Budapest"}}
    parsed = {
        "hard_filters": {
            "city": "Budapest",
            "city_id": "settlement-budapest",
            "transaction_type": "sale",
        },
    }

    assert find_hallucinated_keys(expected, parsed) == []


def test_real_invention_still_flagged_alongside_exempt_keys() -> None:
    expected = {"hard_filters": {"city": "Budapest"}}
    parsed = {
        "hard_filters": {
            "city": "Budapest",
            "city_id": "settlement-budapest",
            "transaction_type": "sale",
            "rooms_min": 4,
        },
    }

    assert find_hallucinated_keys(expected, parsed) == [("hard_filters.rooms_min", 4)]


# --- free-tier quota discipline (design decision 2026-07-03) ---

is_quota_error = _module.is_quota_error
order_records_gate_first = _module.order_records_gate_first


def test_quota_errors_recognised_by_marker() -> None:
    assert is_quota_error(Exception("429 RESOURCE_EXHAUSTED. {'error': {...}}"))
    assert is_quota_error(Exception("got HTTP 429 from upstream"))
    assert not is_quota_error(Exception("invalid JSON in model response"))
    assert not is_quota_error(Exception("connection reset by peer"))


def test_gated_categories_run_first_and_order_is_stable() -> None:
    records = [
        {"query": "a", "category": "fuzzy_geo"},
        {"query": "b", "category": "hard_constraint"},
        {"query": "c", "category": "soft_pref"},
        {"query": "d", "category": "adversarial"},
        {"query": "e", "category": "hard_constraint"},
    ]

    ordered = [r["query"] for r in order_records_gate_first(records)]

    # Gated first, original relative order preserved within each group.
    assert ordered == ["b", "d", "e", "a", "c"]


def test_uncategorized_records_sort_after_gated() -> None:
    records = [{"query": "a"}, {"query": "b", "category": "adversarial"}]

    ordered = [r["query"] for r in order_records_gate_first(records)]

    assert ordered == ["b", "a"]


def test_gate_inconclusive_when_gated_coverage_too_low() -> None:
    passed, reasons = gate_verdict(
        1.0,
        {"hard_constraint": 1.0},
        0,
        gated_coverage=0.5,
    )

    assert not passed
    assert any("inconclusive" in r for r in reasons)


def test_gate_passes_at_full_coverage_with_good_recall() -> None:
    passed, reasons = gate_verdict(
        0.95,
        {"hard_constraint": 0.95},
        0,
        gated_coverage=1.0,
    )

    assert passed
    assert reasons == []


def test_gate_reports_hallucinations_even_when_inconclusive() -> None:
    """Real hallucinations found before quota died must still surface."""
    passed, reasons = gate_verdict(
        1.0,
        {"hard_constraint": 1.0},
        2,
        gated_coverage=0.4,
    )

    assert not passed
    assert any("inconclusive" in r for r in reasons)
    assert any("hallucinated" in r for r in reasons)


def test_invented_poi_near_items_are_hallucinations() -> None:
    """'családbarát környék' must not become a school-proximity request."""
    expected: dict[str, Any] = {"hard_filters": {"property_type_in": ["Ház"]}, "poi_near": []}
    parsed: dict[str, Any] = {
        "hard_filters": {"property_type_in": ["Ház"]},
        "poi_near": [{"category": "school", "max_walk_minutes": None}],
    }

    assert find_hallucinated_keys(expected, parsed) == [("poi_near", {"category": "school", "max_walk_minutes": None})]


def test_expected_poi_near_items_are_not_hallucinations() -> None:
    expected = {"poi_near": [{"category": "transit", "max_walk_minutes": 5}]}
    parsed = {"poi_near": [{"category": "transit", "max_walk_minutes": 5}]}

    assert find_hallucinated_keys(expected, parsed) == []


# --- soft_prefs_freetext stem matching (review follow-up 2026-07-04) ---

freetext_stem_matches = _module.freetext_stem_matches


def test_freetext_stem_matches_inside_longer_phrase() -> None:
    assert freetext_stem_matches("szauna", ["szaunás wellness részleg"])
    assert freetext_stem_matches("kutya", ["kutyabarát"])


def test_freetext_stem_matching_is_accent_and_case_blind() -> None:
    assert freetext_stem_matches("vízpart", ["VIZPARTI panoráma"])
    assert freetext_stem_matches("BÚTOROZOTT", ["butorozott lakás"])


def test_freetext_stem_mismatch_still_misses() -> None:
    assert not freetext_stem_matches("szauna", ["medencés kert"])
    assert not freetext_stem_matches("panel", [])


def test_subset_checker_uses_stems_only_for_freetext_key() -> None:
    expected: dict[str, Any] = {
        "soft_prefs_freetext": ["kutya"],
        "soft_preferences": ["balcony"],
    }
    actual: dict[str, Any] = {
        "soft_prefs_freetext": ["kutyabarát környék"],
        "soft_preferences": ["balcony_large"],  # NOT a stem match — tags stay exact
    }

    matches, total, missing = check_dict_subset(expected, actual)

    assert matches == 1  # freetext stem matched
    assert total == 2
    assert missing == ["soft_preferences[balcony]"]
