"""Unit tests for §17.2.4 assembler + distribution validator."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
from eval.synthetic.assemble import (
    AREA_RANGE_BY_TYPE,
    CANONICAL_PROPERTY_TYPE,
    PROPERTY_TYPE_PROBABILITIES,
    ROOMS_RANGE_BY_TYPE,
    TARGET_COUNT_LOWER,
    TARGET_COUNT_UPPER,
    SyntheticListing,
    assemble_dataset,
    build_seed,
    load_addresses,
    sample_area_m2,
    sample_property_type,
    sample_rooms,
)
from eval.synthetic.describe import (
    BAD_CASE_FRACTION,
    NULL_FIELD_FRACTION,
    TIER_PROBABILITIES,
)
from eval.synthetic.extract_osm import Address
from eval.synthetic.validate_distribution import (
    REQUIRED_CITIES,
    ValidationReport,
    load_listings,
    validate,
)

from ingovo_search.ingestion.types import NormalizedListing
from ingovo_search.lib.llm import LlmClient, LlmRequest

SEED = 99


class _NoopLlmClient:
    async def complete(self, req: LlmRequest) -> str:
        _ = req
        return "GREAT_HU_STUB"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def test_property_type_probabilities_sum_to_one() -> None:
    assert sum(PROPERTY_TYPE_PROBABILITIES.values()) == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


def test_sample_property_type_returns_valid_value() -> None:
    rng = random.Random(SEED)
    for _ in range(10):
        assert sample_property_type(rng=rng) in PROPERTY_TYPE_PROBABILITIES


def test_sample_rooms_respects_per_type_range() -> None:
    rng = random.Random(SEED)
    for ptype in PROPERTY_TYPE_PROBABILITIES:
        low, high = ROOMS_RANGE_BY_TYPE[ptype]
        for _ in range(20):
            rooms = sample_rooms(property_type=ptype, rng=rng)
            assert low <= rooms <= high


def test_sample_area_respects_per_type_range() -> None:
    rng = random.Random(SEED)
    for ptype in PROPERTY_TYPE_PROBABILITIES:
        low, high = AREA_RANGE_BY_TYPE[ptype]
        for _ in range(20):
            area = sample_area_m2(property_type=ptype, rng=rng)
            assert low <= area <= high


def test_build_seed_uses_address_city_and_district() -> None:
    address = Address(city="Budapest", district="13", street="Lehel u. 1", lat=47.5, lng=19.08)
    rng = random.Random(SEED)
    seed = build_seed(address=address, rng=rng)
    assert seed.city == "Budapest"
    assert seed.district == "13"
    assert seed.property_type in PROPERTY_TYPE_PROBABILITIES


# ---------------------------------------------------------------------------
# load_addresses
# ---------------------------------------------------------------------------


def test_load_addresses_round_trips_minimal_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "addresses.json"
    fixture.write_text(
        json.dumps(
            [
                {"city": "Budapest", "district": "13", "street": "Lehel u. 1", "lat": 47.5, "lng": 19.08},
                {"city": "Pécs", "district": None, "street": "Király u. 3", "lat": 46.07, "lng": 18.23},
            ]
        ),
        encoding="utf-8",
    )
    out = load_addresses(fixture)
    assert len(out) == 2
    assert out[0].city == "Budapest"
    assert out[1].district is None


def test_load_addresses_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_addresses(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# assemble_dataset
# ---------------------------------------------------------------------------


def _addresses_for_required_cities() -> list[Address]:
    """One representative address per spec-required city + a couple of BP districts."""
    return (
        [Address(city="Budapest", district="11", street=f"BP-11 utca {i}", lat=47.47, lng=19.05) for i in range(50)]
        + [Address(city="Budapest", district="13", street=f"BP-13 utca {i}", lat=47.51, lng=19.08) for i in range(50)]
        + [
            Address(city=city, district=None, street=f"{city} utca {i}", lat=46.0 + idx, lng=18.0 + idx)
            for idx, city in enumerate(("Pécs", "Szeged", "Debrecen", "Miskolc", "Győr", "Sopron"), start=1)
            for i in range(20)
        ]
    )


async def test_assemble_dataset_produces_one_listing_per_target_address() -> None:
    addresses = _addresses_for_required_cities()
    listings = await assemble_dataset(
        addresses=addresses,
        target_count=50,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )
    assert len(listings) == 50
    assert all(isinstance(row, SyntheticListing) for row in listings)
    assert all(row.source_id.startswith("syn-") for row in listings)
    assert all(row.source_fingerprint.startswith("sha256:") for row in listings)


async def test_assemble_dataset_rows_match_normalized_listing_contract() -> None:
    addresses = _addresses_for_required_cities()
    listings = await assemble_dataset(
        addresses=addresses,
        target_count=25,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )

    for row in listings:
        data = row.to_dict()
        assert "area" in data
        assert "area_m2" not in data
        validated = NormalizedListing.model_validate(data)
        assert validated.source_id == row.source_id
        assert validated.source_fingerprint == row.source_fingerprint
        assert validated.area == row.area
        assert validated.raw_payload["source"] == "synthetic"


async def test_assemble_dataset_oversamples_when_address_pool_too_small() -> None:
    addresses = [Address(city="Budapest", district="13", street="X u. 1", lat=47.5, lng=19.08)]
    listings = await assemble_dataset(
        addresses=addresses,
        target_count=200,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )
    assert len(listings) == 200


async def test_assemble_dataset_zero_target_returns_empty() -> None:
    listings = await assemble_dataset(
        addresses=_addresses_for_required_cities(),
        target_count=0,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )
    assert listings == []


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _balanced_dataset(*, n: int = 1800, rng: random.Random | None = None) -> list[dict[str, object]]:
    """Build a fake listings list with the documented tier / noise distribution."""
    rng = rng if rng is not None else random.Random(SEED)
    tier_counts: dict[str, int] = {tier: round(n * p) for tier, p in TIER_PROBABILITIES.items()}
    # Adjust for rounding so the total matches `n` exactly.
    diff = n - sum(tier_counts.values())
    tier_counts["templated"] += diff

    tiers: list[str] = []
    for tier, count in tier_counts.items():
        tiers.extend([tier] * count)
    rng.shuffle(tiers)

    null_count = round(n * NULL_FIELD_FRACTION)
    bad_count = round(n * BAD_CASE_FRACTION)
    null_indices = set(rng.sample(range(n), k=null_count))
    bad_indices = set(rng.sample(range(n), k=bad_count))

    cities = list(REQUIRED_CITIES)
    return [
        {
            "source_id": f"syn-{i:05d}",
            "tier": tiers[i],
            "city": cities[i % len(cities)],
            "nulled_field": "rooms" if i in null_indices else None,
            "bad_case": i in bad_indices,
        }
        for i in range(n)
    ]


def test_validate_passes_for_balanced_dataset() -> None:
    report = validate(_balanced_dataset())
    assert report.ok, report.summary()
    assert report.total == 1800
    assert not report.issues


def test_validate_flags_total_outside_window() -> None:
    report = validate(_balanced_dataset(n=1000))  # below TARGET_COUNT_LOWER
    assert not report.ok
    assert any("outside target window" in issue for issue in report.issues)


def test_validate_flags_skewed_tier_distribution() -> None:
    skewed = [{"tier": "great", "city": "Budapest", "nulled_field": None, "bad_case": False} for _ in range(1800)]
    report = validate(skewed)
    assert not report.ok
    assert any("tier 'templated'" in issue for issue in report.issues)


def test_validate_flags_missing_required_city() -> None:
    dataset = _balanced_dataset()
    for entry in dataset:
        if entry["city"] == "Pécs":
            entry["city"] = "Budapest"
    report = validate(dataset)
    assert not report.ok
    assert any("missing required cities" in issue for issue in report.issues)
    assert "Pécs" in report.missing_cities


def test_validation_report_summary_is_human_readable() -> None:
    report = ValidationReport(ok=True, total=1800)
    assert "total=1800" in report.summary()


def test_load_listings_returns_list_of_dicts(tmp_path: Path) -> None:
    fixture = tmp_path / "out.json"
    fixture.write_text(json.dumps([{"city": "Budapest"}, {"city": "Pécs"}]), encoding="utf-8")
    rows = load_listings(fixture)
    assert rows == [{"city": "Budapest"}, {"city": "Pécs"}]


def test_load_listings_rejects_non_array(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.json"
    fixture.write_text('{"not": "list"}')
    with pytest.raises(ValueError, match="JSON array"):
        load_listings(fixture)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


async def test_assembled_dataset_passes_validation() -> None:
    """Full pipeline at TARGET_COUNT_LOWER must satisfy the §17.2.4 acceptance bar."""
    addresses = _addresses_for_required_cities()
    listings = await assemble_dataset(
        addresses=addresses,
        target_count=TARGET_COUNT_LOWER,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )
    rows = [row.to_dict() for row in listings]
    report = validate(rows)
    assert report.total == TARGET_COUNT_LOWER
    assert TARGET_COUNT_LOWER <= report.total <= TARGET_COUNT_UPPER
    # The full pipeline draws fresh tiers via the rng; the validator's ±3pp
    # tier tolerance should hold and noise fractions are constructed exactly.
    realised_cities = Counter(row["city"] for row in rows)
    for required in REQUIRED_CITIES:
        assert realised_cities[required] >= 1


async def test_emitted_property_type_is_canonical_and_build_type_preserved() -> None:
    """The KSH building-type slug must never leak into `property_type` — the
    filter/parser stack only knows the canonical labels (2026-07-04 catch:
    1,800 corpus rows were unmatchable by property_type_in=["Lakás"])."""

    listings = await assemble_dataset(
        addresses=_addresses_for_required_cities(),
        target_count=40,
        client=cast(LlmClient, _NoopLlmClient()),
        rng=random.Random(SEED),
    )

    canonical_labels = set(CANONICAL_PROPERTY_TYPE.values())
    for row in listings:
        assert row.property_type in canonical_labels, row.property_type
        build_type = row.features.get("build_type")
        assert build_type in CANONICAL_PROPERTY_TYPE, build_type
        assert CANONICAL_PROPERTY_TYPE[build_type] == row.property_type
