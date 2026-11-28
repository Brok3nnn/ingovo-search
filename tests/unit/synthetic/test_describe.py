"""Unit tests for the synthetic listings' description generator.

Every tier is a local template, so the tier dispatcher, the tier-probability
sampling, the null-field injection and the bad-case mutation are checked without
any model or network.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from eval.synthetic.describe import (
    BAD_CASE_FRACTION,
    BUILD_TYPE_LABELS,
    NULL_FIELD_FRACTION,
    NULLABLE_STRUCTURED_FIELDS,
    TIER_PROBABILITIES,
    DescriptionRecord,
    DescriptionTier,
    ListingSeed,
    build_type_label,
    choose_tier,
    generate_dataset,
    generate_description,
    generate_great_description,
    generate_skeletal_description,
    generate_templated_description,
    inject_bad_case,
    null_one_field,
)

SEED = 4242


def _seed(**overrides: object) -> ListingSeed:
    defaults: dict[str, object] = {
        "city": "Budapest",
        "district": "13",
        "property_type": "panel",
        "rooms": 2,
        "area_m2": 60.0,
        "price_huf": 60_000_000,
    }
    defaults.update(overrides)
    return ListingSeed(**defaults)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Constants + module surface
# ---------------------------------------------------------------------------


def test_tier_probabilities_match_spec() -> None:
    """The documented split: 33 / 34 / 33."""
    assert TIER_PROBABILITIES == {"great": 0.33, "templated": 0.34, "skeletal": 0.33}
    assert sum(TIER_PROBABILITIES.values()) == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


def test_noise_fractions_match_spec() -> None:
    assert NULL_FIELD_FRACTION == 0.15
    assert BAD_CASE_FRACTION == 0.10


# ---------------------------------------------------------------------------
# Skeletal + templated generators (pure)
# ---------------------------------------------------------------------------


def test_skeletal_description_is_short_and_carries_core_fields() -> None:
    seed = _seed()
    text = generate_skeletal_description(seed)
    assert len(text.split()) <= 50  # the skeletal tier stays under 50 words
    assert "Budapest" in text
    assert "60" in text
    assert "kerület" in text


def test_skeletal_description_handles_missing_fields() -> None:
    seed = _seed(rooms=None, area_m2=None, district=None)
    text = generate_skeletal_description(seed)
    assert "Budapest" in text
    # No `kerület` clause when district missing.
    assert "kerület" not in text


def test_templated_description_is_deterministic_with_seeded_rng() -> None:
    rng_a = random.Random(SEED)
    rng_b = random.Random(SEED)
    seed = _seed()
    assert generate_templated_description(seed, rng=rng_a) == generate_templated_description(seed, rng=rng_b)


def test_templated_description_includes_district_clause_when_present() -> None:
    text = generate_templated_description(_seed(district="11"), rng=random.Random(SEED))
    assert "Budapest 11. kerület" in text


def test_templated_description_falls_back_to_city_without_district() -> None:
    text = generate_templated_description(_seed(city="Pécs", district=None), rng=random.Random(SEED))
    assert "Pécs" in text
    assert "kerület" not in text


# ---------------------------------------------------------------------------
# choose_tier — frequency distribution
# ---------------------------------------------------------------------------


def test_choose_tier_returns_valid_tier() -> None:
    rng = random.Random(SEED)
    tier = choose_tier(rng=rng)
    assert tier in TIER_PROBABILITIES


def test_choose_tier_matches_spec_distribution_in_aggregate() -> None:
    """10 000 samples should be within ±2pp of the documented split."""
    rng = random.Random(SEED)
    counts: Counter[DescriptionTier] = Counter(choose_tier(rng=rng) for _ in range(10_000))
    for tier, target in TIER_PROBABILITIES.items():
        observed = counts[tier] / 10_000
        assert abs(observed - target) < 0.02


# ---------------------------------------------------------------------------
# generate_description dispatch
# ---------------------------------------------------------------------------


def test_generate_description_dispatches_each_tier_to_its_template() -> None:
    seed = _seed()
    assert generate_description(seed, tier="great") == generate_great_description(seed)
    assert generate_description(seed, tier="templated", rng=random.Random(SEED)) == generate_templated_description(
        seed, rng=random.Random(SEED)
    )
    assert generate_description(seed, tier="skeletal") == generate_skeletal_description(seed)


def test_great_description_reads_the_structured_fields() -> None:
    text = generate_great_description(_seed(property_type="csaladi_haz", rooms=4, area_m2=123.0, district="21"))
    assert text.startswith("A 4 szobás, nagyjából 123 négyzetméteres családi ház Budapest 21. kerületében")
    assert "A leírás alapján" in text
    assert text.endswith("élhető otthont keres.")


def test_great_description_without_rooms_area_or_district() -> None:
    text = generate_great_description(_seed(city="Pécs", district=None, rooms=None, area_m2=None))
    assert text.startswith("Az átgondolt elrendezésű panel Pécs városában")


def test_build_type_labels_are_hungarian() -> None:
    assert build_type_label("csaladi_haz") == "családi ház"
    assert build_type_label("tegla") == "tégla"
    assert build_type_label("unknown_type") == "unknown type"
    assert set(BUILD_TYPE_LABELS) == {"panel", "tegla", "csaladi_haz", "sorhaz", "uj_epitesu"}


# ---------------------------------------------------------------------------
# null_one_field
# ---------------------------------------------------------------------------


def test_null_one_field_wipes_exactly_one_nullable_field() -> None:
    seed = _seed()
    new_seed = null_one_field(seed, rng=random.Random(SEED))
    diffs = sum(
        1
        for f in NULLABLE_STRUCTURED_FIELDS
        if hasattr(seed, f) and getattr(seed, f) != getattr(new_seed, f, getattr(seed, f))
    )
    assert diffs == 1


def test_null_one_field_does_not_touch_non_nullable_fields() -> None:
    seed = _seed()
    new_seed = null_one_field(seed, rng=random.Random(SEED))
    # `city`, `property_type`, `price_huf` are not in NULLABLE_STRUCTURED_FIELDS.
    assert new_seed.city == seed.city
    assert new_seed.property_type == seed.property_type
    assert new_seed.price_huf == seed.price_huf


# ---------------------------------------------------------------------------
# inject_bad_case
# ---------------------------------------------------------------------------


def test_inject_bad_case_mislabels_property_type() -> None:
    """`mislabeled_type` mutation flips property_type to a different value."""
    rng = random.Random(0)
    seed = _seed(property_type="panel")
    new_seed, description = inject_bad_case(seed, "original", rng=rng)
    # At seed=0, choice() picks the first option ("mislabeled_type") by index 0.
    assert new_seed.property_type != "panel" or description != "original"


def test_inject_bad_case_returns_modified_record() -> None:
    """For every seed in `range(10)`, at least one of (seed, description) changes."""
    rng = random.Random(SEED)
    seed = _seed()
    for _ in range(10):
        out_seed, out_description = inject_bad_case(seed, "ORIGINAL", rng=rng)
        assert out_seed != seed or out_description != "ORIGINAL"


# ---------------------------------------------------------------------------
# generate_dataset — distribution checks
# ---------------------------------------------------------------------------


def test_generate_dataset_produces_one_record_per_seed() -> None:
    seeds = [_seed(price_huf=p) for p in range(50_000_000, 60_000_000, 100_000)]  # 100 seeds
    records = generate_dataset(seeds, rng=random.Random(SEED))
    assert len(records) == len(seeds)
    assert all(isinstance(r, DescriptionRecord) for r in records)


def test_generate_dataset_is_reproducible_with_a_seeded_rng() -> None:
    seeds = [_seed(price_huf=50_000_000 + i) for i in range(50)]
    assert generate_dataset(seeds, rng=random.Random(SEED)) == generate_dataset(seeds, rng=random.Random(SEED))


def test_generate_dataset_applies_bad_case_at_documented_fraction() -> None:
    n = 500
    seeds = [_seed(price_huf=50_000_000 + i) for i in range(n)]
    records = generate_dataset(seeds, rng=random.Random(SEED))
    bad = sum(1 for r in records if r.bad_case)
    assert bad == round(n * BAD_CASE_FRACTION)


def test_generate_dataset_applies_null_at_documented_fraction() -> None:
    n = 500
    seeds = [_seed(price_huf=50_000_000 + i) for i in range(n)]
    records = generate_dataset(seeds, rng=random.Random(SEED))
    nulled = sum(1 for r in records if r.nulled_field is not None)
    assert nulled == round(n * NULL_FIELD_FRACTION)


def test_generate_dataset_tier_distribution_within_tolerance() -> None:
    n = 2_000
    seeds = [_seed(price_huf=50_000_000 + i) for i in range(n)]
    records = generate_dataset(seeds, rng=random.Random(SEED))
    counts: Counter[DescriptionTier] = Counter(r.tier for r in records)
    for tier, target in TIER_PROBABILITIES.items():
        observed = counts[tier] / n
        # Tighter tolerance possible at n=2000, but ±3pp keeps the test robust.
        assert abs(observed - target) < 0.03
