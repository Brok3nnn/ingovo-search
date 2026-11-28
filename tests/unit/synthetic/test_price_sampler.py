"""Unit tests for the KSH price-distribution sampler.

Acceptance per spec: "Generated 1000 prices for Budapest district 12 panel
apartments fall within KSH range." We verify the realised distribution
matches the configured median ± 1 lognormal sigma after the documented
outlier clip.
"""

from __future__ import annotations

import random
import statistics

import pytest
from eval.synthetic.price_sampler import (
    BUDAPEST_DISTRICT_PRICES,
    CITY_PRICES,
    DEFAULT_SIGMA,
    OUTLIER_CLIP_FACTOR,
    PROPERTY_TYPE_MULTIPLIER,
    RURAL_FALLBACK,
    PriceDistribution,
    lookup_distribution,
    sample_price_huf,
)

SEED = 1337


def test_lookup_distribution_budapest_district_match() -> None:
    dist = lookup_distribution(city="Budapest", district="13")
    assert dist == BUDAPEST_DISTRICT_PRICES["13"]


def test_lookup_distribution_budapest_unknown_district_falls_back_to_city() -> None:
    dist = lookup_distribution(city="Budapest", district="42")
    assert dist == CITY_PRICES["Budapest"]


def test_lookup_distribution_non_budapest_city() -> None:
    assert lookup_distribution(city="Pécs", district=None) == CITY_PRICES["Pécs"]


def test_lookup_distribution_unknown_city_falls_back_to_rural() -> None:
    assert lookup_distribution(city="Nagykáta", district=None) == RURAL_FALLBACK


def test_property_type_multiplier_table_covers_documented_types() -> None:
    assert set(PROPERTY_TYPE_MULTIPLIER) == {
        "panel",
        "tegla",
        "csaladi_haz",
        "sorhaz",
        "uj_epitesu",
    }
    # Brick is baseline; everything else scales relative to it.
    assert PROPERTY_TYPE_MULTIPLIER["tegla"] == 1.0
    assert PROPERTY_TYPE_MULTIPLIER["panel"] < 1.0
    assert PROPERTY_TYPE_MULTIPLIER["uj_epitesu"] > 1.0


def test_sample_price_huf_is_deterministic_with_seeded_rng() -> None:
    rng_a = random.Random(SEED)
    rng_b = random.Random(SEED)
    a = sample_price_huf(city="Budapest", district="13", area_m2=60.0, rng=rng_a)
    b = sample_price_huf(city="Budapest", district="13", area_m2=60.0, rng=rng_b)
    assert a == b


def test_sample_price_huf_is_positive_for_zero_area() -> None:
    """area_m2 <= 0 must not produce zero / negative prices."""
    price = sample_price_huf(city="Budapest", district="13", area_m2=0.0, rng=random.Random(SEED))
    assert price >= 1


def test_sample_price_huf_scales_linearly_with_area_at_fixed_seed() -> None:
    """Doubling area at the same seed should roughly double the price."""
    rng_a = random.Random(SEED)
    rng_b = random.Random(SEED)
    p_small = sample_price_huf(city="Budapest", district="13", area_m2=50.0, rng=rng_a)
    p_large = sample_price_huf(city="Budapest", district="13", area_m2=100.0, rng=rng_b)
    # Allow tight tolerance — rounding can shift by a few HUF per m².
    assert p_large == pytest.approx(p_small * 2, rel=1e-3)  # pyright: ignore[reportUnknownMemberType]


def test_sample_price_huf_clips_to_outlier_envelope() -> None:
    """Sampled prices must fall within `median * OUTLIER_CLIP_FACTOR` either way."""
    dist = BUDAPEST_DISTRICT_PRICES["12"]
    median = dist.median_huf_per_m2 * PROPERTY_TYPE_MULTIPLIER["panel"]
    upper = median * OUTLIER_CLIP_FACTOR * 100  # x100 = 100 m^2 area
    lower = (median / OUTLIER_CLIP_FACTOR) * 100

    rng = random.Random(SEED)
    samples = [
        sample_price_huf(city="Budapest", district="12", area_m2=100.0, property_type="panel", rng=rng)
        for _ in range(500)
    ]
    assert all(lower <= s <= upper for s in samples)


def test_acceptance_1000_samples_budapest_12_panel_match_ksh_band() -> None:
    """Acceptance: 1 000 BP-12 panel samples land near the KSH median."""
    rng = random.Random(SEED)
    samples = [
        sample_price_huf(city="Budapest", district="12", area_m2=60.0, property_type="panel", rng=rng)
        for _ in range(1000)
    ]

    dist = BUDAPEST_DISTRICT_PRICES["12"]
    expected_median = dist.median_huf_per_m2 * PROPERTY_TYPE_MULTIPLIER["panel"] * 60

    realised_median = statistics.median(samples)
    # Sample median should be within ±15% of the configured median.
    assert realised_median == pytest.approx(expected_median, rel=0.15)  # pyright: ignore[reportUnknownMemberType]


def test_property_type_multiplier_affects_price() -> None:
    """At the same seed and district, new-build > brick > panel."""
    rngs = {pt: random.Random(SEED) for pt in ("panel", "tegla", "uj_epitesu")}
    prices = {
        pt: sample_price_huf(
            city="Budapest",
            district="5",
            area_m2=60.0,
            property_type=pt,  # pyright: ignore[reportArgumentType]
            rng=rngs[pt],
        )
        for pt in rngs
    }
    assert prices["uj_epitesu"] > prices["tegla"] > prices["panel"]


def test_default_sigma_documented_value() -> None:
    """Pin DEFAULT_SIGMA so callers know what they're getting."""
    assert DEFAULT_SIGMA == 0.22


def test_distribution_dataclass_is_frozen() -> None:
    """Tables must be immutable so test code can't accidentally mutate them."""
    dist = PriceDistribution(median_huf_per_m2=500_000)
    with pytest.raises(AttributeError):
        dist.median_huf_per_m2 = 999_999  # type: ignore[misc]
