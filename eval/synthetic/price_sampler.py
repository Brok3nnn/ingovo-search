"""§17.2.2 KSH price-distribution sampler.

Hard-coded price-per-m² distributions per district / city, drawn from public
KSH (Hungarian Statistical Office) stats and broker surveys for 2024-2025.
Values are *approximate* and intentionally embedded in code so the synthetic
generator stays deterministic and reproducible — founder refines the numbers
as a follow-up tweak.

The sampler draws from a lognormal-with-clipping distribution per
`(city, district, property_type)`, returns the realised price in HUF for a
given area_m2, and clips outliers more than `OUTLIER_CLIP_FACTOR=2.5x` away
from the median (KSH stats have a long noise tail; we keep the synthetic
dataset inside a plausible range for ranking-eval purposes).

Pure-logic module — no I/O, no external services. Reproducibility is the
caller's responsibility via `random.Random(seed)`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

PropertyType = Literal["panel", "tegla", "csaladi_haz", "sorhaz", "uj_epitesu"]

# Lognormal sigma applied to every district's median — captures the
# inter-listing variance broker surveys typically report (~25% std dev).
DEFAULT_SIGMA = 0.22

# Per-property-type multiplier on the base district median. `tegla` (brick) is
# the baseline. Values reflect typical HU broker pricing surveys.
PROPERTY_TYPE_MULTIPLIER: dict[PropertyType, float] = {
    "panel": 0.85,  # 10-15% discount vs brick
    "tegla": 1.0,  # baseline (brick)
    "csaladi_haz": 1.15,  # detached house premium varies; modest +15%
    "sorhaz": 1.05,  # row house slight premium
    "uj_epitesu": 1.40,  # new build commands a heavy premium
}

# Clip samples to median * this factor in either direction so the synthetic
# dataset stays inside a realistic envelope.
OUTLIER_CLIP_FACTOR = 2.5


@dataclass(frozen=True, slots=True)
class PriceDistribution:
    """Per-region price-per-m² distribution (HUF / m²)."""

    median_huf_per_m2: int
    sigma: float = DEFAULT_SIGMA


# Budapest districts — keys are the canonical numeric strings used in the
# `Property.district` column. Medians reflect 2024-2025 broker surveys (rounded
# to the nearest 50K HUF/m²). Founder refines as KSH publishes updates.
BUDAPEST_DISTRICT_PRICES: dict[str, PriceDistribution] = {
    "1": PriceDistribution(median_huf_per_m2=1_250_000),
    "2": PriceDistribution(median_huf_per_m2=1_000_000),
    "3": PriceDistribution(median_huf_per_m2=850_000),
    "4": PriceDistribution(median_huf_per_m2=700_000),
    "5": PriceDistribution(median_huf_per_m2=1_800_000),
    "6": PriceDistribution(median_huf_per_m2=1_350_000),
    "7": PriceDistribution(median_huf_per_m2=1_300_000),
    "8": PriceDistribution(median_huf_per_m2=1_100_000),
    "9": PriceDistribution(median_huf_per_m2=1_150_000),
    "10": PriceDistribution(median_huf_per_m2=750_000),
    "11": PriceDistribution(median_huf_per_m2=900_000),
    "12": PriceDistribution(median_huf_per_m2=1_100_000),
    "13": PriceDistribution(median_huf_per_m2=950_000),
    "14": PriceDistribution(median_huf_per_m2=850_000),
    "15": PriceDistribution(median_huf_per_m2=700_000),
    "16": PriceDistribution(median_huf_per_m2=700_000),
    "17": PriceDistribution(median_huf_per_m2=600_000),
    "18": PriceDistribution(median_huf_per_m2=650_000),
    "19": PriceDistribution(median_huf_per_m2=750_000),
    "20": PriceDistribution(median_huf_per_m2=650_000),
    "21": PriceDistribution(median_huf_per_m2=700_000),
    "22": PriceDistribution(median_huf_per_m2=800_000),
    "23": PriceDistribution(median_huf_per_m2=550_000),
}

# Major regional cities — applied when `district` is `None`.
CITY_PRICES: dict[str, PriceDistribution] = {
    "Budapest": PriceDistribution(median_huf_per_m2=1_000_000),  # fallback when district missing
    "Pécs": PriceDistribution(median_huf_per_m2=500_000),
    "Szeged": PriceDistribution(median_huf_per_m2=600_000),
    "Debrecen": PriceDistribution(median_huf_per_m2=600_000),
    "Miskolc": PriceDistribution(median_huf_per_m2=400_000),
    "Győr": PriceDistribution(median_huf_per_m2=700_000),
    "Sopron": PriceDistribution(median_huf_per_m2=750_000),
    # Balaton + Pest-agglomeration geo-coverage extension (2026-07-15) — the
    # fuzzy_geo eval queries need real supply here; medians are 2025-26 broker
    # ballparks, founder refines with KSH like the rest of the table.
    "Siófok": PriceDistribution(median_huf_per_m2=900_000),
    "Balatonfüred": PriceDistribution(median_huf_per_m2=1_100_000),
    "Keszthely": PriceDistribution(median_huf_per_m2=700_000),
    "Balatonlelle": PriceDistribution(median_huf_per_m2=800_000),
    "Tihany": PriceDistribution(median_huf_per_m2=1_300_000),
    "Balatonalmádi": PriceDistribution(median_huf_per_m2=950_000),
    "Érd": PriceDistribution(median_huf_per_m2=650_000),
    "Dunakeszi": PriceDistribution(median_huf_per_m2=750_000),
    "Budaörs": PriceDistribution(median_huf_per_m2=900_000),
    "Szentendre": PriceDistribution(median_huf_per_m2=900_000),
    "Gödöllő": PriceDistribution(median_huf_per_m2=700_000),
    "Vác": PriceDistribution(median_huf_per_m2=600_000),
}

# Catch-all for smaller towns / villages not in the maps above.
RURAL_FALLBACK = PriceDistribution(median_huf_per_m2=350_000)


def lookup_distribution(*, city: str, district: str | None) -> PriceDistribution:
    """Return the most specific distribution available for `(city, district)`."""
    if city == "Budapest" and district is not None:
        bd = BUDAPEST_DISTRICT_PRICES.get(district)
        if bd is not None:
            return bd
    by_city = CITY_PRICES.get(city)
    if by_city is not None:
        return by_city
    return RURAL_FALLBACK


def sample_price_huf(
    *,
    city: str,
    district: str | None,
    area_m2: float,
    property_type: PropertyType = "tegla",
    rng: random.Random | None = None,
) -> int:
    """Draw a plausible total listing price in HUF.

    `area_m2 <= 0` is treated as 1 m² so callers don't have to guard upstream;
    real downstream filters will discard such rows anyway.
    """
    if area_m2 <= 0:
        area_m2 = 1.0
    rng = rng if rng is not None else random.Random()
    dist = lookup_distribution(city=city, district=district)
    multiplier = PROPERTY_TYPE_MULTIPLIER[property_type]
    median = dist.median_huf_per_m2 * multiplier

    # Lognormal in price-per-m² space — `mu` set so that the median of the
    # distribution lines up with `median` exactly.
    mu = math.log(median)
    sampled_per_m2 = rng.lognormvariate(mu, dist.sigma)

    # Clip to median * OUTLIER_CLIP_FACTOR in either direction.
    upper = median * OUTLIER_CLIP_FACTOR
    lower = median / OUTLIER_CLIP_FACTOR
    clipped = max(lower, min(upper, sampled_per_m2))

    total = round(clipped * area_m2)
    return max(total, 1)  # never zero
