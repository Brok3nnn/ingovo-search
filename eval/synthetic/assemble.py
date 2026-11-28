"""Synthetic dataset assembler.

Turns the OSM address list into a 1 500-2 000 row synthetic dataset. Every row
validates as a `NormalizedListing`, the shape the seed command
(`ingovo-search-seed`) loads into `properties`.

Order of operations:

1. Load the addresses written by `extract_osm.py` (`addresses.json`).
2. For each address, synthesise the structured fields: building type, rooms,
   area, price (via `price_sampler.py`).
3. Generate a description per the tier distribution in `describe.py`, with its
   null-field and bad-case noise.
4. Write the resulting list to `data/synthetic_listings.json`.

Pure orchestration: the addresses, the price sampler and the random generator
are passed in, so tests run without filesystem reads on the real `addresses.json`.
Nothing here calls a model or the network.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.synthetic.describe import (
    BAD_CASE_FRACTION,
    NULL_FIELD_FRACTION,
    TIER_PROBABILITIES,
    DescriptionRecord,
    ListingSeed,
    build_type_label,
    generate_dataset,
)
from eval.synthetic.extract_osm import OUTPUT_PATH as DEFAULT_ADDRESS_PATH
from eval.synthetic.extract_osm import Address
from eval.synthetic.price_sampler import (
    PropertyType,
    sample_price_huf,
)

DEFAULT_OUTPUT_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "synthetic_listings.json"
DEFAULT_TARGET_COUNT = 1800
TARGET_COUNT_LOWER = 1500
TARGET_COUNT_UPPER = 2000
# Fixed so that a regeneration is reproducible. The bundled data file predates
# the seed, so re-running the assembler writes a different (equally valid) corpus.
DEFAULT_RANDOM_SEED = 20260713

# A trailing house-number token: "86/a", "70-72", "19".
_HOUSE_NUMBER_RE = re.compile(r"\s+\d\S*$")


def street_level(street: str) -> str:
    """The street without its house number: "Visegrádi utca 86/a" -> "Visegrádi utca".

    The listings are synthetic, so they name a real street but never a real
    building. Street names that contain a number ("Augusztus 20. utca") keep it.
    """
    return _HOUSE_NUMBER_RE.sub("", street.strip()) or street.strip()


# Property-type sampling distribution. Mirrors what HU broker surveys report
# for the unit / detached / new-build split across the country.
# The sampling vocabulary is a BUILDING-TYPE axis (KSH price statistics
# segment by construction). The listing-taxonomy `properties.property_type`
# must carry the canonical parser/filter labels instead — the building type
# survives as `features["build_type"]` (freetext-matchable: "panel", "tégla").
CANONICAL_PROPERTY_TYPE: dict[PropertyType, str] = {
    "panel": "Lakás",
    "tegla": "Lakás",
    "uj_epitesu": "Lakás",
    "csaladi_haz": "Ház",
    "sorhaz": "Ház",
}

PROPERTY_TYPE_PROBABILITIES: dict[PropertyType, float] = {
    "panel": 0.35,
    "tegla": 0.40,
    "csaladi_haz": 0.15,
    "sorhaz": 0.05,
    "uj_epitesu": 0.05,
}

# Rooms distribution per property type (rooms_min .. rooms_max inclusive).
ROOMS_RANGE_BY_TYPE: dict[PropertyType, tuple[int, int]] = {
    "panel": (1, 4),
    "tegla": (1, 5),
    "csaladi_haz": (3, 7),
    "sorhaz": (2, 5),
    "uj_epitesu": (1, 5),
}

# Area distribution per property type (m², approximate uniform sampling).
AREA_RANGE_BY_TYPE: dict[PropertyType, tuple[float, float]] = {
    "panel": (35.0, 90.0),
    "tegla": (40.0, 130.0),
    "csaladi_haz": (80.0, 250.0),
    "sorhaz": (70.0, 180.0),
    "uj_epitesu": (40.0, 150.0),
}


@dataclass(frozen=True, slots=True)
class SyntheticListing:
    """One row in `synthetic_listings.json`."""

    source_id: str
    source_fingerprint: str
    title: str
    description: str
    price: int
    transaction_type: str
    city: str
    district: str | None
    address: str
    property_type: str
    rooms: int | None
    area: float | None
    lat: float
    lng: float
    features: dict[str, Any]
    raw_payload: dict[str, Any]
    tier: str
    nulled_field: str | None
    bad_case: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Address loading
# ---------------------------------------------------------------------------


def load_addresses(path: Path = DEFAULT_ADDRESS_PATH) -> list[Address]:
    """Parse `addresses.json` into `Address` records."""
    if not path.exists():
        raise FileNotFoundError(f"addresses.json not found at {path}; run `extract_osm.py` first")
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    out: list[Address] = []
    for entry in raw:
        out.append(
            Address(
                city=str(entry["city"]),
                district=entry.get("district"),
                street=str(entry.get("street", "")),
                lat=float(entry["lat"]),
                lng=float(entry["lng"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Structured-field sampling
# ---------------------------------------------------------------------------


def sample_property_type(*, rng: random.Random) -> PropertyType:
    roll = rng.random()
    cumulative = 0.0
    for ptype, probability in PROPERTY_TYPE_PROBABILITIES.items():
        cumulative += probability
        if roll < cumulative:
            return ptype
    return "tegla"


def sample_rooms(*, property_type: PropertyType, rng: random.Random) -> int:
    low, high = ROOMS_RANGE_BY_TYPE[property_type]
    return rng.randint(low, high)


def sample_area_m2(*, property_type: PropertyType, rng: random.Random) -> float:
    low, high = AREA_RANGE_BY_TYPE[property_type]
    return round(rng.uniform(low, high), 1)


def build_seed(
    *,
    address: Address,
    rng: random.Random,
    price_sampler: Callable[..., int] = sample_price_huf,
) -> ListingSeed:
    """Synthesise structured fields + a plausible price for one address."""
    ptype = sample_property_type(rng=rng)
    rooms = sample_rooms(property_type=ptype, rng=rng)
    area = sample_area_m2(property_type=ptype, rng=rng)
    price = price_sampler(
        city=address.city,
        district=address.district,
        area_m2=area,
        property_type=ptype,
        rng=rng,
    )
    return ListingSeed(
        city=address.city,
        district=address.district,
        property_type=ptype,
        rooms=rooms,
        area_m2=area,
        price_huf=price,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_dataset(
    *,
    addresses: Sequence[Address],
    target_count: int = DEFAULT_TARGET_COUNT,
    rng: random.Random,
    price_sampler: Callable[..., int] = sample_price_huf,
) -> list[SyntheticListing]:
    """Build `target_count` synthetic listings from `addresses`.

    Sampling is **with replacement** when `len(addresses) < target_count`. We
    intentionally accept duplicate street + district pairs because the
    description generator's null / bad-case noise will diverge them downstream.
    """
    if target_count <= 0:
        return []

    chosen: list[Address] = (
        list(addresses)
        if len(addresses) >= target_count
        else [addresses[rng.randint(0, len(addresses) - 1)] for _ in range(target_count)]
    )
    if len(addresses) >= target_count:
        rng.shuffle(chosen)
        chosen = chosen[:target_count]

    seeds = [build_seed(address=a, rng=rng, price_sampler=price_sampler) for a in chosen]
    description_records = generate_dataset(seeds, rng=rng)

    return [
        _build_listing(record=record, address=address, index=index)
        for index, (record, address) in enumerate(zip(description_records, chosen, strict=True))
    ]


def _build_listing(*, record: DescriptionRecord, address: Address, index: int) -> SyntheticListing:
    seed = record.seed
    title_parts = [
        f"{seed.rooms} szobás" if seed.rooms is not None else "",
        build_type_label(seed.property_type),
        f"{address.city}",
    ]
    title = " · ".join(p for p in title_parts if p).strip(" ·")
    source_id = f"syn-{index:05d}"
    metadata = {
        "source": "synthetic",
        "lat": address.lat,
        "lng": address.lng,
        "tier": record.tier,
        "nulled_field": record.nulled_field,
        "bad_case": record.bad_case,
    }
    return SyntheticListing(
        source_id=source_id,
        source_fingerprint=_fingerprint_for(source_id=source_id, record=record, address=address),
        title=title or f"Listing {index}",
        description=record.description,
        price=int(seed.price_huf),
        transaction_type="sale",
        city=address.city,
        district=address.district,
        address=street_level(address.street),
        property_type=CANONICAL_PROPERTY_TYPE[seed.property_type],
        rooms=seed.rooms,
        area=seed.area_m2,
        lat=address.lat,
        lng=address.lng,
        features={"synthetic": metadata, "build_type": seed.property_type},
        raw_payload=metadata,
        tier=record.tier,
        nulled_field=record.nulled_field,
        bad_case=record.bad_case,
    )


def _fingerprint_for(*, source_id: str, record: DescriptionRecord, address: Address) -> str:
    seed = record.seed
    payload = {
        "source_id": source_id,
        "title_city": address.city,
        "district": address.district,
        "address": address.street,
        "description": record.description,
        "price": seed.price_huf,
        "property_type": seed.property_type,
        "rooms": seed.rooms,
        "area": seed.area_m2,
        "tier": record.tier,
        "nulled_field": record.nulled_field,
        "bad_case": record.bad_case,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:24]}"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for flags like --target-count, --output
    rng = random.Random(DEFAULT_RANDOM_SEED)
    listings = assemble_dataset(addresses=load_addresses(), rng=rng)
    DEFAULT_OUTPUT_PATH.write_text(
        json.dumps([row.to_dict() for row in listings], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(listings)} synthetic listings to {DEFAULT_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    sys.exit(main(sys.argv[1:]))


__all__ = [
    "AREA_RANGE_BY_TYPE",
    "BAD_CASE_FRACTION",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TARGET_COUNT",
    "NULL_FIELD_FRACTION",
    "PROPERTY_TYPE_PROBABILITIES",
    "ROOMS_RANGE_BY_TYPE",
    "TARGET_COUNT_LOWER",
    "TARGET_COUNT_UPPER",
    "TIER_PROBABILITIES",
    "SyntheticListing",
    "assemble_dataset",
    "build_seed",
    "load_addresses",
    "sample_area_m2",
    "sample_property_type",
    "sample_rooms",
    "street_level",
]
