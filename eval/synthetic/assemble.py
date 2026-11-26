"""§17.2.4 synthetic dataset assembler.

Coordinates §17.2.1-§17.2.3 outputs into a 1 500-2 000 row synthetic dataset.
Every row validates as a `NormalizedListing`, the shape the seed command
(`ingovo-search-seed`) loads into `properties`.

Order of operations:

1. Load addresses from §17.2.1 output (`addresses.json`).
2. For each address, synthesise the structured fields: property type, rooms,
   area, price (via §17.2.2 sampler).
3. Generate description per the §17.2.3 tier distribution + null + bad-case
   noise.
4. Write the resulting list to `data/synthetic_listings.json`.

Pure orchestration — `address_loader` / `price_sampler` / `description_client`
are all injected so tests run without LLM contact or filesystem reads on the
real `addresses.json`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
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
    generate_dataset,
)
from eval.synthetic.extract_osm import OUTPUT_PATH as DEFAULT_ADDRESS_PATH
from eval.synthetic.extract_osm import Address
from eval.synthetic.price_sampler import (
    PropertyType,
    sample_price_huf,
)
from ingovo_search.lib.llm import LlmClient, LlmRequest

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "synthetic_listings.json"
DEFAULT_TARGET_COUNT = 1800
TARGET_COUNT_LOWER = 1500
TARGET_COUNT_UPPER = 2000


class LocalGreatDescriptionClient:
    """Deterministic no-network stand-in for the gated great-tier LLM path.

    The founder-reviewed Gemini prompt can replace this for a small "great"
    sample later. For the bootstrap corpus we need stable, LLM-free descriptions
    that still exercise the richer-description tier distribution.
    """

    async def complete(self, req: LlmRequest) -> str:
        fields = _prompt_fields(req.user)
        city = fields.get("city", "a környék")
        district = fields.get("district")
        property_type = fields.get("property_type", "ingatlan").replace("_", " ")
        rooms = fields.get("rooms")
        area = fields.get("area_m2")

        location = f"Budapest {district}. kerületében" if city == "Budapest" and district else f"{city} városában"
        room_sentence = f"A {rooms} szobás" if rooms and rooms != "None" else "Az átgondolt elrendezésű"
        area_sentence = f", nagyjából {float(area):.0f} négyzetméteres" if area and area != "None" else ""
        return (
            f"{room_sentence}{area_sentence} {property_type} {location} kínál kényelmes, jól használható tereket. "
            "A leírás alapján világos nappali, praktikus tárolási lehetőségek és mindennapi élethez közeli "
            "szolgáltatások várhatók a közelben. "
            "Jó választás lehet annak, aki rendezett környezetet, kiszámítható közlekedést és hosszabb távon is "
            "élhető otthont keres."
        )


def _prompt_fields(user_prompt: str) -> dict[str, str]:
    """Extract the final `key=value` field line from `render_great_user_prompt`."""
    if "\n\n" not in user_prompt:
        return {}
    raw_fields = user_prompt.rsplit("\n\n", 1)[-1]
    out: dict[str, str] = {}
    for part in raw_fields.split(", "):
        key, sep, value = part.partition("=")
        if sep:
            out[key] = value
    return out


# Property-type sampling distribution. Mirrors what HU broker surveys report
# for the unit / detached / new-build split across the country.
# The §17.2 sampling vocabulary is a BUILDING-TYPE axis (KSH price statistics
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
        raise FileNotFoundError(f"addresses.json not found at {path}; run §17.2.1 `extract_osm` first")
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


async def assemble_dataset(
    *,
    addresses: Sequence[Address],
    target_count: int = DEFAULT_TARGET_COUNT,
    client: LlmClient,
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
    description_records = await generate_dataset(seeds, client=client, rng=rng)

    return [
        _build_listing(record=record, address=address, index=index)
        for index, (record, address) in enumerate(zip(description_records, chosen, strict=True))
    ]


def _build_listing(*, record: DescriptionRecord, address: Address, index: int) -> SyntheticListing:
    seed = record.seed
    title_parts = [
        f"{seed.rooms} szobás" if seed.rooms is not None else "",
        seed.property_type.replace("_", " "),
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
        address=address.street,
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


async def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for flags like --target-count, --output
    rng = random.Random()
    addresses = load_addresses()
    listings = await assemble_dataset(
        addresses=addresses,
        client=LocalGreatDescriptionClient(),
        rng=rng,
    )
    await asyncio.to_thread(
        DEFAULT_OUTPUT_PATH.write_text,
        json.dumps([row.to_dict() for row in listings], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(listings)} synthetic listings to {DEFAULT_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    sys.exit(asyncio.run(main(sys.argv[1:])))


__all__ = [
    "AREA_RANGE_BY_TYPE",
    "BAD_CASE_FRACTION",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TARGET_COUNT",
    "NULL_FIELD_FRACTION",
    "PROPERTY_TYPE_PROBABILITIES",
    "ROOMS_RANGE_BY_TYPE",
    "TARGET_COUNT_LOWER",
    "TARGET_COUNT_UPPER",
    "TIER_PROBABILITIES",
    "LocalGreatDescriptionClient",
    "SyntheticListing",
    "assemble_dataset",
    "build_seed",
    "load_addresses",
    "sample_area_m2",
    "sample_property_type",
    "sample_rooms",
]
