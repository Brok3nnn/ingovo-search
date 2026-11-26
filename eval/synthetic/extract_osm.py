"""§17.2.1 OSM Overpass address extractor.

Pulls real building addresses across Hungary from the OSM Overpass API and
writes them to `eval/synthetic/addresses.json` for the §17.2.4 assembler.
The output is opt-in: import the module to use the helpers, but the actual
network fetch only runs when `fetch_all_addresses()` is invoked.

Why not vendor a one-shot data file: OSM data drifts, and we want the
generator script committed so any developer can regenerate the addresses
with one `uv run` (especially when KSH publishes a new district boundary
or smaller-town list changes).

The script enforces a single global timeout + per-region retry. Overpass is
rate-limited (~ 2 concurrent slots); the helpers serialise calls
intentionally so we don't trigger 429s.

Per the design spec (§17.2.1, §7.4):
- Budapest districts 1-23, weighted by population (KSH 2021 census).
- 6 major regional cities: Pécs, Szeged, Debrecen, Miskolc, Győr, Sopron.
- 200 addresses from `RURAL_TARGETS` smaller towns for rural coverage.

Pure Python + httpx (already in §-1 dependency set). Tests stub the httpx
client so they run without network access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# The main endpoint can reject bare Python/httpx requests with 406 when no
# explicit representation is requested. Keep headers centralized so all mirrors
# receive the same polite identity.
OVERPASS_HEADERS = {
    "User-Agent": "IngovoSyntheticAddressExtractor/0.1 (local dev)",
    "Accept": "application/json,text/plain,*/*",
}
OVERPASS_URLS: tuple[str, ...] = (
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
)
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 5.0

# Min entry count for the acceptance test (`addresses.json` ≥ 2 500).
MIN_TOTAL_ADDRESSES = 2500

OUTPUT_PATH: Path = Path(__file__).resolve().parent / "addresses.json"

# District → target sample count, weighted roughly by 2021 KSH population.
# Numbers tuned so the BP total is ~1 800 (rest comes from regional cities +
# rural targets to hit ≥ 2 500).
BUDAPEST_DISTRICT_TARGETS: dict[str, int] = {
    "1": 40,
    "2": 100,
    "3": 150,
    "4": 110,
    "5": 40,
    "6": 60,
    "7": 60,
    "8": 100,
    "9": 90,
    "10": 90,
    "11": 150,
    "12": 80,
    "13": 130,
    "14": 130,
    "15": 90,
    "16": 90,
    "17": 90,
    "18": 100,
    "19": 80,
    "20": 80,
    "21": 90,
    "22": 90,
    "23": 80,
}

REGIONAL_CITY_TARGETS: dict[str, int] = {
    "Pécs": 100,
    "Szeged": 100,
    "Debrecen": 100,
    "Miskolc": 80,
    "Győr": 80,
    "Sopron": 60,
}

# 200 addresses spread across smaller HU towns for rural coverage.
RURAL_TARGETS: dict[str, int] = {
    "Veszprém": 25,
    "Kecskemét": 25,
    "Eger": 20,
    "Nyíregyháza": 25,
    "Székesfehérvár": 25,
    "Tatabánya": 20,
    "Szombathely": 25,
    "Zalaegerszeg": 20,
    "Békéscsaba": 15,
}


class _AsyncHttpClient(Protocol):
    # Protocol mirrors `httpx.AsyncClient.post` exactly; `timeout` is httpx's
    # own kwarg, not ours, so ASYNC109 doesn't apply here.
    async def post(
        self,
        url: str,
        *,
        data: dict[str, str] | None = ...,
        headers: dict[str, str] | None = ...,
        timeout: float | None = ...,  # noqa: ASYNC109
    ) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class Address:
    city: str
    district: str | None
    street: str
    lat: float
    lng: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "district": self.district,
            "street": self.street,
            "lat": self.lat,
            "lng": self.lng,
        }


def build_overpass_query(*, city: str, district: str | None, limit: int) -> str:
    """Construct a single Overpass QL query for one region."""
    address_selectors = (
        'node["addr:housenumber"]["addr:street"](area.searchArea);'
        'way["building"]["addr:housenumber"]["addr:street"](area.searchArea);'
    )
    # Budapest districts are `admin_level=9` relations with exact numeric refs.
    # First scope the relation lookup to Budapest; otherwise refs like "13"
    # match administrative areas all over the world.
    if district is not None:
        area_clause = (
            'area["name"="Budapest"]["boundary"="administrative"]->.budapestArea;'
            f'relation["boundary"="administrative"]["admin_level"="9"]["ref"="{district}"](area.budapestArea);'
            "map_to_area->.searchArea;"
        )
    else:
        area_clause = f'area["name"="{city}"]["admin_level"~"^(6|7|8)$"]->.searchArea;'

    return "[out:json][timeout:30];" + area_clause + f"({address_selectors});" + f"out center {limit};"


def parse_overpass_response(*, city: str, district: str | None, body: dict[str, Any]) -> list[Address]:
    """Walk the Overpass JSON `elements` list into `Address` records."""
    addresses: list[Address] = []
    elements = body.get("elements") or []
    for element in elements:
        tags = element.get("tags") or {}
        street = tags.get("addr:street")
        housenumber = tags.get("addr:housenumber")
        if not street or not housenumber:
            continue
        center = element.get("center") or {}
        lat = element.get("lat", center.get("lat"))
        lng = element.get("lon", center.get("lon"))
        if lat is None or lng is None:
            continue
        addresses.append(
            Address(
                city=city,
                district=district,
                street=f"{street} {housenumber}".strip(),
                lat=float(lat),
                lng=float(lng),
            )
        )
    return addresses


async def fetch_region(
    *,
    city: str,
    district: str | None,
    target: int,
    client: _AsyncHttpClient,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: object | None = None,
) -> list[Address]:
    """Fetch up to `target` addresses for one region with bounded retry."""
    query = build_overpass_query(city=city, district=district, limit=target)
    sleep_fn = sleep if sleep is not None else asyncio.sleep
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        url = OVERPASS_URLS[(attempt - 1) % len(OVERPASS_URLS)]
        try:
            response = await client.post(
                url,
                data={"data": query},
                headers=OVERPASS_HEADERS,
                timeout=timeout_seconds,
            )
            if response.status_code >= 500 or response.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"overpass status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return parse_overpass_response(city=city, district=district, body=body)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "extract_osm: region=%s/%s attempt=%d url=%s failed: %s",
                city,
                district,
                attempt,
                url,
                exc,
            )
            if attempt < retries:
                await sleep_fn(backoff_seconds * attempt)  # pyright: ignore[reportGeneralTypeIssues]
            continue

    logger.error("extract_osm: region=%s/%s gave up after %d retries", city, district, retries)
    assert last_error is not None
    raise last_error


def _iter_targets() -> Iterable[tuple[str, str | None, int]]:
    """Yield `(city, district, target_count)` for every region we sample."""
    for district, target in BUDAPEST_DISTRICT_TARGETS.items():
        yield "Budapest", district, target
    for city, target in REGIONAL_CITY_TARGETS.items():
        yield city, None, target
    for city, target in RURAL_TARGETS.items():
        yield city, None, target


async def fetch_all_addresses(
    *,
    client: _AsyncHttpClient | None = None,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Run the full sweep across BP districts + regional cities + rural towns."""
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    addresses: list[Address] = []
    failed_regions: list[str] = []
    try:
        for city, district, target in _iter_targets():
            try:
                addresses.extend(await fetch_region(city=city, district=district, target=target, client=client))
            except (httpx.HTTPError, ValueError) as exc:
                failed_regions.append(f"{city}/{district or '-'}")
                logger.error("extract_osm: giving up on region=%s/%s: %s", city, district, exc)
                continue
    finally:
        if owns_client:
            # `httpx.AsyncClient.aclose` is the documented shutdown path.
            await client.aclose()  # type: ignore[attr-defined]

    await asyncio.to_thread(
        output_path.write_text,
        json.dumps([a.to_dict() for a in addresses], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failed_regions:
        logger.error("extract_osm: %d failed regions: %s", len(failed_regions), ", ".join(failed_regions))
    logger.info("extract_osm: wrote %d addresses to %s", len(addresses), output_path)
    return output_path


def validate_address_file(*, path: Path = OUTPUT_PATH) -> tuple[bool, str]:
    """Return `(ok, message)` describing whether the file meets the acceptance bar."""
    if not path.exists():
        return False, f"{path} not found"
    raw_text = path.read_text(encoding="utf-8")
    try:
        raw: list[dict[str, Any]] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return False, f"{path} is not valid JSON: {exc}"
    if not isinstance(raw, list):
        return False, f"{path} must be a JSON array, got {type(raw).__name__}"
    if len(raw) < MIN_TOTAL_ADDRESSES:
        return False, f"only {len(raw)} addresses found, need ≥ {MIN_TOTAL_ADDRESSES}"
    return True, f"{len(raw)} addresses ✓"


async def main(argv: list[str] | None = None) -> int:
    mode: Literal["fetch", "validate"] = "fetch"
    if argv:
        if argv[0] == "--validate":
            mode = "validate"
        elif argv[0] == "--help":
            print(
                "usage: python -m eval.synthetic.extract_osm [--fetch|--validate]\n"
                "  --fetch     run the Overpass sweep and write addresses.json (default)\n"
                "  --validate  check addresses.json meets the ≥ 2 500 entry bar"
            )
            return 0

    if mode == "validate":
        ok, message = validate_address_file()
        print(message)
        return 0 if ok else 1

    await fetch_all_addresses()
    ok, message = validate_address_file()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover — CLI shim
    sys.exit(asyncio.run(main(sys.argv[1:])))
