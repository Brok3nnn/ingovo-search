"""Extract the points of interest in `data/osm_pois_*.json` from OpenStreetMap.

One Overpass API query per (category, area) pair, for the ten categories the
parser and the proximity scoring know, in Budapest and six regional cities:

    [out:json][timeout:120];
    area["name"="Budapest"]["boundary"="administrative"]->.searchArea;
    (node["amenity"="school"](area.searchArea);way["amenity"="school"](area.searchArea););
    out center;

Ways and relations (schools or parks drawn as areas) resolve to one point via
`out center`. Calls are serialised (Overpass is rate-limited), each pair gets
three attempts with backoff that rotate between two public mirrors, and a pair
that still fails is reported instead of discarding the whole run. Each record
keeps the OSM element id (`node/123`) and the area it was found in.

    uv run python scripts/extract_osm_pois.py [output.json]

The output defaults to `data/osm_pois_<today>.json`; pass it to the seed command
with `uv run ingovo-search-seed --pois <file>`. The data is © OpenStreetMap
contributors, under the ODbL 1.0 (see `data/README.md`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

logger = logging.getLogger("extract_osm_pois")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# The main instance regularly answers 504 "server too busy"; retries rotate
# through a public mirror instead of hammering the same overloaded host.
OVERPASS_URLS: tuple[str, ...] = (
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
)
# The main endpoint rejects a bare httpx request with 406 when no explicit
# representation is asked for: send a polite identity and an Accept header.
OVERPASS_HEADERS: dict[str, str] = {
    "User-Agent": "ingovo-search-poi-extractor (https://github.com/Brok3nnn/ingovo-search)",
    "Accept": "application/json,text/plain,*/*",
}
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 5.0

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AREAS: tuple[str, ...] = (
    "Budapest",
    "Pécs",
    "Szeged",
    "Debrecen",
    "Miskolc",
    "Győr",
    "Sopron",
)

# category -> Overpass selectors; the categories are the `PoiCategory` values of
# `recommender/parser_schema.py` (without the parser-level "transit" aggregate).
POI_CATEGORY_SELECTORS: dict[str, tuple[str, ...]] = {
    "school": ('node["amenity"="school"]', 'way["amenity"="school"]'),
    "kindergarten": ('node["amenity"="kindergarten"]', 'way["amenity"="kindergarten"]'),
    "metro": ('node["railway"="station"]["station"="subway"]',),
    "tram": ('node["railway"="tram_stop"]',),
    "bus": ('node["highway"="bus_stop"]',),
    "supermarket": ('node["shop"="supermarket"]', 'way["shop"="supermarket"]'),
    "pharmacy": ('node["amenity"="pharmacy"]', 'way["amenity"="pharmacy"]'),
    "park": ('way["leisure"="park"]', 'relation["leisure"="park"]'),
    "hospital": ('node["amenity"="hospital"]', 'way["amenity"="hospital"]'),
    "playground": ('node["leisure"="playground"]', 'way["leisure"="playground"]'),
}

POI_CATEGORIES: tuple[str, ...] = tuple(POI_CATEGORY_SELECTORS)


@dataclass(frozen=True, slots=True)
class PoiRecord:
    category: str
    name: str | None
    lat: float
    lng: float
    source_id: str
    area: str


def build_overpass_query(category: str, area_name: str, *, timeout_s: int = 120) -> str:
    """Overpass QL for one (category, administrative area) pair."""
    selectors = POI_CATEGORY_SELECTORS[category]
    body = "".join(f"{selector}(area.searchArea);" for selector in selectors)
    return (
        f"[out:json][timeout:{timeout_s}];"
        f'area["name"="{area_name}"]["boundary"="administrative"]->.searchArea;'
        f"({body});out center;"
    )


def _record(element: dict[str, Any], category: str, area: str) -> PoiRecord | None:
    """One Overpass element as a record. Nodes carry lat/lon; ways and relations carry `center`."""
    lat_raw: Any = element.get("lat")
    lng_raw: Any = element.get("lon")
    if lat_raw is None or lng_raw is None:
        center = cast("dict[str, Any]", element.get("center") or {})
        lat_raw = center.get("lat")
        lng_raw = center.get("lon")
    if lat_raw is None or lng_raw is None:
        return None
    tags = cast("dict[str, Any]", element.get("tags") or {})
    name_raw = tags.get("name")
    return PoiRecord(
        category=category,
        name=str(name_raw) if name_raw is not None else None,
        lat=float(lat_raw),
        lng=float(lng_raw),
        source_id=f"{element.get('type', 'node')}/{element.get('id')}",
        area=area,
    )


def parse_overpass_elements(payload: dict[str, Any], category: str, area: str) -> list[PoiRecord]:
    """Overpass JSON -> records, skipping elements without coordinates."""
    elements = cast("list[dict[str, Any]]", payload.get("elements", []))
    records = (_record(element, category, area) for element in elements)
    return [record for record in records if record is not None]


class _PostClient(Protocol):
    async def post(self, url: str, *, data: dict[str, str], headers: dict[str, str]) -> httpx.Response: ...


async def fetch_pois(
    client: _PostClient,
    category: str,
    area_name: str,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> list[PoiRecord]:
    """One category x area Overpass call with retry and backoff, rotating mirrors."""
    query = build_overpass_query(category, area_name)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        url = OVERPASS_URLS[(attempt - 1) % len(OVERPASS_URLS)]
        try:
            response = await client.post(url, data={"data": query}, headers=OVERPASS_HEADERS)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            last_error = error
            logger.warning(
                "overpass fetch failed (attempt %d/%d) url=%s category=%s area=%s: %s",
                attempt,
                retries,
                url,
                category,
                area_name,
                error,
            )
            if attempt < retries:
                await asyncio.sleep(backoff_seconds * attempt)
        else:
            return parse_overpass_elements(payload, category, area_name)
    raise RuntimeError(f"Overpass fetch failed for {category}/{area_name}: {last_error}")


async def fetch_all_pois(
    areas: tuple[str, ...] = DEFAULT_AREAS,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[PoiRecord], list[str]]:
    """Serialised fan-out over every (category, area) pair; returns the records and the failed pairs."""
    records: list[PoiRecord] = []
    failed_pairs: list[str] = []
    async with httpx.AsyncClient(timeout=timeout_seconds, headers=OVERPASS_HEADERS) as client:
        for area in areas:
            for category in POI_CATEGORIES:
                try:
                    fetched = await fetch_pois(client, category, area)
                except RuntimeError as error:
                    failed_pairs.append(f"{category}/{area}")
                    logger.warning("giving up on %s/%s: %s", category, area, error)
                    continue
                logger.info("fetched %d %s POIs for %s", len(fetched), category, area)
                records.extend(fetched)
    return records, failed_pairs


def write_records(records: list[PoiRecord], path: Path) -> None:
    """The same shape and formatting as the bundled extract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.now(UTC).date().isoformat()
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / f"osm_pois_{today}.json"
    records, failed_pairs = asyncio.run(fetch_all_pois())
    write_records(records, output)
    print(f"wrote {len(records)} POI records to {output}")  # noqa: T201
    if failed_pairs:
        print(f"FAILED pairs ({len(failed_pairs)}): {', '.join(failed_pairs)}")  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
