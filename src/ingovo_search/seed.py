"""`ingovo-search-seed` — create the schema and load the demo data.

1. Apply `db/schema.sql` (idempotent).
2. Upsert the synthetic listings (`data/synthetic_listings.json`), each validated as a
   `NormalizedListing`, with its PostGIS point.
3. Upsert the OpenStreetMap points of interest (`data/osm_pois_2026-07-13.json`) and
   precompute every listing's distance to the nearest POI per category.
4. Embed every listing that has no embedding yet with Cohere — only when
   `COHERE_API_KEY` is set. Without it the seed still succeeds and says what is missing:
   the hard filters work, but the semantic ranking terms need the embeddings, and a
   search needs the key to embed its own query anyway.

Re-running is safe: rows are keyed by (source, source_id) and only missing
embeddings are computed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

import structlog
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from ingovo_search.db.session import get_async_session_factory, get_engine
from ingovo_search.ingestion.embed_listing import embed_listings_batch
from ingovo_search.ingestion.poi_distances import refresh_poi_distances
from ingovo_search.ingestion.types import NormalizedListing
from ingovo_search.lib.embeddings import CohereEmbeddingClient, EmbeddingClient
from ingovo_search.lib.env import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LISTINGS_PATH = REPO_ROOT / "data" / "synthetic_listings.json"
DEFAULT_POIS_PATH = REPO_ROOT / "data" / "osm_pois_2026-07-13.json"
LISTING_SOURCE = "synthetic"
POI_SOURCE = "osm"
# Stable ids across machines and re-runs: uuid5 of the source and its own id.
_ID_NAMESPACE = uuid.UUID("6f2b9c1e-3d4a-4b5c-9e8f-0a1b2c3d4e5f")
EMBED_BATCH = 96

_logger = structlog.get_logger("ingovo_search.seed")

UPSERT_LISTING_SQL = """
INSERT INTO properties (
    id, title, description, price_huf, currency, transaction_type, city, district, address,
    property_type, rooms, area_m2, floor, total_floors, year_built, heating, status, features,
    images, photo_count, source, source_id, geo
) VALUES (
    :id, :title, :description, :price_huf, 'HUF', :transaction_type, :city, :district, :address,
    :property_type, :rooms, :area_m2, :floor, :total_floors, :year_built, :heating, 'active',
    CAST(:features AS jsonb), CAST(:images AS varchar[]), :photo_count, :source, :source_id,
    CASE WHEN CAST(:lat AS double precision) IS NULL OR CAST(:lng AS double precision) IS NULL THEN NULL
         ELSE ST_SetSRID(ST_MakePoint(CAST(:lng AS double precision), CAST(:lat AS double precision)), 4326)::geography
    END
)
ON CONFLICT (source, source_id) DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    price_huf = EXCLUDED.price_huf,
    transaction_type = EXCLUDED.transaction_type,
    city = EXCLUDED.city,
    district = EXCLUDED.district,
    address = EXCLUDED.address,
    property_type = EXCLUDED.property_type,
    rooms = EXCLUDED.rooms,
    area_m2 = EXCLUDED.area_m2,
    floor = EXCLUDED.floor,
    total_floors = EXCLUDED.total_floors,
    year_built = EXCLUDED.year_built,
    heating = EXCLUDED.heating,
    features = EXCLUDED.features,
    images = EXCLUDED.images,
    photo_count = EXCLUDED.photo_count,
    geo = EXCLUDED.geo,
    updated_at = now()
"""

UPSERT_POI_SQL = """
INSERT INTO poi (id, category, name, geo, source, source_id)
VALUES (
    :id, :category, :name,
    ST_SetSRID(ST_MakePoint(CAST(:lng AS double precision), CAST(:lat AS double precision)), 4326)::geography,
    :source, :source_id
)
ON CONFLICT (source, source_id) DO UPDATE SET
    category = EXCLUDED.category, name = EXCLUDED.name, geo = EXCLUDED.geo
"""

MISSING_EMBEDDINGS_SQL = """
SELECT id FROM properties
WHERE source = :source AND (factual_embedding IS NULL OR descriptive_embedding IS NULL)
ORDER BY id
"""


@dataclass(frozen=True, slots=True)
class SeedSummary:
    listings: int
    skipped: int
    pois: int
    poi_distances: int
    embedded: int
    embeddings_skipped_reason: str | None

    def format(self) -> str:
        lines = [
            f"listings upserted:          {self.listings} (skipped as invalid: {self.skipped})",
            f"points of interest:         {self.pois}",
            f"listings with POI distances: {self.poi_distances}",
            f"listings embedded:          {self.embedded}",
        ]
        if self.embeddings_skipped_reason:
            lines.append(f"embeddings skipped:         {self.embeddings_skipped_reason}")
        return "\n".join(lines)


def stable_id(source: str, source_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source}:{source_id}"))


def _finite_coordinate(value: object, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if low <= number <= high else None


def listing_params(raw: dict[str, Any]) -> dict[str, object] | None:
    """One synthetic row as the bind parameters of `UPSERT_LISTING_SQL`; None if it is invalid."""
    try:
        listing = NormalizedListing.model_validate(raw)
    except ValidationError:
        return None
    return {
        "id": stable_id(LISTING_SOURCE, listing.source_id),
        "title": listing.title,
        "description": listing.description or None,
        "price_huf": listing.price if listing.price > 0 else None,
        "transaction_type": listing.transaction_type,
        "city": listing.city,
        "district": listing.district,
        "address": listing.address,
        "property_type": listing.property_type,
        "rooms": str(listing.rooms) if listing.rooms is not None else None,
        "area_m2": listing.area,
        "floor": listing.floor,
        "total_floors": listing.total_floors,
        "year_built": listing.year_built,
        "heating": listing.heating,
        "features": json.dumps(listing.features, ensure_ascii=False),
        "images": listing.images,
        "photo_count": len(listing.images),
        "source": LISTING_SOURCE,
        "source_id": listing.source_id,
        "lat": _finite_coordinate(raw.get("lat"), -90.0, 90.0),
        "lng": _finite_coordinate(raw.get("lng"), -180.0, 180.0),
    }


def poi_params(raw: dict[str, Any]) -> dict[str, object] | None:
    """One OSM point of interest as the bind parameters of `UPSERT_POI_SQL`; None if unusable."""
    lat = _finite_coordinate(raw.get("lat"), -90.0, 90.0)
    lng = _finite_coordinate(raw.get("lng"), -180.0, 180.0)
    category = raw.get("category")
    source_id = raw.get("source_id")
    if lat is None or lng is None or not isinstance(category, str) or not isinstance(source_id, str):
        return None
    name = raw.get("name")
    return {
        "id": stable_id(POI_SOURCE, source_id),
        "category": category,
        "name": name if isinstance(name, str) else None,
        "lat": lat,
        "lng": lng,
        "source": POI_SOURCE,
        "source_id": source_id,
    }


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    return [cast("dict[str, Any]", row) for row in cast("list[object]", payload) if isinstance(row, dict)]


def schema_sql() -> str:
    return resources.files("ingovo_search.db").joinpath("schema.sql").read_text(encoding="utf-8")


async def apply_schema(connection: AsyncConnection) -> None:
    """Run the whole schema script in one round trip (asyncpg's simple-query protocol)."""
    raw = await connection.get_raw_connection()
    driver = cast("Any", raw.driver_connection)
    await driver.execute(schema_sql())


async def _upsert(session: AsyncSession, sql: str, rows: Sequence[dict[str, object]]) -> None:
    for start in range(0, len(rows), 500):
        await session.execute(text(sql), list(rows[start : start + 500]))


async def _embed_missing(session: AsyncSession, client: EmbeddingClient, limit: int | None) -> int:
    ids = [str(row.id) for row in await session.execute(text(MISSING_EMBEDDINGS_SQL), {"source": LISTING_SOURCE})]
    if limit is not None:
        ids = ids[:limit]
    for start in range(0, len(ids), EMBED_BATCH):
        await embed_listings_batch(ids[start : start + EMBED_BATCH], session=session, client=client)
        await session.commit()
        _logger.info("embedded listings", done=min(start + EMBED_BATCH, len(ids)), total=len(ids))
    return len(ids)


async def seed(
    *,
    listings_path: Path = DEFAULT_LISTINGS_PATH,
    pois_path: Path | None = DEFAULT_POIS_PATH,
    embed: bool = True,
    embed_limit: int | None = None,
) -> SeedSummary:
    engine = get_engine()
    async with engine.connect() as connection:
        await apply_schema(connection)

    listing_rows = load_json_list(listings_path)
    params = [p for p in (listing_params(row) for row in listing_rows) if p is not None]
    poi_rows = [p for p in (poi_params(row) for row in load_json_list(pois_path)) if p is not None] if pois_path else []

    async with get_async_session_factory()() as session:
        await _upsert(session, UPSERT_LISTING_SQL, params)
        await _upsert(session, UPSERT_POI_SQL, poi_rows)
        # Recomputed on every run: the upsert above rewrites `features`, POI distances included.
        with_distances = await refresh_poi_distances(session)
        await session.commit()

        embedded = 0
        skipped_reason: str | None = None
        if not embed:
            skipped_reason = "--skip-embeddings"
        elif get_settings().cohere_api_key is None:
            skipped_reason = "COHERE_API_KEY is not set (semantic ranking needs the embeddings)"
        else:
            embedded = await _embed_missing(session, CohereEmbeddingClient(), embed_limit)

    return SeedSummary(
        listings=len(params),
        skipped=len(listing_rows) - len(params),
        pois=len(poi_rows),
        poi_distances=with_distances,
        embedded=embedded,
        embeddings_skipped_reason=skipped_reason,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingovo-search-seed", description="Create the schema and load the demo data.")
    parser.add_argument("--listings", type=Path, default=DEFAULT_LISTINGS_PATH, help="synthetic listings JSON")
    parser.add_argument("--pois", type=Path, default=DEFAULT_POIS_PATH, help="OSM points of interest JSON")
    parser.add_argument("--no-pois", action="store_true", help="do not load points of interest")
    parser.add_argument("--skip-embeddings", action="store_true", help="never call Cohere, even with a key")
    parser.add_argument("--embed-limit", type=int, default=None, help="embed at most this many listings")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = asyncio.run(
        seed(
            listings_path=args.listings,
            pois_path=None if args.no_pois else args.pois,
            embed=not args.skip_embeddings,
            embed_limit=args.embed_limit,
        )
    )
    print(summary.format())  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
