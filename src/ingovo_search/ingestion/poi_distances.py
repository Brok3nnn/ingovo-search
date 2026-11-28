"""Per-listing POI proximity refresh (run by the seed command; the full product also ran it nightly).

For every geocoded listing, compute the distance in metres to the nearest POI
of each category within `POI_DISTANCE_RADIUS_M`, and store the map in
`properties.features["poi_m"]`, e.g. `{"school": 320, "metro": 850}`.
Categories with no POI inside the radius are simply absent — the scorer
treats a missing key as "no data", never as an exclusion.

One SQL statement keeps the work in the database: the lateral per-category
MIN(ST_Distance) rides the GIST index on `poi.geo`, and `jsonb_set` preserves
whatever else lives in `features`.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.types.errors import DbError

POI_DISTANCE_RADIUS_M = 2000

POI_DISTANCE_SQL = text(
    """
    UPDATE properties AS p
    SET features = jsonb_set(COALESCE(p.features, '{}'::jsonb), '{poi_m}', agg.poi_m)
    FROM (
        SELECT pr.id AS property_id,
               jsonb_object_agg(nearest.category, nearest.min_m) AS poi_m
        FROM properties pr
        JOIN LATERAL (
            SELECT poi.category, MIN(ST_Distance(pr.geo, poi.geo))::int AS min_m
            FROM poi
            WHERE ST_DWithin(pr.geo, poi.geo, :radius_m)
            GROUP BY poi.category
        ) AS nearest ON TRUE
        WHERE pr.geo IS NOT NULL
        GROUP BY pr.id
    ) AS agg
    WHERE p.id = agg.property_id
    RETURNING p.id
    """
)


async def refresh_poi_distances(session: AsyncSession, *, radius_m: int = POI_DISTANCE_RADIUS_M) -> int:
    """Refresh `features["poi_m"]` for every geocoded listing; returns rows touched."""
    try:
        result = await session.execute(POI_DISTANCE_SQL, {"radius_m": radius_m})
        touched = len(list(result.scalars().all()))
    except SQLAlchemyError as error:
        raise DbError(f"POI distance refresh failed: {error}") from error
    return touched
