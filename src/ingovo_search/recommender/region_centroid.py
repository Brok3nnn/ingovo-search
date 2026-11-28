"""Learned region centroids — lookup, scoring, and nightly recompute.

A fuzzy region name ("Budai-hegyvidék") has no fixed polygon. Instead it has a
statistical centroid + spread, learned from the listings users positively
engaged with on queries naming that region.

- `matches_named_region` scores one listing against a region name: a Gaussian
  on the learned centroid once enough signal has accrued, otherwise the
  cold-start LLM district hint.
- `recompute_region_centroid` is the nightly job (the full product scheduled it)
  that refreshes one region's centroid from the `interactions` table.

DB access is raw SQL (PostGIS `ST_X` / `ST_Y` / `ST_MakePoint`); the centroid
math itself (`weighted_centroid`) is pure and unit-tested.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import structlog
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.llm import LlmClient
from ingovo_search.recommender.geo_score import NO_LISTING_GEO, ListingGeo, gaussian_region_score, haversine_km
from ingovo_search.recommender.pooled_signal import DECAY_WINDOW_SECONDS, SIGNAL_WEIGHTS
from ingovo_search.recommender.region_llm_hint import RegionHintDistrict, hint_district_score, region_llm_hint
from ingovo_search.types.errors import DbError, LlmError

_logger = structlog.get_logger("ingovo_search.recommender.region_centroid")

# Minimum positive-engagement count before the learned centroid replaces the
# cold-start LLM hint.
LEARNED_CENTROID_MIN_SIGNALS = 50

# Only positive engagements shape a centroid.
POSITIVE_SIGNAL_WEIGHTS: dict[str, float] = {name: w for name, w in SIGNAL_WEIGHTS.items() if w > 0}


@dataclass(frozen=True)
class RegionCentroidData:
    """Plain snapshot of a `region_centroid` row, coordinates as floats."""

    region_name: str
    lat: float | None
    lng: float | None
    spread_km: float | None
    n_signals: int
    llm_hint_districts: list[RegionHintDistrict]


def weighted_centroid(points: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Weighted centroid of `(lat, lng, weight)` points.

    Returns `(lat, lng, spread_km)` where `spread_km` is the weight-RMS of the
    great-circle distances from the centroid — the sigma that makes the Gaussian
    `exp(-(d/sigma)^2)` in `matches_named_region` meaningful.

    Raises `ValueError` on an empty set or non-positive total weight.
    """
    total_weight = sum(weight for _, _, weight in points)
    if not points or total_weight <= 0:
        raise ValueError("weighted_centroid requires at least one point with positive total weight")
    lat_c = sum(lat * weight for lat, _, weight in points) / total_weight
    lng_c = sum(lng * weight for _, lng, weight in points) / total_weight
    variance = sum(weight * haversine_km(lat_c, lng_c, lat, lng) ** 2 for lat, lng, weight in points) / total_weight
    return lat_c, lng_c, variance**0.5


def _build_signal_case(weights: dict[str, float]) -> str:
    arms = "\n        ".join(f"WHEN '{name}' THEN {value}" for name, value in weights.items())
    return f"CASE i.signal_type\n        {arms}\n        ELSE 0\n      END"


_GET_CENTROID_SQL = """
SELECT
  region_name,
  ST_Y(centroid_geo::geometry) AS lat,
  ST_X(centroid_geo::geometry) AS lng,
  spread_km,
  n_signals,
  llm_hint_districts
FROM region_centroid
WHERE region_name = :region_name
"""

_UPSERT_HINT_SQL = """
INSERT INTO region_centroid (region_name, llm_hint_districts, n_signals, last_updated)
VALUES (:region_name, CAST(:hints AS jsonb), 0, now())
ON CONFLICT (region_name) DO UPDATE
  SET llm_hint_districts = EXCLUDED.llm_hint_districts, last_updated = now()
"""

_UPSERT_CENTROID_SQL = """
INSERT INTO region_centroid (region_name, centroid_geo, spread_km, n_signals, last_updated)
VALUES (
  :region_name,
  CASE WHEN CAST(:lat AS double precision) IS NULL THEN NULL
       ELSE ST_SetSRID(
         ST_MakePoint(CAST(:lng AS double precision), CAST(:lat AS double precision)),
         4326
       )::geography END,
  CAST(:spread_km AS double precision),
  CAST(:n_signals AS integer),
  now()
)
ON CONFLICT (region_name) DO UPDATE SET
  centroid_geo = EXCLUDED.centroid_geo,
  spread_km = EXCLUDED.spread_km,
  n_signals = EXCLUDED.n_signals,
  last_updated = now()
"""


def _parse_hint_districts(raw: object) -> list[RegionHintDistrict]:
    if not isinstance(raw, list):
        return []
    return [RegionHintDistrict.model_validate(entry) for entry in cast("list[object]", raw) if isinstance(entry, dict)]


async def get_region_centroid(session: AsyncSession, region_name: str) -> RegionCentroidData | None:
    """Load one `region_centroid` row as a `RegionCentroidData`, or `None`."""
    try:
        result = await session.execute(text(_GET_CENTROID_SQL), {"region_name": region_name})
        row = result.one_or_none()
    except SQLAlchemyError as exc:
        raise DbError(f"failed to load region centroid for '{region_name}'") from exc
    if row is None:
        return None
    return RegionCentroidData(
        region_name=row.region_name,
        lat=float(row.lat) if row.lat is not None else None,
        lng=float(row.lng) if row.lng is not None else None,
        spread_km=float(row.spread_km) if row.spread_km is not None else None,
        n_signals=int(row.n_signals),
        llm_hint_districts=_parse_hint_districts(row.llm_hint_districts),
    )


async def _cache_llm_hint(session: AsyncSession, region_name: str, hints: list[RegionHintDistrict]) -> None:
    payload = json.dumps([hint.model_dump() for hint in hints])
    try:
        await session.execute(text(_UPSERT_HINT_SQL), {"region_name": region_name, "hints": payload})
    except SQLAlchemyError as exc:
        raise DbError(f"failed to cache LLM hint for region '{region_name}'") from exc


@dataclass(frozen=True)
class ResolvedRegionMatcher:
    """A region name resolved once into the data needed to score any listing.

    `recommend()` runs one `resolve_named_region` per search and then calls
    `score()` per candidate in memory — the per-listing `get_region_centroid`
    DB query (and any cold-start LLM hint) is done a single time, not N times.
    """

    learned_lat: float | None
    learned_lng: float | None
    spread_km: float | None
    learned_ready: bool
    hints: list[RegionHintDistrict]

    def score(
        self,
        *,
        property_lat: float | None = None,
        property_lng: float | None = None,
        property_district: str | None = None,
    ) -> float:
        """Score one listing against the resolved region, in `0..1`. Pure."""
        if self.learned_ready and self.learned_lat is not None and self.learned_lng is not None:
            if property_lat is None or property_lng is None:
                return 0.0
            distance = haversine_km(self.learned_lat, self.learned_lng, property_lat, property_lng)
            return gaussian_region_score(distance, self.spread_km)
        return hint_district_score(property_district, self.hints)


async def resolve_named_region(
    session: AsyncSession,
    region_name: str,
    *,
    client: LlmClient,
) -> ResolvedRegionMatcher:
    """Resolve a fuzzy region name into a reusable, in-memory `score()` matcher.

    Uses the learned Gaussian centroid once `n_signals >= 50`; otherwise the
    cold-start LLM district hint, generating and caching it on first use. The
    DB/LLM work happens here once so callers can score many listings cheaply.
    """
    centroid = await get_region_centroid(session, region_name)

    learned_ready = (
        centroid is not None
        and centroid.n_signals >= LEARNED_CENTROID_MIN_SIGNALS
        and centroid.lat is not None
        and centroid.lng is not None
    )
    if learned_ready and centroid is not None:
        return ResolvedRegionMatcher(
            learned_lat=centroid.lat,
            learned_lng=centroid.lng,
            spread_km=centroid.spread_km,
            learned_ready=True,
            hints=[],
        )

    hints = centroid.llm_hint_districts if centroid is not None else []
    if not hints:
        try:
            generated = await region_llm_hint(region_name, client=client, session=session)
        except LlmError as exc:
            # A model outage or a twice-failed schema: the region name adds no
            # geographic signal to this search, and the hint is retried on the next.
            _logger.warning(
                "region hint unavailable; scoring without it", region_name=region_name, error=type(exc).__name__
            )
        else:
            hints = generated.districts
            await _cache_llm_hint(session, region_name, hints)
    return ResolvedRegionMatcher(
        learned_lat=None,
        learned_lng=None,
        spread_km=None,
        learned_ready=False,
        hints=hints,
    )


async def matches_named_region(
    session: AsyncSession,
    region_name: str,
    listing: ListingGeo = NO_LISTING_GEO,
    *,
    client: LlmClient,
) -> float:
    """Score one listing against a fuzzy region name, in `0..1`.

    Thin wrapper over `resolve_named_region` + `ResolvedRegionMatcher.score` for
    single-listing callers; batch callers should resolve once and reuse the
    matcher across candidates. Only the listing's coordinates and district are read.
    """
    matcher = await resolve_named_region(session, region_name, client=client)
    return matcher.score(
        property_lat=listing.lat,
        property_lng=listing.lng,
        property_district=listing.district,
    )


async def recompute_region_centroid(session: AsyncSession, region_name: str) -> None:
    """Recompute and upsert one region's learned centroid (nightly job).

    Aggregates listings with positive engagement on queries naming `region_name`
    over the last 365 days, weighting each by `signal_value * recency_decay *
    signal_weight`. Writes the weighted centroid + spread; an empty result
    upserts a zero-signal row so cold-start hinting still applies.

    Caller is responsible for `session.commit()`.
    """
    select_sql = f"""
    WITH points AS (
      SELECT
        ST_Y(p.geo::geometry) AS lat,
        ST_X(p.geo::geometry) AS lng,
        ({_build_signal_case(POSITIVE_SIGNAL_WEIGHTS)})
          * EXP(-EXTRACT(EPOCH FROM (now() - i.created_at)) / {DECAY_WINDOW_SECONDS})
          * i.signal_weight AS weight
      FROM interactions i
      JOIN properties p ON p.id = i.property_id
      WHERE p.geo IS NOT NULL
        AND i.created_at > now() - INTERVAL '365 days'
        AND i.signal_type IN ({", ".join(f"'{name}'" for name in POSITIVE_SIGNAL_WEIGHTS)})
        AND i.query_parsed -> 'geo_hints' ->> 'region_name' = :region_name
    ),
    centroid AS (
      SELECT
        COUNT(*)::integer AS n_signals,
        SUM(weight) AS total_weight,
        SUM(lat * weight) / NULLIF(SUM(weight), 0) AS lat,
        SUM(lng * weight) / NULLIF(SUM(weight), 0) AS lng
      FROM points
      WHERE weight > 0
    )
    SELECT
      c.lat,
      c.lng,
      CASE
        WHEN c.n_signals = 0 THEN NULL
        ELSE SQRT(
          SUM(
            p.weight * POWER(
              ST_Distance(
                ST_SetSRID(ST_MakePoint(c.lng, c.lat), 4326)::geography,
                ST_SetSRID(ST_MakePoint(p.lng, p.lat), 4326)::geography
              ) / 1000.0,
              2
            )
          ) / c.total_weight
        )
      END AS spread_km,
      c.n_signals
    FROM centroid c
    LEFT JOIN points p ON c.n_signals > 0 AND p.weight > 0
    GROUP BY c.lat, c.lng, c.total_weight, c.n_signals
    """  # noqa: S608 -- only the module constants (positive signal weights, decay window) are interpolated, the region is bound
    try:
        result = await session.execute(text(select_sql), {"region_name": region_name})
        row = result.one()
    except SQLAlchemyError as exc:
        raise DbError(f"failed to aggregate interactions for region '{region_name}'") from exc

    params: dict[str, object | None]
    if row.n_signals > 0 and row.lat is not None and row.lng is not None:
        params = {
            "region_name": region_name,
            "lat": float(row.lat),
            "lng": float(row.lng),
            "spread_km": float(row.spread_km) if row.spread_km is not None else None,
            "n_signals": int(row.n_signals),
        }
    else:
        params = {"region_name": region_name, "lat": None, "lng": None, "spread_km": None, "n_signals": 0}

    try:
        await session.execute(text(_UPSERT_CENTROID_SQL), params)
    except SQLAlchemyError as exc:
        raise DbError(f"failed to upsert region centroid for '{region_name}'") from exc
