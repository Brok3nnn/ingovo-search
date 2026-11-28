"""Integration tests for the geo scoring + learned centroids.

Requires Postgres + pgvector + postgis. Run via:
    uv run pytest -m integration

Covers what needs a real database: learned-centroid scoring, cold-start hint
generation + caching, `recompute_region_centroid`, and the scale test.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from uuid import uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import Interaction, Property
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url
from ingovo_search.lib.llm import LlmRequest, MockLlmClient, fixture_key
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.lib.vector_db import geo_set_point, geo_set_points_batch, geo_st_dwithin
from ingovo_search.recommender.geo_score import ListingGeo, haversine_km
from ingovo_search.recommender.region_centroid import (
    get_region_centroid,
    matches_named_region,
    recompute_region_centroid,
)

pytestmark = pytest.mark.integration

VALID_HINT = json.dumps({"districts": [{"district": "13", "weight": 0.7}, {"district": "5", "weight": 0.2}]})


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh DB session that rolls back after each test."""
    settings = get_settings()
    engine = create_async_engine(
        sqlalchemy_asyncpg_url(settings.database_url),
        echo=False,
        future=True,
        connect_args=asyncpg_connect_args(),
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        try:
            yield sess
        finally:
            await sess.rollback()
    await engine.dispose()


class _CountingLlmClient:
    """Wraps `MockLlmClient` to count `complete()` invocations."""

    def __init__(self, fixtures: dict[str, str] | None = None) -> None:
        self._inner = MockLlmClient(fixtures)
        self.calls = 0

    async def complete(self, req: LlmRequest) -> str:
        self.calls += 1
        return await self._inner.complete(req)


def _hint_key(region_name: str) -> str:
    return fixture_key(LlmRequest(model=TIER_S_PARSER, system="", user=region_name))


async def _insert_property(sess: AsyncSession, lat: float | None = None, lng: float | None = None) -> str:
    pid = str(uuid4())
    sess.add(Property(id=pid, title="Test property", city="Budapest"))
    await sess.flush()
    if lat is not None and lng is not None:
        await geo_set_point(sess, pid, lat=lat, lng=lng)
        await sess.flush()
    return pid


async def _insert_interaction(sess: AsyncSession, property_id: str, signal_type: str, region_name: str) -> None:
    sess.add(
        Interaction(
            id=str(uuid4()),
            property_id=property_id,
            signal_type=signal_type,
            signal_weight=1.0,
            query_parsed={"geo_hints": {"region_name": region_name}},
        )
    )
    await sess.flush()


@dataclass(frozen=True)
class _CentroidRow:
    """The values of one seeded `region_centroid` row."""

    region_name: str
    lat: float
    lng: float
    spread_km: float
    n_signals: int


async def _insert_region_centroid(sess: AsyncSession, row: _CentroidRow) -> None:
    await sess.execute(
        text(
            "INSERT INTO region_centroid (region_name, centroid_geo, spread_km, n_signals, last_updated) "
            "VALUES (:rn, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography, :spread, :n, now())"
        ),
        {"rn": row.region_name, "lat": row.lat, "lng": row.lng, "spread": row.spread_km, "n": row.n_signals},
    )
    await sess.flush()


async def test_learned_centroid_gaussian_scoring(session: AsyncSession) -> None:
    """A centroid with n_signals=100 scores by Gaussian distance.

    The original design note annotated property (47.5, 19.02) as "≈0.95", but that
    offset is ~1.5 km and the formula exp(-(d/spread)^2) with spread=2 yields ~0.57
    there — see `test_geo_score.test_gaussian_at_the_centroid_and_near_it`.
    This asserts the value the formula actually produces.
    """
    client = MockLlmClient({})  # learned path must not call the LLM
    await _insert_region_centroid(
        session, _CentroidRow("TanultRégió", lat=47.5, lng=19.0, spread_km=2.0, n_signals=100)
    )

    at_centroid = await matches_named_region(session, "TanultRégió", ListingGeo(lat=47.5, lng=19.0), client=client)
    assert abs(at_centroid - 1.0) < 1e-6

    offset = await matches_named_region(session, "TanultRégió", ListingGeo(lat=47.5, lng=19.02), client=client)
    assert abs(offset - 0.569) < 0.01


async def test_cold_start_generates_hint_then_cache_hits(session: AsyncSession) -> None:
    """A brand-new region triggers one LLM hint; later calls cache-hit."""
    region = "ÚjRégió"
    client = _CountingLlmClient({_hint_key(region): VALID_HINT})

    first = await matches_named_region(session, region, ListingGeo(district="13"), client=client)
    second = await matches_named_region(session, region, ListingGeo(district="13"), client=client)

    assert client.calls == 1, "the cached hint must not trigger a second LLM call"
    assert first == second == 0.7  # district 13 carries hint weight 0.7


async def test_recompute_region_centroid_matches_expected(session: AsyncSession) -> None:
    """Recompute produces the expected centroid within 100 m."""
    region = "TesztRégió"
    coords = [(47.50, 19.00), (47.50, 19.02), (47.52, 19.00)]
    for lat, lng in coords:
        pid = await _insert_property(session, lat, lng)
        await _insert_interaction(session, pid, "thumb_up", region)

    await recompute_region_centroid(session, region)
    centroid = await get_region_centroid(session, region)

    assert centroid is not None
    assert centroid.n_signals == 3
    assert centroid.lat is not None
    assert centroid.lng is not None
    # All three interactions are equal-weight, so the centroid is the plain mean.
    expected_lat = sum(c[0] for c in coords) / 3
    expected_lng = sum(c[1] for c in coords) / 3
    assert haversine_km(centroid.lat, centroid.lng, expected_lat, expected_lng) < 0.1


async def test_recompute_with_no_engagement_writes_zero_signal_row(session: AsyncSession) -> None:
    await recompute_region_centroid(session, "ÜresRégió")
    centroid = await get_region_centroid(session, "ÜresRégió")
    assert centroid is not None
    assert centroid.n_signals == 0
    assert centroid.lat is None


async def test_geo_scale_10k_properties(session: AsyncSession) -> None:
    """10K geocoded properties; geo lookup stays fast and indexed."""
    base_lat, base_lng = 47.4979, 19.0402
    property_ids = [str(uuid4()) for _ in range(10_000)]
    await _bulk_insert_properties(session, property_ids, title="Scale property")
    await geo_set_points_batch(
        session,
        [
            # Spread points over ~0.2° so most fall outside the 1 km test radius.
            (pid, base_lat + (i % 100) * 0.002, base_lng + (i // 100) * 0.002)
            for i, pid in enumerate(property_ids)
        ],
    )
    await session.flush()

    start = time.perf_counter()
    nearby = await geo_st_dwithin(session, center_lat=base_lat, center_lng=base_lng, radius_km=1.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert isinstance(nearby, list)
    assert elapsed_ms < 1000.0, f"geo_st_dwithin client roundtrip took {elapsed_ms:.1f}ms"

    plan_rows = await session.execute(
        text(
            "EXPLAIN (ANALYZE, FORMAT TEXT) SELECT id FROM properties "
            "WHERE geo IS NOT NULL AND ST_DWithin(geo, "
            "ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography, :radius_m)"
        ),
        {"lat": base_lat, "lng": base_lng, "radius_m": 1000.0},
    )
    plan = "\n".join(str(row[0]) for row in plan_rows)
    assert "Seq Scan on properties" not in plan, f"geo query seq-scanned properties:\n{plan}"
    execution_ms = _explain_execution_time_ms(plan)
    assert execution_ms < 50.0, f"geo_st_dwithin DB execution took {execution_ms:.1f}ms (budget 50ms)"


async def test_recompute_region_centroid_scale(session: AsyncSession) -> None:
    """Recompute over 10K interactions for one region completes < 5 s."""
    region = "SkálaRégió"
    property_ids = [str(uuid4()) for _ in range(200)]
    await _bulk_insert_properties(session, property_ids, title="P")
    await geo_set_points_batch(
        session,
        [(pid, 47.5 + (i % 20) * 0.001, 19.0 + (i // 20) * 0.001) for i, pid in enumerate(property_ids)],
    )
    await session.flush()

    await _bulk_insert_interactions(session, property_ids, region, count=10_000)
    await session.flush()

    start = time.perf_counter()
    await recompute_region_centroid(session, region)
    elapsed_s = time.perf_counter() - start

    assert elapsed_s < 5.0, f"recompute_region_centroid took {elapsed_s:.2f}s (budget 5s)"
    centroid = await get_region_centroid(session, region)
    assert centroid is not None
    assert centroid.n_signals == 10_000


async def _bulk_insert_properties(session: AsyncSession, property_ids: Sequence[str], *, title: str) -> None:
    for start in range(0, len(property_ids), 1000):
        chunk = property_ids[start : start + 1000]
        await session.execute(
            insert(Property),
            [{"id": pid, "title": title, "city": "Budapest"} for pid in chunk],
        )
    await session.flush()


async def _bulk_insert_interactions(
    session: AsyncSession,
    property_ids: Sequence[str],
    region: str,
    *,
    count: int,
) -> None:
    for start in range(0, count, 1000):
        await session.execute(
            insert(Interaction),
            [
                {
                    "id": str(uuid4()),
                    "property_id": property_ids[i % len(property_ids)],
                    "signal_type": "thumb_up",
                    "signal_weight": 1.0,
                    "query_parsed": {"geo_hints": {"region_name": region}},
                }
                for i in range(start, min(start + 1000, count))
            ],
        )


def _explain_execution_time_ms(plan: str) -> float:
    match = re.search(r"Execution Time: ([0-9.]+) ms", plan)
    assert match is not None, f"EXPLAIN output did not include execution time:\n{plan}"
    return float(match.group(1))
