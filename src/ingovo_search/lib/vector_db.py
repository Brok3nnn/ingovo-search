"""Raw-SQL helpers for pgvector and PostGIS queries.

All non-vector / non-geo queries use the SQLAlchemy ORM via `AsyncSession`.
The helpers here exist because complex kNN and ST_DWithin queries are
expressed more cleanly as raw SQL with parameter binding.

Validation (dimension check, lat/lng range, positive radius) happens before
the DB is touched, so wrong inputs raise `DbError` early — consistent with
the conventions on error handling.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Awaitable, Callable, Sequence
from importlib import import_module
from typing import Any, Literal, NamedTuple, cast

from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.types.errors import DbError

VECTOR_DIM: int = 1024
EmbeddingKind = Literal["factual", "descriptive"]
# Tracked by weak reference, NOT by id(): a GC'd asyncpg connection frees its
# memory address for reuse, and an id()-keyed set then silently skips codec
# registration on a brand-new connection — the vector bind on that connection
# fails ("expected str, got list"), which surfaced as intermittent
# "failed to run batch pooled-signal query" 500s after pool recycles.
_REGISTERED_VECTOR_CODEC_CONNECTIONS: weakref.WeakSet[Any] = weakref.WeakSet()


class KnnRow(NamedTuple):
    """One row returned by `knn_by_query_embedding`."""

    property_id: str
    similarity: float


def validate_vector(vec: Sequence[float]) -> None:
    """Raise `DbError` when the vector dimension does not match `VECTOR_DIM`."""
    if len(vec) != VECTOR_DIM:
        raise DbError(f"vector dim mismatch: expected {VECTOR_DIM}, got {len(vec)}")


def validate_lat_lng(lat: float, lng: float) -> None:
    """Raise `DbError` when lat / lng fall outside the WGS84 valid range."""
    if not -90.0 <= lat <= 90.0:
        raise DbError(f"latitude out of range [-90, 90]: {lat}")
    if not -180.0 <= lng <= 180.0:
        raise DbError(f"longitude out of range [-180, 180]: {lng}")


def vector_literal(vec: Sequence[float]) -> str:
    """Format a sequence of floats as a pgvector literal: `'[1.0,2.0,...]'`."""
    return "[" + ",".join(str(float(v)) for v in vec) + "]"


async def vector_bind_value(session: AsyncSession, vec: Sequence[float]) -> object:
    """Return a pgvector bind value compatible with the current asyncpg connection."""
    if await _ensure_asyncpg_vector_codec(session):
        return list(vec)
    return vector_literal(vec)


def _embedding_column(kind: EmbeddingKind) -> str:
    return f"{kind}_embedding"


async def insert_embedding(
    session: AsyncSession,
    property_id: str,
    kind: EmbeddingKind,
    vec: Sequence[float],
) -> None:
    """Write `vec` to `properties.{kind}_embedding` for the given property.

    Raises `DbError` if `vec` is not `VECTOR_DIM` long.
    """
    validate_vector(vec)
    column = _embedding_column(kind)
    try:
        await session.execute(
            text(f"UPDATE properties SET {column} = CAST(:vec AS vector) WHERE id = :pid"),
            {"vec": vector_literal(vec), "pid": property_id},
        )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to insert {kind} embedding for property {property_id}") from exc


async def insert_embeddings_batch(
    session: AsyncSession,
    kind: EmbeddingKind,
    rows: Sequence[tuple[str, Sequence[float]]],
) -> None:
    """Bulk-write vectors to `properties.{kind}_embedding`.

    This keeps the same validation and raw SQL contract as `insert_embedding`,
    but uses one executemany call for scale rebuilds instead of one DB round
    trip per listing.
    """
    if not rows:
        return
    column = _embedding_column(kind)
    params: list[dict[str, str]] = []
    for property_id, vec in rows:
        validate_vector(vec)
        params.append({"vec": vector_literal(vec), "pid": property_id})

    try:
        if not await _insert_embeddings_asyncpg_binary(session, column, rows):
            await session.execute(
                text(f"UPDATE properties SET {column} = CAST(:vec AS vector) WHERE id = :pid"),
                params,
            )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to insert {kind} embeddings for {len(rows)} properties") from exc
    except Exception as exc:
        raise DbError(f"failed to insert {kind} embeddings for {len(rows)} properties") from exc


async def _insert_embeddings_asyncpg_binary(
    session: AsyncSession,
    column: str,
    rows: Sequence[tuple[str, Sequence[float]]],
) -> bool:
    """Use asyncpg's binary pgvector codec when a real AsyncSession is available."""
    driver_connection = await _asyncpg_driver_connection(session)
    if driver_connection is None:
        return False

    await _register_asyncpg_vector_codec(driver_connection)
    await driver_connection.executemany(
        f"UPDATE properties SET {column} = $1::vector WHERE id = $2",
        [(list(vec), property_id) for property_id, vec in rows],
    )
    return True


async def _ensure_asyncpg_vector_codec(session: AsyncSession) -> bool:
    driver_connection = await _asyncpg_driver_connection(session)
    if driver_connection is None:
        return False
    await _register_asyncpg_vector_codec(driver_connection)
    return True


async def _register_asyncpg_vector_codec(driver_connection: Any) -> None:
    try:
        if driver_connection in _REGISTERED_VECTOR_CODEC_CONNECTIONS:
            return
    except TypeError:  # not weak-referenceable — register unconditionally
        pass
    register_vector_fn = cast(Callable[[Any], Awaitable[None]], import_module("pgvector.asyncpg").register_vector)
    await register_vector_fn(driver_connection)
    with contextlib.suppress(TypeError):
        _REGISTERED_VECTOR_CODEC_CONNECTIONS.add(driver_connection)


async def _asyncpg_driver_connection(session: AsyncSession) -> Any | None:
    try:
        connection = await session.connection()
        raw_connection = await connection.get_raw_connection()
        driver_connection = getattr(raw_connection, "driver_connection", None)
    except (AttributeError, TypeError):
        return None

    if driver_connection is None:
        return None
    if not driver_connection.__class__.__module__.startswith("asyncpg.") or not hasattr(
        driver_connection, "executemany"
    ):
        return None
    return driver_connection


async def knn_by_query_embedding(
    session: AsyncSession,
    query_emb: Sequence[float],
    kind: EmbeddingKind,
    sim_threshold: float,
    limit: int = 500,
    candidate_ids: Sequence[str] | None = None,
) -> list[KnnRow]:
    """Return properties ranked by cosine similarity to `query_emb`.

    Used by the recommender retrieval path. Rows are ordered by similarity descending.
    Raises `DbError` if `query_emb` is not `VECTOR_DIM` long.
    """
    validate_vector(query_emb)
    if limit <= 0:
        raise DbError(f"limit must be positive, got {limit}")
    if not -1.0 <= sim_threshold <= 1.0:
        raise DbError(f"sim_threshold must be between -1.0 and 1.0, got {sim_threshold}")
    if candidate_ids is not None and len(candidate_ids) == 0:
        return []

    column = _embedding_column(kind)
    candidate_filter = ""
    params: dict[str, object] = {
        "q": await vector_bind_value(session, query_emb),
        "threshold": sim_threshold,
        "limit": limit,
    }
    sql = f"""
        SELECT
          p.id AS property_id,
          1 - (p.{column} <=> CAST(:q AS vector)) AS similarity
        FROM properties p
        WHERE p.{column} IS NOT NULL
          AND (1 - (p.{column} <=> CAST(:q AS vector))) >= :threshold
        {candidate_filter}
        ORDER BY p.{column} <=> CAST(:q AS vector)
        LIMIT :limit
        """
    if candidate_ids is not None:
        candidate_filter = "AND p.id = ANY(:candidate_ids)"
        params["candidate_ids"] = list(candidate_ids)
        sql = f"""
        SELECT
          p.id AS property_id,
          1 - (p.{column} <=> CAST(:q AS vector)) AS similarity
        FROM properties p
        WHERE p.{column} IS NOT NULL
          AND (1 - (p.{column} <=> CAST(:q AS vector))) >= :threshold
          {candidate_filter}
        ORDER BY p.{column} <=> CAST(:q AS vector)
        LIMIT :limit
        """
        stmt = text(sql).bindparams(bindparam("candidate_ids", type_=ARRAY(String())))
    else:
        stmt = text(sql)

    try:
        result = await session.execute(stmt, params)
    except SQLAlchemyError as exc:
        raise DbError(f"failed to run {kind} kNN query") from exc

    return [
        KnnRow(
            property_id=row.property_id,
            similarity=float(row.similarity),
        )
        for row in result
    ]


async def geo_set_point(
    session: AsyncSession,
    property_id: str,
    lat: float,
    lng: float,
) -> None:
    """Set `properties.geo` to a POINT at (lng, lat).

    PostGIS uses (longitude, latitude) order for `ST_MakePoint`.
    """
    validate_lat_lng(lat, lng)
    try:
        await session.execute(
            text("UPDATE properties SET geo = ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography WHERE id = :pid"),
            {"pid": property_id, "lat": lat, "lng": lng},
        )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to set geo point for property {property_id}") from exc


async def geo_set_points_batch(
    session: AsyncSession,
    rows: Sequence[tuple[str, float, float]],
) -> None:
    """Bulk-set `properties.geo` for scale ingestion fixtures and jobs."""
    if not rows:
        return
    params: list[dict[str, float | str]] = []
    for property_id, lat, lng in rows:
        validate_lat_lng(lat, lng)
        params.append({"pid": property_id, "lat": lat, "lng": lng})

    sql_named = "UPDATE properties SET geo = ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography WHERE id = :pid"
    try:
        driver_connection = await _asyncpg_driver_connection(session)
        if driver_connection is not None:
            await driver_connection.executemany(
                "UPDATE properties SET geo = ST_SetSRID(ST_MakePoint($3, $2), 4326)::geography WHERE id = $1",
                [(property_id, lat, lng) for property_id, lat, lng in rows],
            )
            return
        await session.execute(text(sql_named), params)
    except SQLAlchemyError as exc:
        raise DbError(f"failed to set geo points for {len(rows)} properties") from exc
    except Exception as exc:
        raise DbError(f"failed to set geo points for {len(rows)} properties") from exc


async def geo_st_dwithin(
    session: AsyncSession,
    center_lat: float,
    center_lng: float,
    radius_km: float,
) -> list[str]:
    """Return property IDs whose `geo` is within `radius_km` of (center_lat, center_lng)."""
    validate_lat_lng(center_lat, center_lng)
    if radius_km <= 0:
        raise DbError(f"radius_km must be positive, got {radius_km}")

    try:
        result = await session.execute(
            text(
                "SELECT id FROM properties "
                "WHERE geo IS NOT NULL "
                "  AND ST_DWithin(geo, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography, :radius_m)"
            ),
            {"lat": center_lat, "lng": center_lng, "radius_m": radius_km * 1000.0},
        )
    except SQLAlchemyError as exc:
        raise DbError("failed to run ST_DWithin query") from exc
    return [row.id for row in result]
