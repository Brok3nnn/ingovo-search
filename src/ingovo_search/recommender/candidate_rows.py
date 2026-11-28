"""The batched candidate fetches behind `recommend`.

Each helper is one SQL round-trip for a whole candidate list: the scoring columns, the
PostGIS coordinates, the embedding similarities and the searchable kNN head. The column
whitelist lives here so the homepage probe and the orchestrator read the same row shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import Property
from ingovo_search.lib.vector_db import EmbeddingKind, validate_vector, vector_bind_value
from ingovo_search.recommender.sql_filter import PUBLIC_SEARCH_VISIBILITY_SQL
from ingovo_search.types.errors import DbError

SCORING_PROPERTY_COLUMNS = [
    "id",
    "created_at",
    "title",
    "description",
    # The stored machine translations. They travel with the scored candidate so a
    # search result card can show the reader's language.
    "title_translations",
    "description_translations",
    "price_huf",
    "currency",
    "transaction_type",
    "property_type",
    "city",
    "district",
    "address",
    "status",
    "expires_at",
    "rooms",
    "area_m2",
    "floor",
    "year_built",
    "features",
    "has_balcony",
    "has_terrace",
    "has_parking",
    "has_garage",
    "has_garden",
    "has_basement",
    "has_pool",
    "elevator",
    "air_conditioning",
    "renovation_state",
    "orientation",
    "noise_score",
    "light_quality_score",
    "red_flags",
    "description_quality_score",
    "templated_phrase_fraction",
    "cleaned_description",
    "one_paragraph_summary",
    "short_summary",
    "images",
    "photo_count",
    "quality_score",
]


async def fetch_properties_in_order(session: AsyncSession, ids: Sequence[str]) -> list[SimpleNamespace]:
    if not ids:
        return []
    try:
        result = await session.execute(
            text(
                f"SELECT {', '.join(SCORING_PROPERTY_COLUMNS)} FROM properties "  # noqa: S608 -- constant column list and visibility predicate, ids bound
                f"WHERE id = ANY(:ids) AND ({PUBLIC_SEARCH_VISIBILITY_SQL})"
            ),
            {"ids": list(ids)},
        )
    except SQLAlchemyError as exc:
        raise DbError("failed to fetch candidate properties") from exc
    by_id: dict[str, SimpleNamespace] = {
        str(row.id): SimpleNamespace(**{column: getattr(row, column) for column in SCORING_PROPERTY_COLUMNS})
        for row in result
    }
    return [by_id[pid] for pid in ids if pid in by_id]


async def fetch_candidate_rows(session: AsyncSession, ids: Sequence[str]) -> list[SimpleNamespace]:
    """Public wrapper: load the scoring columns for `ids`, order preserved.

    Exposes the same lightweight `SimpleNamespace` rows the scorer consumes so
    callers outside the full pipeline (the homepage probe) can score a candidate
    sample with `compute_score` without duplicating the column whitelist.
    """
    return await fetch_properties_in_order(session, ids)


def to_property(row: SimpleNamespace, coordinates: tuple[float, float] | None) -> Property:
    prop = Property(**{column: getattr(row, column, None) for column in SCORING_PROPERTY_COLUMNS})
    # These detached scoring snapshots are never persisted. Carry the already
    # batched PostGIS coordinates into the public card payload as transient fields
    # so the browser does not need one full-property request per map pin.
    latitude, longitude = coordinates if coordinates is not None else (None, None)
    vars(prop).update(latitude=latitude, longitude=longitude)
    return prop


async def fetch_property_coords(session: AsyncSession, ids: Sequence[str]) -> dict[str, tuple[float, float]]:
    if not ids:
        return {}
    try:
        result = await session.execute(
            text(
                "SELECT id, ST_X(geo::geometry) AS lng, ST_Y(geo::geometry) AS lat "  # noqa: S608 -- only the constant visibility predicate is interpolated, ids bound
                f"FROM properties WHERE id = ANY(:ids) AND geo IS NOT NULL AND ({PUBLIC_SEARCH_VISIBILITY_SQL})"
            ),
            {"ids": list(ids)},
        )
    except SQLAlchemyError as exc:
        raise DbError("failed to fetch property coordinates") from exc
    coords: dict[str, tuple[float, float]] = {}
    for row in result:
        if row.lat is not None and row.lng is not None:
            coords[str(row.id)] = (float(row.lat), float(row.lng))
    return coords


async def fetch_embedding_similarities(
    session: AsyncSession,
    ids: Sequence[str],
    query_emb: Sequence[float],
    kind: EmbeddingKind,
) -> dict[str, float]:
    validate_vector(query_emb)
    if not ids:
        return {}
    column = f"{kind}_embedding"
    try:
        result = await session.execute(
            text(
                f"SELECT id, 1 - ({column} <=> CAST(:q AS vector)) AS similarity "  # noqa: S608 -- the column follows the EmbeddingKind literal, the predicate is constant, values bound
                f"FROM properties WHERE id = ANY(:ids) AND {column} IS NOT NULL "
                f"AND ({PUBLIC_SEARCH_VISIBILITY_SQL})"
            ),
            {"ids": list(ids), "q": await vector_bind_value(session, query_emb)},
        )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to fetch {kind} embedding similarities") from exc
    return {str(row.id): float(row.similarity) for row in result if row.similarity is not None}


async def fetch_searchable_knn_ids(
    session: AsyncSession,
    query_emb: Sequence[float],
    *,
    kind: EmbeddingKind,
    limit: int,
) -> list[tuple[str, float]]:
    validate_vector(query_emb)
    column = f"{kind}_embedding"
    try:
        result = await session.execute(
            text(
                f"SELECT id, 1 - ({column} <=> CAST(:q AS vector)) AS similarity "  # noqa: S608 -- the column follows the EmbeddingKind literal, the predicate is constant, values bound
                "FROM properties "
                f"WHERE {column} IS NOT NULL "
                f"AND ({PUBLIC_SEARCH_VISIBILITY_SQL}) "
                # id tiebreak: with tied distances (e.g. placeholder embeddings)
                # LIMIT would otherwise select an arbitrary, run-varying subset.
                f"ORDER BY {column} <=> CAST(:q AS vector), id "
                "LIMIT :limit"
            ),
            {"q": await vector_bind_value(session, query_emb), "limit": limit},
        )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to run searchable {kind} kNN query") from exc
    return [(str(row.id), float(row.similarity)) for row in result if row.similarity is not None]
