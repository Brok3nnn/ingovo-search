"""Unit tests for vector_db helpers — no DB access; mocked AsyncSession."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.vector_db import (
    VECTOR_DIM,
    geo_set_point,
    geo_set_points_batch,
    geo_st_dwithin,
    insert_embedding,
    insert_embeddings_batch,
    knn_by_query_embedding,
    validate_lat_lng,
    validate_vector,
    vector_literal,
)
from ingovo_search.types.errors import DbError


def test_validate_vector_accepts_correct_dim() -> None:
    validate_vector([0.0] * VECTOR_DIM)


def test_validate_vector_rejects_wrong_dim() -> None:
    with pytest.raises(DbError, match="dim mismatch"):
        validate_vector([0.0] * 512)


def test_validate_vector_rejects_empty() -> None:
    with pytest.raises(DbError, match="dim mismatch"):
        validate_vector([])


def test_validate_lat_lng_accepts_in_range() -> None:
    validate_lat_lng(0, 0)
    validate_lat_lng(47.5, 19.04)
    validate_lat_lng(-90, -180)
    validate_lat_lng(90, 180)


def test_validate_lat_lng_rejects_out_of_range() -> None:
    with pytest.raises(DbError, match="latitude out of range"):
        validate_lat_lng(91, 0)
    with pytest.raises(DbError, match="longitude out of range"):
        validate_lat_lng(0, 181)
    with pytest.raises(DbError, match="latitude out of range"):
        validate_lat_lng(-91, 0)


def test_vector_literal_format() -> None:
    assert vector_literal([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"
    assert vector_literal([0.5, -1.5]) == "[0.5,-1.5]"


@pytest.fixture
def mock_session() -> MagicMock:
    """An AsyncSession-shaped mock whose `execute()` is awaitable."""
    sess = MagicMock(spec=AsyncSession)
    sess.execute = AsyncMock()
    return sess


async def test_insert_embedding_rejects_wrong_dim(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="dim mismatch"):
        await insert_embedding(cast(AsyncSession, mock_session), "prop-1", "factual", [0.0] * 100)
    mock_session.execute.assert_not_called()


async def test_knn_by_query_embedding_rejects_wrong_dim(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="dim mismatch"):
        await knn_by_query_embedding(cast(AsyncSession, mock_session), [0.0] * 100, "factual", 0.5)
    mock_session.execute.assert_not_called()


async def test_knn_by_query_embedding_rejects_bad_limit(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="limit must be positive"):
        await knn_by_query_embedding(cast(AsyncSession, mock_session), [0.0] * VECTOR_DIM, "factual", 0.5, limit=0)
    mock_session.execute.assert_not_called()


async def test_knn_by_query_embedding_rejects_bad_threshold(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="sim_threshold"):
        await knn_by_query_embedding(cast(AsyncSession, mock_session), [0.0] * VECTOR_DIM, "factual", 2.0)
    mock_session.execute.assert_not_called()


async def test_geo_set_point_rejects_out_of_range(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="out of range"):
        await geo_set_point(cast(AsyncSession, mock_session), "prop-1", 100.0, 0.0)
    mock_session.execute.assert_not_called()


async def test_geo_st_dwithin_rejects_zero_radius(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="radius_km must be positive"):
        await geo_st_dwithin(cast(AsyncSession, mock_session), 47.5, 19.04, 0.0)
    mock_session.execute.assert_not_called()


async def test_geo_st_dwithin_rejects_negative_radius(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="radius_km must be positive"):
        await geo_st_dwithin(cast(AsyncSession, mock_session), 47.5, 19.04, -1.0)
    mock_session.execute.assert_not_called()


async def test_geo_set_points_batch_uses_executemany_fallback(mock_session: MagicMock) -> None:
    await geo_set_points_batch(
        cast(AsyncSession, mock_session),
        [("prop-1", 47.5, 19.0), ("prop-2", 47.6, 19.1)],
    )

    assert mock_session.execute.call_count == 1
    sql = str(mock_session.execute.call_args.args[0]).lower()
    params = mock_session.execute.call_args.args[1]
    assert "st_makepoint" in sql
    assert params[0]["pid"] == "prop-1"
    assert params[1]["lat"] == 47.6


async def test_geo_set_points_batch_rejects_out_of_range(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="latitude out of range"):
        await geo_set_points_batch(cast(AsyncSession, mock_session), [("prop-1", 91.0, 19.0)])
    mock_session.execute.assert_not_called()


async def test_insert_embedding_kind_is_factual_or_descriptive(mock_session: MagicMock) -> None:
    vec = [1.0] * VECTOR_DIM
    await insert_embedding(cast(AsyncSession, mock_session), "prop-1", "factual", vec)
    await insert_embedding(cast(AsyncSession, mock_session), "prop-1", "descriptive", vec)
    assert mock_session.execute.call_count == 2

    factual_sql = str(mock_session.execute.call_args_list[0].args[0]).lower()
    descriptive_sql = str(mock_session.execute.call_args_list[1].args[0]).lower()
    assert "factual_embedding" in factual_sql
    assert "descriptive_embedding" in descriptive_sql


async def test_insert_embedding_binds_params(mock_session: MagicMock) -> None:
    vec = [0.0] * VECTOR_DIM
    await insert_embedding(cast(AsyncSession, mock_session), "prop-99", "factual", vec)
    params = mock_session.execute.call_args.args[1]
    assert params["pid"] == "prop-99"
    assert params["vec"].startswith("[")
    assert params["vec"].endswith("]")


async def test_insert_embeddings_batch_uses_executemany(mock_session: MagicMock) -> None:
    vec = [0.0] * VECTOR_DIM
    await insert_embeddings_batch(
        cast(AsyncSession, mock_session),
        "descriptive",
        [("prop-1", vec), ("prop-2", vec)],
    )

    assert mock_session.execute.call_count == 1
    sql = str(mock_session.execute.call_args.args[0]).lower()
    params = mock_session.execute.call_args.args[1]
    assert "descriptive_embedding" in sql
    assert params[0]["pid"] == "prop-1"
    assert params[1]["pid"] == "prop-2"


async def test_insert_embeddings_batch_rejects_wrong_dim(mock_session: MagicMock) -> None:
    with pytest.raises(DbError, match="dim mismatch"):
        await insert_embeddings_batch(cast(AsyncSession, mock_session), "factual", [("prop-1", [0.0] * 100)])
    mock_session.execute.assert_not_called()


async def test_knn_by_query_embedding_queries_properties_by_kind(mock_session: MagicMock) -> None:
    mock_session.execute.return_value = [
        SimpleNamespace(property_id="prop-a", similarity=0.93),
        SimpleNamespace(property_id="prop-b", similarity=0.77),
    ]

    rows = await knn_by_query_embedding(
        cast(AsyncSession, mock_session),
        [0.1] * VECTOR_DIM,
        "descriptive",
        sim_threshold=0.5,
        limit=5,
    )

    assert rows[0].property_id == "prop-a"
    assert rows[0].similarity == 0.93
    sql = str(mock_session.execute.call_args.args[0]).lower()
    params = mock_session.execute.call_args.args[1]
    assert "from properties" in sql
    assert "descriptive_embedding" in sql
    assert "limit" in sql
    assert params["threshold"] == 0.5
    assert params["limit"] == 5


async def test_knn_by_query_embedding_filters_candidate_ids(mock_session: MagicMock) -> None:
    mock_session.execute.return_value = []

    await knn_by_query_embedding(
        cast(AsyncSession, mock_session),
        [0.1] * VECTOR_DIM,
        "factual",
        sim_threshold=0.5,
        limit=5,
        candidate_ids=["prop-a", "prop-b"],
    )

    sql = str(mock_session.execute.call_args.args[0]).lower()
    params = mock_session.execute.call_args.args[1]
    assert "any" in sql
    assert params["candidate_ids"] == ["prop-a", "prop-b"]


async def test_knn_by_query_embedding_empty_candidate_ids_returns_empty(mock_session: MagicMock) -> None:
    rows = await knn_by_query_embedding(
        cast(AsyncSession, mock_session),
        [0.1] * VECTOR_DIM,
        "factual",
        sim_threshold=0.5,
        candidate_ids=[],
    )

    assert rows == []
    mock_session.execute.assert_not_called()
