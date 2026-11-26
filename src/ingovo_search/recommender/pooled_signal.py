"""§9 pooled-signal kNN — query-pooled collaborative filtering.

`pooled_signal` aggregates buyer signals (±1 / favorite / contact / view / hide)
that other users left on a property, weighted by how similar *their* query was
to the current one. The query embedding — not the user — is the pooling key
(the design spec §6.6).

The single-property path uses a direct asyncpg scalar aggregate so the hot
request path avoids SQLAlchemy and pgvector codec overhead. A DB-free fallback
keeps unit tests and non-asyncpg sessions covered by fetching property-local
rows and computing the same average in Python. The batch path still uses one
raw SQL aggregate for multi-property scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.vector_db import validate_vector, vector_bind_value, vector_literal
from ingovo_search.types.errors import DbError

# Minimum cosine similarity between a neighbour's query embedding and the
# current query for that neighbour's signals to count.
SIMILARITY_THRESHOLD = 0.85

# Only interactions from the last this-many days are pooled.
TIME_WINDOW_DAYS = 365

# Recency decay e-folding time, in seconds (90 days). A neighbour's weight is
# multiplied by exp(-age_seconds / DECAY_WINDOW_SECONDS).
DECAY_WINDOW_SECONDS = 90 * 24 * 60 * 60  # 7_776_000

# Signal value per interaction type — see the design spec Appendix C.
# `refinement_*` signals are intentionally absent: they never enter the pool.
SIGNAL_WEIGHTS: dict[str, float] = {
    "thumb_up": 1.0,
    "thumb_down": -1.0,
    "favorite": 0.7,
    "save": 0.7,
    "contact": 1.5,
    "viewing_booked": 2.0,
    "view_long": 0.2,
    "view_short": -0.1,
    "hide": -1.5,
}


def _build_signal_case() -> str:
    """Render the `CASE i.signal_type ... END` arms from `SIGNAL_WEIGHTS`.

    The keys are fixed internal constants, never user input, so interpolating
    them into SQL is safe.
    """
    arms = "\n        ".join(f"WHEN '{name}' THEN {value}" for name, value in SIGNAL_WEIGHTS.items())
    return f"CASE i.signal_type\n        {arms}\n        ELSE 0\n      END"


# Bound parameters (`:pid`, `:q`) carry runtime data and are never interpolated.
# The numeric tuning constants are interpolated from the module constants above.
POOLED_SIGNAL_SQL = f"""
WITH neighbours AS (
  SELECT
    1 - (i.query_embedding <=> CAST(:q AS vector)) AS sim,
    EXP(-EXTRACT(EPOCH FROM (now() - i.created_at)) / {DECAY_WINDOW_SECONDS}) AS decay,
    {_build_signal_case()} AS sval,
    i.signal_weight AS sw
  FROM interactions i
  WHERE i.property_id = :pid
    AND i.query_embedding IS NOT NULL
    AND (1 - (i.query_embedding <=> CAST(:q AS vector))) > {SIMILARITY_THRESHOLD}
    AND i.created_at > now() - INTERVAL '{TIME_WINDOW_DAYS} days'
)
SELECT CASE WHEN SUM(sim * decay) = 0 THEN 0
            ELSE SUM(sim * decay * sval * sw) / SUM(sim * decay)
       END AS pooled_signal
FROM neighbours
"""  # noqa: S608 -- only the module constants (decay window, threshold, signal weights) are interpolated, values bound

POOLED_SIGNAL_ASYNCPG_SQL = f"""
WITH neighbours AS (
  SELECT
    1 - (i.query_embedding <=> $2::vector) AS sim,
    EXP(-EXTRACT(EPOCH FROM (now() - i.created_at)) / {DECAY_WINDOW_SECONDS}) AS decay,
    {_build_signal_case()} AS sval,
    i.signal_weight AS sw
  FROM interactions i
  WHERE i.property_id = $1
    AND i.query_embedding IS NOT NULL
    AND (1 - (i.query_embedding <=> $2::vector)) > {SIMILARITY_THRESHOLD}
    AND i.created_at > now() - INTERVAL '{TIME_WINDOW_DAYS} days'
)
SELECT CASE WHEN SUM(sim * decay) = 0 THEN 0
            ELSE SUM(sim * decay * sval * sw) / SUM(sim * decay)
       END AS pooled_signal
FROM neighbours
"""  # noqa: S608 -- only the module constants (decay window, threshold, signal weights) are interpolated, values bound

POOLED_SIGNALS_SQL = f"""
WITH neighbours AS (
  SELECT
    i.property_id,
    1 - (i.query_embedding <=> CAST(:q AS vector)) AS sim,
    EXP(-EXTRACT(EPOCH FROM (now() - i.created_at)) / {DECAY_WINDOW_SECONDS}) AS decay,
    {_build_signal_case()} AS sval,
    i.signal_weight AS sw
  FROM interactions i
  WHERE i.property_id = ANY(:pids)
    AND i.query_embedding IS NOT NULL
    AND (1 - (i.query_embedding <=> CAST(:q AS vector))) > {SIMILARITY_THRESHOLD}
    AND i.created_at > now() - INTERVAL '{TIME_WINDOW_DAYS} days'
)
SELECT property_id,
       CASE WHEN SUM(sim * decay) = 0 THEN 0
            ELSE SUM(sim * decay * sval * sw) / SUM(sim * decay)
       END AS pooled_signal
FROM neighbours
GROUP BY property_id
"""  # noqa: S608 -- only the module constants (decay window, threshold, signal weights) are interpolated, values bound

POOLED_SIGNAL_ROWS_SQL = f"""
SELECT
  i.signal_type,
  i.signal_weight,
  i.query_embedding,
  i.created_at
FROM interactions i
WHERE i.property_id = :pid
  AND i.query_embedding IS NOT NULL
  AND i.created_at > now() - INTERVAL '{TIME_WINDOW_DAYS} days'
"""  # noqa: S608 -- only the module constants (decay window, threshold, signal weights) are interpolated, values bound


def _coerce_embedding(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [float(part) for part in value.split(",")]
    to_list = getattr(raw, "to_list", None)
    if callable(to_list):
        return [float(cast(float, value)) for value in cast(Sequence[object], to_list())]
    if isinstance(raw, Sequence) and not isinstance(raw, bytes):
        return [float(cast(float, value)) for value in cast(Sequence[object], raw)]
    return []


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _age_seconds(created_at: datetime, *, now: datetime) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return max(0.0, (now - created_at).total_seconds())


async def _asyncpg_driver_connection(session: AsyncSession) -> Any | None:
    try:
        connection = await session.connection()
        raw_connection = await connection.get_raw_connection()
        driver_connection = getattr(raw_connection, "driver_connection", None)
    except (AttributeError, TypeError):
        return None

    if driver_connection is None:
        return None
    if not driver_connection.__class__.__module__.startswith("asyncpg.") or not hasattr(driver_connection, "fetchval"):
        return None
    return driver_connection


async def _pooled_signal_asyncpg(
    session: AsyncSession,
    property_id: str,
    query_embedding: Sequence[float],
) -> float | None:
    driver_connection = await _asyncpg_driver_connection(session)
    if driver_connection is None:
        return None
    try:
        value = await driver_connection.fetchval(
            POOLED_SIGNAL_ASYNCPG_SQL, property_id, vector_literal(query_embedding)
        )
    except Exception as exc:
        raise DbError(f"failed to run pooled-signal query for property {property_id}") from exc
    return float(value) if value is not None else 0.0


async def pooled_signal(
    session: AsyncSession,
    property_id: str,
    query_embedding: Sequence[float],
) -> float:
    """Return the pooled buyer-signal score for `property_id` under this query.

    The score is a similarity- and recency-weighted average of signal values
    over qualifying interactions. Returns `0.0` when no interaction qualifies
    (none recent enough, none similar enough, or none at all).

    Raises `DbError` if `query_embedding` has the wrong dimension or the query
    fails.
    """
    validate_vector(query_embedding)
    fast_value = await _pooled_signal_asyncpg(session, property_id, query_embedding)
    if fast_value is not None:
        return fast_value

    try:
        result = await session.execute(
            text(POOLED_SIGNAL_ROWS_SQL),
            {"pid": property_id},
        )
    except SQLAlchemyError as exc:
        raise DbError(f"failed to run pooled-signal query for property {property_id}") from exc

    numerator = 0.0
    denominator = 0.0
    now = datetime.now(tz=UTC)
    for row in result:
        neighbour_embedding = _coerce_embedding(row.query_embedding)
        similarity = _cosine_similarity(query_embedding, neighbour_embedding)
        if similarity <= SIMILARITY_THRESHOLD:
            continue
        decay = math.exp(-_age_seconds(row.created_at, now=now) / DECAY_WINDOW_SECONDS)
        weight = similarity * decay
        signal_value = SIGNAL_WEIGHTS.get(str(row.signal_type), 0.0)
        signal_weight = float(row.signal_weight) if row.signal_weight is not None else 1.0
        numerator += weight * signal_value * signal_weight
        denominator += weight

    return numerator / denominator if denominator else 0.0


async def pooled_signals(
    session: AsyncSession,
    property_ids: Sequence[str],
    query_embedding: Sequence[float],
) -> dict[str, float]:
    """Return pooled buyer-signal scores for all `property_ids` in one query.

    Missing rows are returned as `0.0`, matching `pooled_signal()` semantics
    when a property has no qualifying interactions.
    """
    validate_vector(query_embedding)
    ids = list(dict.fromkeys(property_ids))
    if not ids:
        return {}
    try:
        result = await session.execute(
            text(POOLED_SIGNALS_SQL),
            {"pids": ids, "q": await vector_bind_value(session, query_embedding)},
        )
    except SQLAlchemyError as exc:
        raise DbError("failed to run batch pooled-signal query") from exc

    values = {pid: 0.0 for pid in ids}
    for row in result:
        values[str(row.property_id)] = float(row.pooled_signal) if row.pooled_signal is not None else 0.0
    return values
