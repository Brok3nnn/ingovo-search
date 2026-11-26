"""`db/schema.sql` and the SQLAlchemy models describe the same tables and columns."""

from __future__ import annotations

import re

import pytest

from ingovo_search.db.models import Base
from ingovo_search.seed import schema_sql

# Columns that exist only in SQL on purpose (see the comment on `Property.rooms`).
SQL_ONLY_COLUMNS = {"properties": {"rooms_count"}}
_CONSTRAINT_PREFIXES = ("CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN")


def _sql_tables() -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", schema_sql(), flags=re.DOTALL):
        columns: set[str] = set()
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("--", *_CONSTRAINT_PREFIXES)) or raw_line.startswith("        "):
                continue
            columns.add(line.split()[0])
        tables[name] = columns
    return tables


def test_schema_defines_every_model_table() -> None:
    assert set(_sql_tables()) == set(Base.metadata.tables) | {"embedding_cache"}


@pytest.mark.parametrize("table_name", sorted(Base.metadata.tables))
def test_model_columns_match_the_schema(table_name: str) -> None:
    orm_columns = {column.name for column in Base.metadata.tables[table_name].columns}
    sql_columns = _sql_tables()[table_name] - SQL_ONLY_COLUMNS.get(table_name, set())
    assert orm_columns == sql_columns


def test_schema_creates_the_extensions_and_vector_indexes() -> None:
    sql = schema_sql()
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in sql
    assert "CREATE EXTENSION IF NOT EXISTS postgis;" in sql
    assert "USING hnsw (factual_embedding vector_cosine_ops)" in sql
    assert "USING hnsw (descriptive_embedding vector_cosine_ops)" in sql
    assert "ON properties USING gist (geo)" in sql
    assert "GENERATED ALWAYS AS (ingovo_room_count(rooms)) STORED" in sql
