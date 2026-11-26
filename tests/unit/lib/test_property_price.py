"""Exact money-boundary tests for public property readers and ingestion."""

from __future__ import annotations

from types import SimpleNamespace

from ingovo_search.lib.property_price import (
    EFFECTIVE_PROPERTY_PRICE_HUF_SQL,
    MAX_LEGACY_SAFE_INTEGER,
    canonical_ingestion_price_huf,
    public_price_fields,
    resolve_property_price_huf,
)


def test_canonical_bigint_on_a_row_stays_exact_in_public_json() -> None:
    property_row = SimpleNamespace(price_huf=9_007_199_254_740_993, currency="HUF")

    assert resolve_property_price_huf(property_row) == 9_007_199_254_740_993
    assert public_price_fields(property_row) == ("9007199254740993", None)


def test_a_row_without_a_price_resolves_to_none() -> None:
    # Since Alembic 0061 an ORM row has no `price` attribute at all, so the
    # resolver must not reach for one.
    property_row = SimpleNamespace(price_huf=None, currency="HUF")

    assert resolve_property_price_huf(property_row) is None
    assert public_price_fields(property_row) == (None, None)


def test_canonical_payload_key_wins_over_a_divergent_legacy_price_key() -> None:
    assert (
        resolve_property_price_huf({"priceHuf": "9007199254740993", "price": 84_900_000.0, "currency": "HUF"})
        == 9_007_199_254_740_993
    )
    assert resolve_property_price_huf(
        {"price_huf": 9_007_199_254_740_993, "price": 84_900_000.0, "currency": "HUF"}
    ) == (9_007_199_254_740_993)
    assert resolve_property_price_huf({"price_huf": 0, "price": 84_900_000.0, "currency": "HUF"}) is None


def test_exact_legacy_price_key_is_a_payload_only_huf_fallback() -> None:
    # JSON payloads only (a posted drawer payload, a Node-projected listing).
    payload = {"price": 84_900_000.0, "currency": "HUF"}

    assert resolve_property_price_huf(payload) == 84_900_000
    assert public_price_fields(payload) == ("84900000", 84_900_000.0)


def test_fractional_non_finite_unsafe_and_non_huf_payload_values_are_not_reinterpreted_as_money() -> None:
    for price in (84_900_000.5, float("nan"), float("inf"), float(MAX_LEGACY_SAFE_INTEGER + 2)):
        assert public_price_fields({"price": price, "currency": "HUF"}) == (None, None)

    assert public_price_fields({"price_huf": 84_900_000, "price": 84_900_000.0, "currency": "EUR"}) == (None, None)


def test_ingestion_persists_only_canonical_positive_huf() -> None:
    assert canonical_ingestion_price_huf(84_900_000) == 84_900_000
    assert isinstance(canonical_ingestion_price_huf(84_900_000), int)
    assert canonical_ingestion_price_huf(0) is None


def test_effective_price_sql_reads_only_the_canonical_column() -> None:
    sql = EFFECTIVE_PROPERTY_PRICE_HUF_SQL
    assert "price_huf IS NOT NULL" in sql
    assert "price_huf > 0" in sql
    assert "currency = 'HUF'" in sql
    assert sql.rstrip().endswith("END")
    # The legacy `price` column was dropped by Alembic 0061; any reference here
    # would make every embedding SELECT fail on the live database.
    assert "price::bigint" not in sql
    assert "floor(price)" not in sql
    assert "price IS NOT NULL" not in sql
