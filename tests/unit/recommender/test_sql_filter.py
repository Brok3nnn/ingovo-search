"""Unit tests for the SQL hard-filter builder.

`build_hard_filter_query` is pure — it returns a `TextClause` plus a params
dict — so it is fully covered here without a database. `run_hard_filter`
behaviour against real data lives in `tests/integration/recommender/`.
"""

from __future__ import annotations

from types import SimpleNamespace

from ingovo_search.lib.property_price import EFFECTIVE_PROPERTY_PRICE_HUF_SQL
from ingovo_search.recommender.parser_schema import HardFilters, NearFilter
from ingovo_search.recommender.sql_filter import (
    PUBLIC_SEARCH_VISIBILITY_SQL,
    build_hard_filter_clauses,
    build_hard_filter_query,
    cached_listing_ids_are_publicly_visible,
    district_filter_values,
)


def _sql(hard_filters: HardFilters) -> tuple[str, dict[str, object]]:
    stmt, params = build_hard_filter_query(hard_filters)
    return str(stmt), params


def test_empty_hard_filters_returns_all_searchable_listings() -> None:
    sql, params = _sql(HardFilters())
    assert "SELECT id FROM properties WHERE status = 'active'" in sql
    assert "expires_at IS NULL OR expires_at > now()" in sql
    assert params == {}


def test_price_huf_max_excludes_listings_above_threshold() -> None:
    sql, params = _sql(HardFilters(price_huf_max=50_000_000))
    assert "price_huf IS NOT NULL" in sql
    assert "price_huf > 0" in sql
    assert "currency = 'HUF'" in sql
    # The legacy `price` column was dropped by Alembic 0061 — the filter must
    # not name it (the SELECT would fail), nor fall back to it.
    assert "price::bigint" not in sql
    assert "floor(price)" not in sql
    assert ">= 0" not in sql  # zero is never a real HUF amount
    assert "> :price_max" not in sql
    assert "END <= :price_max" in sql
    assert params["price_max"] == 50_000_000


class _VisibilitySession:
    def __init__(self, visible_ids: list[str]) -> None:
        self.visible_ids = visible_ids
        self.statement: object | None = None
        self.params: dict[str, object] | None = None

    async def execute(self, statement: object, params: dict[str, object]) -> list[SimpleNamespace]:
        self.statement = statement
        self.params = params
        return [SimpleNamespace(id=listing_id) for listing_id in self.visible_ids]


async def test_cached_response_visibility_recheck_uses_the_shared_expiry_predicate_and_fails_closed() -> None:
    session = _VisibilitySession(["visible"])

    assert await cached_listing_ids_are_publicly_visible(session, ["visible"])  # pyright: ignore[reportArgumentType]
    assert session.params == {"ids": ["visible"]}
    assert "expires_at IS NULL OR expires_at > now()" in str(session.statement)

    assert not await cached_listing_ids_are_publicly_visible(session, ["visible", "expired"])  # pyright: ignore[reportArgumentType]
    assert not await cached_listing_ids_are_publicly_visible(session, ["visible", "visible"])  # pyright: ignore[reportArgumentType]


# --- Location fields require verified values ---------------------------------


def test_city_filter_excludes_null_and_other_city_rows() -> None:
    sql, params = _sql(HardFilters(city="Budapest"))
    assert "LOWER(city) = LOWER(:city)" in sql
    assert "city IS NULL" not in sql
    assert params["city"] == "Budapest"


def test_district_in_filter_excludes_null_district_rows() -> None:
    sql, params = _sql(HardFilters(district_in=["12", "13"]))
    assert "LOWER(district) = ANY(:districts)" in sql
    assert "district IS NULL" not in sql
    assert params["districts"] == [
        "12",
        "12. kerület",
        "xii",
        "xii. kerület",
        "13",
        "13. kerület",
        "xiii",
        "xiii. kerület",
    ]


def test_district_filter_values_expands_budapest_roman_aliases() -> None:
    assert district_filter_values(["2", "11"]) == [
        "2",
        "2. kerület",
        "ii",
        "ii. kerület",
        "11",
        "11. kerület",
        "xi",
        "xi. kerület",
    ]


def test_nearby_city_scope_replaces_exact_city_with_only_allowed_names() -> None:
    stmt, params = build_hard_filter_query(
        HardFilters(city="Budapest"),
        city_scope=["Budaörs", "Törökbálint"],
    )
    sql = str(stmt)
    assert "LOWER(city) = ANY(:city_scope)" in sql
    assert "LOWER(city) = LOWER(:city)" not in sql
    assert params["city_scope"] == ["budaörs", "törökbálint"]


def test_nearby_district_scope_replaces_exact_district() -> None:
    stmt, params = build_hard_filter_query(
        HardFilters(city="Budapest", district_in=["13"]),
        city_scope=["Budapest"],
        district_scope=["5"],
    )
    sql = str(stmt)
    assert "LOWER(district) = ANY(:district_scope)" in sql
    assert "LOWER(district) = ANY(:districts)" not in sql
    assert params["district_scope"] == ["5", "5. kerület", "v", "v. kerület"]


def test_property_type_in_and_exclude_clauses() -> None:
    hard = HardFilters(property_type_in=["Lakás"], property_type_exclude=["Ház"])
    sql, params = _sql(hard)
    assert "(LOWER(property_type) = ANY(:property_types_in) OR property_type IS NULL)" in sql
    assert "(LOWER(property_type) <> ALL(:property_types_exclude) OR property_type IS NULL)" in sql
    assert params["property_types_in"] == ["lakás"]
    assert params["property_types_exclude"] == ["ház"]


def test_transaction_type_clause() -> None:
    sql, params = _sql(HardFilters(transaction_type="rent"))
    assert "(LOWER(transaction_type) = LOWER(:transaction_type) OR transaction_type IS NULL)" in sql
    assert params["transaction_type"] == "rent"


def test_rooms_exact_treated_as_categorical() -> None:
    """Exact rooms reads the derived numeric column, keeps the unknown-rows policy."""
    sql, params = _sql(HardFilters(rooms=2))
    assert "(rooms_count = :rooms OR rooms_count IS NULL)" in sql
    assert params["rooms"] == 2
    assert isinstance(params["rooms"], int)
    assert "(rooms = :rooms" not in sql


# --- Numeric fields ---------------------------------------------------------


def test_rooms_min_reads_the_generated_rooms_count_column() -> None:
    """`rooms_min` compares the numeric `rooms_count` (0038), not a CAST of the text.

    The regex-guarded `CAST(REPLACE(rooms, ',', '.') AS numeric)` workaround is
    gone: the database derives `rooms_count` from the source text with
    `ingovo_room_count`, so half rooms ("2,5") compare numerically and
    unparseable shapes ("2 + 1") are NULL — dropped like any numeric filter.
    """
    sql, params = _sql(HardFilters(rooms_min=2))
    assert "rooms_count >= :rooms_min" in sql
    assert params["rooms_min"] == 2
    assert "rooms ~" not in sql
    assert "CAST(REPLACE(rooms" not in sql


def test_area_and_price_range_clauses() -> None:
    sql, params = _sql(
        HardFilters(area_m2_min=40.0, area_m2_max=80.0, price_huf_min=20_000_000, price_huf_max=60_000_000)
    )
    assert "area_m2 >= :area_m2_min" in sql
    assert "area_m2 <= :area_m2_max" in sql
    assert "END >= :price_min" in sql
    assert "END <= :price_max" in sql
    assert params["area_m2_min"] == 40.0
    assert params["area_m2_max"] == 80.0
    assert params["price_min"] == 20_000_000
    assert params["price_max"] == 60_000_000


def test_floor_min_clause() -> None:
    sql, params = _sql(HardFilters(floor_min=3))
    assert "floor >= :floor_min" in sql
    assert params["floor_min"] == 3


def test_near_clause_excludes_null_geo_and_uses_postgis() -> None:
    near = NearFilter(lat=47.5, lng=19.0, radius_km=2.0)
    sql, params = _sql(HardFilters(near=near))
    assert "geo IS NOT NULL" in sql
    assert "ST_DWithin(geo, ST_SetSRID(ST_MakePoint(:near_lng, :near_lat), 4326)::geography, :near_radius_m)" in sql
    assert params["near_lat"] == 47.5
    assert params["near_lng"] == 19.0
    assert params["near_radius_m"] == 2000.0


def test_combined_filters_produce_one_query() -> None:
    hard = HardFilters(
        city="Budapest",
        district_in=["12"],
        property_type_in=["Lakás"],
        rooms=2,
        price_huf_max=80_000_000,
    )
    sql, params = _sql(hard)
    for fragment in [
        "LOWER(city) = LOWER(:city)",
        "LOWER(district) = ANY(:districts)",
        "(LOWER(property_type) = ANY(:property_types_in) OR property_type IS NULL)",
        "(rooms_count = :rooms OR rooms_count IS NULL)",
        "END <= :price_max",
    ]:
        assert fragment in sql
    assert params["city"] == "Budapest"
    assert params["price_max"] == 80_000_000


def test_every_filter_renders_in_a_fixed_clause_param_and_array_order() -> None:
    """The emission order is part of the contract: the probe statements and the golden see the same SQL text."""
    hard = HardFilters(
        city="Budapest",
        district_in=["12"],
        transaction_type="sale",
        property_type_in=["Lakás"],
        property_type_exclude=["Telek"],
        rooms=2,
        rooms_min=1,
        area_m2_min=40.0,
        area_m2_max=90.0,
        price_huf_min=20_000_000,
        price_huf_max=80_000_000,
        floor_min=1,
        near=NearFilter(lat=47.5, lng=19.0, radius_km=1.5),
    )

    clauses, params, array_params = build_hard_filter_clauses(hard)

    assert clauses == [
        PUBLIC_SEARCH_VISIBILITY_SQL,
        "LOWER(city) = LOWER(:city)",
        "LOWER(district) = ANY(:districts)",
        "(LOWER(transaction_type) = LOWER(:transaction_type) OR transaction_type IS NULL)",
        "(LOWER(property_type) = ANY(:property_types_in) OR property_type IS NULL)",
        "(LOWER(property_type) <> ALL(:property_types_exclude) OR property_type IS NULL)",
        "(rooms_count = :rooms OR rooms_count IS NULL)",
        "rooms_count >= :rooms_min",
        "area_m2 >= :area_m2_min",
        "area_m2 <= :area_m2_max",
        f"{EFFECTIVE_PROPERTY_PRICE_HUF_SQL} >= :price_min",
        f"{EFFECTIVE_PROPERTY_PRICE_HUF_SQL} <= :price_max",
        "floor >= :floor_min",
        "geo IS NOT NULL AND "
        "ST_DWithin(geo, ST_SetSRID(ST_MakePoint(:near_lng, :near_lat), 4326)::geography, :near_radius_m)",
    ]
    assert list(params.items()) == [
        ("city", "Budapest"),
        ("districts", ["12", "12. kerület", "xii", "xii. kerület"]),
        ("transaction_type", "sale"),
        ("property_types_in", ["lakás"]),
        ("property_types_exclude", ["telek"]),
        ("rooms", 2),
        ("rooms_min", 1),
        ("area_m2_min", 40.0),
        ("area_m2_max", 90.0),
        ("price_min", 20_000_000),
        ("price_max", 80_000_000),
        ("floor_min", 1),
        ("near_lat", 47.5),
        ("near_lng", 19.0),
        ("near_radius_m", 1500.0),
    ]
    assert array_params == ["districts", "property_types_in", "property_types_exclude"]


def test_scopes_render_before_the_other_filters_and_an_empty_scope_matches_nothing() -> None:
    clauses, params, array_params = build_hard_filter_clauses(
        HardFilters(city="Budapest", district_in=["12"], floor_min=2),
        city_scope=["Budaörs"],
        district_scope=[],
    )

    assert clauses == [PUBLIC_SEARCH_VISIBILITY_SQL, "LOWER(city) = ANY(:city_scope)", "FALSE", "floor >= :floor_min"]
    assert list(params.items()) == [("city_scope", ["budaörs"]), ("floor_min", 2)]
    assert array_params == ["city_scope"]

    clauses, params, array_params = build_hard_filter_clauses(
        HardFilters(city="Budapest"), city_scope=[], district_scope=["5"]
    )

    assert clauses == [PUBLIC_SEARCH_VISIBILITY_SQL, "FALSE", "LOWER(district) = ANY(:district_scope)"]
    assert list(params.items()) == [("district_scope", ["5", "5. kerület", "v", "v. kerület"])]
    assert array_params == ["district_scope"]
