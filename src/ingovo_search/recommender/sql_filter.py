"""§8 SQL hard filter — strict WHERE clause built from `HardFilters`.

`build_hard_filter_query` is pure: it turns a `HardFilters` value into a
SQLAlchemy `TextClause` + params dict, ready to execute. Per §8 step 1 the
missing-data policy is split by field kind. Verified location filters exclude
`NULL` because an unknown city or district is not geographically eligible.
Other categorical fields may retain unknown rows for downstream uncertainty;
numeric filters exclude `NULL` automatically. Only searchable/active status
variants enter the candidate pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

from ingovo_search.lib.property_price import EFFECTIVE_PROPERTY_PRICE_HUF_SQL
from ingovo_search.lib.property_status import ACTIVE, PUBLIC_VISIBILITY_SQL
from ingovo_search.recommender.parser_schema import HardFilters
from ingovo_search.types.errors import DbError

SEARCHABLE_STATUS = ACTIVE
# The canonical visibility predicate (`status = 'active'` is sargable on the status
# indexes). Every public candidate SQL statement must use this exact predicate.
PUBLIC_SEARCH_VISIBILITY_SQL = PUBLIC_VISIBILITY_SQL
DISTRICT_ROMAN = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
    "8": "VIII",
    "9": "IX",
    "10": "X",
    "11": "XI",
    "12": "XII",
    "13": "XIII",
    "14": "XIV",
    "15": "XV",
    "16": "XVI",
    "17": "XVII",
    "18": "XVIII",
    "19": "XIX",
    "20": "XX",
    "21": "XXI",
    "22": "XXII",
    "23": "XXIII",
}


def district_filter_values(districts: list[str]) -> list[str]:
    """Expand canonical Budapest district numbers to DB string variants."""
    values: list[str] = []
    for raw in districts:
        district = str(raw).strip().strip(".")
        if not district:
            continue
        candidates = [district]
        if district in DISTRICT_ROMAN:
            roman = DISTRICT_ROMAN[district]
            candidates.extend([f"{district}. kerület", roman, f"{roman}. kerület"])
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered not in values:
                values.append(lowered)
    return values


@dataclass(slots=True)
class _ClauseSet:
    """What `build_hard_filter_clauses` accumulates, in emission order."""

    clauses: list[str]
    params: dict[str, object] = field(default_factory=dict[str, object])
    array_params: list[str] = field(default_factory=list[str])

    def add(self, clause: str, **params: object) -> None:
        self.clauses.append(clause)
        self.params.update(params)

    def add_array(self, clause: str, name: str, values: list[str]) -> None:
        """A clause over one string-array param (`= ANY(:name)` / `<> ALL(:name)`)."""
        self.clauses.append(clause)
        self.params[name] = values
        self.array_params.append(name)


def _add_city(out: _ClauseSet, hard_filters: HardFilters, city_scope: Sequence[str] | None) -> None:
    if city_scope is not None:
        if city_scope:
            out.add_array("LOWER(city) = ANY(:city_scope)", "city_scope", [city.lower() for city in city_scope])
        else:
            out.add("FALSE")
    elif hard_filters.city is not None:
        out.add("LOWER(city) = LOWER(:city)", city=hard_filters.city)


def _add_district(out: _ClauseSet, hard_filters: HardFilters, district_scope: Sequence[str] | None) -> None:
    if district_scope is not None:
        if district_scope:
            out.add_array(
                "LOWER(district) = ANY(:district_scope)",
                "district_scope",
                district_filter_values(list(district_scope)),
            )
        else:
            out.add("FALSE")
    elif hard_filters.district_in:
        out.add_array(
            "LOWER(district) = ANY(:districts)", "districts", district_filter_values(hard_filters.district_in)
        )


def _add_categorical(out: _ClauseSet, hard_filters: HardFilters) -> None:
    if hard_filters.transaction_type is not None:
        out.add(
            "(LOWER(transaction_type) = LOWER(:transaction_type) OR transaction_type IS NULL)",
            transaction_type=hard_filters.transaction_type,
        )

    if hard_filters.property_type_in:
        out.add_array(
            "(LOWER(property_type) = ANY(:property_types_in) OR property_type IS NULL)",
            "property_types_in",
            [pt.lower() for pt in hard_filters.property_type_in],
        )

    if hard_filters.property_type_exclude:
        out.add_array(
            "(LOWER(property_type) <> ALL(:property_types_exclude) OR property_type IS NULL)",
            "property_types_exclude",
            [pt.lower() for pt in hard_filters.property_type_exclude],
        )

    # `rooms` is the source text ("2", "2,5", "2 + 1", "nappali + 2 szoba");
    # `rooms_count` is the numeric derived from it in the database (GENERATED
    # ALWAYS AS ingovo_room_count(rooms), served by ix_properties_rooms_count).
    # Filtering on the derived column instead of a regex-guarded CAST keeps the
    # parse rules in one
    # place (the `ingovo_room_count` SQL function in schema.sql), half rooms compare
    # numerically (2.5 >= 2), and an unparseable shape is NULL rather than a
    # cast error. Exact match keeps the §8 categorical policy — unknown rows
    # stay for downstream uncertainty — while `rooms_min` is numeric and drops
    # NULL like every other numeric filter. `rooms_count` is deliberately not on
    # the ORM model (fresh-DB create_all hazard), so this is the one reader.
    if hard_filters.rooms is not None:
        out.add("(rooms_count = :rooms OR rooms_count IS NULL)", rooms=hard_filters.rooms)


def _add_numeric(out: _ClauseSet, hard_filters: HardFilters) -> None:
    """The numeric bounds, in emission order; each drops `NULL` rows."""
    bounds: tuple[tuple[str, str, float | None], ...] = (
        ("rooms_count >= :rooms_min", "rooms_min", hard_filters.rooms_min),
        ("area_m2 >= :area_m2_min", "area_m2_min", hard_filters.area_m2_min),
        ("area_m2 <= :area_m2_max", "area_m2_max", hard_filters.area_m2_max),
        (f"{EFFECTIVE_PROPERTY_PRICE_HUF_SQL} >= :price_min", "price_min", hard_filters.price_huf_min),
        (f"{EFFECTIVE_PROPERTY_PRICE_HUF_SQL} <= :price_max", "price_max", hard_filters.price_huf_max),
        ("floor >= :floor_min", "floor_min", hard_filters.floor_min),
    )
    for clause, name, value in bounds:
        if value is not None:
            out.add(clause, **{name: value})


def _add_near(out: _ClauseSet, hard_filters: HardFilters) -> None:
    if hard_filters.near is not None:
        out.add(
            "geo IS NOT NULL AND "
            "ST_DWithin(geo, ST_SetSRID(ST_MakePoint(:near_lng, :near_lat), 4326)::geography, :near_radius_m)",
            near_lat=hard_filters.near.lat,
            near_lng=hard_filters.near.lng,
            near_radius_m=hard_filters.near.radius_km * 1000.0,
        )


def build_hard_filter_clauses(
    hard_filters: HardFilters,
    *,
    city_scope: Sequence[str] | None = None,
    district_scope: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, object], list[str]]:
    """Render `HardFilters` into `(clauses, params, array_param_names)`.

    Pure clause construction shared by every statement that applies the §8 hard
    filter — the canonical `SELECT id` query, plus the homepage probe's `COUNT`,
    price-percentile, and preview-sample statements. Callers join `clauses` with
    ` AND `, bind `array_param_names` as `ARRAY(String())`, and pass `params`.
    """
    out = _ClauseSet(clauses=[PUBLIC_SEARCH_VISIBILITY_SQL])
    _add_city(out, hard_filters, city_scope)
    _add_district(out, hard_filters, district_scope)
    _add_categorical(out, hard_filters)
    _add_numeric(out, hard_filters)
    _add_near(out, hard_filters)
    return out.clauses, out.params, out.array_params


def bind_array_params(stmt: TextClause, array_params: Sequence[str]) -> TextClause:
    """Bind the string-array params so `= ANY(:name)` sees a Postgres array."""
    if not array_params:
        return stmt
    return stmt.bindparams(*(bindparam(name, type_=ARRAY(String())) for name in array_params))


def build_hard_filter_where(
    hard_filters: HardFilters,
    *,
    city_scope: Sequence[str] | None = None,
    district_scope: Sequence[str] | None = None,
) -> tuple[str, dict[str, object], list[str]]:
    """Return the §8 hard filter as `(where_sql, params, array_param_names)`.

    `where_sql` is a ready-to-embed boolean expression (no `WHERE` keyword) that
    any `... FROM properties WHERE {where_sql}` statement can reuse.
    """
    clauses, params, array_params = build_hard_filter_clauses(
        hard_filters, city_scope=city_scope, district_scope=district_scope
    )
    return " AND ".join(clauses), params, array_params


def build_hard_filter_query(
    hard_filters: HardFilters,
    *,
    city_scope: Sequence[str] | None = None,
    district_scope: Sequence[str] | None = None,
) -> tuple[TextClause, dict[str, object]]:
    """Render `HardFilters` into a `SELECT id` query targeting `properties`.

    Returns the `text()`-bound statement and the params dict to pass to
    `session.execute`. City and district are strict verified constraints;
    `city_scope` / `district_scope` provide the separately authorised nearby
    administrative set. Numeric conditions exclude `NULL` because the
    constraint cannot be verified.
    """
    where_sql, params, array_params = build_hard_filter_where(
        hard_filters, city_scope=city_scope, district_scope=district_scope
    )
    stmt = bind_array_params(text("SELECT id FROM properties WHERE " + where_sql), array_params)  # noqa: S608 -- where_sql is the §8 clause list, built from constant fragments with every value bound
    return stmt, params


async def run_hard_filter(
    session: AsyncSession,
    hard_filters: HardFilters,
    *,
    city_scope: Sequence[str] | None = None,
    district_scope: Sequence[str] | None = None,
) -> list[str]:
    """Execute the §8 hard-filter SQL and return matching property IDs."""
    stmt, params = build_hard_filter_query(hard_filters, city_scope=city_scope, district_scope=district_scope)
    try:
        result = await session.execute(stmt, params)
    except SQLAlchemyError as exc:
        raise DbError("hard filter query failed") from exc
    return [str(row.id) for row in result]


async def cached_listing_ids_are_publicly_visible(
    session: AsyncSession,
    listing_ids: Sequence[str],
) -> bool:
    """Fail closed before a cached public response can be replayed.

    A response cache outlives an individual listing's status transition or
    expiry by design.  Its row IDs therefore need a fresh, shared
    status-and-expiry authorization check on every hit; cache TTL is only an
    optimisation, never the visibility boundary.  Invalid/duplicate cache
    payload IDs and database/result-shape failures are cache misses rather than
    reasons to return a possibly stale public result.
    """

    ids = [str(listing_id).strip() for listing_id in listing_ids]
    if not ids:
        return True
    if any(not listing_id for listing_id in ids) or len(set(ids)) != len(ids):
        return False

    stmt = bind_array_params(
        text(f"SELECT id FROM properties WHERE id = ANY(:ids) AND ({PUBLIC_SEARCH_VISIBILITY_SQL})"),  # noqa: S608 -- only the constant visibility predicate is interpolated, the ids are bound
        ["ids"],
    )
    try:
        result = await session.execute(stmt, {"ids": ids})
        visible_ids = {str(row.id) for row in result}
    except (SQLAlchemyError, AttributeError, TypeError):
        return False
    return visible_ids == set(ids)
