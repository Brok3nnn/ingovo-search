"""Homepage index probe — a cheap, grounded preview of the live listing pool.

Where ``/api/search/preview`` only *interprets* the sentence, the probe *touches
the real index*. For the current :class:`ParsedQuery` it:

1. counts the listings that satisfy the strict hard filter (``exact_count``);
2. summarises the market price band for the same segment **with the price
   constraint removed**, plus where the buyer's budget sits inside it
   (``price_stats`` + ``budget_percentile``) — the "is my budget tight or
   generous here" signal the cockpit needs;
3. returns a few *real* preview listings, ranked by the **deterministic half**
   of the scorer;
4. offers grounded **directions** — each widening move annotated with the real
   counterfactual ``COUNT`` delta ("+58"), actionable via a ``RefinementOverrides``
   payload the frontend replays into ``/api/search``;
5. distils one market **insight** sentence.

It deliberately skips every expensive step of the full pipeline — no query
embedding, no pgvector similarity, no LLM rerank, no narrator — so it can run in
tens of milliseconds on every debounced keystroke. Because the semantic
(content/factual) terms are therefore ``0``, preview ``match_percent`` reads the
structured fit (geo · price · area · rooms · type · prefs); a semantic-heavy
query will read a notch below its final ``/talalatok`` percentage, which is the
honest floor rather than an inflated guess.

``nearby_count`` is intentionally left ``None`` here: the strict filter alone
cannot express the authorised nearby administrative set — that is resolved by the
region-scope layer and wired in the geo phase.

The payload models (``SearchProbe`` and its parts) live in
``ingovo_search.recommender.probe_schema`` and are re-exported here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.property_price import (
    EFFECTIVE_PROPERTY_PRICE_HUF_SQL,
    MAX_LEGACY_SAFE_INTEGER,
    public_price_fields,
)
from ingovo_search.lib.session import RefinementOverrides
from ingovo_search.recommender.candidate_rows import fetch_candidate_rows
from ingovo_search.recommender.match_percent import compute_match_breakdown
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.probe_schema import PriceStats, ProbeDirection, ProbePreviewListing, SearchProbe
from ingovo_search.recommender.score import ScoreBreakdown, ScoreInputs, compute_score
from ingovo_search.recommender.search_preview import PreviewLocale, constraint_summary
from ingovo_search.recommender.sql_filter import bind_array_params, build_hard_filter_where
from ingovo_search.types.errors import DbError

# The ranking helpers below read candidate rows purely through
# `getattr(row, name, default)`, and every function they hand the row to
# (`compute_score`, `compute_match_breakdown`, `public_price_fields`) accepts
# `Any`/`object`. The rows are therefore genuinely duck-typed: production
# passes the lightweight `SimpleNamespace` rows from `fetch_candidate_rows`,
# while tests pass real `Property` ORM instances. Annotating them as
# `SimpleNamespace` claimed a precision the code never had and rejected the ORM
# rows; a `Protocol` with declared fields would have failed the other way,
# because `SimpleNamespace` satisfies no declared attribute. A type parameter
# states the actual contract — "whatever row type you pass in comes back out" —
# and keeps the pairing between a row and its score exact for the caller.

DEFAULT_SAMPLE_SIZE = 48
DEFAULT_PREVIEW_COUNT = 3
MAX_WIDEN_DIRECTIONS = 4
# Below this segment size the price distribution is too thin to state percentiles.
MIN_SEGMENT_FOR_INSIGHT = 8


def _intent_hash(query: ParsedQuery) -> str:
    return hashlib.sha256(query.model_dump_json().encode("utf-8")).hexdigest()[:20]


def _market_filters(hard: HardFilters) -> HardFilters:
    """The same hard filter with price cleared — the price-comparison segment."""
    return hard.model_copy(update={"price_huf_min": None, "price_huf_max": None})


def _budget(query: ParsedQuery) -> int | None:
    return query.hard_filters.price_huf_max or query.anchors.price_huf


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_image(row: object) -> str | None:
    images = getattr(row, "images", None)
    if isinstance(images, (list, tuple)) and images:
        first = cast("object", images[0])
        return str(first) if first else None
    return None


def _fmt_huf(value: int, locale: PreviewLocale) -> str:
    if value > MAX_LEGACY_SAFE_INTEGER:
        formatted = f"{value:,}".replace(",", " ")
        return f"{formatted} Ft" if locale == "hu" else f"HUF {formatted}"
    if value >= 1_000_000:
        number = f"{value / 1_000_000:g}".replace(".", "," if locale == "hu" else ".")
        return f"{number} M Ft" if locale == "hu" else f"HUF {number}M"
    formatted = f"{value:,}".replace(",", " ")
    return f"{formatted} Ft" if locale == "hu" else f"HUF {formatted}"


def score_and_rank[RowT](
    rows: Sequence[RowT], query: ParsedQuery, *, now: datetime
) -> list[tuple[RowT, ScoreBreakdown]]:
    """Deterministic ranking without embeddings/LLM. Pure and stable.

    ``geo_score_value=0.0`` and ``explore=0.0`` keep the sample ordering stable
    and cheap: previews are ranked on anchor · quality · soft-pref · recency,
    which is enough to surface a representative handful, not the final order.
    """
    scored = [(row, compute_score(row, query, ScoreInputs(geo_score_value=0.0, explore=0.0, now=now))) for row in rows]
    scored.sort(key=lambda pair: (-pair[1].total, str(getattr(pair[0], "id", ""))))
    return scored


def preview_from(row: object, breakdown: ScoreBreakdown, query: ParsedQuery) -> ProbePreviewListing:
    """Build a preview card using the product's own match-% derivation."""
    match = compute_match_breakdown(row, breakdown, query).match_percent
    rooms = getattr(row, "rooms", None)
    price_huf, legacy_price = public_price_fields(row)
    return ProbePreviewListing(
        listing_id=str(getattr(row, "id", "")),
        match_percent=match,
        title=getattr(row, "title", None),
        price=int(legacy_price) if legacy_price is not None else None,
        price_huf=price_huf,
        area_m2=_as_float(getattr(row, "area_m2", None)),
        rooms=str(rooms) if rooms is not None else None,
        city=getattr(row, "city", None),
        district=getattr(row, "district", None),
        property_type=getattr(row, "property_type", None),
        transaction_type=getattr(row, "transaction_type", None),
        image=_first_image(row),
    )


def widen_candidates(
    hard: HardFilters, median_price: int | None, locale: PreviewLocale
) -> list[tuple[str, str, HardFilters]]:
    """Enumerate single-constraint widenings of `hard` — `(kind, title, widened)`.

    Only moves that a human agent would relax first (never city/district-of-city
    beyond dropping an explicit district list): raise the price ceiling toward the
    market median, drop a district list, loosen exact/min rooms, drop the size
    envelope, accept previously-excluded types. Each is applied *independently* to
    the original filter so its ``COUNT`` delta is attributable to that one move.
    """
    out: list[tuple[str, str, HardFilters]] = []

    if hard.price_huf_max is not None:
        # Prices are integer HUF.  Multiplying by a binary float at this point
        # silently changes a legitimate bigint-sized user budget before the
        # query is widened.  Compute ceil(6/5) and the display step entirely
        # with integers instead.
        expanded_budget = (hard.price_huf_max * 6 + 4) // 5
        target = max(expanded_budget, median_price or 0)
        step = 5_000_000
        target = ((target + step - 1) // step) * step
        if target > hard.price_huf_max:
            title = (
                f"Emeld a keretet {_fmt_huf(target, locale)}-ig"
                if locale == "hu"
                else f"Raise the budget to {_fmt_huf(target, locale)}"
            )
            out.append(("loosen_price", title, hard.model_copy(update={"price_huf_max": target})))

    if hard.district_in:
        title = "Bármelyik kerület" if locale == "hu" else "Any district"
        out.append(("drop_district", title, hard.model_copy(update={"district_in": []})))

    if hard.rooms is not None:
        title = "Bármennyi szoba" if locale == "hu" else "Any number of rooms"
        out.append(("loosen_rooms", title, hard.model_copy(update={"rooms": None})))
    elif hard.rooms_min is not None:
        title = "Kevesebb szoba is" if locale == "hu" else "Fewer rooms allowed"
        out.append(("loosen_rooms_min", title, hard.model_copy(update={"rooms_min": None})))

    if hard.area_m2_min is not None or hard.area_m2_max is not None:
        title = "Bármekkora méret" if locale == "hu" else "Any size"
        out.append(("loosen_area", title, hard.model_copy(update={"area_m2_min": None, "area_m2_max": None})))

    if hard.property_type_exclude:
        title = "Kizárt típusok is" if locale == "hu" else "Include excluded types"
        out.append(("accept_excluded_types", title, hard.model_copy(update={"property_type_exclude": []})))

    return out


def _build_insight(
    price_stats: PriceStats, best_widen: ProbeDirection | None, exact_count: int, locale: PreviewLocale
) -> str | None:
    if exact_count == 0 and best_widen is not None:
        return (
            f"Pontosan illő találat nincs, de „{best_widen.title}” +{best_widen.delta_count} lehetőséget nyit."
            if locale == "hu"
            else f"No exact match, but “{best_widen.title}” opens +{best_widen.delta_count}."
        )
    pct = price_stats.budget_percentile
    if pct is not None and price_stats.budget is not None and price_stats.segment_count >= MIN_SEGMENT_FOR_INSIGHT:
        budget = _fmt_huf(price_stats.budget, locale)
        median = _fmt_huf(price_stats.median, locale) if price_stats.median is not None else None
        if pct < 30 and median is not None:
            return (
                f"A(z) {budget} keret feszes ehhez a szegmenshez — a piaci medián {median}."
                if locale == "hu"
                else f"{budget} is tight for this segment — the market median is {median}."
            )
        if pct > 70:
            return (
                f"A(z) {budget} keret bőséges — a kínálat nagy része belefér."
                if locale == "hu"
                else f"{budget} is generous here — most of the supply fits."
            )
    if best_widen is not None and best_widen.delta_count > 0:
        return (
            f"A legtöbb új találatot ez adná: „{best_widen.title}” (+{best_widen.delta_count})."
            if locale == "hu"
            else f"The biggest gain: “{best_widen.title}” (+{best_widen.delta_count})."
        )
    return None


async def _count(session: AsyncSession, where_sql: str, params: dict[str, object], array_params: list[str]) -> int:
    stmt = bind_array_params(text(f"SELECT COUNT(*) AS n FROM properties WHERE {where_sql}"), array_params)  # noqa: S608 -- where_sql is the clause list, built from constant fragments with every value bound
    try:
        result = await session.execute(stmt, params)
    except SQLAlchemyError as exc:
        raise DbError("probe count query failed") from exc
    row = result.first()
    return int(row.n) if row is not None and row.n is not None else 0


async def _count_hard(session: AsyncSession, hard: HardFilters) -> int:
    where_sql, params, array_params = build_hard_filter_where(hard)
    return await _count(session, where_sql, params, array_params)


async def _price_stats(session: AsyncSession, market_hard: HardFilters, budget: int | None) -> PriceStats:
    where_sql, params, array_params = build_hard_filter_where(market_hard)
    price_sql = EFFECTIVE_PROPERTY_PRICE_HUF_SQL
    budget_select = ""
    if budget is not None:
        budget_select = f", COUNT(*) FILTER (WHERE ({price_sql}) <= :__budget) AS n_below"
        params = {**params, "__budget": budget}
    sql = (
        f"SELECT COUNT({price_sql}) AS n, MIN({price_sql}) AS min_price, "  # noqa: S608 -- the constant price expression and the where_sql (constant fragments, values bound)
        f"percentile_disc(0.25) WITHIN GROUP (ORDER BY {price_sql}) AS p25, "
        f"percentile_disc(0.5) WITHIN GROUP (ORDER BY {price_sql}) AS median, "
        f"percentile_disc(0.75) WITHIN GROUP (ORDER BY {price_sql}) AS p75, "
        f"MAX({price_sql}) AS max_price"
        f"{budget_select} "
        f"FROM properties WHERE {where_sql} AND ({price_sql}) IS NOT NULL"
    )
    stmt = bind_array_params(text(sql), array_params)
    try:
        result = await session.execute(stmt, params)
    except SQLAlchemyError as exc:
        raise DbError("probe price stats query failed") from exc
    row = result.first()
    if row is None or not row.n:
        return PriceStats(segment_count=0, budget=budget)
    percentile: int | None = None
    if budget is not None and row.n:
        percentile = round(int(row.n_below) / int(row.n) * 100)
    return PriceStats(
        segment_count=int(row.n),
        min=_as_int(row.min_price),
        p25=_as_int(row.p25),
        median=_as_int(row.median),
        p75=_as_int(row.p75),
        max=_as_int(row.max_price),
        budget=budget,
        budget_percentile=percentile,
    )


async def _sample_ids(
    session: AsyncSession, where_sql: str, params: dict[str, object], array_params: list[str], limit: int
) -> list[str]:
    stmt = bind_array_params(
        text(
            f"SELECT id FROM properties WHERE {where_sql} "  # noqa: S608 -- where_sql is the clause list, built from constant fragments with every value bound
            "ORDER BY quality_score DESC NULLS LAST, id LIMIT :__sample_limit"
        ),
        array_params,
    )
    try:
        result = await session.execute(stmt, {**params, "__sample_limit": limit})
    except SQLAlchemyError as exc:
        raise DbError("probe sample query failed") from exc
    return [str(row.id) for row in result]


async def _build_directions(
    session: AsyncSession,
    query: ParsedQuery,
    *,
    exact_count: int,
    median_price: int | None,
    locale: PreviewLocale,
) -> list[ProbeDirection]:
    """Exact row + up to `MAX_WIDEN_DIRECTIONS` widenings, each with a real delta."""
    exact_title = "A keresés pontosan így" if locale == "hu" else "Your search as stated"
    directions: list[ProbeDirection] = [
        ProbeDirection(id="exact", kind="exact", title=exact_title, result_count=exact_count, delta_count=0)
    ]

    widened: list[ProbeDirection] = []
    for kind, title, hard in widen_candidates(query.hard_filters, median_price, locale):
        count = await _count_hard(session, hard)
        delta = count - exact_count
        if delta <= 0:
            continue
        widened.append(
            ProbeDirection(
                id=kind,
                kind=kind,
                title=title,
                result_count=count,
                delta_count=delta,
                overrides=RefinementOverrides(hard_filters=hard),
            )
        )
    widened.sort(key=lambda d: -d.delta_count)
    directions.extend(widened[:MAX_WIDEN_DIRECTIONS])
    return directions


async def build_probe(
    session: AsyncSession,
    query: ParsedQuery,
    *,
    request_id: str,
    locale: PreviewLocale = "hu",
    now: datetime | None = None,
) -> SearchProbe:
    """Run the cheap index probe for `query` and assemble the payload.

    Samples `DEFAULT_SAMPLE_SIZE` matching listings and previews the best
    `DEFAULT_PREVIEW_COUNT` of them.
    """
    now = now or datetime.now(UTC)
    hard = query.hard_filters
    where_sql, params, array_params = build_hard_filter_where(hard)

    exact_count = await _count(session, where_sql, params, array_params)
    price_stats = await _price_stats(session, _market_filters(hard), _budget(query))

    previews: list[ProbePreviewListing] = []
    sample_ids: list[str] = []
    if exact_count > 0:
        sample_ids = await _sample_ids(session, where_sql, params, array_params, DEFAULT_SAMPLE_SIZE)
        rows = await fetch_candidate_rows(session, sample_ids)
        ranked = score_and_rank(rows, query, now=now)
        previews = [preview_from(row, breakdown, query) for row, breakdown in ranked[:DEFAULT_PREVIEW_COUNT]]

    directions = await _build_directions(
        session, query, exact_count=exact_count, median_price=price_stats.median, locale=locale
    )
    best_widen = next((d for d in directions if d.kind != "exact"), None)
    insight = _build_insight(price_stats, best_widen, exact_count, locale)

    return SearchProbe(
        request_id=request_id,
        intent_hash=_intent_hash(query),
        parsed_query=query,
        understood_summary=constraint_summary(query, locale),
        exact_count=exact_count,
        price_stats=price_stats,
        previews=previews,
        directions=directions,
        insight=insight,
        sample_size=len(sample_ids),
    )


__all__ = [
    "DEFAULT_PREVIEW_COUNT",
    "DEFAULT_SAMPLE_SIZE",
    "PriceStats",
    "ProbeDirection",
    "ProbePreviewListing",
    "SearchProbe",
    "build_probe",
    "preview_from",
    "score_and_rank",
    "widen_candidates",
]
