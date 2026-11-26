"""Exact HUF price helpers for the property read model.

``properties.price_huf`` is the only price column. A ``price`` *key* still exists
on the wire in two directions: outbound as the JavaScript-safe float alias that
``public_price_fields`` derives from ``price_huf``, and inbound on JSON payloads
from the original product's web backend, where it is read only when it proves to
be a positive JavaScript-safe integer. Keeping these rules in one module prevents
different public endpoints from rounding the same value in different ways.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, cast

MAX_LEGACY_SAFE_INTEGER = 9_007_199_254_740_991
MAX_POSTGRES_BIGINT = 9_223_372_036_854_775_807
_MISSING = object()

# The read-side SQL spelling of "a usable HUF amount": the canonical column, in
# HUF, positive. It fails closed (NULL) for anything else, which is what every
# aggregate and range filter embedding it relies on (the hard filter and the probe).
EFFECTIVE_PROPERTY_PRICE_HUF_SQL = """CASE
    WHEN currency = 'HUF' AND price_huf IS NOT NULL AND price_huf > 0 THEN price_huf
    ELSE NULL
END"""


def _positive_bigint(value: object) -> int | None:
    """Return a positive PostgreSQL bigint without ever rounding a value.

    A decimal string is accepted for the canonical field because the Node
    public JSON serializer deliberately represents unsafe JavaScript BigInts as
    strings.  Floats, signs, whitespace and scientific notation remain
    invalid: accepting any of them would make an exact money contract depend
    on a coercion.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value) is None:
            return None
        value = int(value)
    if not isinstance(value, int):
        return None
    if 0 < value <= MAX_POSTGRES_BIGINT:
        return value
    return None


def legacy_price_to_huf(value: object) -> int | None:
    """Accept only an exact, positive legacy ``price`` value as an integer HUF amount.

    JSON payloads only: the browser and Node still post a numeric ``price`` key,
    and the public alias is a float, so a fractional, non-finite or unsafe value
    must never round into money.  ORM rows have no ``price`` attribute any more.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 < value <= MAX_LEGACY_SAFE_INTEGER else None
    if not isinstance(value, float):
        return None
    if not math.isfinite(value) or value <= 0 or value > MAX_LEGACY_SAFE_INTEGER or not value.is_integer():
        return None
    return int(value)


def _field(property_row: object, name: str, default: object = None) -> object:
    """Read a DB model, a RowMapping, or a JSON-like public payload alike."""

    if isinstance(property_row, Mapping):
        # Narrowing an `object` to `Mapping` leaves the key and value types
        # implicitly unknown, which then travels into every caller. The row is
        # genuinely dynamic (ORM model, RowMapping or JSON payload), so state
        # the element types here, at the boundary, instead of at each use.
        return cast("Mapping[Any, Any]", property_row).get(name, default)
    return getattr(property_row, name, default)


def _has_supported_currency(property_row: object) -> bool:
    """Do not relabel an unresolved historic non-HUF row as canonical HUF."""

    currency = _field(property_row, "currency")
    return currency is None or (isinstance(currency, str) and currency.strip().upper() == "HUF")


def resolve_property_price_huf(property_row: object) -> int | None:
    """Return the canonical amount first, with a rigorously safe payload fallback."""

    if not _has_supported_currency(property_row):
        return None

    # DB rows expose ``price_huf`` while the web backend's JSON exposes
    # ``priceHuf``.  Treat either populated spelling as authoritative.  In
    # particular, an invalid populated canonical value must not silently revive
    # a divergent ``price`` key; migration diagnostics own that remediation.
    for canonical_name in ("price_huf", "priceHuf"):
        canonical = _field(property_row, canonical_name, _MISSING)
        if canonical is not _MISSING and canonical is not None:
            return _positive_bigint(canonical)
    # JSON payloads only: an ORM row or a properties SELECT has no ``price``
    # field, so for those this is simply ``None``.
    return legacy_price_to_huf(_field(property_row, "price"))


def public_price_fields(property_row: object) -> tuple[str | None, float | None]:
    """Return ``(price_huf, legacy_price)`` for a JSON-safe public response.

    ``price_huf`` is always a decimal string so JavaScript clients cannot round
    a valid PostgreSQL bigint. The legacy ``price`` field remains a safe numeric
    compatibility alias only below the same exact-integer boundary.
    """

    price_huf = resolve_property_price_huf(property_row)
    if price_huf is None:
        return None, None
    legacy_price = float(price_huf) if price_huf <= MAX_LEGACY_SAFE_INTEGER else None
    return str(price_huf), legacy_price


def json_safe_huf(value: int | None) -> int | str | None:
    """Serialize a known HUF integer without forcing JavaScript to round it."""

    if value is None:
        return None
    return value if 0 < value <= MAX_LEGACY_SAFE_INTEGER else str(value)


def canonical_ingestion_price_huf(value: int) -> int | None:
    """Map the normalized feed's non-negative amount to the nullable HUF column."""

    # Feed `0` has historically represented "price on request". It is not a
    # valid positive money amount, so retain it as NULL in the canonical column.
    return _positive_bigint(value)


__all__ = [
    "EFFECTIVE_PROPERTY_PRICE_HUF_SQL",
    "MAX_LEGACY_SAFE_INTEGER",
    "MAX_POSTGRES_BIGINT",
    "canonical_ingestion_price_huf",
    "json_safe_huf",
    "legacy_price_to_huf",
    "public_price_fields",
    "resolve_property_price_huf",
]
