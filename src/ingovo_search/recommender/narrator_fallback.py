"""The heuristic narrative for when the narrator model fails.

`fallback_narrative` is deterministic and LLM-free: a plain factual headline and
body, the four rationale chips scored from hard-filter alignment, and a compromise
line only when a chip misses. `format_price_short` is the compact Hungarian price it
quotes; the listing brief in `narrator_prompt.py` renders the price the same way, so
the model and the fallback state an amount identically.
"""

from __future__ import annotations

from typing import Any

from ingovo_search.lib.property_price import MAX_LEGACY_SAFE_INTEGER, resolve_property_price_huf
from ingovo_search.recommender.narrator_schema import AINarrative, ChipStatus, RationaleChip
from ingovo_search.recommender.parser_schema import ParsedQuery


def format_price_short(price: int | float | None, *, is_rent: bool) -> str:
    """Hungarian-formatted price string for the fallback narrative."""
    if price is None:
        return "ár nélkül"
    suffix = " / hó" if is_rent else ""
    # A public model prompt/fallback must not round a valid canonical bigint.
    # Keep the established compact rendering for exact JS-safe values and emit
    # the full integer for the rare larger amount.
    if isinstance(price, int) and not isinstance(price, bool) and price > MAX_LEGACY_SAFE_INTEGER:
        return f"{price:,}".replace(",", " ") + f" Ft{suffix}"
    if price >= 1_000_000:
        m = price / 1_000_000
        rounded = f"{m:g}" if m % 1 == 0 else f"{m:.1f}".replace(".", ",")
        return f"{rounded} M Ft{suffix}"
    return f"{int(price):,} Ft".replace(",", " ") + suffix


def _chip_status_location(listing: Any, query: ParsedQuery) -> ChipStatus:
    """Heuristic: match if the listing's city / district aligns with the hard filters."""
    hf = query.hard_filters
    if (
        hf.city is not None
        and listing.city is not None
        and str(listing.city).strip().lower() != hf.city.strip().lower()
    ):
        return "miss"
    if hf.district_in:
        ld = str(listing.district) if listing.district else ""
        if ld not in hf.district_in:
            return "warning"
    return "match"


def _chip_status_size(listing: Any, query: ParsedQuery) -> ChipStatus:
    hf = query.hard_filters
    area = listing.area_m2 if listing.area_m2 is not None else None
    if hf.area_m2_min is not None and area is not None and area < hf.area_m2_min:
        return "miss"
    if hf.area_m2_max is not None and area is not None and area > hf.area_m2_max:
        return "miss"
    if hf.rooms is not None and listing.rooms is not None:
        try:
            rooms_int = int(str(listing.rooms).strip().split()[0])
        except (ValueError, IndexError):
            return "warning"
        if rooms_int != hf.rooms:
            return "warning"
    return "match"


def _chip_status_price(listing: Any, query: ParsedQuery) -> ChipStatus:
    hf = query.hard_filters
    price = resolve_property_price_huf(listing)
    if price is None or hf.price_huf_max is None:
        return "match"
    if price <= hf.price_huf_max:
        return "match"
    if price <= hf.price_huf_max * 1.10:  # within +10% — close-enough warning
        return "warning"
    return "miss"


def _chip_status_renovation(listing: Any) -> ChipStatus:
    state = (listing.renovation_state or "").strip().lower()
    if state in {"új építésű", "uj epitesu", "felújított", "felujitott"}:
        return "match"
    if state in {"felújítandó", "felujitando"}:
        return "miss"
    return "warning"


def _build_rationale_chips(listing: Any, query: ParsedQuery) -> list[RationaleChip]:
    """Four chips with statuses derived from hard-filter alignment, in fixed order."""
    return [
        RationaleChip(group="location", status=_chip_status_location(listing, query), label="Lokáció"),
        RationaleChip(group="size", status=_chip_status_size(listing, query), label="Méret"),
        RationaleChip(group="price", status=_chip_status_price(listing, query), label="Ár"),
        RationaleChip(group="renovation", status=_chip_status_renovation(listing), label="Felújítás"),
    ]


def fallback_narrative(listing: Any, query: ParsedQuery) -> AINarrative:
    """Deterministic, LLM-free narrative used when the Gemini judge fails.

    The text intentionally stays plain and factual — no fake personality —
    so the UI degrades gracefully and the difference between "Gemini happy"
    and "Gemini outage" is not jarring. Rationale chips are produced from
    actual hard-filter alignment so the user still sees a faithful summary.
    """
    is_rent = listing.transaction_type == "rent"
    price = format_price_short(resolve_property_price_huf(listing), is_rent=is_rent)
    area = f"{listing.area_m2:.0f} m²" if listing.area_m2 else None
    rooms = str(listing.rooms).strip() if listing.rooms else None
    location = str(listing.city or "Helyszín ismeretlen")
    if listing.district:
        location += f", {listing.district}. kerület"
    type_label = str(listing.property_type or "Ingatlan")

    headline = f"{type_label} — {location}"
    body_bits: list[str] = [f"{listing.title or type_label}, {location}, {price}."]
    if area or rooms:
        details = " · ".join(filter(None, [area, f"{rooms} szoba" if rooms else None]))
        body_bits.append(details + ".")
    body = " ".join(body_bits)

    # Compromise: only if a hard filter is clearly violated.
    compromise = ""
    chips = _build_rationale_chips(listing, query)
    miss_chips = [chip.label for chip in chips if chip.status == "miss"]
    if miss_chips:
        compromise = "Kompromisszum a következő szempont(ok)ban: " + ", ".join(miss_chips).lower() + "."

    return AINarrative(
        headline=headline,
        body=body,
        compromise=compromise,
        rationale_chips=chips,
    )
