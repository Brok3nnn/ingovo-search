"""Pure-logic builders for the two texts each listing embeds with Cohere.

`build_factual_text` templates the structured fields (rooms, district, area,
renovation state, amenities, …) into a deterministic HU phrase used as the
"factual" embedding input. `build_descriptive_text` concatenates the
enrichment outputs (`cleaned_description` + `one_paragraph_summary`) for the
"descriptive" embedding.

Both are deterministic given the same `Property` and write no DB rows — keeping
embedding inputs pure lets us snapshot-test them and lets the cache key be a
stable hash of the function's output.
"""

from __future__ import annotations

from ingovo_search.db.models import Property

_RENOVATION_HU: dict[str, str] = {
    "original": "eredeti állapotú",
    "partial": "részben felújított",
    "fully_renovated": "felújított",
    "new_build": "újépítésű",
}

_ORIENTATION_HU: dict[str, str] = {
    "N": "északi tájolású",
    "NE": "északkeleti tájolású",
    "E": "keleti tájolású",
    "SE": "délkeleti tájolású",
    "S": "déli tájolású",
    "SW": "délnyugati tájolású",
    "W": "nyugati tájolású",
    "NW": "északnyugati tájolású",
}


def build_factual_text(prop: Property) -> str:
    """Render the structured-field summary used as the factual embedding input."""
    parts: list[str] = []

    if prop.rooms and prop.property_type:
        parts.append(f"{prop.rooms}-szobás {prop.property_type}")
    elif prop.property_type:
        parts.append(prop.property_type)

    location_bits: list[str] = []
    if prop.district:
        location_bits.append(f"a {prop.district}. kerületben")
    if prop.city:
        location_bits.append(prop.city)
    if location_bits:
        parts.append(", ".join(location_bits))

    if prop.area_m2:
        parts.append(f"{_format_number(prop.area_m2)} m²")

    if prop.floor is not None and prop.total_floors is not None:
        parts.append(f"{prop.floor}/{prop.total_floors}. emelet")
    elif prop.floor is not None:
        parts.append(f"{prop.floor}. emelet")

    if prop.year_built:
        parts.append(f"{prop.year_built}-ben épült")

    renovation_phrase = _RENOVATION_HU.get(prop.renovation_state) if prop.renovation_state else None
    if renovation_phrase:
        parts.append(renovation_phrase)

    if prop.heating:
        parts.append(f"{prop.heating} fűtés")

    orientation_phrase = _ORIENTATION_HU.get(prop.orientation) if prop.orientation else None
    if orientation_phrase:
        parts.append(orientation_phrase)

    amenities: list[str] = []
    if prop.has_balcony:
        amenities.append("erkély")
    if prop.has_terrace:
        amenities.append("terasz")
    if prop.has_garden:
        amenities.append("kert")
    if prop.elevator:
        amenities.append("lift")
    if prop.has_parking:
        amenities.append("parkoló")
    if prop.has_garage:
        amenities.append("garázs")
    if prop.has_basement:
        amenities.append("pince")
    if prop.air_conditioning:
        amenities.append("klíma")
    if amenities:
        parts.append(", ".join(amenities))

    return ", ".join(parts)


def build_descriptive_text(prop: Property) -> str:
    """Concatenate enrichment outputs into the descriptive embedding input."""
    blocks: list[str] = []
    if prop.cleaned_description:
        blocks.append(prop.cleaned_description.strip())
    if prop.one_paragraph_summary:
        blocks.append(prop.one_paragraph_summary.strip())
    if not blocks and prop.description:
        blocks.append(prop.description.strip())
    return "\n\n".join(blocks)


def _format_number(value: float) -> str:
    """Render whole-number floats without a trailing `.0`."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"
