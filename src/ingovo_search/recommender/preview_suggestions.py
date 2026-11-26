"""The preview's image concepts and search directions.

The localized copy of every soft preference — a direction title, the change it makes and the
phrase appended to the query — the stock-photo tags of each visual concept, and the two
builders: eight image briefs, and the exact direction plus four preference directions. The
preference keys come from the parser's soft preferences, folded phrase signals in the query
and the POI list, topped up by a property-type fallback; all of it is pure and deterministic,
so the same parsed query always yields the same payload.
"""

from __future__ import annotations

import re

from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.preview_models import DirectionKind, ImageConcept, PreviewLocale, SearchDirection
from ingovo_search.recommender.preview_summary import constraint_summary_with_keys

_PREFERENCE_LABELS: dict[PreviewLocale, dict[str, tuple[str, str, str]]] = {
    "hu": {
        "quiet": ("Csendesebb környezet", "A csend kap nagyobb súlyt", "csendes környéken"),
        "natural_light": ("Világosabb otthonok", "Több természetes fény", "sok természetes fénnyel"),
        "sunny": ("Napfényes terek", "A benapozottság az elsődleges", "napfényes terekkel"),
        "garden": ("Saját kert vagy kertkapcsolat", "A privát zöldterület kerül előre", "saját kerttel"),
        "balcony": ("Erkélyes alternatívák", "Az erkély kerül előre", "erkéllyel"),
        "terrace": ("Teraszos alternatívák", "A terasz kerül előre", "terasszal"),
        "garage": ("Saját garázs", "A garázs kap nagyobb súlyt", "saját garázzsal"),
        "parking": ("Biztos parkolás", "A parkolás kerül előre", "saját parkolóval"),
        "elevator": ("Liftes épületek", "A lift kap nagyobb súlyt", "liftes épületben"),
        "renovated": ("Felújított otthonok", "A jó állapot kerül előre", "felújított állapotban"),
        "new_build": ("Új építésű irány", "Az új építésű kínálat kerül előre", "új építésűként"),
        "fixer_upper": ("Felújítandó lehetőségek", "Az alakítható ingatlanok kerülnek előre", "felújítandó állapotban"),
        "view": ("Panorámás alternatívák", "A kilátás kap nagyobb súlyt", "panorámával"),
        "central": ("Központi elhelyezkedés", "A belvárosi elérhetőség kerül előre", "központi helyen"),
        "near_metro": ("Metróhoz közel", "A gyors közlekedés kerül előre", "metró közelében"),
        "walkable": ("Jól bejárható környék", "A gyalogos elérhetőség kerül előre", "jól bejárható környéken"),
        "pool": ("Saját medence", "A medence kap nagyobb súlyt", "medencével"),
        "large": ("Tágasabb terek", "A térérzet kerül előre", "tágas terekkel"),
        "small": ("Kompakt, praktikus otthon", "A hatékony alaprajz kerül előre", "praktikus alaprajzzal"),
        "family": ("Családbarát környezet", "A családi élethelyzet kerül előre", "családbarát környéken"),
        "pet_friendly": ("Állatbarát otthonok", "Az állattartás kerül előre", "állatbarát feltételekkel"),
        "home_office": ("Külön dolgozósarok", "Az otthoni munkavégzés kerül előre", "külön dolgozósarokkal"),
        "near_park": ("Parkközeli otthonok", "A közeli zöldterület kerül előre", "park közelében"),
        "near_school": ("Iskolákhoz közel", "A családi infrastruktúra kerül előre", "iskola közelében"),
        "waterfront": ("Vízparti környezet", "A víz közelsége kerül előre", "vízpart közelében"),
    },
    "en": {
        "quiet": ("Quieter surroundings", "Quiet streets receive more weight", "in a quiet area"),
        "natural_light": ("Brighter homes", "Natural light receives more weight", "with abundant natural light"),
        "sunny": ("Sun-filled spaces", "Sun exposure becomes the priority", "with sunny interiors"),
        "garden": ("Private garden or garden access", "Private outdoor space comes first", "with a private garden"),
        "balcony": ("Homes with a balcony", "Balconies receive more weight", "with a balcony"),
        "terrace": ("Homes with a terrace", "Terraces receive more weight", "with a terrace"),
        "garage": ("Private garage", "A garage becomes the priority", "with a private garage"),
        "parking": ("Reliable parking", "Parking receives more weight", "with private parking"),
        "elevator": ("Buildings with a lift", "A lift becomes the priority", "in a building with a lift"),
        "renovated": ("Renovated homes", "Move-in-ready condition comes first", "in renovated condition"),
        "new_build": ("New-build direction", "New-build homes come first", "as a new-build home"),
        "fixer_upper": ("Renovation opportunities", "Adaptable properties come first", "in need of renovation"),
        "view": ("Homes with a view", "Panoramic outlook receives more weight", "with panoramic views"),
        "central": ("Central location", "City-centre access comes first", "in a central location"),
        "near_metro": ("Near the metro", "Fast transport comes first", "near the metro"),
        "walkable": ("Walkable neighbourhood", "Everyday walkability comes first", "in a walkable neighbourhood"),
        "pool": ("Private pool", "A pool receives more weight", "with a private pool"),
        "large": ("More generous spaces", "Space receives more weight", "with generous interiors"),
        "small": ("Compact, practical home", "Efficient layout comes first", "with an efficient layout"),
        "family": ("Family-friendly surroundings", "Family life comes first", "in a family-friendly area"),
        "pet_friendly": ("Pet-friendly homes", "Pet-friendly living comes first", "with pet-friendly terms"),
        "home_office": ("Dedicated workspace", "Working from home comes first", "with a dedicated workspace"),
        "near_park": ("Near a park", "Nearby green space comes first", "near a park"),
        "near_school": ("Near schools", "Family infrastructure comes first", "near a school"),
        "waterfront": ("Waterfront setting", "Proximity to water comes first", "near the waterfront"),
    },
}

_CONCEPT_TAGS: dict[str, tuple[str, ...]] = {
    "quiet": ("quiet", "residential", "street"),
    "natural_light": ("natural-light", "bright", "interior"),
    "sunny": ("natural-light", "sunny", "interior"),
    "garden": ("garden", "green", "outdoor"),
    "balcony": ("balcony", "urban", "outdoor"),
    "terrace": ("terrace", "outdoor", "green"),
    "garage": ("garage", "parking", "house"),
    "parking": ("parking", "exterior", "urban"),
    "elevator": ("apartment", "accessible", "building"),
    "renovated": ("renovated", "interior", "real-photo"),
    "new_build": ("new-build", "modern", "interior"),
    "fixer_upper": ("renovation", "fixer-upper", "interior"),
    "view": ("view", "panorama", "balcony"),
    "central": ("urban", "city", "apartment"),
    "near_metro": ("transit", "urban", "apartment"),
    "walkable": ("street", "urban", "neighbourhood"),
    "pool": ("pool", "garden", "house"),
    "large": ("spacious", "living-room", "interior"),
    "small": ("compact", "kitchen", "apartment"),
    "family": ("family", "living-room", "house"),
    "pet_friendly": ("pet-friendly", "garden", "park"),
    "home_office": ("home-office", "workspace", "natural-light"),
    "near_park": ("park", "green", "neighbourhood"),
    "near_school": ("family", "neighbourhood", "street"),
    "waterfront": ("waterfront", "holiday-home", "quiet"),
    "apartment_interior": ("apartment", "living-room", "real-photo"),
    "house_exterior": ("house", "exterior", "garden"),
    "office_interior": ("office", "workspace", "natural-light"),
    "holiday_home": ("holiday-home", "waterfront", "terrace"),
    "kitchen": ("kitchen", "interior", "natural-light"),
}


def _fold(value: str) -> str:
    replacements = str.maketrans("áéíóöőúüű", "aeiooouuu")
    return re.sub(r"\s+", " ", value.casefold().translate(replacements)).strip()


def _preference_keys(parsed: ParsedQuery, query: str) -> list[str]:
    keys: list[str] = list(parsed.soft_preferences)
    folded = _fold(" ".join([query, parsed.semantic_query, *parsed.soft_prefs_freetext]))
    phrase_signals: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("family", ("csalad", "gyerek", "family", "children")),
        ("pet_friendly", ("kutya", "macska", "allatbarat", "pet", "dog", "cat")),
        ("home_office", ("home office", "dolgozosarok", "otthoni munka", "workspace")),
        ("waterfront", ("vizpart", "balaton", "duna-part", "waterfront", "lakeside")),
        ("near_park", ("park", "jatszoter", "green space", "playground")),
        ("near_school", ("iskola", "ovoda", "school", "kindergarten")),
        ("quiet", ("csendes", "nyugodt", "quiet", "peaceful")),
        ("natural_light", ("vilagos", "termeszetes feny", "natural light", "bright")),
        ("home_office", ("dolgozo", "irodasarok", "study", "work from home")),
    )
    for key, phrases in phrase_signals:
        if any(phrase in folded for phrase in phrases):
            keys.append(key)
    poi_map = {
        "park": "near_park",
        "playground": "near_park",
        "school": "near_school",
        "kindergarten": "near_school",
        "metro": "near_metro",
        "tram": "near_metro",
        "transit": "near_metro",
    }
    keys.extend(poi_map[item.category] for item in parsed.poi_near if item.category in poi_map)
    return list(dict.fromkeys(key for key in keys if key in _PREFERENCE_LABELS["hu"]))


def _fallback_preferences(parsed: ParsedQuery) -> list[str]:
    property_types = set(parsed.hard_filters.property_type_in)
    if "Ház" in property_types:
        return ["garden", "quiet", "garage", "natural_light"]
    if "Iroda" in property_types:
        return ["home_office", "near_metro", "natural_light", "parking"]
    if "Nyaraló" in property_types:
        return ["waterfront", "terrace", "view", "quiet"]
    return ["natural_light", "balcony", "quiet", "near_metro"]


def _append_preference(query: str, phrase: str, _locale: PreviewLocale) -> str:
    clean = query.strip().rstrip(".,; ")
    return f"{clean}, {phrase}" if clean else phrase


def build_directions(parsed: ParsedQuery, query: str, locale: PreviewLocale) -> list[SearchDirection]:
    summary, constraints = constraint_summary_with_keys(parsed, locale)
    directions = [
        SearchDirection(
            id="exact",
            kind="exact",
            title="A keresés pontosan így" if locale == "hu" else "Your search as stated",
            summary=summary,
            delta_label="Minden felismert feltétel megmarad" if locale == "hu" else "Keeps every recognised condition",
            canonical_query=query.strip(),
            retained_constraints=constraints,
        )
    ]
    preferences = list(dict.fromkeys([*_preference_keys(parsed, query), *_fallback_preferences(parsed)]))
    kinds: tuple[DirectionKind, ...] = ("priority", "lifestyle", "alternative", "guided")
    for index, key in enumerate(preferences[:4]):
        title, delta, phrase = _PREFERENCE_LABELS[locale][key]
        directions.append(
            SearchDirection(
                id=f"{index + 1}-{key}",
                kind=kinds[index],
                title=title,
                summary=summary,
                delta_label=delta,
                canonical_query=_append_preference(query, phrase, locale),
                retained_constraints=constraints,
                changed_preferences=[f"{key}:+"],
            )
        )
    return directions[:5]


def _concept_copy(key: str, locale: PreviewLocale) -> tuple[str, str, str]:
    if key in _PREFERENCE_LABELS[locale]:
        title, delta, phrase = _PREFERENCE_LABELS[locale][key]
        return title, delta, phrase
    generic: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
        "apartment_interior": (
            ("Valós lakásbelső", "Az ingatlantípushoz illeszkedik", "valós lakásbelsővel"),
            ("Real apartment interior", "Matches the property type", "with a real apartment interior"),
        ),
        "house_exterior": (
            ("Valós családi ház", "Az ingatlantípushoz illeszkedik", "valós családi házként"),
            ("Real family house", "Matches the property type", "as a real family house"),
        ),
        "office_interior": (
            ("Világos munkatér", "Az irodai használathoz illeszkedik", "világos munkatérrel"),
            ("Bright workspace", "Matches office use", "with a bright workspace"),
        ),
        "holiday_home": (
            ("Nyugodt nyaraló", "A pihenési célhoz illeszkedik", "nyugodt nyaralóként"),
            ("Calm holiday home", "Matches a holiday-home search", "as a calm holiday home"),
        ),
        "kitchen": (
            ("Praktikus konyha", "A mindennapi használhatóságot mutatja", "praktikus konyhával"),
            ("Practical kitchen", "Shows everyday usability", "with a practical kitchen"),
        ),
    }
    hu, en = generic[key]
    return hu if locale == "hu" else en


def build_image_concepts(parsed: ParsedQuery, query: str, locale: PreviewLocale) -> list[ImageConcept]:
    property_types = set(parsed.hard_filters.property_type_in)
    base = (
        "house_exterior"
        if "Ház" in property_types
        else "office_interior"
        if "Iroda" in property_types
        else "holiday_home"
        if "Nyaraló" in property_types
        else "apartment_interior"
    )
    ordered = list(
        dict.fromkeys(
            [
                *_preference_keys(parsed, query),
                base,
                *_fallback_preferences(parsed),
                "kitchen",
                "renovated",
                "home_office",
                "near_park",
                "terrace",
                "view",
            ]
        )
    )
    concepts: list[ImageConcept] = []
    for key in ordered:
        tags = _CONCEPT_TAGS.get(key)
        if not tags:
            continue
        title, reason, phrase = _concept_copy(key, locale)
        concepts.append(
            ImageConcept(
                id=key,
                title=title,
                reason=reason,
                tags=list(tags),
                canonical_query=_append_preference(query, phrase, locale),
            )
        )
        if len(concepts) == 8:
            break
    return concepts
