"""The deterministic parser's lexicon: the word tables, regexes and token recognisers.

Split from `parser_fastpath.py`; the scan passes there read these through one import. The
tables are tight on purpose — a false friend here ("haza" as "ház") would silently change
what the fast path answers without the LLM, and the parser golden (141 questions, strict and
loose) pins every answer.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Collection
from typing import Final

from ingovo_search.recommender.geo_taxonomy import CITY_CANONICAL_BY_FOLD, COUNTY_CITIES
from ingovo_search.recommender.parser_schema import BalatonShore, PropertyType, TransactionType

# Roman Budapest district numerals. The LLM parser's district defaults read the same
# table, which is why it lives here, below both parsers, and not in `parser.py`.
ROMAN_DISTRICTS = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
    "xiii": 13,
    "xiv": 14,
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
}

# Queries longer than this are almost always natural language → defer to the LLM.
MAX_FASTPATH_CHARS = 80

# Tokeniser: numbers (with one decimal separator), word runs, or a lone '+'.
# Punctuation is dropped, so "XIII." -> "xiii" and "50," -> "50".
_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+|\+", re.UNICODE)

# Connectors that may remain after extraction without forcing an LLM fallback.
_FILLER = frozenset(
    {
        "és",
        "es",
        "vagy",
        "meg",
        "a",
        "az",
        "egy",
        "+",
        "keresek",
        "keresünk",
        "keresunk",
        "szeretnék",
        "szeretnek",
        "ingatlan",
        "ingatlant",
    }
)

# Price/area qualifier vocabulary (post-amount words).
_MAX_WORDS = frozenset({"alatt", "alatti", "alá", "ala"})
_MIN_WORDS = frozenset({"felett", "fölött", "felül", "fölül", "felette"})
_ANCHOR_WORDS = frozenset({"körül", "korul", "környékén", "kornyeken", "kb", "nagyjából", "nagyjabol"})
_RANGE_WORDS = frozenset({"között", "kozott"})
_RENT_PRICE_WORDS = frozenset({"havi", "havonta"})
_PRICE_PAYMENT_WORDS = frozenset({"forintért", "forintert", "ftért", "ftert"})
_CURRENCY_WORDS = frozenset({"ft", "huf", "forint", "forintot"})
# Qualifier words that appear *before* the amount.
_MAX_PRE_WORDS = frozenset({"max", "maximum", "legfeljebb"})
_MIN_PRE_WORDS = frozenset({"min", "minimum", "legalább", "legalabb"})

# Negation words that turn a district mention into an exclusion when they
# appear within the three tokens before it ("de nem a 8. kerületben").
_NEGATION_TOKENS = frozenset({"nem", "ne", "sem"})

# Hungarian number words 1-6 (the realistic room range), used to read compound
# room tokens like "kétszobás" / "háromszobás" that the LLM would normally
# handle but the deterministic fallback must also cover when the LLM is slow.
_ROOM_NUMBER_WORDS: dict[str, int] = {
    "egy": 1,
    "két": 2,
    "ket": 2,
    "kettő": 2,
    "ketto": 2,
    "három": 3,
    "harom": 3,
    "négy": 4,
    "negy": 4,
    "öt": 5,
    "ot": 5,
    "hat": 6,
}

_MILLION = 1_000_000
_THOUSAND = 1_000

# Generic Balaton-region cues → geo_hints.region_name="Balaton" (geo_score.py
# scores Balaton-area cities for any region_name containing "balaton"). Specific
# lakeside towns (Balatonfüred, Siófok, Tihany) are caught earlier as exact
# cities, so only the bare/region forms reach here.
_BALATON_REGION_WORDS = frozenset(
    {"balaton", "balatoni", "balatonnál", "balatonnal", "balatonon", "balatonra", "balatonhoz"}
)
_BALATON_AREA_QUALIFIERS = frozenset(
    {
        "környéki",
        "kornyeki",
        "környékén",
        "kornyeken",
        "környék",
        "kornyek",
        "melletti",
        "menti",
        "parti",
        "körüli",
        "koruli",
        "part",
        "parton",
        "partján",
        "partjan",
    }
)
_BALATON_REGION_NAME = "Balaton"
_BALATON_SHORE_WORDS: dict[str, BalatonShore] = {
    "északi": "Északi",
    "eszaki": "Északi",
    "déli": "Déli",
    "deli": "Déli",
}

# Small verified landmark registry for deterministic proximity parsing. These
# coordinates are fixed product data, not guessed from user text.
_KNOWN_POIS: dict[tuple[str, ...], tuple[float, float, str]] = {
    ("deak", "ferenc", "ter"): (47.4983, 19.0541, "Budapest"),
}


def normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def district_number(token: str) -> str | None:
    """Resolve a district token (arabic 1-23 or roman i-xxiii) to its number."""
    clean = token.strip().strip(".")
    if clean.isdigit():
        number = int(clean)
    else:
        number = ROMAN_DISTRICTS.get(clean)
        if number is None:
            return None
    return str(number) if 1 <= number <= 23 else None


def is_kerulet(token: str) -> bool:
    return token.startswith("ker")  # kerület / kerületben / ker. / ker


def transaction_of(token: str) -> TransactionType | None:
    if token.startswith(("kiad", "albérl", "alberl", "bérl", "berl", "bérel", "berel")):
        return "rent"
    if token.startswith(
        ("elad", "vétel", "vetel", "vásárol", "vasarol", "vásárl", "vasarl", "venni", "venné", "venne", "megven")
    ):
        return "sale"
    return None


# Property-type recognisers, first match wins. Each predicate is deliberately
# tight to avoid false friends (e.g. "haza" must not read as "ház", "milliárd"
# handled in price). `search` (not `match`) for Lakás so trailing compounds
# resolve too: "téglalakás", "panellakás", "garzonlakás", "újépítésű lakás".
_PROPERTY_TYPE_RULES: Final[tuple[tuple[Callable[[str], bool], PropertyType], ...]] = (
    (lambda token: re.search(r"lak[aá]s", token) is not None, "Lakás"),  # lakás, lakást, lakasok, téglalakás
    (
        lambda token: token.startswith(
            ("sorház", "sorhaz", "ikerház", "ikerhaz", "társasház", "tarsashaz", "panelház", "panelhaz")
        ),
        "Ház",
    ),
    # ház, házat, házban
    (lambda token: re.match(r"h[aá]z", token) is not None and not token.startswith(("haza", "hazá", "házas")), "Ház"),
    (lambda token: token.startswith("iroda"), "Iroda"),
    (lambda token: token in {"telek", "telket", "telken", "telkek", "telke", "telkén", "telkbe"}, "Telek"),
    (lambda token: re.match(r"gar[aá]zs", token) is not None, "Garázs"),  # garázs, garázst
    (lambda token: token.startswith("nyaral"), "Nyaraló"),  # nyaraló, nyaralót
    (lambda token: token in {"üzlet", "uzlet", "üzlethelyiség", "uzlethelyiseg", "üzlethelyiseg"}, "Üzlethelyiség"),
)


def property_type_of(token: str) -> PropertyType | None:
    return next((kind for matches, kind in _PROPERTY_TYPE_RULES if matches(token)), None)


# "budai/pesti lakás", "Budán", "Pesten" — colloquial halves of the capital.
_BUDAPEST_FOLD_ALIASES: frozenset[str] = frozenset(
    {
        "buda",
        "budán",
        "budan",
        "budai",
        "budára",
        "budara",
        "budáról",
        "budarol",
        "pest",
        "pesten",
        "pesti",
        "pestre",
        "pestről",
        "pestrol",
    }
)
_BUDAPEST_DISTRICT_ALIASES: dict[str, str] = {
    "újlipótváros": "13",
    "ujlipotvaros": "13",
}

# Hungarian locative/adjectival case endings, longest first so we strip the full
# ending (e.g. "-ben") rather than a shorter prefix of it ("-n"). We only accept
# the strip when the stem is an *exact* known city, so an ordinary word can never
# be misread as a city (the fast path's no-misread guarantee holds).
_CITY_INFLECTION_SUFFIXES: tuple[str, ...] = (
    "ból",
    "ből",
    "ról",
    "ről",
    "nál",
    "nél",
    "hoz",
    "hez",
    "höz",
    "ban",
    "ben",
    "ba",
    "be",
    "ra",
    "re",
    "án",
    "én",
    "on",
    "en",
    "ön",
    "n",
    "i",
)


def _strip_accents(text: str) -> str:
    """Drop Hungarian diacritics: 'győr' -> 'gyor', 'pécs' -> 'pecs'."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


_COUNTY_TOKEN_SEQUENCES: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    sorted(
        (
            (tuple(_strip_accents(token) for token in _TOKEN_RE.findall(county.casefold())), county)
            for county in COUNTY_CITIES
            if county != "Budapest"
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _build_deaccented_city_index() -> dict[str, str]:
    """Accent-insensitive city lookup derived from the canonical fold table.

    Users routinely type without diacritics ('siofok', 'gyor', 'pecs'). We index
    each city by its accent-stripped fold form, but DROP any de-accented key that
    two different cities share — an ambiguous key must fall back to the LLM rather
    than risk a misread.
    """
    seen: dict[str, str] = {}
    ambiguous: set[str] = set()
    for fold, canonical in CITY_CANONICAL_BY_FOLD.items():
        key = _strip_accents(fold)
        if key in seen and seen[key] != canonical:
            ambiguous.add(key)
        seen[key] = canonical
    return {key: value for key, value in seen.items() if key not in ambiguous}


_CITY_BY_DEACCENTED: dict[str, str] = _build_deaccented_city_index()


def _lookup_city(token: str) -> str | None:
    """Exact city by fold form, then by accent-insensitive fold form."""
    canonical = CITY_CANONICAL_BY_FOLD.get(token)
    if canonical is not None:
        return canonical
    return _CITY_BY_DEACCENTED.get(_strip_accents(token))


def city_of(token: str) -> str | None:
    """Resolve a (possibly inflected) settlement token to its canonical city.

    Handles the bare fold form ("debrecen"), the adjectival "-i" form
    ("szegedi", "tihanyi"), Hungarian locative endings ("debrecenben",
    "szegeden", "miskolcon", "budapesten"), the colloquial Budapest halves
    ("budai", "pesti"), and accent-free spellings ("siofok", "gyor"). Returns
    `None` unless the (possibly stripped, possibly de-accented) stem is an exact
    known city, so the fast path stays incapable of misreading a query.
    """
    if token in _BUDAPEST_FOLD_ALIASES:
        return "Budapest"
    direct = _lookup_city(token)
    if direct is not None:
        return direct
    for suffix in _CITY_INFLECTION_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            stem = token[: -len(suffix)]
            hit = _lookup_city(stem)
            if hit is not None:
                return hit
            if stem in _BUDAPEST_FOLD_ALIASES:
                return "Budapest"
    return None


def budapest_district_alias(token: str) -> str | None:
    aliases = {_strip_accents(key): value for key, value in _BUDAPEST_DISTRICT_ALIASES.items()}
    direct = aliases.get(_strip_accents(token))
    if direct is not None:
        return direct
    for suffix in _CITY_INFLECTION_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            hit = aliases.get(_strip_accents(token[: -len(suffix)]))
            if hit is not None:
                return hit
    return None


def million_suffix(token: str) -> str | None:
    """If `token` is a million unit, return its trailing suffix ('' if none)."""
    m = re.match(r"milli[oó]", token)  # millió/milliós/millióig — excludes 'milliárd'
    if m:
        return token[m.end() :]
    if token.startswith("mft"):
        return token[3:]
    return None


def thousand_suffix(token: str) -> str | None:
    if token.startswith("ezer"):
        return token[4:]
    if token.startswith("eft"):
        return token[3:]
    return None


# The neighbour-word tables `classify_bound` walks, in priority order.
_NEXT_TOKEN_BOUNDS: Final[tuple[tuple[Collection[str], str], ...]] = (
    (_MAX_WORDS, "max"),
    (_MIN_WORDS, "min"),
    (_ANCHOR_WORDS, "anchor"),
)
_PREV_TOKEN_BOUNDS: Final[tuple[tuple[Collection[str], str], ...]] = (
    (_MAX_PRE_WORDS, "max"),
    (_RENT_PRICE_WORDS, "max"),
    (_MIN_PRE_WORDS, "min"),
)


def classify_bound(suffix: str, next_token: str | None, prev_token: str | None) -> tuple[str | None, bool, bool]:
    """Decide max/min/anchor for an amount. Returns (bound, consume_next, consume_prev)."""
    if suffix.startswith("ig"):
        return "max", False, False
    if suffix.startswith(("tól", "tol", "től", "tel")):
        return "min", False, False
    # An unknown glued suffix (e.g. "milliós") falls through to the neighbour words.
    for words, bound in _NEXT_TOKEN_BOUNDS:
        if next_token in words:
            return bound, True, False
    if next_token in _PRICE_PAYMENT_WORDS:
        return "max", True, prev_token in _RENT_PRICE_WORDS
    for words, bound in _PREV_TOKEN_BOUNDS:
        if prev_token in words:
            return bound, False, True
    return None, False, False


_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")


def is_number(token: str) -> bool:
    return _NUMBER_RE.fullmatch(token) is not None


def number_value(token: str) -> float:
    return float(token.replace(",", "."))
