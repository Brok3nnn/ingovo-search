"""Data-driven Hungarian county taxonomy and curated Balaton shores.

The parser extracts fuzzy geographic cues (`geo_hints.county` and
`geo_hints.balaton_shore`) that resolve to **sets of settlement names** rather
than coordinates. This module is the single source of truth for those sets, so
`geo_score` can score a listing's city against the requested county / shore
without an extra DB round-trip.

County membership is derived from the versioned KSH settlement asset, so small
villages receive the same administrative treatment as major cities. Balaton
shore membership remains a separate curated product taxonomy.

All lookups are case-insensitive: keys and inputs are lower-cased on access.
"""

from __future__ import annotations

from typing import Literal

from ingovo_search.recommender.settlements import SettlementKind, all_settlements, county_city_index

type CountyName = Literal[
    "Bács-Kiskun",
    "Baranya",
    "Békés",
    "Borsod-Abaúj-Zemplén",
    "Budapest",
    "Csongrád-Csanád",
    "Fejér",
    "Győr-Moson-Sopron",
    "Hajdú-Bihar",
    "Heves",
    "Jász-Nagykun-Szolnok",
    "Komárom-Esztergom",
    "Nógrád",
    "Pest",
    "Somogy",
    "Szabolcs-Szatmár-Bereg",
    "Tolna",
    "Vas",
    "Veszprém",
    "Zala",
]

type BalatonShore = Literal["Északi", "Déli"]


# County -> canonical settlement names. Lower-cased on lookup so the source
# preserves the readable Hungarian form for diagnostics / future export.
_COUNTY_CITIES_RAW: dict[str, tuple[str, ...]] = {
    "Bács-Kiskun": (
        "Kecskemét",
        "Baja",
        "Kiskunfélegyháza",
        "Kiskunhalas",
        "Kalocsa",
        "Kiskőrös",
        "Bácsalmás",
        "Lajosmizse",
        "Tiszakécske",
        "Solt",
    ),
    "Baranya": (
        "Pécs",
        "Mohács",
        "Komló",
        "Siklós",
        "Szigetvár",
        "Pécsvárad",
        "Sásd",
        "Sellye",
        "Bóly",
        "Villány",
    ),
    "Békés": (
        "Békéscsaba",
        "Gyula",
        "Békés",
        "Orosháza",
        "Szarvas",
        "Mezőkovácsháza",
        "Szeghalom",
        "Mezőberény",
        "Békésszentandrás",
    ),
    "Borsod-Abaúj-Zemplén": (
        "Miskolc",
        "Kazincbarcika",
        "Ózd",
        "Tiszaújváros",
        "Sátoraljaújhely",
        "Mezőkövesd",
        "Encs",
        "Edelény",
        "Szerencs",
        "Tokaj",
    ),
    "Budapest": ("Budapest",),
    "Csongrád-Csanád": (
        "Szeged",
        "Hódmezővásárhely",
        "Makó",
        "Szentes",
        "Csongrád",
        "Mórahalom",
    ),
    "Fejér": (
        "Székesfehérvár",
        "Dunaújváros",
        "Bicske",
        "Mór",
        "Sárbogárd",
        "Velence",
        "Gárdony",
        "Pákozd",
        "Pusztaszabolcs",
        "Etyek",
        "Mány",
    ),
    "Győr-Moson-Sopron": (
        "Győr",
        "Sopron",
        "Mosonmagyaróvár",
        "Csorna",
        "Kapuvár",
        "Pannonhalma",
        "Tét",
    ),
    "Hajdú-Bihar": (
        "Debrecen",
        "Hajdúböszörmény",
        "Hajdúszoboszló",
        "Hajdúnánás",
        "Berettyóújfalu",
        "Püspökladány",
        "Balmazújváros",
        "Hajdúdorog",
    ),
    "Heves": (
        "Eger",
        "Gyöngyös",
        "Hatvan",
        "Heves",
        "Füzesabony",
        "Pétervására",
        "Noszvaj",
    ),
    "Jász-Nagykun-Szolnok": (
        "Szolnok",
        "Jászberény",
        "Karcag",
        "Mezőtúr",
        "Törökszentmiklós",
        "Kunszentmárton",
        "Tiszafüred",
        "Kunhegyes",
    ),
    "Komárom-Esztergom": (
        "Tatabánya",
        "Esztergom",
        "Tata",
        "Komárom",
        "Oroszlány",
        "Nyergesújfalu",
        "Dorog",
        "Lábatlan",
    ),
    "Nógrád": (
        "Salgótarján",
        "Balassagyarmat",
        "Bátonyterenye",
        "Pásztó",
        "Rétság",
        "Szécsény",
    ),
    "Pest": (
        # Agglomeration + major towns where listings concentrate.
        "Érd",
        "Cegléd",
        "Gödöllő",
        "Vác",
        "Szentendre",
        "Dunakeszi",
        "Vecsés",
        "Budakeszi",
        "Gyál",
        "Budaörs",
        "Pilisvörösvár",
        "Pomáz",
        "Nagykovácsi",
        "Solymár",
        "Veresegyház",
        "Pécel",
        "Monor",
        "Üllő",
        "Gyömrő",
        "Aszód",
        "Albertirsa",
        "Dabas",
        "Nagykáta",
        "Nagykőrös",
        "Ráckeve",
        "Szigetszentmiklós",
        "Szigethalom",
        "Tököl",
        "Diósd",
        "Százhalombatta",
        "Tárnok",
        "Biatorbágy",
        "Páty",
        "Telki",
        "Zsámbék",
        "Bag",
        "Mende",
        "Pilisborosjenő",
        "Üröm",
        "Visegrád",
        "Nagymaros",
        "Fót",
        "Csömör",
        "Mogyoród",
        "Kistarcsa",
        "Kerepes",
        "Tahitótfalu",
        "Leányfalu",
        "Dunabogdány",
        "Zebegény",
        "Maglód",
        "Halásztelek",
        "Sóskút",
        "Pusztazámor",
        "Kosd",
        "Sződ",
        "Sződliget",
        "Verőce",
        "Pilisszentkereszt",
        "Pilisszántó",
        "Csomád",
        "Őrbottyán",
    ),
    "Somogy": (
        "Kaposvár",
        "Siófok",
        "Marcali",
        "Nagyatád",
        "Barcs",
        "Tab",
        "Csurgó",
        "Fonyód",
        "Balatonboglár",
        "Balatonlelle",
        "Balatonföldvár",
        "Balatonszemes",
        "Zamárdi",
    ),
    "Szabolcs-Szatmár-Bereg": (
        "Nyíregyháza",
        "Mátészalka",
        "Kisvárda",
        "Vásárosnamény",
        "Fehérgyarmat",
        "Csenger",
        "Tiszavasvári",
        "Nagykálló",
    ),
    "Tolna": (
        "Szekszárd",
        "Paks",
        "Dombóvár",
        "Bonyhád",
        "Tamási",
        "Tolna",
        "Simontornya",
    ),
    "Vas": (
        "Szombathely",
        "Sárvár",
        "Körmend",
        "Kőszeg",
        "Celldömölk",
        "Vasvár",
    ),
    "Veszprém": (
        "Veszprém",
        "Pápa",
        "Ajka",
        "Várpalota",
        "Balatonfüred",
        "Tapolca",
        "Sümeg",
        "Zirc",
        "Berhida",
        "Balatonalmádi",
        "Tihany",
        "Csopak",
        "Balatonakali",
        "Balatonszepezd",
        "Révfülöp",
        "Badacsonytomaj",
        "Szigliget",
        "Balatongyörök",
        "Ábrahámhegy",
        "Balatonrendes",
        "Köveskál",
        "Salföld",
        "Balatonederics",
    ),
    "Zala": (
        "Zalaegerszeg",
        "Nagykanizsa",
        "Keszthely",
        "Lenti",
        "Letenye",
        "Hévíz",
        "Zalaszentgrót",
        "Vonyarcvashegy",
        "Gyenesdiás",
    ),
}

# Public canonical -> lower-cased set, generated from the complete KSH asset.
COUNTY_CITIES: dict[str, frozenset[str]] = county_city_index()

# Case-folded settlement name -> canonical spelling, for recognising an explicit
# city token in a query (e.g. the deterministic parser fast path). Flattened
# across all counties; later counties win on the rare duplicate, which is fine
# since the canonical spelling is identical.
CITY_CANONICAL_BY_FOLD: dict[str, str] = {
    settlement.name.casefold(): settlement.name
    for settlement in all_settlements()
    if settlement.kind is SettlementKind.MUNICIPALITY
}

# Settlements along Balaton, by shore. Includes the lake-front strip used in
# colloquial searches like "balatoni északi parton" / "déli parton".
_BALATON_NORTH_RAW: tuple[str, ...] = (
    "Balatonalmádi",
    "Balatonfüred",
    "Tihany",
    "Csopak",
    "Balatonakali",
    "Balatonszepezd",
    "Révfülöp",
    "Badacsonytomaj",
    "Szigliget",
    "Balatongyörök",
    "Keszthely",
    "Vonyarcvashegy",
    "Gyenesdiás",
    "Ábrahámhegy",
    "Balatonrendes",
    "Balatonederics",
    "Hévíz",
)
_BALATON_SOUTH_RAW: tuple[str, ...] = (
    "Siófok",
    "Zamárdi",
    "Balatonföldvár",
    "Balatonszárszó",
    "Balatonőszöd",
    "Balatonszemes",
    "Balatonlelle",
    "Balatonboglár",
    "Fonyód",
    "Balatonfenyves",
    "Balatonmáriafürdő",
    "Balatonkeresztúr",
    "Balatonberény",
    "Balatonszentgyörgy",
)
BALATON_NORTH_SHORE: frozenset[str] = frozenset(c.lower() for c in _BALATON_NORTH_RAW)
BALATON_SOUTH_SHORE: frozenset[str] = frozenset(c.lower() for c in _BALATON_SOUTH_RAW)
BALATON_AREA_CITIES: frozenset[str] = BALATON_NORTH_SHORE | BALATON_SOUTH_SHORE


def matches_county(city: str | None, county: str | None) -> float:
    """Return 1.0 when `city` is in `county`'s settlement set, else 0.0.

    Both arguments are matched case-insensitively. Unknown counties or missing
    inputs return 0.0 so the score blend simply drops the cue.
    """
    if city is None or county is None:
        return 0.0
    cities = COUNTY_CITIES.get(county)
    if cities is None:
        return 0.0
    return 1.0 if city.lower() in cities else 0.0


def matches_balaton_shore(city: str | None, shore: str | None) -> float:
    """Return 1.0 when `city` sits on the requested Balaton shore, else 0.0."""
    if city is None or shore is None:
        return 0.0
    target = city.lower()
    if shore == "Északi":
        return 1.0 if target in BALATON_NORTH_SHORE else 0.0
    if shore == "Déli":
        return 1.0 if target in BALATON_SOUTH_SHORE else 0.0
    return 0.0


def matches_balaton_area(city: str | None) -> float:
    """Return 1.0 when `city` is a Balaton-area settlement, else 0.0."""
    if city is None:
        return 0.0
    return 1.0 if city.lower() in BALATON_AREA_CITIES else 0.0


def is_balaton_region_name(region_name: str | None) -> bool:
    """Recognise generic region phrases like "Balaton", "Balaton környéke"."""
    return bool(region_name and "balaton" in region_name.casefold())
