"""Data-driven Hungarian settlement resolution and distance helpers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).with_name("data") / "hungary_settlements_2025.json"
_EARTH_RADIUS_KM = 6371.0088
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_GENERIC_BALATON_FORMS = frozenset({"balaton", "balatoni", "balatonon", "balatonra", "balatonnal", "balatonhoz"})

_INFLECTION_SUFFIXES = (
    "bol",
    "rol",
    "nal",
    "hoz",
    "hez",
    "hoz",
    "ban",
    "ben",
    "ott",
    "ett",
    "ba",
    "be",
    "ra",
    "re",
    "an",
    "en",
    "on",
    "n",
    "i",
)

# Normalized street-type words. A settlement-shaped token immediately followed
# by one of these is a street name, not a settlement mention — Budapest is full
# of streets built from settlement adjectives ("Váci út", "Szentendrei út",
# "Budaörsi út", "Csömöri út"), and resolving them as settlements corrupts the
# city hard filter (2026-07-03 eval regression: "a Váci úton" -> city=Vác).
_STREET_FOLLOWER_FORMS = frozenset(
    {
        "ut",
        "uton",
        "utra",
        "utnal",
        "utrol",
        "uthoz",
        "utja",
        "utjan",
        "utca",
        "utcaban",
        "utcan",
        "utcara",
        "utcanal",
        "utcaba",
        "utcaja",
        "utcajaban",
        "ter",
        "teren",
        "terre",
        "ternel",
        "tere",
        "korut",
        "koruton",
        "korutra",
        "korutnal",
        "setany",
        "setanyon",
        "setanyra",
        "koz",
        "sor",
        "soron",
        "rakpart",
        "rakparton",
        "sugarut",
        "sugaruton",
        "fasor",
        "fasoron",
        "lakotelep",
        "lakotelepen",
    }
)

_COMMON_TYPO_IDS = {
    "budapset": "ksh:13578",
    "budapestt": "ksh:13578",
    "debrecn": "ksh:15130",
    "deberecen": "ksh:15130",
    "miskloc": "ksh:30456",
    "szged": "ksh:33367",
    "szegedd": "ksh:33367",
    "nyiregyhza": "ksh:17206",
}


class SettlementKind(StrEnum):
    MUNICIPALITY = "municipality"
    BUDAPEST_DISTRICT = "budapest_district"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class Settlement:
    id: str
    name: str
    kind: SettlementKind
    legal_status: str
    county: str
    district_id: str | None
    district_name: str | None
    district_seat: str | None
    parent_id: str | None
    budapest_district: int | None
    latitude: float | None
    longitude: float | None
    geonames_id: int | None


@dataclass(frozen=True, slots=True)
class NearbySettlement:
    """A verified settlement inside an authorised distance from the origin."""

    settlement: Settlement
    distance_km: float


@dataclass(frozen=True, slots=True)
class LocationResolution:
    status: ResolutionStatus
    candidates: tuple[Settlement, ...] = ()
    matched_text: str | None = None

    @property
    def settlement(self) -> Settlement | None:
        return self.candidates[0] if self.status is ResolutionStatus.RESOLVED and len(self.candidates) == 1 else None


def normalize_settlement_name(value: str) -> str:
    """Case-fold, remove accents and normalise punctuation for deterministic lookup."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    deaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", deaccented).split())


@lru_cache(maxsize=1)
def _raw_dataset() -> dict[str, Any]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def all_settlements() -> tuple[Settlement, ...]:
    return tuple(
        Settlement(
            id=row["id"],
            name=row["name"],
            kind=SettlementKind(row["kind"]),
            legal_status=row["legal_status"].strip(),
            county=row["county"],
            district_id=row["district_id"],
            district_name=row["district_name"],
            district_seat=row["district_seat"],
            parent_id=row["parent_id"],
            budapest_district=row["budapest_district"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            geonames_id=row["geonames_id"],
        )
        for row in _raw_dataset()["settlements"]
    )


def dataset_metadata() -> dict[str, Any]:
    return dict(_raw_dataset()["metadata"])


@lru_cache(maxsize=1)
def _by_id() -> dict[str, Settlement]:
    return {settlement.id: settlement for settlement in all_settlements()}


@lru_cache(maxsize=1)
def _by_name() -> dict[str, tuple[Settlement, ...]]:
    grouped: dict[str, list[Settlement]] = {}
    for settlement in all_settlements():
        if settlement.kind is not SettlementKind.MUNICIPALITY:
            continue
        grouped.setdefault(normalize_settlement_name(settlement.name), []).append(settlement)
    return {key: tuple(sorted(values, key=lambda item: item.id)) for key, values in grouped.items()}


@lru_cache(maxsize=1)
def _district_by_number() -> dict[int, Settlement]:
    return {
        settlement.budapest_district: settlement
        for settlement in all_settlements()
        if settlement.budapest_district is not None
    }


def settlement_by_id(settlement_id: str) -> Settlement | None:
    return _by_id().get(settlement_id)


def budapest_district(number: int) -> Settlement | None:
    return _district_by_number().get(number)


def _resolution_for_key(key: str, *, matched_text: str) -> LocationResolution:
    typo_id = _COMMON_TYPO_IDS.get(key)
    if typo_id is not None:
        settlement = settlement_by_id(typo_id)
        if settlement is not None:
            return LocationResolution(ResolutionStatus.RESOLVED, (settlement,), matched_text)

    candidates = _by_name().get(key, ())
    if len(candidates) == 1:
        return LocationResolution(ResolutionStatus.RESOLVED, candidates, matched_text)
    if len(candidates) > 1:
        return LocationResolution(ResolutionStatus.AMBIGUOUS, candidates, matched_text)
    return LocationResolution(ResolutionStatus.NOT_FOUND, (), matched_text)


def resolve_settlement_name(value: str) -> LocationResolution:
    """Resolve a canonical, accent-free, inflected or known-typo settlement name."""

    key = normalize_settlement_name(value)
    if not key:
        return LocationResolution(ResolutionStatus.NOT_FOUND, (), value)
    direct = _resolution_for_key(key, matched_text=value)
    if direct.status is not ResolutionStatus.NOT_FOUND:
        return direct
    if " " in key:
        return direct
    for suffix in _INFLECTION_SUFFIXES:
        if len(key) <= len(suffix) + 2 or not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]
        for candidate_stem in (stem, f"{stem}a", f"{stem}e"):
            stripped = _resolution_for_key(candidate_stem, matched_text=value)
            if stripped.status is not ResolutionStatus.NOT_FOUND:
                return stripped
    return direct


def resolve_location_text(text: str) -> LocationResolution:
    """Resolve explicit settlement mentions; multiple distinct hits are ambiguous."""

    hits: dict[str, Settlement] = {}
    matched: list[str] = []
    tokens = _WORD_RE.findall(text)
    for index, token in enumerate(tokens):
        if normalize_settlement_name(token) in _GENERIC_BALATON_FORMS:
            continue
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if normalize_settlement_name(next_token) in _STREET_FOLLOWER_FORMS:
            continue
        resolution = resolve_settlement_name(token)
        if resolution.status is ResolutionStatus.AMBIGUOUS:
            return resolution
        settlement = resolution.settlement
        if settlement is None:
            continue
        hits[settlement.id] = settlement
        matched.append(token)
    candidates = tuple(sorted(hits.values(), key=lambda item: item.id))
    if len(candidates) == 1:
        return LocationResolution(ResolutionStatus.RESOLVED, candidates, " ".join(matched))
    if len(candidates) > 1:
        return LocationResolution(ResolutionStatus.AMBIGUOUS, candidates, " ".join(matched))
    return LocationResolution(ResolutionStatus.NOT_FOUND)


def county_city_index() -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = {}
    for settlement in all_settlements():
        if settlement.kind is not SettlementKind.MUNICIPALITY or not settlement.county:
            continue
        grouped.setdefault(settlement.county, set()).add(settlement.name.casefold())
    return {county: frozenset(names) for county, names in grouped.items()}


def distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS84 coordinates."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    value = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(value))


def settlement_distance_km(first: Settlement, second: Settlement) -> float | None:
    """Distance for verified coordinates, or ``None`` when either point is missing."""

    if first.latitude is None or first.longitude is None or second.latitude is None or second.longitude is None:
        return None
    return distance_km(first.latitude, first.longitude, second.latitude, second.longitude)


def is_within_distance(first: Settlement, second: Settlement, maximum_km: float) -> bool:
    measured = settlement_distance_km(first, second)
    return measured is not None and measured <= maximum_km


@lru_cache(maxsize=4096)
def nearby_settlements(origin_id: str, maximum_km: float) -> tuple[NearbySettlement, ...]:
    """Return verified same-level neighbours, nearest first.

    Budapest districts are compared only with other Budapest districts;
    municipalities are compared only with municipalities. Records without
    verified coordinates are excluded instead of being treated as nearby.
    """

    if maximum_km <= 0:
        return ()
    origin = settlement_by_id(origin_id)
    if origin is None or origin.latitude is None or origin.longitude is None:
        return ()

    nearby: list[NearbySettlement] = []
    for candidate in all_settlements():
        if candidate.id == origin.id or candidate.kind is not origin.kind:
            continue
        measured = settlement_distance_km(origin, candidate)
        if measured is None or measured > maximum_km:
            continue
        nearby.append(NearbySettlement(settlement=candidate, distance_km=measured))
    return tuple(sorted(nearby, key=lambda item: (item.distance_km, item.settlement.name, item.settlement.id)))
