"""§7 deterministic parser fast path — rule-based parsing for simple queries.

`deterministic_parse` turns a short, fully-explicit Hungarian property query
("eladó lakás Budapest 50 millió alatt") into a `ParsedQuery` with **no LLM
call**. It is intentionally conservative: it only returns a result when *every*
token is accounted for by a recognised field (transaction, property type,
city/district, price, room count) plus a tiny set of connectors. Anything left
over — a soft preference ("napos", "felújított"), a construction type ("panel"),
a free-text phrase, an unknown city, a spelled-out number, an explicit area —
makes it return `None`, and `parse_query` falls back to the Gemini parser
unchanged. This guarantees the fast path can never *misread* a query: at worst
it declines and the LLM handles it.

Fast-path results carry an empty `semantic_query`, which the search pipeline
already treats as a "filter-only" path (no embedding, no rerank).
"""

# pyright: reportPrivateUsage=false
# The lexicon's tables and recognisers are this module's own implementation detail;
# they stay private to the pair rather than becoming API.

from __future__ import annotations

from dataclasses import dataclass, field

from ingovo_search.recommender.fastpath_lexicon import (
    _BALATON_AREA_QUALIFIERS,
    _BALATON_REGION_NAME,
    _BALATON_REGION_WORDS,
    _BALATON_SHORE_WORDS,
    _COUNTY_TOKEN_SEQUENCES,
    _CURRENCY_WORDS,
    _FILLER,
    _KNOWN_POIS,
    _MILLION,
    _MIN_PRE_WORDS,
    _NEGATION_TOKENS,
    _RANGE_WORDS,
    _ROOM_NUMBER_WORDS,
    _THOUSAND,
    _TOKEN_RE,
    MAX_FASTPATH_CHARS,
    _strip_accents,
    budapest_district_alias,
    city_of,
    classify_bound,
    district_number,
    is_kerulet,
    is_number,
    million_suffix,
    normalise,
    number_value,
    property_type_of,
    thousand_suffix,
    transaction_of,
)
from ingovo_search.recommender.parser_schema import (
    Anchors,
    BalatonShore,
    GeoHints,
    HardFilters,
    NearFilter,
    ParsedQuery,
    PropertyType,
    TransactionType,
)


@dataclass(slots=True)
class _Scan:
    """The token scan of one query: what each pass found, and which tokens it consumed."""

    tokens: list[str]
    folded: list[str]
    covered: list[bool]
    city: str | None = None
    region_name: str | None = None
    county: str | None = None
    balaton_shore: BalatonShore | None = None
    near: NearFilter | None = None
    districts: list[str] = field(default_factory=list[str])
    transaction: TransactionType | None = None
    property_types: list[PropertyType] = field(default_factory=list[PropertyType])
    rooms: int | None = None
    rooms_min: int | None = None
    price_max: int | None = None
    price_min: int | None = None
    price_anchor: int | None = None

    def free(self, index: int) -> bool:
        """An existing token no earlier pass has consumed."""
        return 0 <= index < len(self.tokens) and not self.covered[index]

    def cover(self, start: int, stop: int) -> None:
        for index in range(start, stop):
            self.covered[index] = True


def _scan_tokens(text: str, *, allow_uncovered: bool) -> _Scan | None:
    """Tokenise a query worth scanning; ``None`` when it is empty or too long for the strict path."""
    if not text or not text.strip():
        return None
    norm = normalise(text)
    # The length gate guards the *strict* path from misreading a long natural-
    # language query as a complete filter-only parse. The loose fallback keeps
    # the full text as `semantic_query` and tolerates uncovered tokens, so it
    # must still salvage hard filters (district, type, price, rooms) from long
    # queries — that is exactly when the LLM parser has timed out and this
    # fallback is all that stands between the user and garbage results.
    if not allow_uncovered and len(norm) > MAX_FASTPATH_CHARS:
        return None
    tokens = _TOKEN_RE.findall(norm)
    if not tokens:
        return None
    return _Scan(tokens=tokens, folded=[_strip_accents(token) for token in tokens], covered=[False] * len(tokens))


def _scan_county(scan: _Scan) -> None:
    """0.5 Explicit county: "Pest vármegye", including multi-token counties."""
    for marker_index, token in enumerate(scan.folded):
        if not token.startswith(("varmegy", "megy")):
            continue
        for county_tokens, canonical_county in _COUNTY_TOKEN_SEQUENCES:
            start = marker_index - len(county_tokens)
            if start < 0 or tuple(scan.folded[start:marker_index]) != county_tokens:
                continue
            scan.county = canonical_county
            scan.cover(start, marker_index + 1)
            break


def _poi_radius_km(scan: _Scan, radius_index: int) -> float | None:
    """The explicit "<n> km" after a POI, or ``None`` when the tokens there are not a radius."""
    if radius_index + 1 >= len(scan.tokens):
        return None
    radius_token = scan.tokens[radius_index]
    unit_token = scan.folded[radius_index + 1]
    if not is_number(radius_token) or not unit_token.startswith(("kilometer", "km")):
        return None
    radius_km = number_value(radius_token)
    return radius_km if radius_km > 0 else None


def _scan_poi(scan: _Scan) -> None:
    """0.6 Verified POI + explicit kilometre radius."""
    for poi_tokens, (lat, lng, poi_city) in _KNOWN_POIS.items():
        for start in range(0, len(scan.tokens) - len(poi_tokens) + 1):
            if tuple(scan.folded[start : start + len(poi_tokens)]) != poi_tokens:
                continue
            radius_index = start + len(poi_tokens)
            radius_km = _poi_radius_km(scan, radius_index)
            if radius_km is None:
                continue
            scan.near = NearFilter(lat=lat, lng=lng, radius_km=radius_km)
            scan.city = poi_city
            scan.cover(start, radius_index + 2)
            if radius_index + 2 < len(scan.tokens) and scan.folded[radius_index + 2].startswith("korzet"):
                scan.covered[radius_index + 2] = True
            break


def _scan_districts(scan: _Scan) -> None:
    """1. Budapest districts: "<n> kerület". Implies city = Budapest.

    Negated mentions ("nem a 8. kerületben", "a 8. kerület kivételével") are
    exclusions — leave them uncovered so the strict path defers to the LLM
    and the loose path never turns them into an inclusive district filter.
    """
    for i in range(len(scan.tokens) - 1):
        if scan.covered[i] or scan.covered[i + 1] or not is_kerulet(scan.tokens[i + 1]):
            continue
        preceding = scan.tokens[max(0, i - 3) : i]
        following = scan.folded[i + 2] if i + 2 < len(scan.tokens) else ""
        if any(token in _NEGATION_TOKENS for token in preceding) or following.startswith("kivetel"):
            continue
        district = district_number(scan.tokens[i])
        if district is None:
            continue
        if district not in scan.districts:
            scan.districts.append(district)
        scan.covered[i] = scan.covered[i + 1] = True
        scan.city = "Budapest"


def _scan_transaction(scan: _Scan) -> bool:
    """2. Transaction intent; ``False`` on conflicting rent/sale words (let the LLM judge)."""
    for i, token in enumerate(scan.tokens):
        if scan.covered[i]:
            continue
        tx = transaction_of(token)
        if tx is None:
            continue
        if scan.transaction is not None and scan.transaction != tx:
            return False
        scan.transaction = tx
        scan.covered[i] = True
    return True


def _scan_property_types(scan: _Scan) -> None:
    """3. Property types; a "garzon" is a one-room flat."""
    for i, token in enumerate(scan.tokens):
        if scan.covered[i]:
            continue
        if token.startswith("garzon"):
            if "Lakás" not in scan.property_types:
                scan.property_types.append("Lakás")
            if scan.rooms is None:
                scan.rooms = 1
            scan.covered[i] = True
            continue
        ptype = property_type_of(token)
        if ptype is not None:
            if ptype not in scan.property_types:
                scan.property_types.append(ptype)
            scan.covered[i] = True


def _scan_cities(scan: _Scan) -> bool:
    """3.5 Known settlement names ("Szeged", "Debrecen", "Budapest", …); ``False`` on two different cities."""
    for i, token in enumerate(scan.tokens):
        if scan.covered[i] or token in _BALATON_REGION_WORDS:
            continue
        district_alias = budapest_district_alias(token)
        if district_alias is not None:
            scan.city = "Budapest"
            if district_alias not in scan.districts:
                scan.districts.append(district_alias)
            scan.covered[i] = True
            continue
        canonical = city_of(token)
        if canonical is None:
            continue
        if scan.city is not None and scan.city.casefold() != canonical.casefold():
            return False  # two different cities → ambiguous, defer to the LLM
        scan.city = canonical
        scan.covered[i] = True
    return True


def _scan_balaton(scan: _Scan) -> None:
    """3.6 Generic Balaton region: "balatoni", "Balaton környéki", "Balaton menti"."""
    for i, token in enumerate(scan.tokens):
        if scan.covered[i] or token not in _BALATON_REGION_WORDS:
            continue
        scan.region_name = _BALATON_REGION_NAME
        scan.covered[i] = True
        nxt = i + 1
        if scan.free(nxt) and scan.tokens[nxt] in _BALATON_SHORE_WORDS:
            scan.balaton_shore = _BALATON_SHORE_WORDS[scan.tokens[nxt]]
            scan.covered[nxt] = True
            nxt += 1
        if scan.free(nxt) and scan.tokens[nxt] in _BALATON_AREA_QUALIFIERS:
            scan.covered[nxt] = True


def _scan_rooms(scan: _Scan) -> None:
    """4. Room counts: "<n> szoba(s)" / "legalább <n> szoba" / "<n> szobától"."""
    for i in range(len(scan.tokens) - 1):
        if scan.covered[i] or scan.covered[i + 1]:
            continue
        nxt = scan.tokens[i + 1]
        if not nxt.startswith("szob") or not scan.tokens[i].isdigit():
            continue  # decimals / half-rooms → defer to the LLM
        value = int(scan.tokens[i])
        prev = scan.tokens[i - 1] if scan.free(i - 1) else None
        if nxt.endswith(("tól", "tol", "től", "tel")) or prev in _MIN_PRE_WORDS:
            scan.rooms_min = value
            if prev in _MIN_PRE_WORDS:
                scan.covered[i - 1] = True
        else:
            scan.rooms = value
        scan.covered[i] = scan.covered[i + 1] = True


def _scan_room_words(scan: _Scan) -> None:
    """4b. Word-number room forms the digit scan misses: "kétszobás", and split "két szobás"."""
    if scan.rooms is not None or scan.rooms_min is not None:
        return
    for i, token in enumerate(scan.tokens):
        if scan.covered[i]:
            continue
        # Single token: "<numberword>szob..." (kétszobás, háromszobás).
        for word, value in _ROOM_NUMBER_WORDS.items():
            if token.startswith(word) and "szob" in token[len(word) :]:
                scan.rooms = value
                scan.covered[i] = True
                break
        if scan.rooms is not None:
            break
        # Split form: "<numberword>" followed by "szob..." token.
        if token in _ROOM_NUMBER_WORDS and scan.free(i + 1) and scan.tokens[i + 1].startswith("szob"):
            scan.rooms = _ROOM_NUMBER_WORDS[token]
            scan.covered[i] = scan.covered[i + 1] = True
            break


def _unit_multiplier(token: str) -> int | None:
    """What a "millió" / "ezer" unit token multiplies the amount by."""
    if million_suffix(token) is not None:
        return _MILLION
    if thousand_suffix(token) is not None:
        return _THOUSAND
    return None


def _scan_price_range(scan: _Scan) -> None:
    """5. Price ranges: "50 és 80 millió között", "150 és 250 ezer Ft között"."""
    for i in range(len(scan.tokens) - 4):
        if any(scan.covered[j] for j in range(i, i + 5)):
            continue
        if not is_number(scan.tokens[i]) or scan.tokens[i + 1] not in {"és", "es"} or not is_number(scan.tokens[i + 2]):
            continue
        multiplier = _unit_multiplier(scan.tokens[i + 3])
        if multiplier is None:
            continue
        range_index = i + 4
        if scan.tokens[range_index] in _CURRENCY_WORDS:
            range_index += 1
        if range_index >= len(scan.tokens) or scan.tokens[range_index] not in _RANGE_WORDS:
            continue
        lower = round(number_value(scan.tokens[i]) * multiplier)
        upper = round(number_value(scan.tokens[i + 2]) * multiplier)
        scan.price_min = min(lower, upper)
        scan.price_max = max(lower, upper)
        scan.cover(i, range_index + 1)


def _price_unit(token: str) -> tuple[str, int] | None:
    """The recognised unit suffix and its multiplier, million first."""
    suffix = million_suffix(token)
    if suffix is not None:
        return suffix, _MILLION
    suffix = thousand_suffix(token)
    if suffix is not None:
        return suffix, _THOUSAND
    return None


def _record_bound(scan: _Scan, bound: str, value: int) -> None:
    if bound == "max":
        scan.price_max = value
    elif bound == "min":
        scan.price_min = value
    else:
        scan.price_anchor = value


def _price_at(scan: _Scan, i: int, unit: tuple[str, int]) -> bool:
    """Read "<n> millió|ezer [Ft|HUF] [qualifier]" at ``i``; ``False`` when the price has no clear direction."""
    suffix, multiplier = unit
    qualifier_index = i + 2
    has_currency = scan.free(qualifier_index) and scan.tokens[qualifier_index] in _CURRENCY_WORDS
    if has_currency:
        qualifier_index += 1
    next_token = scan.tokens[qualifier_index] if scan.free(qualifier_index) else None
    prev_token = scan.tokens[i - 1] if scan.free(i - 1) else None
    bound, consume_next, consume_prev = classify_bound(suffix, next_token, prev_token)
    if bound is None:
        return False
    _record_bound(scan, bound, round(number_value(scan.tokens[i]) * multiplier))
    consumed = [i, i + 1]
    if has_currency:
        consumed.append(i + 2)
    if consume_next:
        consumed.append(qualifier_index)
    if consume_prev:
        consumed.append(i - 1)
    for index in consumed:
        scan.covered[index] = True
    return True


def _scan_prices(scan: _Scan) -> bool:
    """5.5 Price: "<n> millió|ezer [Ft|HUF] [qualifier]"; ``False`` on a price with no direction."""
    for i in range(len(scan.tokens) - 1):
        if scan.covered[i] or scan.covered[i + 1] or not is_number(scan.tokens[i]):
            continue
        unit = _price_unit(scan.tokens[i + 1])
        if unit is None:
            continue  # the next token is not a price unit
        if not _price_at(scan, i, unit):
            return False  # a price with no clear direction → let the LLM decide
    return True


def _coverage_ok(scan: _Scan, *, allow_uncovered: bool) -> bool:
    """6. Coverage gate: on the strict path every remaining token must be harmless filler."""
    if allow_uncovered:
        return True
    return all(covered or token in _FILLER for token, covered in zip(scan.tokens, scan.covered, strict=True))


def _has_constraint(scan: _Scan) -> bool:
    """7. At least one constraint beyond a bare transaction word."""
    return bool(
        scan.city
        or scan.region_name
        or scan.county
        or scan.balaton_shore
        or scan.near is not None
        or scan.districts
        or scan.property_types
        or scan.rooms is not None
        or scan.rooms_min is not None
        or scan.price_max is not None
        or scan.price_min is not None
        or scan.price_anchor is not None
    )


def _build_parsed_query(scan: _Scan, text: str, *, allow_uncovered: bool) -> ParsedQuery:
    hard_filters = HardFilters(
        city=scan.city,
        district_in=scan.districts,
        transaction_type=scan.transaction,
        property_type_in=scan.property_types,
        rooms=scan.rooms,
        rooms_min=scan.rooms_min,
        price_huf_min=scan.price_min,
        price_huf_max=scan.price_max,
        near=scan.near,
    )
    anchors = Anchors(price_huf=scan.price_anchor)
    geo_hints = GeoHints(
        region_name=None if scan.balaton_shore else scan.region_name,
        county=scan.county,
        balaton_shore=scan.balaton_shore,
    )
    semantic_query = text.strip() if allow_uncovered else ""
    return ParsedQuery(hard_filters=hard_filters, anchors=anchors, geo_hints=geo_hints, semantic_query=semantic_query)


def _deterministic_parse(text: str, *, allow_uncovered: bool) -> ParsedQuery | None:
    """Return a `ParsedQuery` for a simple explicit query, or `None` to defer.

    `None` means "not confidently a simple query" — the caller must run the LLM
    parser. A non-`None` result is safe to use directly. The passes run in a
    fixed order; a pass that finds the query ambiguous ends the scan.
    """
    scan = _scan_tokens(text, allow_uncovered=allow_uncovered)
    if scan is None:
        return None
    _scan_county(scan)
    _scan_poi(scan)
    _scan_districts(scan)
    if not _scan_transaction(scan):
        return None
    _scan_property_types(scan)
    if not _scan_cities(scan):
        return None
    _scan_balaton(scan)
    _scan_rooms(scan)
    _scan_room_words(scan)
    _scan_price_range(scan)
    if not _scan_prices(scan) or not _coverage_ok(scan, allow_uncovered=allow_uncovered) or not _has_constraint(scan):
        return None
    return _build_parsed_query(scan, text, allow_uncovered=allow_uncovered)


def deterministic_parse(text: str) -> ParsedQuery | None:
    """Strict fast path: every meaningful token must be understood."""
    return _deterministic_parse(text, allow_uncovered=False)


def deterministic_parse_loose(text: str) -> ParsedQuery | None:
    """Latency fallback: keep understood hard filters and preserve the full text.

    This is intentionally not used by the normal parser fast path. It is only a
    fallback when the LLM parser would exceed the request latency budget.
    """
    return _deterministic_parse(text, allow_uncovered=True)
