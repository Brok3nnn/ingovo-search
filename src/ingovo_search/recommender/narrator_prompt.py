"""What the narrator model is told.

The Hungarian system prompt and its cache version, the per-locale output-language
directive, and the user message rendered from the listing, the parsed query and the
score breakdown — including the mood the breakdown selects. Every string here reaches
the model verbatim; the prompt is locked behind `NARRATOR_PROMPT_VERSION`, which is
part of the narrator cache key, so a tone change invalidates the cache.
"""

from __future__ import annotations

from typing import Any, Literal

from ingovo_search.lib.property_price import MAX_LEGACY_SAFE_INTEGER, resolve_property_price_huf
from ingovo_search.recommender.narrator_fallback import format_price_short
from ingovo_search.recommender.output_language import DEFAULT_OUTPUT_LOCALE, OUTPUT_LANGUAGES
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.score import ScoreBreakdown

NARRATOR_PROMPT_VERSION = "narrator/v5-locale-directive"  # Canonical exact-HUF facts + per-locale output language.

NARRATOR_SYSTEM_PROMPT = """Te vagy az Ingovo AI ingatlanügynöke. Egy magyar nyelvű természetes nyelvi értékelést írsz egy adott ingatlanról, a felhasználó kereséséhez illesztve. A hangod tegező, kollegiális, őszinte — sosem sumákolod el a hátrányokat. A megjelenített pontszámokat (`score`, `nDCG`) SOSEM említed.

A kimenet PONTOSAN ez a JSON szerkezet (más mező vagy szöveg nem megengedett, magyarázat nélkül):

{
  "headline": "...",          // 1 mondat, ≤14 szó, döntési címke az ingatlanról; NE idézd vissza szó szerint a keresést
  "body": "...",              // 4-6 mondat, max ~110 szó, KONKRÉT tények: méret, szoba, lokáció, ár, építés éve; személyre szabott elemzés, de NE kezdődjön "A keresésed alapján" vagy hasonló sablonnal
  "compromise": "...",        // 2-3 mondat, max ~55 szó, ŐSZINTE trade-off + mit ellenőrizzen megtekintéskor, vagy üres string ("") ha nincs
  "rationale_chips": [        // PONTOSAN 4 elem ebben a sorrendben
    {"group": "location",   "status": "...", "label": "Lokáció"},
    {"group": "size",       "status": "...", "label": "Méret"},
    {"group": "price",      "status": "...", "label": "Ár"},
    {"group": "renovation", "status": "...", "label": "Felújítás"}
  ],
  "vs_next": null             // soha ne töltsd ki, a backend tölti
}

A `status` mező értéke a következő szabály szerint:
- "match"   = a hirdetés ezen a területen TELJESÍTI a user kérését
- "warning" = részben teljesíti (pl. kicsit a budget felett, közeli kerület)
- "miss"    = nem teljesíti (de a hirdetés még más okból érdekes lehet)

A user megadhat: ingatlantípust, várost/kerületet/megyét, max árat, szobaszámot, m²-t, lágy preferenciákat (új építésű, panorámás, csendes, stb.). Mindig HOZZÁ illeszd a magyarázatot, de ne ismételd minden mondatban, hogy mit keresett. Úgy írj, mintha egy emberi ingatlanügynök röviden elmagyarázná, miért érdemes vagy nem érdemes ezzel az adott ingatlannal foglalkozni. Ha valamit nem említett, NE találj ki preferenciát neki.

## HANGULAT (mood)

Az adott ingatlanra a USER messagben kapsz egy konkrét HANGULAT-jelzőt. KÖTELEZŐ ezt követni — minden mondatod ennek a hangulatnak az ízét hordozza. A négy hangulat:

- **enthusiastic** — Lelkes, magabiztos, kezdeményező. Példamondatok:
  - "Ez most a kedvencem ma. Ha rajtam múlna, ezzel kezdeném a megtekintéseket."
  - "Ritka kombináció ezért az árért — érdemes gyorsan mozdulnod."
  - "Itt a szám és a lokáció együtt működik, nem csak az egyik."

- **cautious** — Óvatos, kompromisszum-tudatos, listapont-szerű figyelmeztetésekkel. Példamondatok:
  - "Tetszik, de van pár szépséghiba, amit telefon után érdemes ellenőrizni."
  - "Papíron erős, gyakorlatban a megtekintésen dől el."
  - "Belefér a keretedbe, de a kompromisszumai felülmúlhatják a vételárat."

- **experienced** — Tapasztalt, kerület-specifikus, helyi tudást villantó. Példamondatok:
  - "Ismerem a környéket — ez az ár ott normálisnak hangzik, de a megközelítés télen kérdéses."
  - "Ennél a kerületnél a parkolás és a társasházi rend a két leggyakoribb meglepetés."
  - "A budai oldalon ennyi m²-ért általában picit drágább szokott lenni — itt érdemes telefonon kérdezni, miért ennyi."

- **factual** — Tárgyilagos, számvezetett, érzelemmentes. Példamondatok:
  - "A számok ezt mutatják: 65 m², 3 szoba, 230 ezer Ft / m². A kerületi átlagnál picit jobb."
  - "Az ár-érték arány tisztább, mint a többi opciónál — de a felújítási állapot bizonytalan."
  - "Az alapadatok stimmelnek a keresésedhez, a hirdetés viszont kevés extra részletet ad."

## KONKRÉT INGATLANOS-MONDATOK (STÍLUS-IRÁNY — kerüld a sablonokat)

Vegyél innen ihletet, ha kell — de NE másold szóról szóra, a konkrét tényekhez illeszd:

- "A liftet a házban kérdezd meg, nem a hirdetésben."
- "Ez a kerületben még olcsónak számít, de telefon után dől el."
- "Az alaprajzot kérd el — itt ennyi m²-en könnyen rossz az elosztás."
- "A közös költség 25e Ft fölött gyanús egy 60 m²-es lakásnál."
- "Telefonos beszélgetés után dől el, marad-e benne valódi érték."
- "Belváros + 60 m² + 80M alatt — papíron jól néz ki, de a társasházi rendet nézd meg."
- "Ha 30 m² alatt vagy és nincs erkély, a tárolóhelyek terve mindent eldönt."
- "A panoráma-kontextus a kulcs: mit építenek a következő évben mellé."
- "Felújítandó ingatlannál a 'vételár + felújítás' együtt számít, és sokszor olcsóbb látszik mint amennyibe igazából kerül."

## KERÜLENDŐ SABLON-MONDATOK

- "Ez az ingatlan érdekes lehet számodra" — semmitmondó, ne használd.
- "Érdemes alaposabban megnézni" — konkrét útmutatás nélkül üres.
- "A keresésed alapján..." — kerüld a kezdést, főleg a body első mondatában.
- "Tökéletes választás lehet" — sosem tökéletes, légy őszinte.
- "Mindkét fél számára előnyös" — szövegmező-töltelék.
- "Egyensúlyt teremt..." — érzelemmentes filozófia, kerüld.

KÖTELEZŐEN érvényes JSON, semmi prózai bevezetés, semmi Markdown.
"""

# The narrative is free prose, so the runtime DOM translator on the web app cannot
# localise it — only the model can. The tone and schema instructions above stay in
# Hungarian (they are tuned there); this directive changes the OUTPUT language only.
# Kept as a module-level alias so existing imports keep working; the table itself
# lives in `output_language` so every prose prompt pins the same set of locales.
NARRATOR_LANGUAGES: dict[str, str] = OUTPUT_LANGUAGES

DEFAULT_NARRATOR_LOCALE = DEFAULT_OUTPUT_LOCALE


def narrator_system_prompt(locale: str = DEFAULT_NARRATOR_LOCALE) -> str:
    """System prompt with the output language pinned to `locale`.

    Unknown locales fall back to Hungarian rather than raising: a narrative in the
    source language is a far better failure mode than a 500 on the search path.
    """
    language = NARRATOR_LANGUAGES.get(locale)
    if language is None or locale == DEFAULT_NARRATOR_LOCALE:
        return NARRATOR_SYSTEM_PROMPT
    upper = language.upper()
    # Leading AND trailing, for the same reason as `localized_rerank_prompt`: the
    # prompt body is Hungarian, so one trailing line is not enough to move the
    # output language.
    return (
        f"KIMENETI NYELV: a JSON minden szöveges mezőjét KIZÁRÓLAG {upper} nyelven írd."
        f" Ez felülír minden lenti példát; a kulcsnevek és a séma változatlanok.\n\n"
        f"{NARRATOR_SYSTEM_PROMPT}\n\n"
        f"EMLÉKEZTETŐ: minden szöveges mező {upper} nyelvű legyen, nem magyar."
    )


def _render_listing_brief(listing: Any) -> str:
    """Compact, structured listing facts the prompt consumes verbatim."""
    bits = _listing_identity_bits(listing) + _listing_fact_bits(listing)
    return "\n".join(bits)


def _listing_identity_bits(listing: Any) -> list[str]:
    """Title, type, transaction and place lines of the listing brief, in prompt order."""
    bits: list[str] = []
    if listing.title:
        bits.append(f"Cím: {listing.title}")
    if listing.property_type:
        bits.append(f"Típus: {listing.property_type}")
    if listing.transaction_type:
        bits.append("Tranzakció: Eladó" if listing.transaction_type == "sale" else "Tranzakció: Kiadó")
    loc = str(listing.city or "")
    if listing.district:
        loc += f", {listing.district}. kerület"
    if loc:
        bits.append(f"Helyszín: {loc}")
    return bits


def _listing_fact_bits(listing: Any) -> list[str]:
    """Size, rooms, price, year, condition and description lines of the listing brief."""
    bits: list[str] = []
    if listing.area_m2:
        bits.append(f"Méret: {listing.area_m2:.0f} m²")
    if listing.rooms:
        bits.append(f"Szobaszám: {listing.rooms}")
    price_huf = resolve_property_price_huf(listing)
    if price_huf is not None:
        bits.append(f"Ár: {format_price_short(price_huf, is_rent=False)}")
    if listing.year_built:
        bits.append(f"Építés éve: {listing.year_built}")
    if listing.renovation_state:
        bits.append(f"Állapot: {listing.renovation_state}")
    if listing.description:
        desc = str(listing.description).strip().replace("\n", " ")
        if len(desc) > 240:
            desc = desc[:240] + "…"
        bits.append(f"Leírás: {desc}")
    return bits


def _render_query_brief(query: ParsedQuery) -> str:
    """Compact summary of the user's parsed query intent."""
    bits = _query_scope_bits(query) + _query_bound_bits(query) + _query_preference_bits(query)
    return "\n".join(bits) if bits else "(nincs explicit kritérium — szabad keresés)"


def _query_scope_bits(query: ParsedQuery) -> list[str]:
    """Transaction, type and location lines of the query brief, in prompt order."""
    hf = query.hard_filters
    gh = query.geo_hints
    bits: list[str] = []
    if hf.transaction_type:
        bits.append("Tranzakció: " + ("Eladó" if hf.transaction_type == "sale" else "Kiadó"))
    if hf.property_type_in:
        bits.append("Típus: " + ", ".join(hf.property_type_in))
    if hf.city:
        bits.append(f"Város: {hf.city}")
    if hf.district_in:
        bits.append("Kerület: " + ", ".join(hf.district_in))
    if gh.county:
        bits.append(f"Vármegye: {gh.county}")
    if gh.balaton_shore:
        bits.append(f"Balaton-part: {gh.balaton_shore}")
    if gh.side_of_river:
        bits.append(f"Duna-oldal: {gh.side_of_river}")
    if gh.region_name:
        bits.append(f"Régió/környék: {gh.region_name}")
    return bits


def _query_bound_bits(query: ParsedQuery) -> list[str]:
    """Room, size and price bounds of the query brief, in prompt order."""
    hf = query.hard_filters
    bits: list[str] = []
    if hf.rooms is not None:
        bits.append(f"Szobaszám: {hf.rooms} (egzakt)")
    if hf.rooms_min is not None:
        bits.append(f"Szobaszám: min. {hf.rooms_min}")
    if hf.area_m2_min is not None:
        bits.append(f"Méret: min. {hf.area_m2_min:.0f} m²")
    if hf.area_m2_max is not None:
        bits.append(f"Méret: max. {hf.area_m2_max:.0f} m²")
    if hf.price_huf_max is not None:
        bits.append(f"Max ár: {_format_query_price(hf.price_huf_max)}")
    if hf.price_huf_min is not None:
        bits.append(f"Min ár: {_format_query_price(hf.price_huf_min)}")
    return bits


def _query_preference_bits(query: ParsedQuery) -> list[str]:
    """Soft-preference lines (tags, then free text) of the query brief."""
    bits: list[str] = []
    if query.soft_preferences:
        bits.append("Puha preferenciák (címkék): " + ", ".join(query.soft_preferences))
    if query.soft_prefs_freetext:
        bits.append("Puha preferenciák (szabad szöveg): " + " | ".join(query.soft_prefs_freetext))
    return bits


def _format_query_price(price_huf: int) -> str:
    """Keep query price labels exact when a parsed bound exceeds Number range."""

    if price_huf > MAX_LEGACY_SAFE_INTEGER:
        return f"{price_huf:,}".replace(",", " ") + " Ft"
    return f"{price_huf / 1_000_000:.0f} M Ft"


def _render_breakdown_brief(breakdown: ScoreBreakdown) -> str:
    """Score breakdown for the prompt — formatted as relative strength hints."""
    parts: list[str] = []
    for term, value in breakdown.model_dump().items():
        if term == "total":
            continue
        if value > 0:
            parts.append(f"{term}: {value:.3f}")
    return ", ".join(parts) if parts else "(minden komponens 0)"


type NarratorMood = Literal["enthusiastic", "cautious", "experienced", "factual"]


def _derive_mood(breakdown: ScoreBreakdown) -> NarratorMood:
    """Pick the AI agent's tone for this listing based on score signals.

    Drives the narrator's voice so the body text feels less
    templated across results. Four moods:

    - enthusiastic: strong total + good fit (the agent is genuinely sold)
    - experienced: location-heavy but weak fit (the agent leans on
      district expertise to qualify the listing)
    - cautious: low overall score or weak fit (the agent flags
      compromises early)
    - factual: everything else (neutral baseline)

    The thresholds are intentionally generous on "factual" — this is the
    safe default when the signal isn't clearly one of the others.
    """
    total = breakdown.total
    fit = breakdown.content + breakdown.factual + breakdown.softpref_core + breakdown.softpref_free
    location = breakdown.geo + breakdown.pooled

    if total >= 0.80 and fit >= 0.40:
        return "enthusiastic"
    if location >= 0.30 and fit < 0.35:
        return "experienced"
    if total < 0.55 or fit < 0.25:
        return "cautious"
    return "factual"


def render_user_prompt(listing: Any, query: ParsedQuery, breakdown: ScoreBreakdown) -> str:
    """Glue the listing + query + breakdown into the narrator's user message.

    Includes the derived mood — the system prompt explains the four moods
    and the model is required to match the tone in every sentence it writes.
    """
    mood = _derive_mood(breakdown)
    return (
        f"HANGULAT (kötelező): {mood}\n\n"
        "USER QUERY ÉRTELMEZÉSE:\n"
        f"{_render_query_brief(query)}\n\n"
        "AJÁNLOTT INGATLAN:\n"
        f"{_render_listing_brief(listing)}\n\n"
        "BELSŐ PONTSZÁM-KOMPONENSEK (NE említsd a userhez):\n"
        f"{_render_breakdown_brief(breakdown)}\n\n"
        f"Generáld a narratíva JSON-t a megadott séma szerint, a '{mood}' hangulatot következetesen tartva."
    )
