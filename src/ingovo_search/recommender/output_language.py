"""Output language for model-written prose.

Every prompt in the recommender is authored in Hungarian, because Hungarian is
the source language of the product and of the listing data. The *reader*,
however, may be on any of the ten site locales, and free prose (search
rationales, hero narratives, the listing analysis, the chat answers) cannot be
localised by the runtime DOM translator the way UI labels are — the model has to
write it in the reader's language in the first place.

This module is the single place that knows which locales that covers and how the
directive is phrased. `narrator.py`, `rerank_prompt.py` and `ai_prompt.py` all
pin their output language through it.

Unknown locales fall back to Hungarian rather than raising: prose in the source
language is a far better failure mode than a 500 on the search path.
"""

from __future__ import annotations

from typing import Final

#: Site locale → the Hungarian name of that language, as used inside the prompts.
OUTPUT_LANGUAGES: Final[dict[str, str]] = {
    "hu": "magyar",
    "en": "angol",
    "de": "német",
    "ro": "román",
    "hr": "horvát",
    "sk": "szlovák",
    "sl": "szlovén",
    "cs": "cseh",
    "bg": "bolgár",
    "pl": "lengyel",
}

DEFAULT_OUTPUT_LOCALE: Final = "hu"


def output_language_name(locale: str) -> str | None:
    """The Hungarian name of `locale`'s language, or None when it needs no directive.

    Returns None for Hungarian itself and for anything unknown — both mean "leave
    the prompt as authored".
    """
    if locale == DEFAULT_OUTPUT_LOCALE:
        return None
    return OUTPUT_LANGUAGES.get(locale)


def localized_prose_prompt(prompt: str, locale: str, *, subject: str = "a teljes válaszod") -> str:
    """Wrap a Hungarian system prompt so the model answers in `locale` instead.

    Leading AND trailing, the same way `localized_rerank_prompt` does it: the body
    of every one of these prompts is Hungarian throughout and several of them
    contain an explicit "KIZÁRÓLAG magyarul írj" tone rule, so a single trailing
    line loses the argument and the model keeps answering in Hungarian.
    """
    language = output_language_name(locale)
    if language is None:
        return prompt
    upper = language.upper()
    return (
        f"KIMENETI NYELV: {subject} KIZÁRÓLAG {upper} nyelven írd. Ez FELÜLÍRJA a lenti"
        f" prompt minden magyar nyelvre vonatkozó szabályát és példáját — a szerkezet,"
        f" a formázás és a szakmai tartalom változatlan, csak a nyelv más."
        f" A nyelvválasztást SOHA ne említsd és ne vezesd fel: tilos az olyan bevezető"
        f" mondat, hogy Íme az elemzés, kizárólag németül. Egyszerűen ezen a nyelven"
        f" kezdd a választ.\n\n"
        f"{prompt}\n\n"
        f"EMLÉKEZTETŐ: {subject} {upper} nyelvű legyen, nem magyar. A magyar helynevek,"
        f" utcanevek és a hirdetés saját címe maradjon az eredeti alakjában."
    )
