"""Compact deterministic response-language policy and compliance checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.doubt_solver import CanonicalLanguage

_POLICIES: dict[CanonicalLanguage, str] = {
    "english": (
        "Respond in clear, natural English suitable for competitive-exam preparation."
    ),
    "hinglish": (
        "Respond in natural Latin-script Hinglish. Keep familiar academic, technical "
        "and mathematical terms in English. Preserve formulas, variables, numerals, "
        "units and option labels exactly."
    ),
    "hindi": (
        "Respond in simple, natural modern Hindi using Devanagari. Preserve formulas, "
        "variables, numerals, units, proper nouns and option labels. Use familiar English "
        "technical terms when forced Hindi would reduce clarity. Avoid literary Hindi."
    ),
}

_PROTECTED_CONTENT = re.compile(
    r"```.*?```|`[^`]*`|https?://\S+|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
_LATIN_WORD = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class ResolvedLanguagePolicy:
    language: CanonicalLanguage
    instruction: str


class LanguagePolicyResolver:
    """Resolve one canonical language to one immutable compact instruction."""

    def resolve(self, language: CanonicalLanguage) -> ResolvedLanguagePolicy:
        try:
            instruction = _POLICIES[language]
        except KeyError as exc:
            raise ValueError(f"Unsupported canonical language: {language!r}") from exc
        return ResolvedLanguagePolicy(language=language, instruction=instruction)


def is_language_compliant(content: str, language: CanonicalLanguage) -> bool:
    """Detect only obvious selected-language script violations."""
    if not content or not content.strip():
        return False

    prose = _PROTECTED_CONTENT.sub(" ", content)
    latin = 0
    devanagari = 0
    other = 0
    for char in prose:
        if not char.isalpha():
            continue
        if "A" <= char <= "Z" or "a" <= char <= "z":
            latin += 1
        elif "\u0900" <= char <= "\u097f":
            devanagari += 1
        else:
            other += 1

    script_letters = latin + devanagari + other
    if script_letters < 12:
        return True
    if language == "english":
        unexpected = devanagari + other
        return unexpected < 12 or unexpected / script_letters < 0.70
    if language == "hinglish":
        return devanagari < 12 or devanagari / script_letters < 0.60

    latin_words = len(_LATIN_WORD.findall(prose))
    obvious_latin_prose = (
        latin >= 24
        and latin_words >= 4
        and latin / script_letters >= 0.85
        and devanagari < 8
    )
    return not obvious_latin_prose


def language_policy_char_count(language: CanonicalLanguage) -> int:
    """Return deterministic prompt overhead for observability and tests."""
    return len(_POLICIES[language])
