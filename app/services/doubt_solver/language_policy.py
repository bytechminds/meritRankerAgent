"""Compact deterministic response-language policy and compliance checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.doubt_solver import CanonicalLanguage

_POLICIES: dict[CanonicalLanguage, str] = {
    "english": "",
    "hinglish": (
        "Answer natural, simple Hinglish mein dein using Roman script only. Common exam "
        "aur technical English terms, formulas, equations, numbers, option labels, "
        "official names, source titles, citations aur URLs ko exact preserve karein. "
        "Explanation student-friendly Hindi-English mix mein rakhein; Devanagari aur "
        "overly formal English use na karein."
    ),
    "hindi": (
        "उत्तर सरल, स्वाभाविक और स्पष्ट हिंदी में दें और देवनागरी प्रयोग करें। परीक्षा के "
        "सामान्य अंग्रेज़ी संक्षेप, तकनीकी शब्द, सूत्र, समीकरण, संख्याएँ, विकल्प लेबल, "
        "आधिकारिक नाम, source titles, citations और URLs सटीक रखें। कठिन हिंदी और अनावश्यक "
        "पूरे अंग्रेज़ी वाक्यों से बचें। केवल सूत्र/संख्या हो तो भी एक छोटा हिंदी वाक्य जोड़ें।"
    ),
}

_PROTECTED_CONTENT = re.compile(
    r"```.*?```|`[^`]*`|https?://\S+|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_DEVANAGARI_WORD = re.compile(r"[\u0900-\u097f]+")
_HINGLISH_MARKERS = frozenset(
    {
        "aap",
        "aapka",
        "aapko",
        "agar",
        "ab",
        "aur",
        "bas",
        "hai",
        "hain",
        "hoga",
        "hogi",
        "isliye",
        "iska",
        "ismein",
        "ka",
        "kar",
        "karke",
        "karna",
        "karne",
        "ke",
        "ki",
        "ko",
        "liye",
        "mein",
        "nahi",
        "phir",
        "rakhein",
        "samjhein",
        "se",
        "simple",
        "to",
        "wala",
        "yahan",
        "ye",
        "yeh",
    }
)


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
    """Detect high-confidence selected-language violations without parsing formulas."""
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
    if language == "english":
        if script_letters < 12:
            return True
        return devanagari < 12
    if language == "hinglish":
        if devanagari >= 2:
            return False
        latin_words = [word.lower() for word in _LATIN_WORD.findall(prose)]
        if len(latin_words) < 12:
            return True
        return any(word in _HINGLISH_MARKERS for word in latin_words)

    latin_words = len(_LATIN_WORD.findall(prose))
    devanagari_words = len(_DEVANAGARI_WORD.findall(prose))
    if devanagari < 4:
        return script_letters == 0
    return not (latin_words >= 8 and devanagari_words <= 3)


def language_policy_char_count(language: CanonicalLanguage) -> int:
    """Return deterministic prompt overhead for observability and tests."""
    return len(_POLICIES[language])
