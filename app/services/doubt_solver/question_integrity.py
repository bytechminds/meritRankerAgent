"""Conditional normalization of structurally malformed student questions.

A deterministic integrity gate looks only for high-confidence formatting damage
(formula/encoding debris, broken math delimiters, equation fragments). A clean question
never reaches the normalizer. A flagged question gets one bounded structured call that
may only restructure it; a deterministic guard rejects any output that changes numbers,
option labels, units, operators, comparisons, or negations, and a repair that is still
structurally incomplete is treated as missing information.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from observability import log_event
from schemas.doubt_solver import CanonicalLanguage
from schemas.llm_routing import RouteRequest
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.structured_output import StructuredOutputError, parse_structured_output

logger = logging.getLogger(__name__)

QUESTION_NEEDS_CLARIFICATION = "QUESTION_NEEDS_CLARIFICATION"
_NORMALIZER_PROMPT = "question_normalizer.md"

# Bracket/brace pieces (⎧ ⎨ ⎩ ⎪ …) only appear when a rendered multi-line formula was
# copied as text; the replacement character, private-use code points, and combining
# marks with no base character are encoding debris.
_OCR_DEBRIS = re.compile(r"[⎛-⎳�-]|(?:^|\s)[̀-ͯ]")
_OPTION_LABEL = re.compile(r"\(([a-eA-E])\)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_UNIT = re.compile(
    r"(?<![A-Za-z])(?:km/h|km/hr|m/s|km|cm|mm|kg|mg|ml|sec|min|hrs?|hours?|minutes?|"
    r"seconds?|days?|weeks?|months?|years?|rs|%|₹|°)(?![A-Za-z])",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|none|neither|nor|except|without|cannot|can't|isn't|doesn't|"
    r"don't|won't)\b|नहीं",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:greater|less|more|fewer|larger|smaller|higher|lower|maximum|minimum|most|least|"
    r"older|younger|faster|slower|above|below|increased?|decreased?)\b",
    re.IGNORECASE,
)
_OPERATOR = re.compile(r"[+\-−–=<>*×/÷^]")
_OPERATOR_ALIASES = str.maketrans({"−": "-", "–": "-", "×": "*", "÷": "/"})
# LaTeX delimiters not preceded by another backslash (so `\\[2pt]` row spacing is not one).
_OPEN_INLINE = re.compile(r"(?<!\\)\\\(")
_CLOSE_INLINE = re.compile(r"(?<!\\)\\\)")
_OPEN_DISPLAY = re.compile(r"(?<!\\)\\\[")
_CLOSE_DISPLAY = re.compile(r"(?<!\\)\\\]")
_OPERATOR_ONLY_LINE = re.compile(r"^[=+\-−×÷*/^]{1,3}$")
# A repair that still ends a line on an operator, puts an operator straight before "=",
# or ends on a bare ask ("find") is missing data the model claimed was recoverable.
_DANGLING_STRUCTURE = re.compile(
    r"[=+\-−×÷*/^]\s*$|[+\-−×÷*/^]\s*[=+×÷*/^]|"
    r"[=<>≤≥≠+\-−×÷*/^]\s*(?:\.{3,}|…|_{2,})|"
    r"\([a-eA-E]\)\s*(?=\([a-eA-E]\)|$)|"
    r"\b(?:find|calculate|determine|evaluate|solve|what is)\s*[.:]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Semantic-safety guard: a repair may change spacing, line breaks, punctuation, and
# operator glyphs, never which words and numbers appear or their order. Order carries
# meaning a multiset cannot see: which quantity belongs to which entity, which option
# holds which value, and every direction, multiplicity, or ordering word. Devanagari vowel
# signs are not word characters to `\w`, so they are listed to keep Hindi words whole.
_WORD_OR_NUMBER = re.compile(r"\d+(?:[.,]\d+)*|(?:[^\W\d_]|[\u0300-\u036f\u0900-\u0963])+")
_GUARD_OPERATOR = re.compile(r"[+\-−–=<>≤≥≠*×/÷^]")
_PLACEHOLDER = re.compile(r"\?|\.{3,}|…|_{2,}")
# Layout guard. Curly-brace pieces of a rendered equation system and LaTeX math delimiters
# carry no meaning and are dropped. Every other character — letter case, vowel signs,
# brackets, √ ! | ' : and all math symbols — must survive in order, and only the break
# between two characters may change:
# - a space may vanish (not beside a "." or "," that is part of a value, so "Rs .50" stays)
#   or become a line break; a space may be added only beside an operator, never inside a
#   compound operator ("5 != x" is not "5! = x");
# - a line break or a phrase-ending . , ; (followed by a break, not before a digit, sign,
#   or punctuation, so ".5", "5, -3", "1,500" stay characters) may change form but not
#   vanish, because "x = 2, y = 3" and "x = 2y = 3" differ; nothing may become punctuation;
# - a line break may be dropped after an operator, "(" or option label, before = ≤ ≥ ≠ ^ / ),
#   or before + - * < > standing alone on their line, since "-x + y = 2", "* y = 3", or
#   "> y" starting a line can begin a new equation;
# - the repair may end a lead-in line with a colon ("equations:") before a line that
#   starts with a digit, sign, or "("; a colon in the student's text is always kept.
_LAYOUT_DEBRIS = re.compile(r"[⎧-⎭]|\\[()\[\]]")
_LINE_CHARS = r"\n\r\v\f\x1c-\x1e\x85  "
_LAYOUT_TOKEN = re.compile(
    r"(?P<phrase>[.,;](?=\s|$)(?!\s*[\d+\-.,;:]))"
    rf"|(?P<line>[{_LINE_CHARS}])"
    r"|(?P<space>\s)"
    r"|(?P<char>.)",
    re.DOTALL,
)
_LEAD_IN_COLON = re.compile(rf"(?<=[^\W\d_]{{2}}[^\W\d_A-Z]):(?=[ \t]*[{_LINE_CHARS}]\s*[\d(+\-])")
_NO_BREAK, _SPACE_BREAK, _LINE_BREAK, _PHRASE_BREAK = 0, 1, 2, 3
_LAYOUT_OPERATORS = frozenset("+-*/=<>≤≥≠^")
_COMPOUND_OPERATORS = frozenset({"!=", "<=", ">=", "==", "**", "->", "=>", "<>", ":="})
_VALUE_PUNCTUATION = frozenset(".,")
_BREAK_ABSORBING_LEFT = re.compile(r"(?:[+\-*/=<>≤≥≠^(]|(?<![^\W\d_])\([a-eA-E]\))$")
_BREAK_ABSORBING_RIGHT = frozenset("=≤≥≠^/)")
_LINE_LEADING_MARKERS = frozenset("+-*<>")
_MIN_FRAGMENT_TOKENS = 25
_FRAGMENT_TOKEN_RATIO = 0.7
_MIN_FRAGMENT_OPERATORS = 3
_MIN_FRAGMENT_LINES = 4
_MIN_OPERATOR_ONLY_LINES = 2


def assess_question_integrity(query: str) -> tuple[str, ...]:
    """Return high-confidence formatting-damage signals; empty means a clean question.

    Each signal must be rare in well-formed questions: option or label order, vertical
    option lists, and letter series are ordinary, so they never flag on their own.
    """
    signals: list[str] = []
    if _OCR_DEBRIS.search(query):
        signals.append("ocr_symbol_corruption")
    if len(_OPEN_INLINE.findall(query)) != len(_CLOSE_INLINE.findall(query)) or len(
        _OPEN_DISPLAY.findall(query)
    ) != len(_CLOSE_DISPLAY.findall(query)):
        signals.append("unbalanced_math_delimiters")
    short_lines = [
        line.strip() for line in query.splitlines() if 0 < len(line.strip()) <= 3
    ]
    if (
        len(short_lines) >= _MIN_FRAGMENT_LINES
        and sum(1 for line in short_lines if _OPERATOR_ONLY_LINE.match(line))
        >= _MIN_OPERATOR_ONLY_LINES
        and any(re.search(r"[0-9A-Za-z]", line) for line in short_lines)
    ):
        signals.append("fragmented_equation_lines")
    tokens = query.split()
    if (
        len(tokens) >= _MIN_FRAGMENT_TOKENS
        and sum(1 for token in tokens if len(token) == 1) / len(tokens) >= _FRAGMENT_TOKEN_RATIO
        and sum(1 for token in tokens if _OPERATOR.fullmatch(token)) >= _MIN_FRAGMENT_OPERATORS
    ):
        signals.append("excessive_isolated_fragments")
    return tuple(signals)


class _NormalizerOutput(BaseModel):
    status: Literal["RECOVERABLE_FORMATTING", "AMBIGUOUS", "MISSING_INFORMATION"]
    normalized_question: str | None = Field(default=None)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}


NormalizationOutcome = Literal[
    "normalized",
    "needs_clarification",
    "guard_rejected",
    "unavailable",
]


@dataclass(frozen=True)
class QuestionNormalization:
    outcome: NormalizationOutcome
    normalized_question: str | None = None


def _counts(pattern: re.Pattern[str], text: str) -> Counter[str]:
    return Counter(match.lower().translate(_OPERATOR_ALIASES) for match in pattern.findall(text))


def _sequence(pattern: re.Pattern[str], text: str) -> list[str]:
    return [match.lower().translate(_OPERATOR_ALIASES) for match in pattern.findall(text)]


def _layout(text: str, *, lead_in_colon: bool = False) -> tuple[str, list[int]]:
    """Return the meaningful characters and the break strength between each adjacent pair."""
    text = _LAYOUT_DEBRIS.sub("", unicodedata.normalize("NFC", text).translate(_OPERATOR_ALIASES))
    if lead_in_colon:
        text = _LEAD_IN_COLON.sub("\n", text)
    chars: list[str] = []
    breaks: list[int] = []
    pending = _NO_BREAK
    for match in _LAYOUT_TOKEN.finditer(text):
        if match.lastgroup == "char":
            if chars:
                breaks.append(pending)
            chars.append(match.group())
            pending = _NO_BREAK
        elif match.lastgroup == "space":
            pending = max(pending, _SPACE_BREAK)
        elif match.lastgroup == "line":
            pending = max(pending, _LINE_BREAK)
        elif match.lastgroup == "phrase":
            pending = _PHRASE_BREAK
    return "".join(chars), breaks


def _preserves_layout(original: str, normalized: str) -> bool:
    chars, original_breaks = _layout(original)
    normalized_chars, normalized_breaks = _layout(normalized)
    if normalized_chars != chars:
        normalized_chars, normalized_breaks = _layout(normalized, lead_in_colon=True)
        if normalized_chars != chars:
            return False
    for index, (before, after) in enumerate(zip(original_breaks, normalized_breaks, strict=True)):
        left, right = chars[index], chars[index + 1]
        if before == _NO_BREAK and after == _SPACE_BREAK:
            if (
                left not in _LAYOUT_OPERATORS and right not in _LAYOUT_OPERATORS
            ) or left + right in _COMPOUND_OPERATORS:
                return False
            continue
        if before == _SPACE_BREAK and after == _NO_BREAK:
            if left in _VALUE_PUNCTUATION or right in _VALUE_PUNCTUATION:
                return False
            continue
        if (
            _BREAK_ABSORBING_LEFT.search(chars, 0, index + 1)
            or right in _BREAK_ABSORBING_RIGHT
            or (
                right in _LINE_LEADING_MARKERS
                and index + 1 < len(original_breaks)
                and original_breaks[index + 1] >= _LINE_BREAK
            )
        ):
            continue
        if after == _PHRASE_BREAK and before <= _SPACE_BREAK:
            return False
        if after == _LINE_BREAK and before == _NO_BREAK:
            return False
        if before >= _LINE_BREAK and after <= _SPACE_BREAK:
            return False
    return True


def _preserves_facts(original: str, normalized: str) -> bool:
    """Restructuring may not add, drop, reorder, or change a word, number, option, unit,
    operator, comparison, or negation, and may not add a placeholder. Anything the guard
    cannot prove unchanged is treated as changed."""
    return (
        _preserves_layout(original, normalized)
        and _counts(_NUMBER, original) == _counts(_NUMBER, normalized)
        and set(_counts(_OPTION_LABEL, original)) == set(_counts(_OPTION_LABEL, normalized))
        and _counts(_UNIT, original) == _counts(_UNIT, normalized)
        and _counts(_OPERATOR, original) == _counts(_OPERATOR, normalized)
        and _counts(_COMPARISON, original) == _counts(_COMPARISON, normalized)
        and _counts(_NEGATION, original) == _counts(_NEGATION, normalized)
        and _sequence(_WORD_OR_NUMBER, original) == _sequence(_WORD_OR_NUMBER, normalized)
        and _sequence(_GUARD_OPERATOR, original) == _sequence(_GUARD_OPERATOR, normalized)
        and len(_PLACEHOLDER.findall(normalized)) <= len(_PLACEHOLDER.findall(original))
    )


def compose_normalized_query(original: str, normalized: str) -> str:
    """Keep the student's text verbatim and add the restructured reading after it."""
    return (
        f"{original}\n\n"
        "[Same question with its formatting repaired — no information added or removed]\n"
        f"{normalized}"
    )


def clarification_message(language: CanonicalLanguage = "english") -> str:
    if language == "hindi":
        return (
            "यह प्रश्न अधूरा या अस्पष्ट लग रहा है, इसलिए इसे भरोसे के साथ हल नहीं किया जा सकता। "
            "कृपया पूरा प्रश्न फिर से टाइप करें या साफ़ इमेज अपलोड करें।"
        )
    if language == "hinglish":
        return (
            "Yeh question adhoora ya unclear lag raha hai, isliye ise reliably solve nahi kiya "
            "ja sakta. Please poora question dobara type karein ya clear image upload karein."
        )
    return (
        "This question looks incomplete or garbled, so it can't be solved reliably. "
        "Please re-type the full question or re-upload a clear image."
    )


def ambiguous_question_message(
    language: CanonicalLanguage = "english", *, multiple_answers: bool = False
) -> str:
    """Ask for what is missing when a well-formed question still cannot be pinned down.

    Separate from `clarification_message`, which addresses text that arrived garbled.
    """
    if multiple_answers:
        if language == "hindi":
            return (
                "इस प्रश्न के एक से अधिक सही उत्तर बन सकते हैं, इसलिए एक निश्चित उत्तर नहीं दिया जा सकता। "
                "कृपया पूरा प्रश्न सभी शर्तों और विकल्पों के साथ भेजें।"
            )
        if language == "hinglish":
            return (
                "Is question ke ek se zyada sahi jawab ban sakte hain, isliye ek final answer "
                "nahi diya ja sakta. Please poora question saari conditions aur options ke "
                "saath bhejein."
            )
        return (
            "This question can be read in more than one way, so a single answer can't be given. "
            "Please send the complete question with all conditions and options."
        )
    if language == "hindi":
        return (
            "इस प्रश्न में कुछ जानकारी छूट रही है, इसलिए इसका उत्तर नहीं दिया जा सकता। "
            "कृपया सभी दी गई मात्राएँ, शर्तें और विकल्प भेजें।"
        )
    if language == "hinglish":
        return (
            "Is question mein kuch information missing hai, isliye answer nahi diya ja sakta. "
            "Please saari di gayi values, conditions aur options bhejein."
        )
    return (
        "Some information is missing from this question, so it can't be answered. "
        "Please send all the given values, conditions and options."
    )


class QuestionNormalizer:
    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def normalize(
        self,
        *,
        request_id: str,
        query: str,
        language: CanonicalLanguage,
        signals: tuple[str, ...],
    ) -> QuestionNormalization:
        started = time.monotonic()
        outcome: NormalizationOutcome
        normalized: str | None = None
        status: str | None = None
        try:
            result = self._orchestrator.generate_structured(
                route_request=RouteRequest(
                    request_id=request_id,
                    subject="general",
                    task_role="classifier_strong",
                    difficulty="default",
                    language=language,
                ),
                user_content=query,
                prompt=_NORMALIZER_PROMPT,
            )
            parsed = parse_structured_output(result.content, _NormalizerOutput)
        except StructuredOutputError as exc:
            outcome = "unavailable"
            failure = exc.diagnostic.failure_kind
        except Exception as exc:  # noqa: BLE001 - normalization never blocks the existing path
            outcome = "unavailable"
            failure = type(exc).__name__
        else:
            failure = None
            status = parsed.status
            if parsed.status != "RECOVERABLE_FORMATTING":
                outcome = "needs_clarification"
            elif not parsed.normalized_question or not _preserves_facts(
                query, parsed.normalized_question
            ):
                outcome = "guard_rejected"
            elif _DANGLING_STRUCTURE.search(parsed.normalized_question):
                outcome = "needs_clarification"
            else:
                outcome = "normalized"
                normalized = parsed.normalized_question
        log_event(
            "QUESTION_NORMALIZATION_DECISION",
            component="doubt_solver.question_integrity",
            stage="understanding",
            status=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            details={
                "signals": ",".join(signals),
                "normalizer_status": status,
                "failure": failure,
            },
            level=logging.WARNING if outcome in {"guard_rejected", "unavailable"} else logging.INFO,
        )
        return QuestionNormalization(outcome=outcome, normalized_question=normalized)
