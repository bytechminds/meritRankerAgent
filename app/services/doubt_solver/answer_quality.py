"""Deterministic generator answer validation, sanitization, and rewrite support."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from config import Settings, get_settings
from observability import log_event, update_request_summary
from schemas.doubt_solver import CanonicalLanguage
from schemas.llm import LlmMessage
from services.doubt_solver.language_policy import is_language_compliant
from services.doubt_solver.markdown_replay import iter_markdown_segments

logger = logging.getLogger(__name__)

Severity = Literal["clean", "minor", "rewrite_required", "unsafe", "error"]

GENERATION_FAILURE_MESSAGE = (
    "I could not generate a reliable answer for this question. Please try again."
)

REWRITE_USER_PROMPT = (
    "Rewrite the answer into the required compact format. Keep only the final clean "
    "solution with one consistent answer. Preserve normal prose spacing and every "
    "number, unit, punctuation mark, and math, statistics, or chemistry symbol. Do not "
    "show failed attempts. Follow every original system instruction, including the "
    "requested response language and script. Use valid Markdown. Use \\(...\\) and "
    "\\[...\\] only for math. Do not use $ or $$. "
    "End with <ANSWER_DONE>."
)

_BAD_PHRASES: tuple[str, ...] = (
    "contradiction",
    "impossible",
    "check the setup",
    "re-express carefully",
    "reconsider approach",
    "actually",
    "close enough",
    "approximately confirms",
    "continuing from",
    "slight difference due to rounding",
    "failed attempt",
)

_RAW_HTML_PATTERN = re.compile(
    r"<\s*(script|iframe|object|embed|style|link|meta|form|input|button)\b",
    re.IGNORECASE,
)
_HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>\n]*>")
_DOLLAR_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_DOLLAR_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_QUAD_DOLLAR = re.compile(r"\${4,}")
_ANSWER_HEADING_LINE = re.compile(
    r"(?im)^\s*\*{0,2}\s*(?P<label>(?:final\s+)?answer)\s*:?\s*\*{0,2}"
    r"\s*(?P<value>[^\n]*)$"
)
_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_JOINED_MONTH_DATE = re.compile(
    r"(?i)\b(?:"
    r"(?:on|from|since|until)\d{1,2}(?:st|nd|rd|th)?(?:january|february|march|"
    r"april|may|june|july|august|september|october|november|december)(?:\d{2,4})?"
    r"|"
    r"\d{1,2}(?:st|nd|rd|th)?(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)(?:\d{2,4})?"
    r"|(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\d{2,4}"
    r")\b"
)
_JOINED_PROSE_NUMBER = re.compile(
    r"(?i)\b(?:on|from|since|until|article|section|rule|chapter|option)\d+\b"
)
_FENCE_LINE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_NUMBERED_STEP = re.compile(r"(?m)^\s*\d+\.\s+")
_DISPLAY_MATH = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_INCOMPLETE_ENDINGS = re.compile(
    r"(?i)\b(calculate|therefore|actually|let's|lets)\s*[.:]?\s*$"
)
_ANSWER_ONLY_REQUEST = re.compile(
    r"\b(?:only\s+(?:the\s+)?answer|just\s+(?:the\s+)?answer|"
    r"answer\s+only|no\s+(?:solution|steps?|explanation)|one[- ]word\s+answer)\b",
    re.IGNORECASE,
)
_REASONING_MARKER = re.compile(
    r"(?:[=+\-*/×÷]|\\(?:frac|times|div|sqrt)|"
    r"\b(?:because|since|therefore|thus|using|formula|substitut|"
    r"let|given|condition|case|step)\b)",
    re.IGNORECASE,
)

_MAX_MATH_LINE_CHARS_DEFAULT = 300


@dataclass(frozen=True)
class AnswerQualityPolicy:
    validation_enabled: bool
    rewrite_enabled: bool
    max_rewrite_attempts: int
    math_intermediate_max_chars: int
    max_visible_steps: int
    max_display_math_blocks: int
    max_math_line_chars: int
    completion_marker: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AnswerQualityPolicy:
        cfg = settings or get_settings()
        return cls(
            validation_enabled=cfg.answer_quality_validation_enabled,
            rewrite_enabled=cfg.answer_quality_rewrite_enabled,
            max_rewrite_attempts=cfg.answer_quality_max_rewrite_attempts,
            math_intermediate_max_chars=cfg.answer_quality_math_intermediate_max_chars,
            max_visible_steps=cfg.answer_quality_max_visible_steps,
            max_display_math_blocks=cfg.answer_quality_max_display_math_blocks,
            max_math_line_chars=cfg.answer_quality_max_math_line_chars,
            completion_marker=cfg.answer_completion_marker,
        )


@dataclass(frozen=True)
class AnswerQualityResult:
    is_valid: bool
    severity: Severity
    reason_codes: list[str]
    sanitized_text: str | None = None
    language_compliant: bool = True


# Findings about how an answer is laid out, not whether its markup, structure,
# language, or safety is sound. After the single rewrite, an answer failing only on
# these may continue to the correctness verifier instead of failing terminally.
PRESENTATION_ONLY_REASON_CODES = frozenset(
    {"too_many_display_math_blocks", "too_many_visible_steps"}
)


def is_presentation_only_failure(result: AnswerQualityResult) -> bool:
    """True when a rejected answer's every reason code is presentation-only."""
    return (
        not result.is_valid
        and result.language_compliant
        and bool(result.reason_codes)
        and set(result.reason_codes) <= PRESENTATION_ONLY_REASON_CODES
    )


def detect_final_answer(content: str) -> bool:
    """Return True when an Answer or Final Answer section is present."""
    if not content or not content.strip():
        return False
    return _ANSWER_HEADING_LINE.search(content) is not None


def count_final_answer_sections(content: str) -> int:
    return len(_ANSWER_HEADING_LINE.findall(content))


# A compact scalar answer: an option letter, or a number with an optional short unit.
# Anything longer is prose.  The distinction matters because an "Answer:" heading is
# routinely used for explanation or an intermediate elimination step
# ("**Answer:** Eliminate options A and B."), and comparing prose against a final value
# is exactly what produced the historical false positives this module guards against.
_SCALAR_ANSWER_VALUE = re.compile(
    r"^\s*(?:option\s+)?"
    r"(?:(?P<option>[A-D])(?![A-Za-z0-9])"
    r"|(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|[A-Za-z/]{1,12})?)"
    r"\s*[.\u2014-]?\s*$",
    re.IGNORECASE,
)


def _scalar_answer(value: str) -> tuple[str, str] | None:
    """Reduce an answer heading to its comparable scalar, or None when it is prose."""
    cleaned = re.sub(r"[*\\()$]", "", value).strip()
    match = _SCALAR_ANSWER_VALUE.match(cleaned)
    if match is None:
        return None
    if match.group("option"):
        return ("option", match.group("option").upper())
    return ("number", match.group("number").replace(",", ""))


# A LaTeX command, or a sub/superscript that is braced or digit-led. A letter-led one is
# excluded because "price_limit" and "_the rate_" are ordinary prose, not math.
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]|[\^_][{0-9]")


def _has_single_dollar_math(content: str) -> bool:
    """Is there a bare `$...$` span the renderer would actually treat as math?

    Currency prose is not math. The renderer requires the opening `$` to be followed by a
    non-space and the closing `$` to be preceded by one, on the same line and unescaped, so
    "The cost is $5 and the price is $12." carries no math span while "$x+1$" does. Asking
    the renderer's own matcher keeps this rule and the renderer from disagreeing.
    """
    for line in content.splitlines():
        dollars = [
            index
            for index, char in enumerate(line)
            if char == "$"
            and not _is_escaped(line, index)
            and not line.startswith("$$", index)
            and not (index and line.startswith("$$", index - 1))
        ]
        if len(dollars) < 2:
            continue
        # A span exists when some opener has some closer after it. The earliest valid
        # opener sees every candidate closer, so one pass settles the line.
        opener = next(
            (i for i in dollars if i + 1 < len(line) and not line[i + 1].isspace()), None
        )
        if opener is not None and any(
            j > opener and not line[j - 1].isspace() for j in dollars
        ):
            return True
        # "$ \frac{d}{t} $" is padded, so the renderer prints it literally rather than
        # typesetting it. The student still sees raw LaTeX, so it is still a defect. Both
        # ends must be padded: a currency amount always binds to its sign ("$5"), so a
        # LaTeX command merely sitting between two prices is not a delimiter pair.
        if any(
            line[first + 1].isspace()
            and line[second - 1].isspace()
            and _LATEX_COMMAND.search(line[first + 1 : second])
            for first, second in zip(dollars, dollars[1:], strict=False)
            if first + 1 < len(line) and second > first + 1
        ):
            return True
    return False


# A repair that returns half the answer is not a reformat. The floor is deliberately loose:
# reflowing math and dropping a duplicated answer section trim a little, not most, of a draft.
_MIN_REPAIR_LENGTH_RATIO = 0.5


def _normalized_answer_value(value: str) -> str:
    """The answer as the student reads it, free of the markup a reformat may change."""
    cleaned = html.unescape(value)
    cleaned = re.sub(r"\\[()\[\]]|[*`$\\]", " ", cleaned)
    cleaned = cleaned.replace("%", " percent ")
    cleaned = re.sub(r"[\s,]+", " ", cleaned).strip()
    # Only trailing punctuation is decoration; a leading "-" is the sign of the answer.
    return cleaned.rstrip(" .;:\u2014-").casefold()


def _answer_values(content: str) -> set[str]:
    values = {
        _normalized_answer_value(match.group("value"))
        for match in _ANSWER_HEADING_LINE.finditer(content)
    }
    return {value for value in values if value}


def preserves_answer_surface(draft: str, candidate: str) -> bool:
    """Does a repaired answer still state the same answer, and still show its work?

    Presentation repair must return the same solution in different formatting, never a
    different one, so every answer the draft states must survive verbatim once markup is
    normalized away — a unit, an option, a ratio or a prose conclusion included, and a
    draft that states two conflicting answers is not resolved here because choosing
    between them is a semantic decision this gate cannot make. The length floor catches
    the rest: a repair that discards most of the draft has stopped reformatting it, even
    when the final answer happens to survive.
    """
    draft_values = _answer_values(draft)
    candidate_values = _answer_values(candidate)
    if _contradicting_answer_surfaces(draft):
        # A draft whose scalar answers disagree is the one case the existing gate asks the
        # rewrite to resolve. It may drop a surface but never introduce a new one. A
        # genuinely multi-part answer is not this shape, and must keep every part.
        if not candidate_values or not candidate_values <= draft_values:
            return False
    elif draft_values and candidate_values != draft_values:
        return False
    stripped_draft = draft.strip()
    if not stripped_draft:
        return True
    return len(candidate.strip()) >= _MIN_REPAIR_LENGTH_RATIO * len(stripped_draft)


def _answer_scalars(content: str) -> set[tuple[str, str]]:
    scalars: set[tuple[str, str]] = set()
    for match in _ANSWER_HEADING_LINE.finditer(content):
        value = match.group("value").strip()
        if not value:
            continue
        scalar = _scalar_answer(value)
        if scalar is not None:
            scalars.add(scalar)
    return scalars


def _contradicting_answer_surfaces(content: str) -> bool:
    """Do two scalar answer surfaces state different values?

    Every answer heading counts, whatever its label. The generator contract writes a
    plain `**Answer:**`, so two of those that disagree are as visible to the student as
    one that disagrees with `**Final Answer:**`; requiring a "final" label let exactly
    that shape pass as clean. Neither surface is treated as the true one: the conflict
    itself is the defect.

    Only compact scalars are compared, and the unit is deliberately ignored so that
    "12 s" and "12 seconds", or "20%" and "20 percent", stay consistent. A part-labelled
    surface such as `**Answer (a):**` is not a compact scalar, so the answers to a
    genuinely multi-part question are never compared with each other.
    """
    return len(_answer_scalars(content)) > 1


def _explicit_final_answer_values(content: str) -> list[str]:
    values: list[str] = []
    lines = content.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_HEADING_LINE.fullmatch(line)
        if match is None or not match.group("label").casefold().startswith("final"):
            continue
        value = match.group("value").strip()
        if not value:
            for following in lines[index + 1 :]:
                candidate = following.strip()
                if candidate:
                    value = candidate
                    break
        if value:
            values.append(re.sub(r"\s+", " ", value).casefold())
    return values


def _mask_spacing_protected_regions(content: str) -> str:
    parts: list[str] = []
    for segment, protected in iter_markdown_segments(content):
        if protected:
            parts.append("".join("\n" if char == "\n" else " " for char in segment))
        else:
            parts.append(segment)
    plain = "".join(parts)
    return _URL_PATTERN.sub(lambda match: " " * len(match.group(0)), plain)


def _has_unclosed_fence(content: str) -> bool:
    opening_char: str | None = None
    opening_length = 0
    for line in content.splitlines():
        match = _FENCE_LINE.match(line)
        if match is None:
            continue
        marker = match.group(1)
        if opening_char is None:
            opening_char = marker[0]
            opening_length = len(marker)
        elif marker[0] == opening_char and len(marker) >= opening_length:
            opening_char = None
            opening_length = 0
    return opening_char is not None


def _has_malformed_markdown_link(content: str) -> bool:
    cursor = 0
    while cursor < len(content):
        bracket_start = content.find("[", cursor)
        if bracket_start < 0:
            return False
        bracket_end = content.find("]", bracket_start + 1)
        if bracket_end < 0:
            return False
        if bracket_end + 1 >= len(content) or content[bracket_end + 1] != "(":
            cursor = bracket_end + 1
            continue
        depth = 1
        link_cursor = bracket_end + 2
        while link_cursor < len(content) and depth:
            if content[link_cursor] == "(" and not _is_escaped(content, link_cursor):
                depth += 1
            elif content[link_cursor] == ")" and not _is_escaped(content, link_cursor):
                depth -= 1
            link_cursor += 1
        if depth:
            return True
        cursor = link_cursor
    return False


def _is_escaped(content: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and content[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def validate_answer_quality(
    content: str,
    *,
    subject: str,
    difficulty: str,
    intent: str | None,
    query: str | None = None,
    language: CanonicalLanguage = "english",
    policy: AnswerQualityPolicy | None = None,
) -> AnswerQualityResult:
    """Run deterministic checks on generator output (before marker strip)."""
    pol = policy or AnswerQualityPolicy.from_settings()
    if not content.strip():
        return AnswerQualityResult(
            is_valid=False,
            severity="error",
            reason_codes=["empty_answer"],
        )
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        return AnswerQualityResult(
            is_valid=False,
            severity="unsafe",
            reason_codes=["invalid_utf8"],
        )
    language_compliant = is_language_compliant(content, language)
    if not pol.validation_enabled:
        return AnswerQualityResult(
            is_valid=language_compliant,
            severity="clean" if language_compliant else "rewrite_required",
            reason_codes=[] if language_compliant else ["language_mismatch"],
            language_compliant=language_compliant,
        )

    reasons: list[str] = []
    severity: Severity = "clean"
    check_content = content.replace(pol.completion_marker, "")

    def _flag(code: str, level: Severity) -> None:
        nonlocal severity
        reasons.append(code)
        if level == "unsafe":
            severity = "unsafe"
        elif level == "rewrite_required" and severity not in ("unsafe",):
            severity = "rewrite_required"
        elif level == "minor" and severity == "clean":
            severity = "minor"

    if not language_compliant:
        _flag("language_mismatch", "rewrite_required")

    if _QUAD_DOLLAR.search(content):
        _flag("math_quad_dollar", "rewrite_required")
    if _DOLLAR_DISPLAY.search(content):
        _flag("math_double_dollar", "rewrite_required")
    if _has_single_dollar_math(content):
        _flag("math_single_dollar", "rewrite_required")

    if _count_unbalanced(content, r"\(", r"\)"):
        _flag("math_unbalanced_inline", "rewrite_required")
    if _count_unbalanced(content, r"\[", r"\]"):
        _flag("math_unbalanced_display", "rewrite_required")

    if (
        _has_unclosed_fence(content)
        or content.count("```") % 2 != 0
        or content.count("~~~") % 2 != 0
    ):
        _flag("markdown_unclosed_fence", "rewrite_required")
    if _has_malformed_markdown_link(check_content):
        _flag("markdown_malformed_link", "rewrite_required")

    if _RAW_HTML_PATTERN.search(check_content):
        _flag("raw_html_script", "unsafe")
    elif _HTML_TAG_PATTERN.search(check_content):
        _flag("raw_html_tag", "rewrite_required")

    for line in content.splitlines():
        if len(line) > pol.max_math_line_chars and ("\\(" in line or "\\[" in line or "$" in line):
            _flag("math_line_too_long", "rewrite_required")
            break
        if re.search(r"\\\[.+\\\]", line) and len(line) > 120:
            prose = re.sub(r"\\\[.+?\\\]", "", line).strip()
            if len(prose) > 40:
                _flag("display_math_with_prose", "rewrite_required")
                break

    lowered = content.lower()
    for phrase in _BAD_PHRASES:
        if phrase in lowered:
            _flag(f"bad_phrase_{phrase.replace(' ', '_')}", "rewrite_required")

    explicit_final_values = _explicit_final_answer_values(content)
    if count_final_answer_sections(content) > 1 and len(explicit_final_values) <= 1:
        logger.info("quality_false_positive_regression count=1")
    if len(explicit_final_values) > 1:
        _flag("duplicate_final_answer", "rewrite_required")
        answer_values = set(explicit_final_values)
        if len(answer_values) > 1:
            _flag("conflicting_answer_values", "rewrite_required")
    if _contradicting_answer_surfaces(content):
        # The visible headline states one scalar while the final answer states another.
        # The student reads the headline, so the response must be repaired or refused.
        _flag("conflicting_answer_values", "rewrite_required")

    spacing_content = _mask_spacing_protected_regions(check_content)
    if _JOINED_MONTH_DATE.search(spacing_content):
        _flag("joined_date_tokens", "rewrite_required")
    if _JOINED_PROSE_NUMBER.search(spacing_content):
        _flag("suspicious_word_number_join", "rewrite_required")

    if content.count(pol.completion_marker) > 1:
        _flag("duplicate_completion_marker", "minor")

    if intent in ("solve", "solve_question") and not detect_final_answer(content):
        _flag("missing_final_answer", "rewrite_required")
    if (
        intent in ("solve", "solve_question")
        and query is not None
        and not _ANSWER_ONLY_REQUEST.search(query)
        and not has_sufficient_solve_working(content, difficulty=difficulty)
    ):
        _flag("insufficient_solve_working", "rewrite_required")

    if _INCOMPLETE_ENDINGS.search(content.rstrip()):
        _flag("incomplete_ending", "rewrite_required")

    display_blocks = len(_DISPLAY_MATH.findall(content))
    if display_blocks > pol.max_display_math_blocks:
        _flag("too_many_display_math_blocks", "rewrite_required")

    visible_steps = len(_NUMBERED_STEP.findall(content))
    if visible_steps > pol.max_visible_steps:
        _flag("too_many_visible_steps", "rewrite_required")

    if subject == "math" and difficulty == "intermediate":
        if len(content) > pol.math_intermediate_max_chars:
            _flag("math_intermediate_too_long", "rewrite_required")

    sanitized = try_sanitize_minor(content, pol, reasons)
    if severity == "clean":
        is_valid = True
    elif severity == "minor" and sanitized is not None:
        is_valid = True
    else:
        is_valid = False

    return AnswerQualityResult(
        is_valid=is_valid,
        severity=severity,
        reason_codes=reasons,
        sanitized_text=sanitized,
        language_compliant=language_compliant,
    )


def has_sufficient_solve_working(content: str, *, difficulty: str) -> bool:
    """Require compact reproducible working, scaled to the classified difficulty."""
    working_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "<ANSWER_DONE>":
            continue
        answer_match = _ANSWER_HEADING_LINE.fullmatch(stripped)
        if answer_match is not None:
            inline_value = answer_match.group("value").strip()
            if inline_value:
                working_lines.append(inline_value)
            continue
        if re.fullmatch(r"#{1,6}\s*(?:answer|final answer)\s*", stripped, re.IGNORECASE):
            continue
        working_lines.append(stripped)
    supporting = " ".join(working_lines)
    words = re.findall(r"[A-Za-z0-9\u0900-\u097f]+", supporting)
    marker_count = len(_REASONING_MARKER.findall(supporting))
    if difficulty == "advanced":
        return len(words) >= 15 and (marker_count >= 2 or len(working_lines) >= 3)
    if difficulty == "intermediate":
        return len(words) >= 8 and (marker_count >= 1 or len(working_lines) >= 2)
    return len(words) >= 4 or marker_count >= 1


def try_sanitize_minor(
    content: str,
    policy: AnswerQualityPolicy,
    reason_codes: list[str],
) -> str | None:
    """Apply safe minor fixes only."""
    if not content:
        return None
    changed = False
    text = content

    if (
        reason_codes.count("duplicate_completion_marker")
        or text.count(policy.completion_marker) > 1
    ):
        parts = text.split(policy.completion_marker)
        text = parts[0].rstrip() + policy.completion_marker
        changed = True

    if _HTML_TAG_PATTERN.search(text.replace(policy.completion_marker, "")):
        text = html.escape(text.replace(policy.completion_marker, ""))
        changed = True

    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if text != content:
        changed = True

    return text if changed else None


def apply_safe_sanitizer(content: str, *, marker: str) -> str:
    """Best-effort sanitizer for fallback output."""
    text = content.replace(marker, "").strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if _HTML_TAG_PATTERN.search(text):
        text = html.escape(text)
    return text


def strip_duplicate_final_answer_section(content: str) -> str:
    """Remove a repeated Final Answer block if duplicated verbatim."""
    matches = [
        match
        for match in _ANSWER_HEADING_LINE.finditer(content)
        if match.group("label").casefold().startswith("final")
    ]
    if len(matches) < 2:
        return content
    first_start = matches[0].start()
    second_start = matches[1].start()
    tail = content[second_start:]
    first_block = content[first_start:second_start]
    if tail.strip() == first_block.strip():
        return content[:second_start].rstrip()
    return content


def parse_rewrite_output(content: str, *, marker: str) -> tuple[str | None, str]:
    """Return a usable rewrite and a safe parse outcome."""
    if not content or not content.strip():
        return None, "provider_empty"
    marker_found = marker in content
    candidate = apply_safe_sanitizer(content, marker=marker)
    if not candidate or not detect_final_answer(candidate):
        logger.warning("rewrite_parse_failure metric_count=1 outcome=parse_failed")
        return None, "parse_failed"
    if not marker_found:
        return candidate, "marker_missing_but_complete"
    return candidate, "rewrite_accepted"


def build_rewrite_messages(
    base_messages: list[LlmMessage],
    *,
    draft_answer: str,
    reason_codes: Sequence[str] | None = None,
) -> list[LlmMessage]:
    """Compose the bounded rewrite request, naming what the gate actually rejected.

    Without the reason codes the rewrite is effectively a blind retry: the generic
    prompt asks only for a concise rewrite, so a presentation defect such as
    math_line_too_long survives into the second candidate and the request fails.
    """
    instruction = REWRITE_USER_PROMPT
    if reason_codes:
        instruction = (
            f"{REWRITE_USER_PROMPT}\n\n"
            f"The previous answer was rejected for: {','.join(reason_codes[:8])}. "
            "Fix exactly that presentation defect. Keep every fact, step, equation and "
            "the final answer unchanged, and do not shorten the explanation. Put display "
            "math on its own line and never leave a long sentence and math markup on the "
            "same line."
        )
    return [
        *base_messages,
        LlmMessage(role="assistant", content=draft_answer),
        LlmMessage(role="user", content=instruction),
    ]


def rewrite_max_tokens(*, difficulty: str, route_subject: str) -> int:
    if difficulty == "advanced" or route_subject == "practice":
        return 1000
    if difficulty == "intermediate":
        return 700
    return 500


def plain_text_fallback(
    *, subject: str, language: CanonicalLanguage = "english"
) -> str:
    if language == "hindi":
        return "उत्तर को विश्वसनीय रूप से तैयार नहीं किया जा सका। कृपया फिर प्रयास करें।"
    if language == "hinglish":
        return "Answer reliably format nahi ho saka. Please dobara try karein."
    return (
        "A compact answer could not be formatted reliably. "
        "Please try asking again with a shorter question."
        if subject == "math"
        else "The answer could not be formatted reliably. Please try again."
    )


def generation_failure_message(
    language: CanonicalLanguage = "english",
) -> str:
    """Safe user-facing message when all generation attempts fail or return empty."""
    if language == "hindi":
        return "इस प्रश्न का विश्वसनीय उत्तर तैयार नहीं हो सका। कृपया फिर प्रयास करें।"
    if language == "hinglish":
        return "Is question ka reliable answer generate nahi ho saka. Please dobara try karein."
    return GENERATION_FAILURE_MESSAGE


def fallback_required_for_result(result: AnswerQualityResult) -> bool:
    """True when quality result indicates model-level fallback should be attempted."""
    return "empty_answer" in result.reason_codes


def log_answer_quality_validation(
    *,
    request_id: str,
    route_id: str,
    subject: str,
    difficulty: str,
    intent: str | None,
    result: AnswerQualityResult,
    output_chars: int,
    rewrite_required: bool,
    sanitized: bool,
) -> None:
    logger.debug(
        "answer_quality_validation  request_id=%s  route_id=%s  subject=%s  "
        "difficulty=%s  intent=%s  is_valid=%s  severity=%s  reasons_count=%d  "
        "reason_codes=%s  output_chars=%d  rewrite_required=%s  sanitized=%s  "
        "fallback_required=%s",
        request_id,
        route_id,
        subject,
        difficulty,
        intent or "",
        result.is_valid,
        result.severity,
        len(result.reason_codes),
        ",".join(result.reason_codes[:8]),
        output_chars,
        rewrite_required,
        sanitized,
        fallback_required_for_result(result),
    )
    log_event(
        "quality_decision",
        component="doubt_solver.quality",
        stage="validate_quality",
        status="passed" if result.is_valid else "failed",
        error_code=None if result.is_valid else "QUALITY_REWRITE_REQUIRED",
        details={
            "passed": result.is_valid,
            "reason_code": ",".join(result.reason_codes[:8]) or "none",
            "repair_required": rewrite_required,
        },
    )


def log_answer_quality_rewrite(
    *,
    request_id: str,
    used: bool,
    attempt_count: int,
    success: bool,
    final_output_chars: int,
    outcome: str,
) -> None:
    logger.debug(
        "answer_quality_rewrite  request_id=%s  used=%s  attempt_count=%d  "
        "success=%s  final_output_chars=%d  outcome=%s",
        request_id,
        used,
        attempt_count,
        success,
        final_output_chars,
        outcome,
    )
    update_request_summary(rewrite_attempted=used)
    log_event(
        "quality_rewrite_completed",
        component="doubt_solver.quality",
        stage="validate_quality",
        status="completed" if success else "failed",
        error_code=None if success else "QUALITY_REWRITE_FAILED",
        details={
            "used": used,
            "attempt_count": attempt_count,
            "success": success,
            "outcome": outcome,
        },
        level=logging.INFO if success else logging.WARNING,
    )


def _count_unbalanced(text: str, open_delim: str, close_delim: str) -> bool:
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith(open_delim, i):
            depth += 1
            i += len(open_delim)
        elif text.startswith(close_delim, i):
            depth -= 1
            i += len(close_delim)
            if depth < 0:
                return True
        else:
            i += 1
    return depth != 0
