"""Deterministic generator answer validation, sanitization, and rewrite support."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Literal

from config import Settings, get_settings
from schemas.llm import LlmMessage
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
    "show failed attempts. Use valid Markdown. Use \\(...\\) and \\[...\\] only for "
    "math. Do not use $ or $$. Keep it concise. "
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
_HTML_TAG_PATTERN = re.compile(r"<\s*[a-zA-Z][^>]*>")
_DOLLAR_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_DOLLAR_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_QUAD_DOLLAR = re.compile(r"\${4,}")
_ANSWER_HEADING_LINE = re.compile(
    r"(?im)^\s*\*{0,2}\s*(?:final\s+)?answer\s*:?\s*\*{0,2}"
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


def detect_final_answer(content: str) -> bool:
    """Return True when an Answer or Final Answer section is present."""
    if not content or not content.strip():
        return False
    return _ANSWER_HEADING_LINE.search(content) is not None


def count_final_answer_sections(content: str) -> int:
    return len(_ANSWER_HEADING_LINE.findall(content))


def _answer_heading_values(content: str) -> list[str]:
    values: list[str] = []
    lines = content.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_HEADING_LINE.fullmatch(line)
        if match is None:
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
    if not pol.validation_enabled:
        return AnswerQualityResult(is_valid=True, severity="clean", reason_codes=[])

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

    if _QUAD_DOLLAR.search(content):
        _flag("math_quad_dollar", "rewrite_required")
    if _DOLLAR_DISPLAY.search(content):
        _flag("math_double_dollar", "rewrite_required")
    if _DOLLAR_INLINE.search(content):
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

    if count_final_answer_sections(content) > 1:
        _flag("duplicate_final_answer", "rewrite_required")
        answer_values = set(_answer_heading_values(content))
        if len(answer_values) > 1:
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
    )


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
    matches = list(_ANSWER_HEADING_LINE.finditer(content))
    if len(matches) < 2:
        return content
    first_start = matches[0].start()
    second_start = matches[1].start()
    tail = content[second_start:]
    first_block = content[first_start:second_start]
    if tail.strip() == first_block.strip():
        return content[:second_start].rstrip()
    return content


def build_rewrite_messages(
    base_messages: list[LlmMessage],
    *,
    draft_answer: str,
) -> list[LlmMessage]:
    return [
        *base_messages,
        LlmMessage(role="assistant", content=draft_answer),
        LlmMessage(role="user", content=REWRITE_USER_PROMPT),
    ]


def rewrite_max_tokens(*, difficulty: str, route_subject: str) -> int:
    if difficulty == "advanced" or route_subject == "practice":
        return 1000
    if difficulty == "intermediate":
        return 700
    return 500


def plain_text_fallback(*, subject: str) -> str:
    return (
        "A compact answer could not be formatted reliably. "
        "Please try asking again with a shorter question."
        if subject == "math"
        else "The answer could not be formatted reliably. Please try again."
    )


def generation_failure_message() -> str:
    """Safe user-facing message when all generation attempts fail or return empty."""
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
    logger.info(
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


def log_answer_quality_rewrite(
    *,
    request_id: str,
    used: bool,
    attempt_count: int,
    success: bool,
    final_output_chars: int,
) -> None:
    logger.info(
        "answer_quality_rewrite  request_id=%s  used=%s  attempt_count=%d  "
        "success=%s  final_output_chars=%d",
        request_id,
        used,
        attempt_count,
        success,
        final_output_chars,
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
