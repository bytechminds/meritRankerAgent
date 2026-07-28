"""Guard generation when required live web evidence is unavailable."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from observability import log_event
from schemas.doubt_solver import CanonicalLanguage

_LIMITED_RESPONSES: dict[CanonicalLanguage, str] = {
    "english": (
        "I could not verify the requested current information from reliable live "
        "sources, so I cannot provide factual current-affairs content right now."
    ),
    "hinglish": (
        "Main reliable live sources se requested current information verify nahi "
        "kar saka, isliye abhi factual current-affairs content dena safe nahi hai."
    ),
    "hindi": (
        "मैं विश्वसनीय लाइव स्रोतों से मांगी गई वर्तमान जानकारी सत्यापित नहीं कर सका, "
        "इसलिए अभी तथ्यात्मक समसामयिक सामग्री देना सुरक्षित नहीं है।"
    ),
}
_EXECUTED_REASONS = frozenset(
    {"web_context_selected", "weak_web_context", "retrieval_error"}
)
_CONTEXT_URL_RE = re.compile(r"(?m)^\s*URL:\s*(https?://\S+)\s*$")
_ANSWER_URL_RE = re.compile(r"https?://[^\s)>}\]]+")
_NUMBERED_QUESTION_RE = re.compile(r"(?m)^\s*\d{1,2}[.)]\s+")
_QUESTION_BLOCK_RE = re.compile(
    r"(?ms)^\s*(\d{1,2})[.)]\s+(.*?)(?=^\s*\d{1,2}[.)]\s+|\Z)"
)


def _retrieval_reason(state: Mapping[str, object]) -> object:
    retrieval_context = state.get("retrieval_context")
    retrieval = retrieval_context if isinstance(retrieval_context, Mapping) else {}
    trace_value = retrieval.get("retrievalTrace") or retrieval.get("retrieval_trace")
    trace = trace_value if isinstance(trace_value, Mapping) else {}
    return trace.get("fallbackReason") or trace.get("fallback_reason")


def required_web_context_verified(
    classification: Mapping[str, object],
    state: Mapping[str, object],
) -> bool:
    """Return true unless required web evidence is absent or weak."""
    if not bool(classification.get("need_web_search")):
        return True
    reason = _retrieval_reason(state)
    context_text = str(state.get("context_text") or "")
    return (
        reason == "web_context_selected"
        and "[Web Context]" in context_text
        and bool(re.search(r"https?://", context_text))
    )


def verification_limited_response(language: CanonicalLanguage) -> str:
    return _LIMITED_RESPONSES[language]


def _normalize_url(value: str) -> str:
    cleaned = value.rstrip(".,;:")
    parsed = urlsplit(cleaned)
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def selected_source_urls(state: Mapping[str, object]) -> tuple[str, ...]:
    """Return unique selected-source URLs retained in bounded web context."""
    context_text = str(state.get("context_text") or "")
    return tuple(
        dict.fromkeys(
            _normalize_url(match.group(1))
            for match in _CONTEXT_URL_RE.finditer(context_text)
        )
    )


def required_web_answer_verified(
    classification: Mapping[str, object],
    state: Mapping[str, object],
    answer: str,
) -> bool:
    """Reject required-web output that is not traceable to selected evidence."""
    if not bool(classification.get("need_web_search")):
        return True
    source_urls = set(selected_source_urls(state))
    answer_urls = {
        _normalize_url(match.group(0)) for match in _ANSWER_URL_RE.finditer(answer)
    }
    if not source_urls or not answer_urls or not answer_urls.issubset(source_urls):
        return False
    if not answer_urls.intersection(source_urls):
        return False
    if classification.get("intent") in {"practice", "practice_question"}:
        question_count = len(_NUMBERED_QUESTION_RE.findall(answer))
        if question_count == 0 or question_count > len(source_urls):
            return False
    return True


def sanitize_required_web_answer(
    classification: Mapping[str, object],
    state: Mapping[str, object],
    answer: str,
) -> str | None:
    """Keep at most one cited practice-question block per selected source."""
    if not bool(classification.get("need_web_search")):
        return answer
    if classification.get("intent") not in {"practice", "practice_question"}:
        return answer if required_web_answer_verified(classification, state, answer) else None

    selected_urls = set(selected_source_urls(state))
    used_urls: set[str] = set()
    grounded_blocks: list[str] = []
    for match in _QUESTION_BLOCK_RE.finditer(answer):
        block = match.group(2).strip()
        block_urls = {
            _normalize_url(url.group(0)) for url in _ANSWER_URL_RE.finditer(block)
        }
        matching_urls = block_urls.intersection(selected_urls) - used_urls
        if len(matching_urls) != 1 or not block_urls.issubset(selected_urls):
            continue
        used_urls.update(matching_urls)
        grounded_blocks.append(f"{len(grounded_blocks) + 1}. {block}")
    if not grounded_blocks:
        return None
    return "\n\n".join(grounded_blocks)


def log_grounding_status(
    classification: Mapping[str, object],
    state: Mapping[str, object],
    *,
    verified: bool,
    answer: str = "",
) -> None:
    web_required = bool(classification.get("need_web_search"))
    reason = _retrieval_reason(state)
    selected_urls = set(selected_source_urls(state))
    answer_urls = {
        _normalize_url(match.group(0)) for match in _ANSWER_URL_RE.finditer(answer)
    }
    citation_count = len(selected_urls.intersection(answer_urls))
    log_event(
        "grounding_completed",
        component="doubt_solver.grounding",
        stage="generate",
        status="grounded" if verified else "verification_limited",
        details={
            "web_required": web_required,
            "web_executed": web_required and reason in _EXECUTED_REASONS,
            "web_context_used": verified and web_required,
            "citation_count": citation_count,
            "verification_status": "grounded" if verified else "verification_limited",
        },
    )
