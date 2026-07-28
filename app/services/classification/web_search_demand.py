"""Normalize unmistakable web-search demand after either classifier."""

from __future__ import annotations

import re

from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import QueryClassification

_CURRENT_AFFAIRS = re.compile(r"\bcur+r?e?nt\s+affairs?\b", re.IGNORECASE)
_STATIC_DEFINITION = re.compile(
    r"\b(?:meaning|definition)\s+of\s+cur+r?e?nt\s+affairs?\b|"
    r"\bwhat\s+(?:does|do)\s+cur+r?e?nt\s+affairs?\s+mean\b|"
    r"\bdefine\s+cur+r?e?nt\s+affairs?\b",
    re.IGNORECASE,
)
_EXPLICIT_WEB = re.compile(
    r"\b(?:search|check|verify)\s+(?:online|the\s+web|web|internet)\b|"
    r"\b(?:online|web|internet)\s+(?:search|verification)\b",
    re.IGNORECASE,
)
_MONTH_YEAR = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+20\d{2}\b",
    re.IGNORECASE,
)
_WEB_CONTEXT_ACTIONS = frozenset(
    {"ANSWER_WITH_CONTEXT", "CONTINUE_PREVIOUS", "GENERATE_SIMILAR"}
)


def normalize_web_search_demand(
    query: str,
    classification: QueryClassification,
    *,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
) -> QueryClassification:
    """Preserve model output and force only unambiguous current/web requests."""
    evidence = query
    if classification.requested_action in _WEB_CONTEXT_ACTIONS:
        selected = next(
            (
                card
                for card in candidate_cards
                if card.turn_id == classification.selected_turn_id
            ),
            None,
        )
        if selected is not None:
            evidence = f"{query} {selected.question_preview} {selected.topic or ''}"

    if _STATIC_DEFINITION.search(query):
        return classification

    reason: str | None = None
    if _CURRENT_AFFAIRS.search(evidence):
        reason = "current_affairs"
    elif classification.need_web_search:
        return classification
    elif _EXPLICIT_WEB.search(query):
        reason = "user_requested_web"
    if reason is None:
        return classification

    month_year = _MONTH_YEAR.search(evidence)
    search_query = (
        f"current affairs {month_year.group(0)}"
        if reason == "current_affairs" and month_year is not None
        else query.strip()
    )
    return classification.model_copy(
        update={
            "need_web_search": True,
            "web_search_reason": (
                classification.web_search_reason or reason
            ),
            "web_search_query": search_query[:256],
        }
    )
