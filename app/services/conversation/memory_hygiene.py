"""Single policy for selecting substantive academic conversation turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from schemas.conversation import ConversationTurnType, RecentConversationTurn

MemoryHygieneReason = Literal[
    "accepted_academic_pair",
    "clarification_response",
    "unresolved_reference_response",
    "generic_acknowledgement",
    "technical_failure",
    "failed_quality",
    "non_substantive_answer",
]

_RESPONSE_HEADING = re.compile(
    r"^\s*(?:[*_`>#-]+\s*)*(?:answer|final answer|response|solution)\s*"
    r"[:\-–—]\s*",
    re.IGNORECASE,
)
_GENERIC_ACKNOWLEDGEMENT = re.compile(
    r"^\s*(?:"
    r"i\s+(?:understand|see)\b|"
    r"okay|ok|got\s+it|thanks?|thank\s+you|"
    r"samajh\s+gaya|ठीक\s+है|समझ\s+गया"
    r")",
    re.IGNORECASE,
)
_CLARIFICATION_RESPONSE = re.compile(
    r"^\s*(?:"
    r"please\s+(?:clarify|provide\s+(?:more|additional)\s+(?:details|information))|"
    r"could\s+you\s+(?:clarify|provide\s+more\s+(?:details|information))|"
    r"which\s+(?:earlier|previous)\s+(?:question|answer|step|option)|"
    r"i\s+need\s+more\s+(?:details|information)|"
    r"(?:kripya|please)\s+(?:clear|clarify)\s+(?:karein|karo)|"
    r"thodi\s+aur\s+jaankari|"
    r"कृपया\s+स्पष्ट|थोड़ी\s+और\s+जानकारी"
    r")\b",
    re.IGNORECASE,
)
_UNRESOLVED_REFERENCE_RESPONSE = re.compile(
    r"^\s*(?:"
    r"(?:the|this|your)\s+question.{0,100}"
    r"(?:is\s+incomplete|is\s+unclear|does\s+not\s+specify\s+(?:who|what))|"
    r"(?:it|the\s+question)\s+does\s+not\s+specify\s+(?:who|what)|"
    r"(?:who|what)\s+does\s+[\"“']?(?:he|she|it|they|this|that)[\"”']?"
    r"\s+refer\s+to|"
    r"cannot\s+(?:determine|identify|resolve)\s+(?:who|what|the\s+reference)|"
    r"unclear\s+(?:who|what|which\s+person|which\s+thing)|"
    r"(?:question|sawal)\s+(?:incomplete|adhura|clear\s+nahi).{0,80}"
    r"(?:kaun|kya|kis|refer)|"
    r"(?:he|she|it|woh|wo|yeh|ye)\s+(?:kis|kisko|kise|kaun|kya)"
    r".{0,40}(?:refer|matlab)|"
    r"प्रश्न.{0,80}(?:अधूरा|स्पष्ट\s+नहीं).{0,80}(?:कौन|क्या|किस)|"
    r"[\"“]?(?:वह|यह)[\"”]?\s+किस.{0,40}(?:संदर्भ|व्यक्ति|चीज़)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_TECHNICAL_FAILURE = re.compile(
    r"^\s*(?:"
    r"(?:i|we)\s+(?:could\s+not|couldn't|cannot|can't|was\s+unable\s+to)"
    r"\s+(?:generate|complete|process|verify)|"
    r"(?:the\s+)?provider\s+(?:failed|is\s+unavailable)|"
    r"(?:an\s+)?internal\s+error|"
    r"(?:please\s+)?try\s+again\s+later|"
    r"service\s+(?:is\s+)?temporarily\s+unavailable"
    r")\b",
    re.IGNORECASE,
)
_CORRECTION_ONLY_QUERY = re.compile(
    r"^\s*(?:"
    r"(?:your|the|this|that)?\s*(?:answer|solution)?\s*(?:is\s+)?"
    r"(?:wrong|incorrect)|"
    r"(?:again\s+)?wrong(?:\s*,?\s*(?:solve|resolve)\s+it\s+from\s+scratch)?|"
    r"(?:solve|resolve)\s+it\s+(?:again|from\s+scratch)|"
    r"answer\s+galat\s+hai|गलत\s+है"
    r")[.!]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryHygieneDecision:
    turn: RecentConversationTurn
    turn_type: ConversationTurnType
    usable: bool
    reason: MemoryHygieneReason


def _primary_response(answer: str) -> str:
    normalized = " ".join(answer.replace("\r", " ").split())
    previous = ""
    while normalized != previous:
        previous = normalized
        normalized = _RESPONSE_HEADING.sub("", normalized).lstrip("*_`>#- ")
    return normalized[:800]


def classify_non_substantive_response(
    answer: str,
) -> MemoryHygieneReason | None:
    """Classify the response's primary function without relying on one token."""
    primary = _primary_response(answer)
    if not primary:
        return "non_substantive_answer"
    if _UNRESOLVED_REFERENCE_RESPONSE.search(primary):
        return "unresolved_reference_response"
    if _CLARIFICATION_RESPONSE.search(primary):
        return "clarification_response"
    if _TECHNICAL_FAILURE.search(primary):
        return "technical_failure"
    if _GENERIC_ACKNOWLEDGEMENT.match(primary) and len(primary) < 200:
        return "generic_acknowledgement"
    return None


def classify_stored_turn(turn: RecentConversationTurn) -> MemoryHygieneDecision:
    if turn.turn_type != "academic_question_answer":
        reason: MemoryHygieneReason = {
            "clarification": "clarification_response",
            "acknowledgement": "generic_acknowledgement",
            "error": "technical_failure",
            "failed_quality": "failed_quality",
            "correction_only": "non_substantive_answer",
            "meta_response": "non_substantive_answer",
        }[turn.turn_type]
        return MemoryHygieneDecision(
            turn=turn,
            turn_type=turn.turn_type,
            usable=False,
            reason=reason,
        )
    answer = turn.final_answer.strip()
    query = turn.original_query.strip()
    if _CORRECTION_ONLY_QUERY.fullmatch(query):
        return MemoryHygieneDecision(
            turn,
            "correction_only",
            False,
            "non_substantive_answer",
        )
    rejection = classify_non_substantive_response(answer)
    if rejection is not None:
        turn_type: ConversationTurnType = {
            "clarification_response": "clarification",
            "unresolved_reference_response": "clarification",
            "generic_acknowledgement": "acknowledgement",
            "technical_failure": "error",
            "failed_quality": "failed_quality",
            "non_substantive_answer": "meta_response",
            "accepted_academic_pair": "academic_question_answer",
        }[rejection]
        return MemoryHygieneDecision(turn, turn_type, False, rejection)
    return MemoryHygieneDecision(
        turn=turn,
        turn_type="academic_question_answer",
        usable=True,
        reason="accepted_academic_pair",
    )


def filter_substantive_turns(
    turns: tuple[RecentConversationTurn, ...],
) -> tuple[tuple[RecentConversationTurn, ...], tuple[MemoryHygieneDecision, ...]]:
    decisions = tuple(classify_stored_turn(turn) for turn in turns)
    return (
        tuple(decision.turn for decision in decisions if decision.usable),
        decisions,
    )
