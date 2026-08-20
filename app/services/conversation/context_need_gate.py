"""Deterministic decision for whether recent conversation may be needed."""

from __future__ import annotations

import re
import time
import unicodedata

from schemas.conversation import ContextNeedAssessment
from services.conversation.reference_resolution import analyze_reference

_PUNCTUATION = re.compile(r"[^\w%₹$€£+\-*/=?.\u0900-\u097f]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9\u0900-\u097f]+", re.IGNORECASE)

_EXPLICIT_REFERENCE_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "previous_reference",
        re.compile(
            r"\b(?:previous|previos|last|earlier|above|pichla|pichhle|"
            r"pehle wala|पिछला|आखिरी)\s+"
            r"(?:answer|solution|soluton|question|step|formula|उत्तर|सवाल|चरण)\b"
        ),
    ),
    (
        "assistant_reference",
        re.compile(
            r"\b(?:why|how)\s+(?:did\s+)?(?:you|u)\s+"
            r"(?:calculate|calculated|replace|replaced|remove|removed|"
            r"get|got|use|used|apply|applied|divide|divided|multiply|multiplied|"
            r"add|added|subtract|subtracted)\b|"
            r"\b(?:what|which)\s+(?:formula|method|operation)\s+"
            r"(?:did\s+)?(?:you|u)\s+(?:use|apply)\b|"
            r"\b(?:aapne|आपने)\b.{0,48}\b(?:kaise|kyu|nikala|lagaya|कैसे|क्यों)\b"
        ),
    ),
    (
        "selected_object_reference",
        re.compile(
            r"\b(?:explain|check|verify|rethink)\s+"
            r"(?:this|that|it|the last step|the answer)\b|"
            r"\bhighlight\s+the\s+[\w-]+(?:\s+[\w-]+){0,4}\s+solution\b|"
            r"\bwhat was the pattern\b|"
            r"\b(?:ye|yeh|woh|इसे|यह)\s+(?:kaise|kyu|कैसे|क्यों)\b|"
            r"\b(?:isko|ise|usko|इसे|उसको)\s+"
            r"(?:explain|samjhao|highlight|check|समझाओ)\b"
        ),
    ),
    (
        "continue_reference",
        re.compile(
            r"\b(?:continue|continue karo|continue karein|go on|carry on|"
            r"aage badhao|जारी रखो)\b"
        ),
    ),
    (
        # Anaphoric by wording only. This family is the one that a current-turn
        # source can satisfy on its own, so the gate consults the local antecedent
        # before treating a match here as a history reference.
        "similar_reference",
        re.compile(
            r"\b(?:same type|same method|similar questions?|"
            r"similar (?:practice |mock |quiz )?questions?|"
            r"more like this|more like that|(?:questions?|problems?) like (?:this|that)|"
            r"another question like this|what we just did|aise aur|isi method)\b"
        ),
    ),
    (
        "correction_reference",
        re.compile(
            r"\b(?:your|ur|the|this)?\s*(?:answer|solution)?\s*"
            r"(?:is\s+)?(?:wrong|worng|incorrect|galat)\b|"
            r"\b(?:answer|solution)\s+गलत\b"
        ),
    ),
    (
        "resolve_again_reference",
        re.compile(
            r"\b(?:solve|solv|resolve)\s+(?:it\s+)?(?:again|agane|from scratch)\b|"
            r"\b(?:agane solve it|do it again|recheck the previous solution|dobara solve|"
            r"phir se check|फिर से हल)\b|"
            r"\bagain wrong\b"
        ),
    ),
    (
        "transform_reference",
        re.compile(
            r"\b(?:make|convert|change|translate)\s+"
            r"(?:the\s+)?(?:previous|last|this|that)\s+(?:question|problem|it)\b|"
            r"\b(?:make it harder|convert it into|change the values)\b"
        ),
    ),
)

_UNCERTAIN_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "elliptical_why",
        re.compile(r"^(?:why|why is that|why this|why [₹$€£]?\s*\d+(?:\.\d+)?%?)\??$"),
    ),
    (
        "elliptical_what_about",
        re.compile(r"^(?:and\s+)?what about\b.{0,48}\??$"),
    ),
    (
        "elliptical_another",
        re.compile(r"^(?:can|could|would)\s+(?:you|u)\s+(?:do|give|create)\s+another\b"),
    ),
    (
        "elliptical_method",
        re.compile(r"^(?:do|use)\s+(?:the\s+)?(?:second|other|another)\s+method\b"),
    ),
    (
        "elliptical_correctness",
        re.compile(r"^(?:is|was)\s+(?:that|this|it)\s+correct\??$"),
    ),
    (
        "elliptical_object",
        re.compile(r"^(?:and\s+)?(?:the\s+)?(?:second|third|other)\s+one\??$"),
    ),
)

_ACADEMIC_INSTRUCTION = re.compile(
    r"\b(?:solve|calculate|compute|find|evaluate|determine|simplify|prove|"
    r"explain|define|describe|compare|identify|name|write|show|"
    r"who|what|where|when|which|how many|how much|"
    r"हल|गणना|ज्ञात|समझाइए|बताइए|कौन|क्या|कहाँ)\b"
)
_PROBLEM_STRUCTURE = re.compile(
    r"(?:[=?]|\b(?:given|if|when|where|such that|in the set|"
    r"at\s+\d|for\s+\d|of\s+\d|with\s+\d)\b)"
)
_NUMERIC_PROBLEM = re.compile(r"(?:\d|[₹$€£%])")
_COMPLETE_PERCENTAGE_PROBLEM = re.compile(
    r"\d+(?:\.\d+)?\b.{0,40}\b\d+(?:\.\d+)?\s*(?:%|percent\b)",
    re.IGNORECASE,
)
_EQUATION_OR_INEQUALITY = re.compile(
    r"(?:[a-z\u0900-\u097f]\s*(?:[+\-*/]\s*)?\d*\s*(?:=|≤|≥|<|>)|"
    r"(?:=|≤|≥|<|>)\s*-?\d)",
    re.IGNORECASE,
)
_TARGET_CUE = re.compile(
    r"\b(?:required|asked|unknown|value|amount|number|percentage|percent|"
    r"profit|loss|distance|time|speed|age|ratio|answer|result|final)\b"
)
_COMPLETE_CONDITION = re.compile(
    r"\b(?:contains?|holds?|has|have|given|if|when|where|such that|"
    r"respectively|increased|decreased|removed|replaced|sold|bought)\b"
)
_MCQ_STRUCTURE = re.compile(
    r"(?:\boptions?\b|(?:^|\s)[(]?[a-d1-4][).:-]\s)",
    re.IGNORECASE,
)
_INCOMPLETE_OPENING = re.compile(
    r"^(?:why(?:\s+is)?\s+(?:this|that|it|\d)|what about|"
    r"(?:and\s+)?(?:the\s+)?(?:second|third|other)\s+one|"
    r"(?:this|that|it|ye|yeh|woh|isko|ise)\b)"
)
_EXPLICIT_STANDALONE_TASK = re.compile(
    r"\b(?:give|create|generate|make|prepare|write)\s+(?:me\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*"
    r"(?:practice\s+)?(?:questions?|problems?|quiz|mock)\s+"
    r"(?:on|about|for|from)\s+\w",
    re.IGNORECASE,
)
_EXPLICIT_TOPIC_REQUEST = re.compile(
    r"\b(?:practice|questions?|problems?|quiz|mock)\b.{0,48}\b"
    r"(?:grammar|narration|speech|interest|algebra|percentage|ratio|"
    r"seating|ranking|reasoning|history|polity|science)\b",
    re.IGNORECASE,
)
_COUNTED_TOPIC_PRACTICE = re.compile(
    r"\b(?:give|create|generate|make|prepare|write)\s+(?:me\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:english\s+)?(?:grammar|narration|speech|interest|algebra|percentage|"
    r"ratio|seating|ranking|reasoning|history|polity|science)\s+"
    r"(?:questions?|problems?)\b",
    re.IGNORECASE,
)
_NAMED_STANDALONE_TOPIC = re.compile(
    r"\b(?:simple[- ]interest|compound[- ]interest|algebra|percentage|ratio|"
    r"seating(?:\s+arrangement)?|ranking|reasoning|grammar|narration|"
    r"history|polity|science)\s+(?:question|problem|practice|quiz)\b",
    re.IGNORECASE,
)
_LOCAL_SOURCE_INTRODUCER = re.compile(
    r"\b(?:here (?:is|are)|consider|take|given)\s+"
    r"(?:the |this |my |a |an )?"
    # Allows a short qualifier such as "a grammar sentence".
    r"(?:\w+\s+){0,2}"
    r"(?:question|problem|sentence|passage|paragraph|statement|equation|sum|"
    r"exercise|example)s?\b|"
    r"\b(?:similar|like|same as)\s+(?:to\s+)?(?:this|the following)\b",
    re.IGNORECASE,
)

_LOCAL_DEMONSTRATIVE_TASK = re.compile(
    r"^(?:solve|calculate|compute|find|evaluate|determine|simplify|prove|"
    r"explain)\s+(?:this|that)\b",
    re.IGNORECASE,
)


def normalize_context_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = _PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def is_short_incomplete_topic_phrase(query: str) -> bool:
    """Return true for a short non-solvable phrase that needs interpretation."""
    normalized = normalize_context_query(query)
    words = _WORD.findall(normalized)
    if not 1 <= len(words) <= 7:
        return False
    if _ACADEMIC_INSTRUCTION.search(normalized):
        return False
    if _NUMERIC_PROBLEM.search(normalized) or _EQUATION_OR_INEQUALITY.search(normalized):
        return False
    return not normalized.endswith("?")


def _is_clearly_self_contained(normalized: str) -> bool:
    words = _WORD.findall(normalized)
    if not words or _INCOMPLETE_OPENING.search(normalized):
        return False
    if _COMPLETE_PERCENTAGE_PROBLEM.search(normalized):
        return True
    if (
        _EXPLICIT_STANDALONE_TASK.search(normalized)
        or _EXPLICIT_TOPIC_REQUEST.search(normalized)
        or _COUNTED_TOPIC_PRACTICE.search(normalized)
        or _NAMED_STANDALONE_TOPIC.search(normalized)
    ):
        return True
    if _ACADEMIC_INSTRUCTION.search(normalized):
        if normalized.startswith(
            (
                "solve ",
                "calculate ",
                "compute ",
                "explain ",
                "define ",
                "describe ",
                "who ",
                "where ",
                "when ",
                "find ",
                "evaluate ",
                "determine ",
                "simplify ",
                "prove ",
            )
        ):
            return len(words) >= 2
        return len(words) >= 5 and bool(
            _PROBLEM_STRUCTURE.search(normalized)
            or _NUMERIC_PROBLEM.search(normalized)
            or normalized.endswith("?")
        )
    if _EQUATION_OR_INEQUALITY.search(normalized) and len(words) >= 2:
        return True
    if _MCQ_STRUCTURE.search(normalized) and len(words) >= 8:
        return True
    if normalized.endswith("?") and len(words) >= 6:
        return True
    return (
        len(words) >= 12
        and bool(_NUMERIC_PROBLEM.search(normalized))
        and bool(_COMPLETE_CONDITION.search(normalized))
        and bool(_TARGET_CUE.search(normalized))
    )


def _has_local_antecedent(normalized: str) -> bool:
    """Does this turn carry its own educational object for "similar" to point at?

    A word like "similar" is anaphoric by wording, not by meaning.  When the same
    message already supplies the source question, the reference resolves locally and
    conversation history is not required.
    """
    if _is_clearly_self_contained(normalized):
        return True
    if not _LOCAL_SOURCE_INTRODUCER.search(normalized):
        return False
    # An introducer alone is not enough; it must actually introduce something.
    return len(_WORD.findall(normalized)) >= 8


class ContextNeedGate:
    """Return only whether bounded recent context should be made available."""

    def evaluate(self, query: str) -> ContextNeedAssessment:
        started = time.monotonic()
        normalized = normalize_context_query(query)
        local_antecedent = _has_local_antecedent(normalized)
        for reason, pattern in _EXPLICIT_REFERENCE_FAMILIES:
            if pattern.search(normalized):
                if reason == "similar_reference" and local_antecedent:
                    # The source lives in this turn; history adds nothing.
                    return ContextNeedAssessment(
                        decision="CONTEXT_NOT_NEEDED",
                        reason_codes=("local_antecedent_resolved",),
                        matched_signals=(reason,),
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
                return ContextNeedAssessment(
                    decision="CONTEXT_REQUIRED",
                    reason_codes=("explicit_context_reference",),
                    matched_signals=(reason,),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
        for reason, pattern in _UNCERTAIN_FAMILIES:
            if pattern.search(normalized):
                return ContextNeedAssessment(
                    decision="UNCERTAIN",
                    reason_codes=("possible_context_dependency",),
                    matched_signals=(reason,),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
        if _LOCAL_DEMONSTRATIVE_TASK.search(
            normalized
        ) and _is_clearly_self_contained(normalized):
            return ContextNeedAssessment(
                decision="CONTEXT_NOT_NEEDED",
                reason_codes=("complete_self_contained_request",),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        reference = analyze_reference(query)
        # A bare demonstrative ("like this") is satisfied by a local source, but an
        # ordinal or explicitly prior reference ("the previous one") never is.
        if (
            reference.external_reference_detected
            and local_antecedent
            and set(reference.reference_types) <= {"object", "demonstrative"}
        ):
            return ContextNeedAssessment(
                decision="CONTEXT_NOT_NEEDED",
                reason_codes=("local_antecedent_resolved",),
                matched_signals=tuple(
                    f"{reference_type}_reference"
                    for reference_type in reference.reference_types
                )[:8],
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        if reference.external_reference_detected:
            return ContextNeedAssessment(
                decision="CONTEXT_REQUIRED",
                reason_codes=("external_reference_detected",),
                matched_signals=tuple(
                    f"{reference_type}_reference"
                    for reference_type in reference.reference_types
                )[:8],
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        if _is_clearly_self_contained(normalized):
            return ContextNeedAssessment(
                decision="CONTEXT_NOT_NEEDED",
                reason_codes=("complete_self_contained_request",),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        return ContextNeedAssessment(
            decision="UNCERTAIN",
            reason_codes=("insufficient_standalone_evidence",),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
