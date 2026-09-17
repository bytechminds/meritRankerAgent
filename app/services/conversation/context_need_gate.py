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
        # "explain this" points outward only while the object stays unresolved.  When
        # the turn goes on to supply the material ("explain this sentence - He does
        # not go"), the demonstrative is satisfied locally and the existing
        # _LOCAL_DEMONSTRATIVE_TASK / local-antecedent checks below settle it.  So the
        # unresolved shape is spelled out in full: the demonstrative, optionally a bare
        # conversational placeholder, optionally a delivery modifier, then end of turn.
        "selected_object_reference",
        re.compile(
            r"\b(?:explain|check|verify|rethink)\s+(?:this|that|it)"
            r"(?:\s+(?:answer|solution|question|result|step|one|part|option))?"
            r"(?:\s+(?:again|once more|simpler|simply|properly|briefly|clearly|"
            r"step by step|in (?:hindi|english)|dobara|phir se))?"
            r"\s*[?.!]?$|"
            r"\b(?:explain|check|verify|rethink)\s+(?:the last step|the answer)\b|"
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
        # Every qualifier used to be optional, so the pattern degenerated to the bare
        # word "wrong" and captured students asking about material they supplied
        # themselves ("which part is wrong").  A correction reference now needs the
        # conversational object it corrects — the assistant, or its answer/solution —
        # to be named alongside the correction word.
        "correction_reference",
        re.compile(
            r"\b(?:your|ur|the|this|that)\s+"
            r"(?:previous\s+|last\s+|earlier\s+)?"
            r"(?:answer|solution|response|result|reply|working)"
            # The stated value may sit between the object and the copula:
            # "the answer 57 is incorrect".
            r"(?:\s+[\w.,%₹$-]+){0,2}\s+"
            r"(?:is|was)\s+(?:wrong|worng|incorrect|galat)\b|"
            r"\bwhy\s+(?:is|was)\s+(?:your|ur|the|this|that)\s+"
            r"(?:previous\s+|last\s+|earlier\s+)?"
            r"(?:answer|solution|response|result|reply)\s+"
            r"(?:wrong|worng|incorrect|galat)\b|"
            r"\b(?:you|u)\s+(?:are|were|r)\s+(?:wrong|worng|incorrect|galat)\b|"
            # Hinglish and Hindi place the correction word straight after the object
            # with no copula: "answer galat hai", "solution गलत hai".
            r"\b(?:answer|solution|jawab|jawaab)\s+(?:galat|गलत)\b"
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
# "explain <object>" carries its own subject; "explain <modifier>" only restates how a
# previous answer should be delivered and names nothing to work on.  The remainder after
# the verb must be a modifier IN FULL — "explain photosynthesis in detail" still supplies
# an object and stays self-contained.  Matching here only withholds CONTEXT_NOT_NEEDED, so
# the gate falls through to UNCERTAIN and the existing classifier makes the final call.
_MODIFIER_ONLY_REMAINDER = re.compile(
    r"^\w+\s+(?:"
    r"again|once more|simpler|simply|differently|briefly|slowly|clearly|more|better|"
    r"another way|in (?:a )?(?:simpler|easier|short|detail|brief) ?(?:way|form)?|"
    r"in (?:hindi|english)|step by step|step \d+|"
    r"dobara|phir se|aur (?:simple|easy|aasan|asaan)|"
    r"(?:hindi|english|aasan|asaan|simple) (?:me|mein)"
    r")\s*[?.!]?$",
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

# Grammar, not subject vocabulary: an anaphor points backwards, so a turn that offers no
# content word before it is pointing outside itself.
_FUNCTION_WORD = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being "
    "do does did can could would should will shall may might must what which who whom "
    "whose when where why how if then than that this these those there here it its "
    "they them their he she his her not no nor yes all any both each more most some "
    "such as so too very just only also about into over under again only".split()
)
_ANAPHOR = re.compile(
    r"\b(?:it|its|they|them|their|theirs|these|those|he|him|his|she|her|hers)\b"
)
# "is it true that ...", "is it safe to ..." — the clause the pronoun stands for follows it
# in the same turn, so nothing points outside.  The linking verb may sit on either side of
# the pronoun, since a question inverts it.
_LINK_VERB = frozenset(
    "is was are were be been seems seem appears appear takes take matters matter "
    "helps help means mean follows follow".split()
)
_EXPLETIVE_CLAUSE = re.compile(r"[^?.!]{0,64}?\b(?:that|to|whether)\b")
# A bare label — "F", "H", "G1" — names something the turn must have introduced, or that
# an earlier turn did.  "A" and "I" are excluded: they are an article and a pronoun.
_SYMBOLIC_ENTITY = re.compile(r"(?<![A-Za-z0-9])[B-HJ-Z](?:\d{1,2})?(?![A-Za-z0-9])")
_SYMBOL_BINDING = re.compile(r"[=<>+\-*/^]|\d")
# "Q1." or "b)" at the start of a line numbers the question; it names nothing.
_ENUMERATION_LABEL = re.compile(r"(?:^|\n)\s*[A-Za-z]?\d{0,2}\s*[.):]")
# "Class B", "Theory X", "Hepatitis B" — a capitalised noun in front makes the letter part
# of a name, not a label standing on its own.
_NAMED_BY_PRECEDING_NOUN = re.compile(r"[A-Z][a-z]+\s+$")
_REFERENCED_LABELS = 2
# A turn this long is stating its own material; a turn leaning on an earlier one is short.
_SELF_SUPPLYING_WORDS = 40
_CONDITIONAL_SELF_SUPPLYING_WORDS = 20


def _is_expletive(normalized: str, match: re.Match[str]) -> bool:
    """Does this "it" stand for a clause the same turn goes on to state?"""
    if not _EXPLETIVE_CLAUSE.match(normalized, match.end()):
        return False
    before = _WORD.findall(normalized[: match.start()])
    after = _WORD.findall(normalized[match.end() :])
    return bool(before and before[-1] in _LINK_VERB) or bool(after and after[0] in _LINK_VERB)


def _has_unresolved_anaphor(normalized: str) -> bool:
    for match in _ANAPHOR.finditer(normalized):
        if match.group(0) == "it" and _is_expletive(normalized, match):
            continue
        preceding = _WORD.findall(normalized[: match.start()])
        if not any(word not in _FUNCTION_WORD for word in preceding):
            return True
    return False


def _is_named_by_preceding_noun(before: str) -> bool:
    """Is the letter part of a name, as in "Class B"? A sentence-initial word is not one."""
    match = _NAMED_BY_PRECEDING_NOUN.search(before)
    return match is not None and match.start() > 0


def _has_unbound_symbolic_entity(query: str, normalized: str) -> bool:
    # Cheapest test first: a turn long enough to state its own material carries the labels
    # it uses, whatever they are, so the per-label work below would be discarded anyway.
    words = len(_WORD.findall(normalized))
    if words >= _SELF_SUPPLYING_WORDS or (
        words >= _CONDITIONAL_SELF_SUPPLYING_WORDS and _COMPLETE_CONDITION.search(normalized)
    ):
        return False
    matches = [
        match
        for match in _SYMBOLIC_ENTITY.finditer(query)
        if not _ENUMERATION_LABEL.match(query, max(0, match.start() - 1))
        and not _is_named_by_preceding_noun(query[: match.start()])
    ]
    # One letter on its own is usually ordinary content ("vitamin C", "the X chromosome").
    # A turn that leans on an earlier turn's cast names more than one of its members, and a
    # short single-label turn is already short of standalone evidence without this signal.
    if len({match.group(0) for match in matches}) < _REFERENCED_LABELS:
        return False
    return any(
        not _SYMBOL_BINDING.search(
            _SYMBOLIC_ENTITY.sub(" ", query[max(0, match.start() - 12) : match.end() + 12])
        )
        for match in matches
    )


def _unresolved_reference_signals(query: str, normalized: str) -> tuple[str, ...]:
    """Name the material this turn leans on but never supplies.

    Surface completeness is not semantic completeness: a turn can be a well-formed
    question of any length and still be unanswerable without the entities or rules the
    previous turn established.  Naming a signal here only withholds the standalone
    certification, so the gate falls through to UNCERTAIN and the existing candidate
    loading and compatibility checks decide whether any history is actually relevant.
    """
    signals = []
    if _has_unbound_symbolic_entity(query, normalized):
        signals.append("unbound_symbolic_entity")
    if _has_unresolved_anaphor(normalized):
        signals.append("unresolved_anaphor")
    return tuple(signals)


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
            return len(words) >= 2 and not _MODIFIER_ONLY_REMAINDER.match(normalized)
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
        assessment = self._decide(query, started)
        if assessment.decision != "CONTEXT_NOT_NEEDED":
            return assessment
        signals = _unresolved_reference_signals(query, normalize_context_query(query))
        if not signals:
            return assessment
        # The turn looked complete but points at material it never supplies.  Checking
        # recent context is cheaper than answering without the rules it depends on.
        return ContextNeedAssessment(
            decision="UNCERTAIN",
            reason_codes=("unresolved_local_reference", *assessment.reason_codes),
            matched_signals=(*signals, *assessment.matched_signals)[:8],
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _decide(self, query: str, started: float) -> ContextNeedAssessment:
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
