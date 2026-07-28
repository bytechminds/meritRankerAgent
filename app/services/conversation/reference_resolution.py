"""Bounded analysis for local and conversation-dependent references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.conversation import ConversationCandidateCard

_WORD = re.compile(r"[A-Za-z\u0900-\u097f]+")
_PERSON_REFERENCES = frozenset(
    {
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "woh",
        "wo",
        "usne",
        "uska",
        "uski",
        "unka",
        "unhone",
    }
)
_OBJECT_REFERENCES = frozenset(
    {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "ye",
        "yeh",
        "isko",
        "ise",
        "iska",
        "usko",
    }
)
_ORDINAL_REFERENCE = re.compile(
    r"\b(?:the\s+)?(?:first|second|third|last|other|previous)\s+"
    r"(?:one|option|answer|method|step)\b|"
    r"\b(?:pichla wala|dusra option|doosra option|last option)\b",
    re.IGNORECASE,
)
_METHOD_REFERENCE = re.compile(
    r"\b(?:this|that|same|other|previous|yeh|ye)\s+"
    r"(?:method|formula|process|step|operation|concept|tarika)\b|"
    r"\b(?:why|how)\s+(?:was|is|did)?\s*(?:it|this|that|isko|ise)\s+"
    r"(?:used|applied|divided|subtracted|added|multiplied|removed|replaced)\b",
    re.IGNORECASE,
)
_EVENT_REFERENCE = re.compile(
    r"\b(?:after (?:that|his|her|their)|before (?:this|that)|"
    r"during (?:his|her|their)|what happened next|"
    r"(?:his|her|their)\s+(?:death|reign|rule))\b",
    re.IGNORECASE,
)
_PERSON_EVIDENCE = re.compile(
    r"\b(?:king|queen|ruler|emperor|empress|leader|president|minister|"
    r"historical figure|son|daughter|father|mother|wife|wives|husband|"
    r"reign|rule|ruled|battle|born|died|death|married|succeeded|lived)\b",
    re.IGNORECASE,
)
_METHOD_EVIDENCE = re.compile(
    r"\b(?:method|formula|equation|step|operation|process|procedure|"
    r"calculate|solve|subtract|divide|multiply|percentage|ratio)\b|[%=+\-*/]",
    re.IGNORECASE,
)
_ORDINAL_EVIDENCE = re.compile(
    r"\b(?:option|choice|method|step|first|second|third|last|"
    r"[a-d]\)|[1-9]\.)\b",
    re.IGNORECASE,
)
_EVENT_EVIDENCE = re.compile(
    r"\b(?:history|reign|ruled|death|battle|war|event|timeline|"
    r"before|after|during|became|ended|began)\b",
    re.IGNORECASE,
)
_CAPITALIZED_NAME = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:(?:of|the|al|bin|de)\s+)?[A-Z][a-z]+){0,3}\b"
)
_NON_ENTITY_WORDS = frozenset(
    {
        "a",
        "an",
        "answer",
        "answers",
        "assistant",
        "based",
        "brief",
        "the",
        "this",
        "that",
        "these",
        "those",
        "to",
        "please",
        "question",
        "questions",
        "response",
        "solution",
        "someone",
        "person",
        "thing",
        "reference",
        "explain",
        "describe",
        "compare",
        "calculate",
        "can",
        "could",
        "compute",
        "find",
        "final",
        "solve",
        "will",
        "would",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "did",
        "does",
        "do",
        "was",
        "were",
        "is",
        "are",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "i",
        "you",
        "yes",
        "no",
        "okay",
        "however",
        "therefore",
        "because",
        "after",
        "before",
        "during",
        "while",
        "then",
    }
)
_GENERIC_ENTITY_LABELS = frozenset(
    {
        "academic",
        "any",
        "answer",
        "concept",
        "context",
        "example",
        "general",
        "history",
        "information",
        "method",
        "option",
        "question",
        "response",
        "result",
        "solution",
        "step",
        "topic",
        "unknown",
    }
)
_LABEL_WORD = re.compile(r"[A-Za-z\u0900-\u097f]+")
_SENTENCE_REFERENCE = re.compile(
    r"[.!?]\s+(?:did|does|do|was|were|is|are|can|could|would|why|how|what)?"
    r"\s*(?:he|him|his|she|her|they|them|their|it|this|that)\b",
    re.IGNORECASE,
)
_LOCAL_NOUN_ANTECEDENT = re.compile(
    r"\b(?:a|an|the|this|that)\s+[a-z\u0900-\u097f]{3,}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReferenceAnalysis:
    """Safe structural facts about references in the current message."""

    local_reference: bool
    external_reference_detected: bool
    reference_types: tuple[str, ...]
    reference_terms: tuple[str, ...]


@dataclass(frozen=True)
class CandidateCompatibility:
    """Compatibility result used only for validation and bounded fallback."""

    turn_id: str
    compatible: bool
    reason: str
    recency_rank: int
    grounded_antecedent: bool
    label: str | None = None


@dataclass(frozen=True)
class CompatibleEntityGroup:
    """Compatible candidates sharing one grounded identity."""

    identity: str
    label: str | None
    turn_ids: tuple[str, ...]
    selected_turn_id: str


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in _WORD.findall(value))


def _reference_terms(query: str) -> tuple[str, ...]:
    words = _normalized_words(query)
    terms = {
        word
        for word in words
        if word in _PERSON_REFERENCES or word in _OBJECT_REFERENCES
    }
    for pattern in (_ORDINAL_REFERENCE, _METHOD_REFERENCE, _EVENT_REFERENCE):
        terms.update(match.group(0).casefold() for match in pattern.finditer(query))
    return tuple(sorted(terms))


def _has_local_antecedent(query: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    if _SENTENCE_REFERENCE.search(query):
        return True
    words = _WORD.findall(query)
    lowered = [word.casefold() for word in words]
    reference_indexes = [
        index
        for index, word in enumerate(lowered)
        if word in _PERSON_REFERENCES or word in _OBJECT_REFERENCES
    ]
    if not reference_indexes:
        return False
    first_reference = min(reference_indexes)
    if first_reference < 1:
        return False
    prefix = words[:first_reference]
    prefix_text = " ".join(prefix)
    return bool(
        _entity_names(prefix_text)
        or _LOCAL_NOUN_ANTECEDENT.search(prefix_text)
    )


def analyze_reference(query: str) -> ReferenceAnalysis:
    """Distinguish locally resolved references from external references."""
    terms = _reference_terms(query)
    words = set(_normalized_words(query))
    reference_types: set[str] = set()
    if words & _PERSON_REFERENCES:
        reference_types.add("person")
    if words & _OBJECT_REFERENCES:
        reference_types.add("object")
    if _ORDINAL_REFERENCE.search(query):
        reference_types.add("ordinal")
    if _METHOD_REFERENCE.search(query):
        reference_types.add("concept")
    if _EVENT_REFERENCE.search(query):
        reference_types.add("event")
    local = _has_local_antecedent(query, terms)
    return ReferenceAnalysis(
        local_reference=local,
        external_reference_detected=bool(terms) and not local,
        reference_types=tuple(sorted(reference_types)),
        reference_terms=terms,
    )


def _candidate_text(card: ConversationCandidateCard) -> str:
    return " ".join(
        (
            card.question_preview,
            card.answer_clue,
            card.subject,
            card.topic or "",
        )
    )


def candidate_label(card: ConversationCandidateCard) -> str | None:
    """Return a bounded human-safe candidate label when one is evident."""
    names = grounded_entity_labels(card)
    if names:
        return names[0]
    return None


def validate_entity_label(value: str | None) -> str | None:
    """Validate one bounded entity label; never repair or invent a label."""
    if value is None:
        return None
    candidate = " ".join(value.replace("\r", " ").replace("\n", " ").split()).strip(
        " *_`#>:;,.!?\"'“”‘’()[]{}"
    )
    if not candidate or len(candidate) > 48:
        return None
    words = _LABEL_WORD.findall(candidate)
    if not words or len(words) > 6:
        return None
    meaningful = [
        word
        for word in words
        if word.casefold() not in _NON_ENTITY_WORDS
        and word.casefold() not in {"of", "the", "al", "bin", "de"}
    ]
    if not meaningful:
        return None
    normalized = " ".join(word.casefold() for word in meaningful)
    if normalized in _GENERIC_ENTITY_LABELS:
        return None
    if len(meaningful) == 1 and len(meaningful[0]) < 3:
        return None
    return candidate


def grounded_entity_labels(card: ConversationCandidateCard) -> tuple[str, ...]:
    """Extract only explicit, validated entity labels in documented priority order."""
    labels: list[str] = []
    for source in (card.question_preview, card.answer_clue):
        for raw_label in _entity_names(source):
            label = validate_entity_label(raw_label)
            if label is not None and label.casefold() not in {
                existing.casefold() for existing in labels
            }:
                labels.append(label)
    if not labels:
        readable_topic = card.topic.replace("_", " ").title() if card.topic else None
        metadata_label = validate_entity_label(readable_topic)
        if metadata_label is not None:
            labels.append(metadata_label)
    return tuple(labels)


def _candidate_grounding(
    analysis: ReferenceAnalysis,
    card: ConversationCandidateCard,
) -> tuple[bool, str]:
    text = _candidate_text(card)
    label = candidate_label(card)
    if "ordinal" in analysis.reference_types:
        grounded = bool(_ORDINAL_EVIDENCE.search(text))
        return grounded, "ordered_content" if grounded else "no_ordered_content"
    if "concept" in analysis.reference_types:
        grounded = bool(_METHOD_EVIDENCE.search(text))
        return grounded, "method_or_formula" if grounded else "no_method_or_formula"
    if "event" in analysis.reference_types:
        grounded = bool(label and (_EVENT_EVIDENCE.search(text) or _PERSON_EVIDENCE.search(text)))
        return grounded, "grounded_event_or_person" if grounded else "no_grounded_event_or_person"
    if "person" in analysis.reference_types:
        grounded = bool(label and _PERSON_EVIDENCE.search(text))
        if grounded:
            return True, "grounded_person_entity"
        if _METHOD_EVIDENCE.search(text):
            return False, "non_person_method_content"
        return False, "grounded_person_unproven"
    if "object" in analysis.reference_types:
        grounded = bool(
            label
            or _METHOD_EVIDENCE.search(text)
            or _ORDINAL_EVIDENCE.search(text)
            or _EVENT_EVIDENCE.search(text)
        )
        return grounded, "grounded_object_or_concept" if grounded else "no_grounded_object"
    return False, "no_reference_grounding"


def _explicit_current_names(query: str) -> frozenset[str]:
    return frozenset(name.casefold() for name in _entity_names(query))


def _entity_names(value: str) -> tuple[str, ...]:
    names: list[str] = []
    for match in _CAPITALIZED_NAME.findall(value):
        parts = match.split()
        while parts and parts[0].casefold() in _NON_ENTITY_WORDS:
            parts.pop(0)
        while parts and parts[-1].casefold() in _NON_ENTITY_WORDS:
            parts.pop()
        if not parts:
            continue
        candidate = validate_entity_label(" ".join(parts))
        if candidate is not None:
            names.append(candidate)
    return tuple(names)


def assess_candidate_compatibility(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> tuple[CandidateCompatibility, ...]:
    """Apply conservative generic type compatibility before recency."""
    analysis = analyze_reference(query)
    explicit_names = _explicit_current_names(query)
    results: list[CandidateCompatibility] = []
    total = len(cards)
    for index, card in enumerate(cards):
        label = candidate_label(card)
        compatible, reason = _candidate_grounding(analysis, card)
        grounded_antecedent = compatible
        if (
            compatible
            and label is not None
            and label.casefold() in explicit_names
            and len(explicit_names) == 1
        ):
            compatible = False
            reason = "explicit_comparison_entity"
        results.append(
            CandidateCompatibility(
                turn_id=card.turn_id,
                compatible=compatible,
                reason=reason,
                recency_rank=total - index,
                grounded_antecedent=grounded_antecedent,
                label=label,
            )
        )
    return tuple(results)


def group_compatible_candidates(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> tuple[CompatibleEntityGroup, ...]:
    """Group compatible cards by grounded identity, preserving newest selection."""
    grouped: dict[str, list[CandidateCompatibility]] = {}
    order: list[str] = []
    for result in assess_candidate_compatibility(query, cards):
        if not result.compatible or not result.grounded_antecedent:
            continue
        identity = (
            f"label:{result.label.casefold()}"
            if result.label is not None
            else f"turn:{result.turn_id}"
        )
        if identity not in grouped:
            grouped[identity] = []
            order.append(identity)
        grouped[identity].append(result)
    return tuple(
        CompatibleEntityGroup(
            identity=identity,
            label=members[0].label,
            turn_ids=tuple(member.turn_id for member in members),
            selected_turn_id=members[-1].turn_id,
        )
        for identity in order
        for members in (grouped[identity],)
    )


def compatible_turn_ids(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> tuple[str, ...]:
    return tuple(
        result.turn_id
        for result in assess_candidate_compatibility(query, cards)
        if result.compatible
    )


def unambiguous_compatible_turn_id(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> str | None:
    """Return one candidate, collapsing only repeated cards for the same entity."""
    groups = group_compatible_candidates(query, cards)
    return groups[0].selected_turn_id if len(groups) == 1 else None


def has_multiple_compatible_entities(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> bool:
    """Return true when compatible candidates expose different entity labels."""
    return len(group_compatible_candidates(query, cards)) > 1


def definitively_incompatible_turn_ids(
    query: str,
    cards: tuple[ConversationCandidateCard, ...],
) -> tuple[str, ...]:
    """Return only candidates whose structure clearly conflicts with the reference."""
    definitive_reasons = {
        "no_ordered_content",
        "no_method_or_formula",
        "no_grounded_event_or_person",
        "non_person_method_content",
        "grounded_person_unproven",
        "no_grounded_object",
        "explicit_comparison_entity",
    }
    return tuple(
        result.turn_id
        for result in assess_candidate_compatibility(query, cards)
        if not result.compatible and result.reason in definitive_reasons
    )


def resolved_reference_text(
    query: str,
    card: ConversationCandidateCard,
) -> str | None:
    """Derive one bounded referent pair for generation without persistence."""
    analysis = analyze_reference(query)
    label = candidate_label(card)
    if not analysis.external_reference_detected or not label:
        return None
    reference = next(
        (
            term
            for term in analysis.reference_terms
            if term in _PERSON_REFERENCES or term in _OBJECT_REFERENCES
        ),
        analysis.reference_terms[0] if analysis.reference_terms else "reference",
    )
    return f"{reference} → {label}"[:128]
