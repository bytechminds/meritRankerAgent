"""Bounded, Memory-first conversation relationship understanding."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from observability import (
    log_event,
    record_local_preview,
    update_request_summary,
    update_request_type,
)
from schemas.conversation import (
    ConversationRelation,
    ConversationUnderstandingResult,
    RecentConversationTurn,
)
from services.conversation.persistence import ConversationPersistenceService
from services.conversation.recent_context import format_recent_conversation

_WORD_PATTERN = re.compile(r"[a-zA-Z\u0900-\u097f]{2,}")
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:[$₹€£]\s*)?-?\d+(?:\.\d+)?\s*(?:%|percent|प्रतिशत)?",
    re.IGNORECASE,
)
_OPTION_PATTERN = re.compile(r"\boption\s*([a-d1-4])\b", re.IGNORECASE)
_FORMULA_SYMBOL_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,8}|sqrt|sin|cos|tan|log|[xyz])\b"
)
_STOP_WORDS = {
    "and",
    "about",
    "answer",
    "answers",
    "apply",
    "applied",
    "asked",
    "calculate",
    "calculated",
    "calculating",
    "can",
    "could",
    "did",
    "explain",
    "from",
    "formula",
    "have",
    "how",
    "into",
    "is",
    "it",
    "mark",
    "marks",
    "of",
    "out",
    "please",
    "question",
    "relevant",
    "score",
    "scored",
    "student",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "used",
    "using",
    "was",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "would",
    "you",
    "your",
}
_REFERENCE_WORDS = {
    "aapne",
    "above",
    "add",
    "aaya",
    "answer",
    "apply",
    "calculate",
    "calculated",
    "clarify",
    "continue",
    "divide",
    "earlier",
    "explain",
    "formula",
    "get",
    "got",
    "kaise",
    "kahan",
    "kiya",
    "kyu",
    "last",
    "mean",
    "multiply",
    "nikala",
    "operation",
    "pattern",
    "previous",
    "step",
    "subtract",
    "use",
    "used",
    "value",
    "why",
}

_ASSISTANT_ACTION = re.compile(
    r"\b(?:"
    r"how\s+(?:did\s+)?(?:you|u)\s+(?:calculate|calculated|get|got)|"
    r"how\s+(?:was|is)\s+(?:this|that|it|ye|yeh)?\s*(?:calculated|calculate)|"
    r"why\s+did\s+(?:you|u)\s+(?:use|apply|divide)|"
    r"which\s+operation\s+did\s+(?:you|u)\s+(?:use|apply)|"
    r"where\s+did\s+(?:this|that|the)?\s*value\s+come\s+from|"
    r"(?:you|u)\s+(?:calculated|applied|said|used)|"
    r"aapne\s+.*(?:kaise|kyu|nikala|lagaya|calculate)|"
    r"(?:ye|yeh)\s+(?:value|formula)\s+(?:kahan|kyu)|"
    r"(?:kaise\s+nikala|kaise\s+aaya)"
    r")\b",
    re.IGNORECASE,
)
_PREVIOUS_REFERENCE = re.compile(
    r"\b(?:last|previous|earlier|above|pichla|pichli|pichle|pehle\s+wala)"
    r"\s*(?:answer|question|step|pattern|formula|उत्तर|सवाल|चरण)?\b",
    re.IGNORECASE,
)
_PRONOUN_REFERENCE = re.compile(
    r"\b(?:this|that|it|these|those|ye|yeh|isko|usko)\b",
    re.IGNORECASE,
)
_CLARIFICATION_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:explain|samjhao|samjhaao|what\s+does).*(?:step|formula|mean|मतलब)|"
    r"what\s+was\s+the\s+pattern"
    r")\b",
    re.IGNORECASE,
)
_CONTINUATION_REFERENCE = re.compile(
    r"\b(?:continue|go\s+on|carry\s+on|aage|आगे)\b",
    re.IGNORECASE,
)
_CORRECTION_REFERENCE = re.compile(
    r"\b(?:that(?:'s|\s+is)\s+wrong|you\s+are\s+wrong|गलत|galat|correct\s+that)\b",
    re.IGNORECASE,
)
_REGENERATION_REFERENCE = re.compile(
    r"\b(?:regenerate|try\s+again|another\s+method|different\s+method|dobara)\b",
    re.IGNORECASE,
)
_BARE_RESULT_REFERENCE = re.compile(
    r"^\s*(?:why|how|kaise|kyu|क्यों|कैसे)\s+"
    r"(?:[$₹€£]?\s*-?\d+(?:\.\d+)?\s*(?:%|percent|प्रतिशत)?|option\s*[a-d1-4])"
    r"\s*\??\s*$",
    re.IGNORECASE,
)
_CALCULATION_REFERENCE = re.compile(
    r"\b(?:"
    r"how\s+(?:did|do)?\s*(?:you|u)?\s*(?:calculate|calculated|get|got)|"
    r"why\s+(?:did\s+you\s+)?(?:divide|multiply|add|subtract|use)|"
    r"(?:kaise|कैसे)\s+(?:nikala|aaya|calculate)"
    r")\b",
    re.IGNORECASE,
)
_ELLIPTICAL_REFERENCE = re.compile(
    r"^\s*(?:why|how|explain|what\s+about|kyu|kaise|क्यों|कैसे)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _TurnScore:
    turn: RecentConversationTurn
    position: str
    score: float
    signals: tuple[str, ...]


def _normalized_number(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.lower())
    normalized = normalized.replace("percent", "%").replace("प्रतिशत", "%")
    return normalized


def _entities(text: str) -> set[str]:
    values = {_normalized_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(text)}
    values.update(f"option:{match.group(1).lower()}" for match in _OPTION_PATTERN.finditer(text))
    values.update(
        f"formula:{match.group(0).lower()}"
        for match in _FORMULA_SYMBOL_PATTERN.finditer(text)
    )
    return values


def _concepts(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_PATTERN.findall(text)
        if token.lower() not in _STOP_WORDS
    }


def _ordered_concepts(text: str) -> list[str]:
    values: list[str] = []
    for token in _WORD_PATTERN.findall(text):
        normalized = token.lower()
        if normalized in _STOP_WORDS or normalized in values:
            continue
        values.append(normalized)
    return values


def _base_signals(query: str) -> list[str]:
    signals: list[str] = []
    for name, pattern in (
        ("assistant_action_reference", _ASSISTANT_ACTION),
        ("previous_turn_reference", _PREVIOUS_REFERENCE),
        ("pronoun_reference", _PRONOUN_REFERENCE),
        ("clarification_reference", _CLARIFICATION_REFERENCE),
        ("continuation_reference", _CONTINUATION_REFERENCE),
        ("correction_reference", _CORRECTION_REFERENCE),
        ("regeneration_reference", _REGENERATION_REFERENCE),
        ("bare_result_reference", _BARE_RESULT_REFERENCE),
        ("calculation_reference", _CALCULATION_REFERENCE),
        ("elliptical_reference", _ELLIPTICAL_REFERENCE),
    ):
        if pattern.search(query):
            signals.append(name)
    return signals


def _relation_name(signals: set[str]) -> str:
    if "regeneration_reference" in signals:
        return "regeneration_request"
    if "correction_reference" in signals:
        return "correction"
    if "continuation_reference" in signals:
        return "continuation"
    if "clarification_reference" in signals:
        return "clarification_of_previous"
    return "follow_up"


def _resolve_query(
    query: str,
    selected: tuple[RecentConversationTurn, ...],
    relation: str,
) -> str:
    concepts: list[str] = []
    for turn in selected:
        for concept in _ordered_concepts(turn.original_query):
            if concept not in concepts:
                concepts.append(concept)
    concepts = concepts[:12]
    referenced_entities = sorted(_entities(query))[:8]
    selected_concepts = set().union(
        *(_concepts(turn.original_query) | _concepts(turn.final_answer) for turn in selected)
    )
    referenced_concepts = [
        concept
        for concept in _ordered_concepts(query)
        if concept not in _REFERENCE_WORDS and concept in selected_concepts
    ][:8]
    topic = ", ".join(concepts) or "the selected academic problem"
    values = ", ".join(
        (
            value.removeprefix("formula:").upper()
            if value.startswith("formula:")
            else (
                f"option {value.removeprefix('option:').upper()}"
                if value.startswith("option:")
                else value
            )
        )
        for value in referenced_entities
    )
    if relation == "correction":
        instruction = f"Re-evaluate and correct the solution for: {topic}."
    elif relation == "regeneration_request":
        instruction = f"Provide a different valid solution method for: {topic}."
    elif relation == "continuation":
        instruction = f"Continue the solution for: {topic}."
    elif values:
        instruction = (
            f"Explain how or why the referenced value or result {values} "
            f"is obtained in: {topic}."
        )
    elif referenced_concepts:
        instruction = (
            f"Explain how or why {', '.join(referenced_concepts)} applies in: {topic}."
        )
    else:
        instruction = f"Explain the relevant formula, operation, or step for: {topic}."
    return instruction[:5000]


class ConversationUnderstandingService:
    """Load bounded context, establish relevance, and resolve contextual input."""

    def __init__(self, *, persistence: ConversationPersistenceService) -> None:
        self._persistence = persistence

    def understand(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        query: str,
    ) -> ConversationUnderstandingResult:
        loaded = self._persistence.load_recent_context(actor_id, conversation_id, 2)
        started = time.monotonic()
        base_signals = _base_signals(query)
        signal_set = set(base_signals)
        query_entities = _entities(query)
        query_concepts = _concepts(query)
        specific_query_concepts = query_concepts - _REFERENCE_WORDS

        scored: list[_TurnScore] = []
        turns = tuple(loaded.turns[-2:])
        for index, turn in enumerate(turns):
            position = "latest_turn" if index == len(turns) - 1 else "previous_of_two"
            combined = f"{turn.original_query}\n{turn.final_answer}"
            entity_overlap = sorted(query_entities & _entities(combined))
            concept_overlap = specific_query_concepts & _concepts(combined)
            signals = list(base_signals)
            score = 0.08 if position == "latest_turn" else 0.04
            if "assistant_action_reference" in signal_set:
                score += 0.45
            if "previous_turn_reference" in signal_set:
                score += 0.4
            if "clarification_reference" in signal_set:
                score += 0.4
            if signal_set & {
                "continuation_reference",
                "correction_reference",
                "regeneration_reference",
            }:
                score += 0.5
            if "pronoun_reference" in signal_set:
                score += 0.4
            if "bare_result_reference" in signal_set:
                score += 0.3
            if "calculation_reference" in signal_set:
                score += 0.25
            if "elliptical_reference" in signal_set:
                score += 0.15
            if entity_overlap:
                for value in entity_overlap:
                    if value.startswith("formula:"):
                        signals.append(f"formula_reference={value.removeprefix('formula:')}")
                    elif value.startswith("option:"):
                        signals.append(f"option_reference={value.removeprefix('option:')}")
                    else:
                        signals.append(f"numeric_reference={value}")
                score += min(0.4, 0.25 + 0.05 * len(entity_overlap))
            if concept_overlap:
                signals.append("semantic_relevance")
                score += min(0.15, len(concept_overlap) * 0.03)
            scored.append(
                _TurnScore(
                    turn=turn,
                    position=position,
                    score=min(score, 1.0),
                    signals=tuple(dict.fromkeys(signals)),
                )
            )

        contextual_signal = bool(
            signal_set
            & {
                "assistant_action_reference",
                "previous_turn_reference",
                "clarification_reference",
                "continuation_reference",
                "correction_reference",
                "regeneration_reference",
                "bare_result_reference",
                "calculation_reference",
                "pronoun_reference",
                "elliptical_reference",
            }
        )
        ranked = sorted(scored, key=lambda candidate: candidate.score, reverse=True)
        best = ranked[0] if ranked else None
        selected: tuple[RecentConversationTurn, ...] = ()
        selection_signals: tuple[str, ...] = tuple(base_signals)
        selection_confidence = 0.0
        position = "none"

        has_relevance_overlap = bool(
            best
            and (
                "semantic_relevance" in best.signals
                or any(
                    signal.startswith(
                        ("numeric_reference=", "formula_reference=", "option_reference=")
                    )
                    for signal in best.signals
                )
            )
        )
        self_contained_topic_mismatch = bool(
            specific_query_concepts
            and not any(
                specific_query_concepts
                & _concepts(f"{candidate.turn.original_query}\n{candidate.turn.final_answer}")
                for candidate in ranked
            )
            and not has_relevance_overlap
        )
        competing_context_is_ambiguous = bool(
            len(ranked) > 1
            and best is not None
            and not has_relevance_overlap
            and ranked[0].score - ranked[1].score < 0.1
            and not signal_set
            & {
                "previous_turn_reference",
                "continuation_reference",
                "correction_reference",
                "regeneration_reference",
            }
        )
        minimum_score = 0.25 if has_relevance_overlap else 0.45

        if (
            best is not None
            and contextual_signal
            and best.score >= minimum_score
            and not self_contained_topic_mismatch
            and not competing_context_is_ambiguous
        ):
            selected = (best.turn,)
            selection_signals = best.signals
            selection_confidence = max(0.75, best.score)
            position = best.position

        if selected:
            relation_name = _relation_name(set(selection_signals))
            relation = ConversationRelation(
                relation=relation_name,
                requires_recent_conversation=True,
                referenced_turn_id=selected[0].turn_id,
                referenced_turn_position=position,
                confidence=selection_confidence,
                decision_source="deterministic_signal",
                matched_signals=selection_signals,
            )
            context = format_recent_conversation(list(selected), source={
                "agentcore_memory": "agentcore",
                "dynamodb_fallback": "dynamodb",
                "none": "none",
            }[loaded.source])
            resolution_started = time.monotonic()
            resolved_query = _resolve_query(query, selected, relation_name)
            resolution_duration_ms = int(
                (time.monotonic() - resolution_started) * 1000
            )
        elif contextual_signal and not self_contained_topic_mismatch:
            relation = ConversationRelation(
                relation="ambiguous",
                requires_recent_conversation=True,
                confidence=0.55 if turns else 0.35,
                decision_source="deterministic_signal" if turns else "no_context",
                matched_signals=tuple(base_signals),
            )
            context = format_recent_conversation([], source="none")
            resolved_query = None
            resolution_duration_ms = 0
        else:
            relation = ConversationRelation(
                relation="independent",
                requires_recent_conversation=False,
                confidence=0.9,
                decision_source="deterministic_signal" if turns else "no_context",
                matched_signals=(),
            )
            context = format_recent_conversation([], source="none")
            resolved_query = query
            resolution_duration_ms = 0

        duration_ms = int((time.monotonic() - started) * 1000)
        final_type = "follow_up" if relation.requires_recent_conversation else "standalone"
        update_request_type(final_type)
        update_request_summary(
            request_type=final_type,
            context_required=relation.requires_recent_conversation,
            context_source=loaded.source,
            usable_recent_turns=loaded.usable_turn_count,
        )
        log_event(
            "conversation_relation_completed",
            component="conversation.understanding",
            stage="conversation_relation",
            status="completed",
            duration_ms=duration_ms,
            details={
                "relation": relation.relation,
                "confidence": relation.confidence,
                "decision_source": relation.decision_source,
                "matched_signals": ",".join(relation.matched_signals),
            },
        )
        if selected:
            log_event(
                "conversation_context_selected",
                component="conversation.understanding",
                stage="select_context",
                status="completed",
                details={
                    "selected_turn_id": selected[0].turn_id,
                    "selected_turn_position": position,
                    "selection_reason": ",".join(selection_signals),
                    "selection_confidence": selection_confidence,
                },
            )
            record_local_preview("selected_turn_id", selected[0].turn_id)
            record_local_preview("selection_reason", ",".join(selection_signals))
            record_local_preview("selected_previous_user", selected[0].original_query)
            record_local_preview("selected_previous_assistant", selected[0].final_answer)
        if resolved_query and relation.requires_recent_conversation:
            record_local_preview("resolved_query", resolved_query)
            log_event(
                "follow_up_resolution_completed",
                component="conversation.follow_up",
                stage="resolve_follow_up",
                status="completed",
                duration_ms=resolution_duration_ms,
                details={"confidence": relation.confidence},
            )

        return ConversationUnderstandingResult(
            relation=relation,
            context_load=loaded,
            selected_turns=selected,
            selection_reason=selection_signals,
            selection_confidence=selection_confidence,
            resolved_query=resolved_query,
            conversation_context=context.formatted_reference,
        )
