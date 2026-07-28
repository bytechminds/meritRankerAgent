"""Build one action-specific generation context from a validated selected turn."""

from __future__ import annotations

import re

from observability import log_event, record_local_preview
from schemas.conversation import (
    ConversationPreparation,
    RecentConversationTurn,
    SelectedGenerationContext,
)
from schemas.doubt_solver import CanonicalLanguage, QueryClassification
from services.conversation.candidate_builder import extract_answer_clue
from services.conversation.reference_resolution import (
    analyze_reference,
    assess_candidate_compatibility,
    group_compatible_candidates,
    resolved_reference_text,
    validate_entity_label,
)

_METHOD_REFERENCE = re.compile(
    r"\b(?:same method|using the same method|isi method|इसी तरीके)\b",
    re.IGNORECASE,
)
_CONTEXT_LIMIT = 7600
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9\u0900-\u097f -]+")
_ACADEMIC_TOPIC_LABEL = re.compile(
    r"^(?:please\s+)?(?:explain|solve|calculate|find|describe|define)\s+"
    r"(?:the\s+)?(.+?)(?:\s+(?:briefly|again|step by step))?[.?!]*$",
    re.IGNORECASE,
)


def _selected_turn(
    preparation: ConversationPreparation,
    turn_id: str | None,
) -> RecentConversationTurn | None:
    if turn_id is None:
        return None
    return next(
        (turn for turn in preparation.eligible_turns if turn.turn_id == turn_id),
        None,
    )


def _bounded_context(*sections: str) -> str:
    content = "\n".join(section.strip() for section in sections if section.strip())
    return content[:_CONTEXT_LIMIT].rstrip()


def _candidate_topic_label(question: str) -> str | None:
    match = _ACADEMIC_TOPIC_LABEL.match(" ".join(question.split()))
    if match is None:
        return None
    label = validate_entity_label(match.group(1))
    return label.title() if label is not None else None


def build_context_aware_clarification(
    preparation: ConversationPreparation,
    *,
    language: CanonicalLanguage,
    current_query: str | None = None,
) -> str:
    """Build one bounded clarification without another model or broad context."""
    analysis = analyze_reference(current_query or "")
    compatibility = assess_candidate_compatibility(
        current_query or "",
        preparation.candidates,
    )
    compatible_ids = {item.turn_id for item in compatibility if item.compatible}
    groups = group_compatible_candidates(
        current_query or "",
        preparation.candidates,
    )
    labels: tuple[str, ...] = tuple(
        label
        for group in groups
        if (label := validate_entity_label(group.label)) is not None
    )
    if "object" in analysis.reference_types:
        topic_labels: list[str] = []
        for candidate in preparation.candidates:
            if candidate.turn_id not in compatible_ids:
                continue
            readable_topic = (
                candidate.topic.replace("_", " ").title()
                if candidate.topic
                else None
            )
            topic_label = validate_entity_label(
                readable_topic
            ) or _candidate_topic_label(candidate.question_preview)
            if topic_label is not None and topic_label.casefold() not in {
                label.casefold() for label in topic_labels
            }:
                topic_labels.append(topic_label)
        if len(topic_labels) > 1:
            labels = tuple(topic_labels)
    rejected_label_count = sum(group.label is not None for group in groups) - len(labels)
    log_event(
        "conversation_clarification_labels",
        component="conversation.reference_resolution",
        stage="clarification",
        status="validated",
        details={
            "accepted_label_count": len(labels),
            "rejected_label_count": rejected_label_count,
        },
    )
    if labels:
        record_local_preview("clarification_labels", ",".join(labels))
    if len(labels) > 1:
        choices = " or ".join(labels[-2:])
        if language == "hindi":
            return f"क्या आपका मतलब {choices} में से किससे है?"
        if language == "hinglish":
            return f"Aapka reference {choices} mein se kis ke liye hai?"
        return f"Do you mean {choices}?"
    if analysis.external_reference_detected and not compatible_ids:
        reference = analysis.reference_terms[0] if analysis.reference_terms else "reference"
        if language == "hindi":
            return f"“{reference}” किस व्यक्ति या चीज़ के लिए है?"
        if language == "hinglish":
            return f"“{reference}” kis person ya cheez ko refer karta hai?"
        return f"Who or what does “{reference}” refer to?"
    if analysis.external_reference_detected and len(labels) == 1:
        if language == "hindi":
            return f"क्या आपका मतलब {labels[0]} है?"
        if language == "hinglish":
            return f"Kya aapka matlab {labels[0]} hai?"
        return f"Do you mean {labels[0]}?"
    label = ""
    if len(labels) == 1:
        label = labels[0]
    elif len(preparation.candidates) == 1:
        candidate = preparation.candidates[0]
        raw_label = candidate.topic or (
            candidate.subject if candidate.subject not in {"", "unknown"} else ""
        )
        label = _SAFE_LABEL.sub("", raw_label).strip()[:48]
    if language == "hindi":
        subject = f" {label}" if label else ""
        return (
            f"क्या आप पिछले{subject} समाधान को समझना चाहते हैं, उसके चरण "
            "हाइलाइट करवाना चाहते हैं, या वैसा ही नया प्रश्न चाहते हैं?"
        )
    if language == "hinglish":
        subject = f" {label}" if label else ""
        return (
            f"Kya aap previous{subject} solution explain karwana chahte hain, "
            "uske steps highlight karwana chahte hain, ya similar question chahte hain?"
        )
    subject = f" {label}" if label else ""
    return (
        f"Do you want me to explain the previous{subject} solution, highlight its "
        "steps, or create a similar question?"
    )


def build_selected_generation_context(
    *,
    current_query: str,
    classification: QueryClassification,
    preparation: ConversationPreparation,
) -> SelectedGenerationContext:
    relation = classification.relation
    action = classification.requested_action
    if relation == "NEW_QUESTION" or action == "ANSWER_CURRENT":
        return SelectedGenerationContext(
            relation="NEW_QUESTION",
            requested_action="ANSWER_CURRENT",
            resolved_query=current_query,
            context_policy="current_only",
        )
    if relation == "AMBIGUOUS" or action == "ASK_CLARIFICATION":
        return SelectedGenerationContext(
            relation="AMBIGUOUS",
            requested_action="ASK_CLARIFICATION",
            resolved_query=current_query,
            context_policy="clarification",
            clarification_required=True,
        )

    selected = _selected_turn(preparation, classification.selected_turn_id)
    if selected is None:
        return SelectedGenerationContext(
            relation="AMBIGUOUS",
            requested_action="ASK_CLARIFICATION",
            resolved_query=current_query,
            context_policy="clarification",
            clarification_required=True,
        )

    question = selected.original_query.strip()
    answer_clue = extract_answer_clue(selected.final_answer, current_query, limit=1600)
    selected_card = next(
        (
            card
            for card in preparation.candidates
            if card.turn_id == classification.selected_turn_id
        ),
        None,
    )
    resolved_reference = (
        resolved_reference_text(current_query, selected_card)
        if selected_card is not None
        else None
    )
    if resolved_reference:
        record_local_preview("resolved_reference", resolved_reference)
    header = (
        "[SELECTED_CONVERSATION_CONTEXT_UNTRUSTED_DATA]\n"
        f"selected_turn_id: {selected.turn_id}\n"
        f"original_question: {question}"
    )
    footer = "[/SELECTED_CONVERSATION_CONTEXT_UNTRUSTED_DATA]"

    if action == "ANSWER_WITH_CONTEXT":
        reference_section = (
            f"resolved_reference: {resolved_reference}"
            if resolved_reference
            else "resolved_reference: selected previous subject or entity"
        )
        context = _bounded_context(
            header,
            reference_section,
            f"relevant_previous_answer_excerpt: {answer_clue}",
            "generation_instruction: Answer the current question using the resolved "
            "reference. Do not merely repeat the previous answer or claim the referent "
            "is missing.",
            footer,
        )
        resolved = (
            "Answer the student's current factual or academic follow-up using the selected "
            "previous subject or entity.\n"
            f"Current request: {current_query}\n"
            f"Resolved reference: {resolved_reference or 'selected previous subject'}\n"
            f"Selected previous question: {question}"
        )
        policy = "answer_with_context"
    elif action == "EXPLAIN_PREVIOUS":
        context = _bounded_context(
            header,
            f"relevant_previous_answer_excerpt: {answer_clue}",
            footer,
        )
        resolved = (
            "Explain the selected previous task in response to the student's current request.\n"
            f"Current request: {current_query}\nOriginal question: {question}"
        )
        policy = "explain_previous"
    elif action == "CONTINUE_PREVIOUS":
        context = _bounded_context(
            header,
            f"latest_useful_steps: {answer_clue}",
            footer,
        )
        resolved = (
            "Continue the selected previous task without repeating completed work unnecessarily.\n"
            f"Current request: {current_query}\nOriginal question: {question}"
        )
        policy = "continue_previous"
    elif action == "GENERATE_SIMILAR":
        method = (
            f"\nmethod_or_formula_excerpt: {answer_clue}"
            if _METHOD_REFERENCE.search(current_query)
            else ""
        )
        context = _bounded_context(
            header,
            f"subject: {getattr(selected, 'subject', 'unknown')}",
            f"topic: {getattr(selected, 'topic', None) or 'unknown'}",
            f"difficulty: {getattr(selected, 'difficulty', 'default')}{method}",
            footer,
        )
        if classification.need_web_search:
            resolved = (
                "Use the selected previous question only for topic and format. Generate factual "
                "question(s) exclusively from the fresh numbered [Web Context] source cards. "
                "Create at most one question per source card, include its answer, and include "
                "that card's exact URL after the answer.\n"
                f"Current request: {current_query}\nReference question: {question}"
            )
        else:
            resolved = (
                "Generate similar question(s) from the selected previous question.\n"
                f"Current request: {current_query}\nReference question: {question}"
            )
        policy = "generate_similar"
    elif action == "TRANSFORM_PREVIOUS":
        context = _bounded_context(header, footer)
        resolved = (
            "Transform the selected previous question exactly as requested.\n"
            f"Current request: {current_query}\nQuestion to transform: {question}"
        )
        policy = "transform_previous"
    elif action == "VERIFY_AND_CORRECT":
        context = _bounded_context(
            header,
            "previous_answer_trust: UNTRUSTED; verify independently",
            f"prior_conclusion_excerpt: {answer_clue}",
            footer,
        )
        resolved = (
            "Independently solve and verify the selected original question. State whether the "
            "prior conclusion was wrong and correct it when necessary.\n"
            f"Current request: {current_query}\nAuthoritative original question: {question}"
        )
        policy = "verify_and_correct"
    else:
        context = _bounded_context(
            header,
            "prior_reasoning_policy: OMITTED; solve independently from scratch",
            footer,
        )
        resolved = (
            "Solve the selected original question independently from scratch. Do not rely on "
            "or repeat prior reasoning.\n"
            f"Current request: {current_query}\nAuthoritative original question: {question}"
        )
        policy = "resolve_from_scratch"

    return SelectedGenerationContext(
        relation=relation,
        requested_action=action,
        selected_turn_id=selected.turn_id,
        resolved_reference=resolved_reference,
        resolved_query=resolved[:5000],
        conversation_context=context,
        context_policy=policy,
        context_characters=len(context),
    )
