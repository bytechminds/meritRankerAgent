"""Conditional conversation loading and compact candidate preparation."""

from __future__ import annotations

import logging

from observability import log_event, record_local_preview, update_request_summary
from schemas.conversation import (
    ContextNeedAssessment,
    ConversationPreparation,
    RecentContextLoadResult,
)
from schemas.doubt_solver import CanonicalLanguage
from services.conversation.candidate_builder import (
    EXPANDED_CANDIDATE_LIMIT,
    NORMAL_CANDIDATE_LIMIT,
    build_candidate_cards,
    format_candidate_cards,
)
from services.conversation.context_need_gate import ContextNeedGate
from services.conversation.memory_hygiene import (
    MemoryHygieneDecision,
    filter_substantive_turns,
)
from services.conversation.persistence import ConversationPersistenceService
from services.conversation.reference_resolution import (
    analyze_reference,
    assess_candidate_compatibility,
    grounded_entity_labels,
    group_compatible_candidates,
)

logger = logging.getLogger(__name__)


class ConversationUnderstandingService:
    """Gate context reads and prepare untrusted candidates; never classify them."""

    def __init__(
        self,
        *,
        persistence: ConversationPersistenceService,
        gate: ContextNeedGate | None = None,
    ) -> None:
        self._persistence = persistence
        self._gate = gate or ContextNeedGate()

    def understand(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        query: str,
        request_id: str = "conversation-understanding",
        exam_id: str | None = None,
        language: CanonicalLanguage = "english",
    ) -> ConversationPreparation:
        """Compatibility name used by graph/stream; returns preparation only."""
        del exam_id, language
        return self.prepare(
            actor_id=actor_id,
            conversation_id=conversation_id,
            query=query,
            request_id=request_id,
        )

    def prepare(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        query: str,
        request_id: str,
    ) -> ConversationPreparation:
        gate = self._gate.evaluate(query)
        self._log_gate(gate, request_id=request_id)
        if gate.decision == "CONTEXT_NOT_NEEDED":
            preparation = ConversationPreparation(
                gate=gate,
                context_load=RecentContextLoadResult(),
            )
            self._log_reference_analysis(
                query=query,
                preparation=preparation,
            )
            self._log_preparation(preparation, (), request_id=request_id)
            return preparation
        correction_chain = bool(
            {"correction_reference", "resolve_again_reference"}
            & set(gate.matched_signals)
        )
        return self._load_candidates(
            actor_id=actor_id,
            conversation_id=conversation_id,
            query=query,
            request_id=request_id,
            gate=gate,
            limit=(
                EXPANDED_CANDIDATE_LIMIT
                if correction_chain
                else NORMAL_CANDIDATE_LIMIT
            ),
        )

    def expand(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        query: str,
        request_id: str,
        gate: ContextNeedAssessment,
    ) -> ConversationPreparation:
        """Exceptional correction/re-solve expansion, capped at five clean pairs."""
        return self._load_candidates(
            actor_id=actor_id,
            conversation_id=conversation_id,
            query=query,
            request_id=request_id,
            gate=gate,
            limit=EXPANDED_CANDIDATE_LIMIT,
        )

    def _load_candidates(
        self,
        *,
        actor_id: str,
        conversation_id: str,
        query: str,
        request_id: str,
        gate: ContextNeedAssessment,
        limit: int,
    ) -> ConversationPreparation:
        loaded = self._persistence.load_recent_context(
            actor_id,
            conversation_id,
            limit,
        )
        substantive, hygiene = filter_substantive_turns(tuple(loaded.turns))
        cards = build_candidate_cards(
            substantive,
            current_query=query,
            limit=limit,
        )
        rejected = tuple(item.turn.turn_id for item in hygiene if not item.usable)
        candidate_characters = len(format_candidate_cards(cards))
        preparation = ConversationPreparation(
            gate=gate,
            context_load=loaded,
            eligible_turns=substantive[-limit:],
            candidates=cards,
            rejected_turn_ids=rejected,
            candidate_characters=candidate_characters,
        )
        self._log_reference_analysis(
            query=query,
            preparation=preparation,
        )
        self._log_preparation(preparation, hygiene, request_id=request_id)
        return preparation

    @staticmethod
    def _log_reference_analysis(
        *,
        query: str,
        preparation: ConversationPreparation,
    ) -> None:
        analysis = analyze_reference(query)
        compatibility = assess_candidate_compatibility(query, preparation.candidates)
        log_event(
            "conversation_reference_analyzed",
            component="conversation.reference_resolution",
            stage="select_context",
            status="completed",
            details={
                "local_reference": analysis.local_reference,
                "external_reference_detected": analysis.external_reference_detected,
                "reference_types": ",".join(analysis.reference_types) or "none",
            },
        )
        for card in preparation.candidates:
            labels = grounded_entity_labels(card)
            log_event(
                "conversation_grounded_entity_extracted",
                component="conversation.reference_resolution",
                stage="select_context",
                status="grounded" if labels else "ungrounded",
                details={
                    "turn_id": card.turn_id,
                    "entity_count": len(labels),
                    "validated_label_count": len(labels),
                },
            )
            if labels:
                record_local_preview(
                    f"grounded_labels_{card.turn_id[:8]}",
                    ",".join(labels),
                )
        for item in compatibility:
            log_event(
                "conversation_candidate_compatibility",
                component="conversation.reference_resolution",
                stage="select_context",
                status="compatible" if item.compatible else "incompatible",
                details={
                    "turn_id": item.turn_id,
                    "compatible": item.compatible,
                    "grounded_antecedent": item.grounded_antecedent,
                    "compatibility_reason": item.reason,
                    "recency_rank": item.recency_rank,
                },
            )
        for index, group in enumerate(
            group_compatible_candidates(query, preparation.candidates),
            start=1,
        ):
            log_event(
                "conversation_entity_grouped",
                component="conversation.reference_resolution",
                stage="select_context",
                status="grouped",
                details={
                    "entity_index": index,
                    "member_count": len(group.turn_ids),
                    "selected_turn_id": group.selected_turn_id,
                },
            )
            if group.label:
                record_local_preview(
                    f"entity_group_{index}",
                    f"{group.label}:{','.join(group.turn_ids)}",
                )

    @staticmethod
    def _log_gate(gate: ContextNeedAssessment, *, request_id: str) -> None:
        log_event(
            "context_gate_completed",
            component="conversation.context_need_gate",
            stage="context_gate",
            status="completed",
            duration_ms=gate.duration_ms,
            details={
                "decision": gate.decision,
                "reason_codes": ",".join(gate.reason_codes),
                "matched_signals": ",".join(gate.matched_signals),
            },
        )
        logger.debug(
            "context_gate request_id=%s decision=%s reasons=%s duration_ms=%d",
            request_id,
            gate.decision,
            ",".join(gate.reason_codes),
            gate.duration_ms,
        )

    @staticmethod
    def _log_preparation(
        preparation: ConversationPreparation,
        hygiene: tuple[MemoryHygieneDecision, ...],
        *,
        request_id: str,
    ) -> None:
        loaded = preparation.context_load
        update_request_summary(
            context_required=preparation.gate.decision != "CONTEXT_NOT_NEEDED",
            context_source=loaded.source,
            usable_recent_turns=len(preparation.eligible_turns),
        )
        log_event(
            "conversation_candidates_prepared",
            component="conversation.context_candidates",
            stage="load_recent_context",
            status="completed",
            duration_ms=loaded.latency_ms,
            details={
                "gate_decision": preparation.gate.decision,
                "source": loaded.source,
                "fetched": len(loaded.turns),
                "eligible": len(preparation.eligible_turns),
                "rejected": len(preparation.rejected_turn_ids),
                "candidate_count": len(preparation.candidates),
                "candidate_characters": preparation.candidate_characters,
                "turn_ids": ",".join(card.turn_id for card in preparation.candidates),
                "query_match_indicators": ",".join(
                    sorted(
                        {
                            indicator
                            for card in preparation.candidates
                            for indicator in card.query_match_indicators
                        }
                    )
                ),
                "memory_attempted": loaded.memory_attempted,
                "dynamodb_attempted": loaded.dynamodb_attempted,
            },
        )
        for item in hygiene:
            log_event(
                "conversation_context_selected",
                component="conversation.memory_hygiene",
                stage="select_context",
                status="selected" if item.usable else "rejected",
                details={
                    "turn_id": item.turn.turn_id,
                    "turn_type": item.turn_type,
                    "reason": item.reason,
                },
            )
        for card in preparation.candidates:
            record_local_preview("selected_turn_id", card.turn_id)
        if preparation.rejected_turn_ids:
            record_local_preview(
                "rejected_turn_ids",
                ",".join(preparation.rejected_turn_ids),
            )
        logger.debug(
            "conversation_candidates request_id=%s source=%s fetched=%d eligible=%d "
            "rejected=%d chars=%d",
            request_id,
            loaded.source,
            len(loaded.turns),
            len(preparation.eligible_turns),
            len(preparation.rejected_turn_ids),
            preparation.candidate_characters,
        )
