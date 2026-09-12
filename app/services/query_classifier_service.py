"""
app/services/query_classifier_service.py
-----------------------------------------
Query classifier service with deterministic fallback and optional LLM path.

Dispatch rules:
  ENABLE_REAL_LLM=false                              → deterministic
  ENABLE_REAL_LLM=true, role not in config           → deterministic
  ENABLE_REAL_LLM=true, role configured, call fails  → fallback (deterministic + low confidence)
  ENABLE_REAL_LLM=true, role configured, call OK     → llm

Public API:
    classify_query(query: str) -> QueryClassification
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import QueryClassification
from services.conversation.context_need_gate import is_short_incomplete_topic_phrase
from services.conversation.reference_resolution import (
    analyze_reference,
    assess_candidate_compatibility,
    definitively_incompatible_turn_ids,
    has_multiple_compatible_entities,
    unambiguous_compatible_turn_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLASSIFIER_ROLE = "doubt_solver_classifier"
_CLASSIFIER_STRONG_TASK_ROLE = "classifier_strong"

_FOLLOW_UP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^why did you\b", re.IGNORECASE),
    re.compile(r"^explain (?:the )?(?:second|third|last|previous) step\b", re.IGNORECASE),
    re.compile(r"^(?:please )?use another method\b", re.IGNORECASE),
    re.compile(r"^(?:please )?continue (?:from )?(?:above|there|the previous)\b", re.IGNORECASE),
    re.compile(r"^what about option [A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"^(?:give|create) (?:me )?\w+ more questions? like (?:that|this)\b", re.IGNORECASE),
    re.compile(r"^(?:can you )?(?:simplify|shorten) (?:that|the) explanation\b", re.IGNORECASE),
    re.compile(r"^explain (?:it|that)\??$", re.IGNORECASE),
    re.compile(r"\bwhat formula (?:did )?you (?:use|apply|applied)\b", re.IGNORECASE),
    re.compile(r"\bhow did you get\s+\S+", re.IGNORECASE),
    re.compile(r"\bexplain (?:that|this) step\b", re.IGNORECASE),
    re.compile(r"\b(?:previous|last) (?:answer|question|pattern)\b", re.IGNORECASE),
    re.compile(r"\b(?:that|this) (?:formula|step|option|answer)\b", re.IGNORECASE),
    re.compile(r"\bwhy (?:this|that) option\b", re.IGNORECASE),
    re.compile(r"\bhow was (?:this|that|it) calculated\b", re.IGNORECASE),
    re.compile(r"\bwhich operation did you (?:use|apply)\b", re.IGNORECASE),
    re.compile(r"\bwhat was the pattern\b", re.IGNORECASE),
    re.compile(r"(?:पिछला|पिछले|आखिरी) (?:सवाल|उत्तर|चरण)", re.IGNORECASE),
    re.compile(r"(?:यह|वह|इस|उस) (?:सूत्र|चरण|विकल्प)", re.IGNORECASE),
    re.compile(r"\b(?:pichla|pichhle|last) (?:sawal|question|answer|step)\b", re.IGNORECASE),
    re.compile(r"\b(?:yeh|woh|is|us) (?:formula|step|option)\b", re.IGNORECASE),
    re.compile(r"\bkaise (?:mila|nikala|calculate kiya)\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+(?:did\s+)?(?:you|u)\s+"
        r"(?:calculate|calculated|get|got)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:why\s+[$₹€£]?\s*\d+(?:\.\d+)?\s*(?:%|percent)?|"
        r"why\s+option\s+[A-D1-4])\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:aapne\s+.*(?:kaise|kyu|nikala|lagaya|calculate)|"
        r"\d+(?:\.\d+)?\s*(?:%|percent)?\s+kaise\s+nikala|"
        r"(?:ye|yeh)\s+value\s+kahan\s+se\s+aayi)\b",
        re.IGNORECASE,
    ),
)


def requires_recent_conversation(query: str) -> bool:
    """Return true only for high-confidence conversational references."""
    normalized = " ".join(query.strip().split())
    return any(pattern.search(normalized) for pattern in _FOLLOW_UP_PATTERNS)


@dataclass(frozen=True)
class ClassifierRunResult:
    """Classifier output plus whether strong-model escalation ran."""

    classification: QueryClassification
    strong_classifier_used: bool = False
    primary_decision: PrimaryClassificationDecision | None = None
    primary_classification: QueryClassification | None = None


class StrongFallbackReason(StrEnum):
    """Typed reason why the primary classifier was not accepted."""

    PRIMARY_ACCEPTED = "PRIMARY_ACCEPTED"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    UNSUPPORTED_ENUM = "UNSUPPORTED_ENUM"
    LOW_MATERIAL_CONFIDENCE = "LOW_MATERIAL_CONFIDENCE"
    CONTEXT_SELECTION_INVALID = "CONTEXT_SELECTION_INVALID"
    RELATION_ACTION_CONFLICT = "RELATION_ACTION_CONFLICT"
    WEB_SEARCH_CONFLICT = "WEB_SEARCH_CONFLICT"
    PRACTICE_INTENT_CONFLICT = "PRACTICE_INTENT_CONFLICT"
    CORRECTION_RESOLVE_CONFLICT = "CORRECTION_RESOLVE_CONFLICT"
    SUBJECT_INTENT_CONFLICT = "SUBJECT_INTENT_CONFLICT"
    REFERENCE_RESOLUTION_CONFLICT = "REFERENCE_RESOLUTION_CONFLICT"
    PRIMARY_PROVIDER_FAILURE = "PRIMARY_PROVIDER_FAILURE"


@dataclass(frozen=True)
class PrimaryClassificationDecision:
    """Typed, testable primary acceptance decision."""

    accepted: bool
    strong_required: bool
    reasons: tuple[StrongFallbackReason, ...]
    material_conflicts: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()

    @property
    def primary_reason(self) -> StrongFallbackReason:
        return self.reasons[0]


class ConversationClassificationConflict(ValueError):
    """A structured classifier result violates the supplied candidate contract."""

    def __init__(self, reason: str, classification: QueryClassification) -> None:
        super().__init__(reason)
        self.reason = reason
        self.classification = classification


# ---------------------------------------------------------------------------
# Lazy orchestrator singleton — orchestrated path
# ---------------------------------------------------------------------------

# Initialised on first call to _get_classifier_orchestrator().
# Reset to None in tests that need a clean slate.
_classifier_orchestrator: object | None = None


def _get_classifier_orchestrator() -> object:
    """Build (or return cached) LlmOrchestrator for the orchestrated classifier path.

    Wires the same provider chain as main.py when ENABLE_REAL_LLM=true.
    Called only when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true and ENABLE_REAL_LLM=true.
    """
    global _classifier_orchestrator  # noqa: PLW0603
    if _classifier_orchestrator is not None:
        return _classifier_orchestrator

    # Deferred heavy imports — only loaded when orchestrated path is active.
    from services.llm.orchestration.model_config_resolver import (  # noqa: PLC0415
        ModelConfigResolver,
    )
    from services.llm.orchestration.model_execution import (  # noqa: PLC0415
        ProviderAdapterExecutor,
        RegistryBackedModelExecutor,
    )
    from services.llm.orchestration.orchestrator import LlmOrchestrator  # noqa: PLC0415
    from services.llm.providers.provider_factory import (  # noqa: PLC0415
        ProviderAdapterFactory,
    )
    from services.secrets.env_secret_resolver import EnvSecretResolver  # noqa: PLC0415
    from services.secrets.provider_credentials import (  # noqa: PLC0415
        ProviderCredentialResolver,
    )

    _secret_resolver = EnvSecretResolver()
    _credential_resolver = ProviderCredentialResolver(secret_resolver=_secret_resolver)
    _adapter_executor = ProviderAdapterExecutor(
        credential_resolver=_credential_resolver,
        provider_factory=ProviderAdapterFactory(),
    )
    _model_executor = RegistryBackedModelExecutor(
        provider_executor=_adapter_executor,
        model_config_resolver=ModelConfigResolver(),
    )

    # Preflight: block orchestrator construction if any Azure deployment is a
    # placeholder.  Only reached when ENABLE_REAL_LLM=true (callers check first).
    # [SECURITY] Error includes only model alias names — no keys or secrets.
    from services.llm.orchestration.config_registry import get_registry  # noqa: PLC0415

    get_registry().validate_real_mode_deployments()  # raises LlmConfigValidationError

    _classifier_orchestrator = LlmOrchestrator(model_executor=_model_executor)
    return _classifier_orchestrator


# ---------------------------------------------------------------------------
# Keyword maps — deterministic path
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "practice_question": [
        "practice question",
        "practice problems",
        "quiz",
        "mock test",
        "questions on",
        "questions about",
        "generate questions",
        "create questions",
    ],
    "solve_question": ["solve", "calculate", "find", "compute", "evaluate", "what is", "answer"],
    "explain_concept": [
        "explain",
        "concept",
        "why",
        "what does",
        "what are",
        "define",
        "describe",
        "meaning of",
    ],
    "explain_option": ["option", "choice", "correct", "why is option", "why option", "why answer"],
}

_SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "math": [
        "profit",
        "loss",
        "equation",
        "calculate",
        "compute",
        "percentage",
        "ratio",
        "fraction",
        "algebra",
        "geometry",
        "number",
        "sum",
        "product",
        "divide",
        "sqrt",
        "square root",
        "exponent",
    ],
    "english": [
        "grammar",
        "narration",
        "direct speech",
        "indirect speech",
        "vocabulary",
        "sentence",
        "word",
        "synonym",
        "antonym",
        "tense",
    ],
    "reasoning": ["series", "pattern", "sequence", "analogy", "odd one out", "logical"],
    "general": [
        "history",
        "polity",
        "science",
        "physics",
        "chemistry",
        "biology",
        "force",
        "energy",
        "atom",
        "molecule",
    ],
}

_DIFFICULTY_KEYWORDS: dict[str, list[str]] = {
    "advanced": [
        "advanced",
        "hard",
        "tough",
        "tricky",
        "high level",
        "ssc cgl level",
        "cat level",
        "upsc level",
    ],
    "basic": [
        "basic",
        "simple",
        "beginner",
        "easy",
    ],
    "intermediate": [
        "intermediate",
        "moderate",
    ],
}

_POLICY_ADVANCED_SIGNALS: tuple[str, ...] = (
    "advanced",
    "sbi po",
    "ibps po",
    "banking exam level",
    "bank exam",
    "mains level",
    "cat level",
    "upsc level",
    "hard",
    "tricky",
    "tough",
    "high level",
    "puzzle",
    "seating arrangement",
    "floor puzzle",
    "coded inequality",
    "caselet",
)

_POLICY_BASIC_SIGNALS: tuple[str, ...] = (
    "basic",
    "beginner",
    "easy",
    "simple explanation",
)

_POLICY_MATH_SIGNALS: tuple[str, ...] = (
    "profit",
    "loss",
    "discount",
    "marked price",
    "selling price",
    "cost price",
    "percentage",
    "ratio",
    "average",
    "mixture",
    "alligation",
    "time and work",
    "time speed distance",
    "train",
    "boat",
    "simple interest",
    "compound interest",
    "partnership",
    "mensuration",
    "algebra",
    "quadratic",
    "number system",
)

_POLICY_REASONING_SIGNALS: tuple[str, ...] = (
    "seating arrangement",
    "floor puzzle",
    "coded inequality",
    "direction sense",
    "blood relation",
    "related to",
    "syllogism",
    "puzzle",
    "arrangement",
    "coding decoding",
    "input output",
    "series",
    "analogy",
    "turned right",
    "turned left",
    "turns right",
    "turns left",
    "facing north",
    "facing south",
)

_POLICY_ENGLISH_SIGNALS: tuple[str, ...] = (
    "grammar",
    "narration",
    "direct speech",
    "indirect speech",
    "synonym",
    "antonym",
    "sentence correction",
    "error spotting",
    "vocabulary",
    "comprehension",
    "fill in the blank",
    "cloze test",
    "reading comprehension",
    "para jumble",
)

_POLICY_GENERAL_SIGNALS: tuple[str, ...] = (
    "history",
    "polity",
    "static gk",
    "general science",
    "physics",
    "chemistry",
    "biology",
)

_MATH_TSD_METHOD_SIGNALS: tuple[str, ...] = (
    "km/hr",
    "km/h",
    "kmph",
    "m/s",
    "relative speed",
    "time speed distance",
    "speed of",
    "crosses",
    "crossing",
)

_QUANT_MOTION_CONTEXT_SIGNALS: tuple[str, ...] = (
    "train",
    "trains",
    "boat",
    "length",
    "distance",
    "hour",
    "metre",
    "meter",
    "km",
    "opposite direction",
    "opposite directions",
)

_AGE_EQUATION_METHOD_SIGNALS: tuple[str, ...] = (
    "years old",
    "year old",
    "older than",
    "younger than",
    "thrice",
    "birth",
    "currently",
    "age of",
    "age is",
    "age was",
)

_FAMILY_CONTEXT_TERMS: tuple[str, ...] = (
    "father",
    "mother",
    "daughter",
    "son",
    "sister",
    "brother",
)

_BROAD_REASONING_SUBJECT_SIGNALS: frozenset[str] = frozenset(
    {"direction sense", "related to", "father", "mother", "daughter", "son", "sister", "brother"}
)


_STRONG_EXPLICIT_REASONING_SIGNALS: frozenset[str] = frozenset(
    {
        "coded inequality",
        "seating arrangement",
        "floor puzzle",
        "blood relation",
        "direction sense",
        "syllogism",
        "coding decoding",
        "input output",
        "caselet",
    }
)


def get_classifier_confidence_threshold() -> float:
    """Return configured primary-classifier confidence threshold for strong fallback."""
    from config import get_settings  # noqa: PLC0415

    return get_settings().classifier_confidence_fallback_threshold


_SANITY_LOW_CONFIDENCE_THRESHOLD = 0.70
_STANDALONE_MATERIAL_CONFIDENCE_FLOOR = 0.85

_PATTERN_FAMILY_SUBJECT: dict[str, str] = {
    "MATH": "math",
    "QUANT": "math",
    "QUANTITATIVE_APTITUDE": "math",
    "REASONING": "reasoning",
    "LOGICAL_REASONING": "reasoning",
    "ENGLISH": "english",
    "GENERAL": "general",
    "GENERAL_AWARENESS": "general",
    "GK": "general",
}


def _append_unique_reason(
    reasons: list[StrongFallbackReason],
    reason: StrongFallbackReason,
) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _evaluate_primary_classification(
    query: str,
    classification: QueryClassification,
    *,
    context_gate: str,
    candidate_cards: tuple[ConversationCandidateCard, ...],
    candidate_turn_ids: tuple[str, ...],
) -> PrimaryClassificationDecision:
    """Accept a valid primary unless routing-critical uncertainty remains."""
    reasons: list[StrongFallbackReason] = []
    material_conflicts: list[str] = []

    family = (classification.pattern_family_candidate or "").strip().upper()
    family_subject = _PATTERN_FAMILY_SUBJECT.get(family)
    if family_subject is not None and family_subject != classification.subject:
        material_conflicts.append("subject,pattern_family")
        _append_unique_reason(reasons, StrongFallbackReason.SUBJECT_INTENT_CONFLICT)

    if _classifier_conflicts_with_signals(query, classification):
        material_conflicts.append("subject,intent")
        _append_unique_reason(reasons, StrongFallbackReason.SUBJECT_INTENT_CONFLICT)

    if classification.need_web_search:
        if not classification.web_search_reason or not classification.web_search_query:
            material_conflicts.append("need_web_search,web_search_metadata")
            _append_unique_reason(reasons, StrongFallbackReason.WEB_SEARCH_CONFLICT)
    elif classification.web_search_reason or classification.web_search_query:
        material_conflicts.append("need_web_search,web_search_metadata")
        _append_unique_reason(reasons, StrongFallbackReason.WEB_SEARCH_CONFLICT)

    from services.classification.web_search_demand import (  # noqa: PLC0415
        normalize_web_search_demand,
    )

    normalized_web = normalize_web_search_demand(
        query,
        classification,
        candidate_cards=candidate_cards,
    )
    if normalized_web.need_web_search != classification.need_web_search:
        material_conflicts.append("need_web_search")
        _append_unique_reason(reasons, StrongFallbackReason.WEB_SEARCH_CONFLICT)

    if (
        classification.requested_action == "GENERATE_SIMILAR"
        and classification.intent != "practice_question"
    ):
        material_conflicts.append("intent,requested_action")
        _append_unique_reason(reasons, StrongFallbackReason.PRACTICE_INTENT_CONFLICT)
    if _is_explicit_standalone_practice(query) and classification.intent != "practice_question":
        material_conflicts.append("intent,explicit_practice")
        _append_unique_reason(reasons, StrongFallbackReason.PRACTICE_INTENT_CONFLICT)

    if len(candidate_turn_ids) > 3:
        material_conflicts.append("candidate_selection")
        _append_unique_reason(reasons, StrongFallbackReason.CONTEXT_SELECTION_INVALID)

    high_risk_context = (
        classification.relation in {"CORRECTION", "RESOLVE_AGAIN", "AMBIGUOUS"}
        or classification.requested_action == "TRANSFORM_PREVIOUS"
        or len(candidate_turn_ids) > 1
        or (
            context_gate != "CONTEXT_NOT_NEEDED"
            and len(candidate_turn_ids) != 1
        )
    )
    confidence_floor = (
        get_classifier_confidence_threshold()
        if high_risk_context
        else min(
            get_classifier_confidence_threshold(),
            _STANDALONE_MATERIAL_CONFIDENCE_FLOOR,
        )
    )
    if classification.confidence < confidence_floor:
        material_conflicts.append("routing_confidence")
        _append_unique_reason(reasons, StrongFallbackReason.LOW_MATERIAL_CONFIDENCE)

    if not reasons:
        return PrimaryClassificationDecision(
            accepted=True,
            strong_required=False,
            reasons=(StrongFallbackReason.PRIMARY_ACCEPTED,),
        )
    return PrimaryClassificationDecision(
        accepted=False,
        strong_required=True,
        reasons=tuple(reasons),
        material_conflicts=tuple(material_conflicts),
    )


def evaluate_primary_classification(
    query: str,
    classification: QueryClassification,
) -> PrimaryClassificationDecision:
    """Evaluate standalone routing fields with the existing materiality policy."""
    return _evaluate_primary_classification(
        query,
        classification,
        context_gate="CONTEXT_NOT_NEEDED",
        candidate_cards=(),
        candidate_turn_ids=(),
    )


def _decision_for_primary_exception(exc: BaseException) -> PrimaryClassificationDecision:
    if isinstance(exc, ConversationClassificationConflict):
        reason_map = {
            "relation_action_incompatible": StrongFallbackReason.RELATION_ACTION_CONFLICT,
            "selected_turn_unknown": StrongFallbackReason.CONTEXT_SELECTION_INVALID,
            "new_question_selected_turn_present": StrongFallbackReason.CONTEXT_SELECTION_INVALID,
            "ambiguous_selected_turn_present": StrongFallbackReason.CONTEXT_SELECTION_INVALID,
            "generic_latest_reference_selected_older_turn": (
                StrongFallbackReason.CORRECTION_RESOLVE_CONFLICT
            ),
            "incomplete_topic_routed_as_solve": StrongFallbackReason.SUBJECT_INTENT_CONFLICT,
        }
        reason = reason_map.get(
            exc.reason,
            StrongFallbackReason.REFERENCE_RESOLUTION_CONFLICT,
        )
        return PrimaryClassificationDecision(
            accepted=False,
            strong_required=True,
            reasons=(reason,),
            material_conflicts=(exc.reason,),
        )
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        error_types = tuple(str(item.get("type", "validation_error")) for item in errors)
        validation_errors = tuple(
            f"{'.'.join(str(part) for part in item.get('loc', ()))}:"
            f"{item.get('type', 'validation_error')}"
            for item in errors
        )
        if "missing" in error_types:
            reason = StrongFallbackReason.MISSING_REQUIRED_FIELD
        elif any("literal" in item or "enum" in item for item in error_types):
            reason = StrongFallbackReason.UNSUPPORTED_ENUM
        else:
            reason = StrongFallbackReason.INVALID_SCHEMA
        return PrimaryClassificationDecision(
            accepted=False,
            strong_required=True,
            reasons=(reason,),
            validation_errors=validation_errors,
        )

    from services.doubt_solver.classifier_json import ClassifierJsonError  # noqa: PLC0415

    reason = (
        StrongFallbackReason.INVALID_SCHEMA
        if isinstance(exc, ClassifierJsonError)
        else StrongFallbackReason.PRIMARY_PROVIDER_FAILURE
    )
    return PrimaryClassificationDecision(
        accepted=False,
        strong_required=True,
        reasons=(reason,),
        validation_errors=(type(exc).__name__,),
    )


def _classifier_conflicts_with_signals(query: str, classification: QueryClassification) -> bool:
    """True when LLM labels conflict with strong deterministic subject signals."""
    query_lower = query.lower()
    explicit_subject = _detect_obvious_subject(query_lower)
    if explicit_subject is not None and classification.subject != explicit_subject:
        return True
    if (
        _has_rank_order_inference_structure(query_lower)
        and classification.subject != "reasoning"
    ):
        return True
    if classification.confidence >= get_classifier_confidence_threshold():
        return False
    if _requires_quantitative_solving_method(query_lower) and classification.subject in {
        "general",
        "unknown",
        "reasoning",
    }:
        return True
    if classification.intent in {"explain_concept", "general_doubt"} and (
        _requires_quantitative_solving_method(query_lower)
        or _requires_logical_inference_method(query_lower)
    ):
        return True
    if _requires_logical_inference_method(query_lower) and classification.subject in {
        "general",
        "unknown",
    }:
        return True
    return False


def _log_classifier_primary_result(
    classification: QueryClassification,
    *,
    valid_json: bool,
    route_id: str,
    decision: PrimaryClassificationDecision | None = None,
) -> None:
    material_conflicts = (
        len(decision.material_conflicts) if decision is not None else 0
    )
    logger.info(
        "classifier_primary_result  route_id=%s  schema_valid=%s  confidence=%.2f  "
        "subject=%s  intent=%s  difficulty=%s  relation=%s  action=%s  "
        "need_web_search=%s  material_conflicts=%d",
        route_id,
        str(valid_json).lower(),
        classification.confidence,
        classification.subject,
        classification.intent,
        classification.difficulty,
        classification.relation,
        classification.requested_action,
        str(classification.need_web_search).lower(),
        material_conflicts,
    )


def _log_classifier_json_error(exc: BaseException, *, route_id: str) -> None:
    from services.doubt_solver.classifier_json import ClassifierJsonError  # noqa: PLC0415

    if isinstance(exc, ConversationClassificationConflict):
        error_type = "conversation_contract_conflict"
        message_short = exc.reason
    elif isinstance(exc, ValidationError):
        error_type = "schema_validation_error"
        message_short = ",".join(
            f"{'.'.join(str(part) for part in item.get('loc', ()))}:"
            f"{item.get('type', 'validation_error')}"
            for item in exc.errors()
        )[:120]
    elif isinstance(exc, ClassifierJsonError):
        error_type = exc.error_type
        message_short = exc.message[:120]
    else:
        error_type = type(exc).__name__
        message_short = type(exc).__name__
    logger.warning(
        "classifier_json_error  route_id=%s  error_type=%s  error_message_short=%s",
        route_id,
        error_type,
        message_short,
    )


def _log_classifier_fallback_decision(
    *,
    request_id: str,
    reason: str,
    fallback_used: bool,
    strong_classifier_used: bool,
) -> None:
    logger.info(
        "classifier_fallback_decision  request_id=%s  reason=%s  "
        "fallback_used=%s  strong_classifier_used=%s",
        request_id,
        reason,
        str(fallback_used).lower(),
        str(strong_classifier_used).lower(),
    )


def _log_classifier_strong_decision(
    decision: PrimaryClassificationDecision,
    *,
    primary_confidence: float | None,
) -> None:
    from observability import log_event  # noqa: PLC0415

    logger.info(
        "classifier_strong_decision  role=classifier_strong attempt=strong "
        "triggered=%s fallback_reason=%s  "
        "primary_confidence=%s  material_fields=%s  material_conflicts=%d",
        str(decision.strong_required).lower(),
        decision.primary_reason.value,
        (
            f"{primary_confidence:.2f}"
            if primary_confidence is not None
            else "unavailable"
        ),
        ",".join(decision.material_conflicts) or "none",
        len(decision.material_conflicts),
    )
    log_event(
        "classifier_primary_decision",
        component="doubt_solver.classifier",
        stage="classify",
        status="accepted" if decision.accepted else "rejected",
        details={
            "primary_confidence": primary_confidence,
            "primary_accepted": decision.accepted,
            "strong_triggered": decision.strong_required,
            "strong_reason": decision.primary_reason.value,
            "material_conflicts": len(decision.material_conflicts),
        },
    )


def apply_classification_sanity(
    query: str,
    classification: dict[str, Any],
    *,
    request_id: str = "",
    classifier_confidence: float | None = None,
) -> dict[str, Any]:
    """Reroute low-confidence general/explain when strong subject signals exist."""
    old_subject = str(classification.get("subject") or "general")
    old_intent = str(classification.get("intent") or "explain")
    confidence = (
        classifier_confidence
        if classifier_confidence is not None
        else _SANITY_LOW_CONFIDENCE_THRESHOLD
    )
    query_lower = query.lower()

    should_check = (
        old_subject == "general"
        and old_intent == "explain"
        and confidence < _SANITY_LOW_CONFIDENCE_THRESHOLD
    )
    if not should_check:
        logger.info(
            "classification_sanity  request_id=%s  applied=false  old_subject=%s  "
            "new_subject=%s  reason=none",
            request_id,
            old_subject,
            old_subject,
        )
        return classification

    new_subject = old_subject
    new_intent = old_intent
    reason = ""
    if _requires_quantitative_solving_method(query_lower) or _detect_policy_subject(
        query_lower
    ) == "math":
        new_subject = "math"
        new_intent = "solve"
        reason = "route_sanity_math_signal"
    elif _requires_logical_inference_method(query_lower) or _detect_policy_subject(
        query_lower
    ) == "reasoning":
        new_subject = "reasoning"
        new_intent = "solve"
        reason = "route_sanity_reasoning_signal"

    applied = new_subject != old_subject or new_intent != old_intent
    logger.info(
        "classification_sanity  request_id=%s  applied=%s  old_subject=%s  "
        "new_subject=%s  reason=%s",
        request_id,
        str(applied).lower(),
        old_subject,
        new_subject if applied else old_subject,
        reason if applied else "none",
    )
    if not applied:
        return classification

    updated = dict(classification)
    updated["subject"] = new_subject
    updated["intent"] = new_intent
    if new_subject == "math" and str(updated.get("difficulty") or "default") in {
        "default",
        "basic",
    }:
        updated["difficulty"] = "intermediate"
    return updated


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _detect_intent(query_lower: str) -> tuple[str, float]:
    """Return (intent, confidence) from keyword matching."""
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return intent, 0.75
    return "general_doubt", 0.55


def _is_explicit_standalone_practice(query_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:give|create|generate|make|prepare|write)\b.{0,32}\b"
            r"(?:questions?|problems?|quiz|mock)\b",
            query_lower,
        )
        or re.search(r"\b(?:practice|quiz|mock test)\b", query_lower)
    )


def _detect_subject(query_lower: str) -> str:
    """Return subject string from keyword matching."""
    for subject, keywords in _SUBJECT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return subject
    return "unknown"


def _detect_style(query_lower: str, intent: str) -> str:
    """Return response_style from keyword matching, falling back to intent default."""
    if "short" in query_lower:
        return "short_answer"
    if "simple" in query_lower:
        return "simple_explanation"
    mapping = {
        "solve_question": "step_by_step",
        "explain_concept": "simple_explanation",
        "explain_option": "short_answer",
        "general_doubt": "simple_explanation",
        "unknown": "step_by_step",
    }
    return mapping.get(intent, "step_by_step")


def _detect_difficulty(query_lower: str) -> str:
    """Return difficulty from keyword matching. Returns 'default' when no signal."""
    for difficulty, keywords in _DIFFICULTY_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return difficulty
    return "default"


def _query_has_advanced_signal(query_lower: str) -> bool:
    return any(signal in query_lower for signal in _POLICY_ADVANCED_SIGNALS)


def _query_has_basic_signal(query_lower: str) -> bool:
    return any(signal in query_lower for signal in _POLICY_BASIC_SIGNALS)


def _detect_policy_subject(query_lower: str) -> str | None:
    """Return subject for the strongest explicit signal match, if any."""
    best_len = 0
    best_subject: str | None = None
    for subject, signals in (
        ("reasoning", _POLICY_REASONING_SIGNALS),
        ("math", _POLICY_MATH_SIGNALS),
        ("english", _POLICY_ENGLISH_SIGNALS),
        ("general", _POLICY_GENERAL_SIGNALS),
    ):
        for signal in signals:
            if signal in query_lower and len(signal) > best_len:
                best_len = len(signal)
                best_subject = subject
    return best_subject


def _detect_obvious_subject(query_lower: str) -> str | None:
    matched = _find_matched_signal(query_lower, _POLICY_ENGLISH_SIGNALS)
    if matched is not None:
        return "english"
    if _has_rank_order_inference_structure(query_lower):
        return "reasoning"
    matched = _find_matched_signal(query_lower, tuple(_STRONG_EXPLICIT_REASONING_SIGNALS))
    if matched is not None:
        return "reasoning"
    matched = _find_matched_signal(query_lower, _POLICY_GENERAL_SIGNALS)
    if matched is not None:
        return "general"
    return None


def _find_matched_signal(query_lower: str, signals: tuple[str, ...]) -> str | None:
    best: str | None = None
    for signal in signals:
        if signal in query_lower and (best is None or len(signal) > len(best)):
            best = signal
    return best


def _requires_quantitative_solving_method(query_lower: str) -> bool:
    """True when solving requires numeric/formula work — guardrail for policy overrides."""
    tsd_rate_hits = sum(1 for signal in _MATH_TSD_METHOD_SIGNALS if signal in query_lower)
    motion_context = sum(
        1 for signal in _QUANT_MOTION_CONTEXT_SIGNALS if signal in query_lower
    )
    if tsd_rate_hits >= 1 and motion_context >= 1:
        return True
    if tsd_rate_hits >= 2:
        return True

    age_hits = sum(1 for signal in _AGE_EQUATION_METHOD_SIGNALS if signal in query_lower)
    if age_hits >= 2:
        return True
    if age_hits >= 1 and (
        "age" in query_lower
        or any(term in query_lower for term in _FAMILY_CONTEXT_TERMS)
    ):
        return True
    return False


def _has_rank_order_inference_structure(query_lower: str) -> bool:
    """Detect two-sided ordinal positioning without relying on an exact question."""
    ordinal_count = len(re.findall(r"\b\d+(?:st|nd|rd|th)\b", query_lower))
    opposing_positions = (
        ("left" in query_lower and "right" in query_lower)
        or ("top" in query_lower and "bottom" in query_lower)
        or ("before" in query_lower and "after" in query_lower)
    )
    return ordinal_count >= 2 and opposing_positions and any(
        term in query_lower for term in ("row", "rank", "position", "queue")
    )


def _requires_logical_inference_method(query_lower: str) -> bool:
    """True when solving is primarily inference/navigation, not numeric calculation."""
    if _requires_quantitative_solving_method(query_lower):
        return False
    if _has_rank_order_inference_structure(query_lower):
        return True
    inference_markers = (
        "how is",
        "related to",
        "facing north",
        "facing south",
        "turned right",
        "turned left",
        "turns right",
        "turns left",
        "statements",
        "conclusions",
        "all some",
        "coded inequality",
        "seating arrangement",
        "floor puzzle",
    )
    return any(marker in query_lower for marker in inference_markers)


def _should_apply_subject_correction(
    query_lower: str,
    old_subject: str,
    detected_subject: str,
    *,
    classifier_confidence: float | None,
) -> bool:
    """Guard subject overrides — policy is safety net, not primary classifier."""
    if detected_subject == old_subject:
        return False

    threshold = get_classifier_confidence_threshold()
    matched = _find_matched_signal(
        query_lower,
        _POLICY_REASONING_SIGNALS
        + _POLICY_MATH_SIGNALS
        + _POLICY_ENGLISH_SIGNALS
        + _POLICY_GENERAL_SIGNALS,
    )

    if detected_subject == "reasoning" and _requires_quantitative_solving_method(query_lower):
        return False

    if (
        old_subject == "math"
        and detected_subject == "reasoning"
        and _requires_quantitative_solving_method(query_lower)
    ):
        return False

    if (
        classifier_confidence is not None
        and classifier_confidence >= threshold
        and old_subject not in {"general", "unknown"}
    ):
        if matched in _STRONG_EXPLICIT_REASONING_SIGNALS and detected_subject == "reasoning":
            return True
        if matched in _POLICY_MATH_SIGNALS and detected_subject == "math":
            return True
        if matched in _POLICY_ENGLISH_SIGNALS and detected_subject == "english":
            return True
        if matched in _POLICY_GENERAL_SIGNALS and detected_subject == "general":
            return True
        return False

    if (
        detected_subject == "reasoning"
        and matched in _BROAD_REASONING_SUBJECT_SIGNALS
        and _requires_quantitative_solving_method(query_lower)
    ):
        return False

    return True


def _count_constraint_clauses(query_lower: str) -> int:
    markers = (
        " if ",
        " then ",
        " who ",
        " which ",
        " given that ",
        " such that ",
        " does not ",
        " cannot ",
        " neither ",
        " either ",
    )
    count = sum(1 for marker in markers if marker in query_lower)
    count += query_lower.count(";")
    count += query_lower.count(" and ")
    return count


def _count_named_entities(query: str) -> int:
    tokens = query.split()
    entities = 0
    for idx, token in enumerate(tokens):
        cleaned = token.strip(".,;:!?()[]\"'")
        if not cleaned or not cleaned[0].isupper():
            continue
        if idx == 0 and len(tokens) > 1:
            continue
        if cleaned.lower() in {"the", "a", "an", "if", "then", "who", "which"}:
            continue
        entities += 1
    return entities


def _has_statements_conclusions_structure(query_lower: str) -> bool:
    if "statement" in query_lower and "conclusion" in query_lower:
        return True
    return "all " in query_lower and "some " in query_lower


def _has_multi_direction_turns(query_lower: str) -> bool:
    turn_markers = (
        "turns left",
        "turns right",
        "facing north",
        "facing south",
        "facing east",
        "facing west",
    )
    return sum(1 for marker in turn_markers if marker in query_lower) >= 2


def _has_structural_complexity(query: str, query_lower: str) -> bool:
    if _count_constraint_clauses(query_lower) >= 4:
        return True
    if _count_named_entities(query) >= 5:
        return True
    if _has_statements_conclusions_structure(query_lower):
        return True
    if _has_multi_direction_turns(query_lower):
        return True
    return len(query.strip()) >= 280


def _has_moderate_quant_complexity(query_lower: str) -> bool:
    quant_ops = (
        "profit",
        "loss",
        "discount",
        "percentage",
        "ratio",
        "mixture",
        "average",
        "interest",
        "partnership",
    )
    hits = sum(1 for op in quant_ops if op in query_lower)
    return hits >= 2 and not _query_has_advanced_signal(query_lower)


def _detect_structural_difficulty(
    query: str,
    subject: str,
    *,
    pattern_topic_key: str | None,
) -> tuple[str | None, str]:
    """Return (new_difficulty, matched_signal_category) or (None, 'none')."""
    from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
        _ADVANCED_REASONING_TOPICS,
        _INTERMEDIATE_REASONING_TOPICS,
    )

    query_lower = query.lower()
    topic = (pattern_topic_key or "").upper()

    if subject == "reasoning" and topic in _ADVANCED_REASONING_TOPICS:
        return "advanced", "reasoning_pattern_advanced"

    if _has_structural_complexity(query, query_lower):
        if subject == "reasoning" or topic in _ADVANCED_REASONING_TOPICS:
            return "advanced", "structural_complexity"

    if subject == "reasoning" and topic in _INTERMEDIATE_REASONING_TOPICS:
        return "intermediate", "intermediate_reasoning_pattern"

    if subject == "math" and _has_moderate_quant_complexity(query_lower):
        return "intermediate", "intermediate_quant_pattern"

    return None, "none"


def apply_classification_policy(
    query: str,
    classification: dict[str, Any],
    *,
    request_id: str = "",
    classifier_confidence: float | None = None,
) -> dict[str, Any]:
    """Deterministic post-classification safety net — not the primary classifier.

    Subject correction is blocked when the primary classifier is high-confidence,
    and math-priority queries must not be overridden to reasoning via broad keywords.
    """
    query_lower = query.lower()
    old_subject = str(classification.get("subject") or "general")
    old_difficulty = str(classification.get("difficulty") or "default")
    new_subject = old_subject
    new_difficulty = old_difficulty
    matched_signal = "none"
    reason = ""
    threshold = get_classifier_confidence_threshold()

    detected_subject = _detect_policy_subject(query_lower)
    if detected_subject and _should_apply_subject_correction(
        query_lower,
        old_subject,
        detected_subject,
        classifier_confidence=classifier_confidence,
    ):
        new_subject = detected_subject
        matched_signal = _find_matched_signal(
            query_lower,
            _POLICY_REASONING_SIGNALS
            + _POLICY_MATH_SIGNALS
            + _POLICY_ENGLISH_SIGNALS
            + _POLICY_GENERAL_SIGNALS,
        ) or "subject"
        reason = "explicit_subject_signal"

    from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
        derive_pattern_hints,
    )

    hints = derive_pattern_hints(query, new_subject, classification)

    if _query_has_advanced_signal(query_lower):
        new_difficulty = "advanced"
        if matched_signal == "none":
            matched_signal = (
                _find_matched_signal(query_lower, _POLICY_ADVANCED_SIGNALS) or "advanced"
            )
        reason = reason or "explicit_advanced_signal"
    else:
        structural_difficulty, structural_signal = _detect_structural_difficulty(
            query,
            new_subject,
            pattern_topic_key=hints.pattern_topic_key,
        )
        if structural_difficulty == "advanced" and new_difficulty != "advanced":
            new_difficulty = "advanced"
            if matched_signal == "none":
                matched_signal = structural_signal
            reason = reason or structural_signal
        elif (
            structural_difficulty == "intermediate"
            and new_difficulty in {"default", "basic"}
        ):
            new_difficulty = "intermediate"
            if matched_signal == "none":
                matched_signal = structural_signal
            reason = reason or structural_signal
        elif _query_has_basic_signal(query_lower) and old_difficulty == "default":
            new_difficulty = "basic"
            if matched_signal == "none":
                matched_signal = _find_matched_signal(query_lower, _POLICY_BASIC_SIGNALS) or "basic"
            reason = reason or "explicit_basic_signal"

    applied = new_subject != old_subject or new_difficulty != old_difficulty
    logger.info(
        "classification_policy  request_id=%s  policy_checked=true  policy_applied=%s  "
        "confidence_threshold=%.2f  classifier_confidence=%s  "
        "old_subject=%s  new_subject=%s  old_difficulty=%s  new_difficulty=%s  "
        "matched_signal=%s  reason=%s",
        request_id,
        str(applied).lower(),
        threshold,
        f"{classifier_confidence:.2f}"
        if classifier_confidence is not None
        else "none",
        old_subject,
        new_subject,
        old_difficulty,
        new_difficulty,
        matched_signal,
        reason if applied else "none",
    )
    logger.info(
        "classification_policy_summary  request_id=%s  policy_applied=%s  "
        "subject=%s  difficulty=%s  pattern_topic=%s  matched_signal=%s",
        request_id,
        str(applied).lower(),
        new_subject,
        new_difficulty,
        hints.pattern_topic_key or "none",
        matched_signal,
    )

    if not applied:
        return classification

    updated = dict(classification)
    updated["subject"] = new_subject
    updated["difficulty"] = new_difficulty
    return updated


def _build_query_classification(
    parsed: QueryClassification,
    *,
    classification_source: str | None = None,
    confidence_override: float | None = None,
) -> QueryClassification:
    """Copy validated classifier fields including retrieval hints."""
    return QueryClassification(
        intent=parsed.intent,
        subject=parsed.subject,
        topic=parsed.topic,
        topic_confidence=parsed.topic_confidence,
        pattern_topic_candidate=parsed.pattern_topic_candidate,
        pattern_family_candidate=parsed.pattern_family_candidate,
        retrieval_tags=parsed.retrieval_tags,
        difficulty=parsed.difficulty,
        response_style=parsed.response_style,
        confidence=confidence_override if confidence_override is not None else parsed.confidence,
        retrieval_need=parsed.retrieval_need,
        reasoning_summary=parsed.reasoning_summary,
        need_web_search=parsed.need_web_search,
        web_search_reason=parsed.web_search_reason,
        web_search_query=parsed.web_search_query,
        requires_recent_conversation=parsed.requires_recent_conversation,
        relation=parsed.relation,
        selected_turn_id=parsed.selected_turn_id,
        requested_action=parsed.requested_action,
        classification_source=classification_source or parsed.classification_source,  # type: ignore[arg-type]
    )


def _classify_deterministic(query: str) -> QueryClassification:
    """Classify a student query using keyword matching. No network calls."""
    query_lower = query.lower()
    policy_subject = _detect_policy_subject(query_lower)
    quant_method = _requires_quantitative_solving_method(query_lower)
    logic_method = _requires_logical_inference_method(query_lower)
    intent_from_kw, conf_from_kw = _detect_intent(query_lower)

    if policy_subject == "math" or quant_method:
        subject = "math"
        if quant_method or intent_from_kw == "solve_question":
            intent = "solve_question"
            confidence = 0.82 if quant_method else max(conf_from_kw, 0.75)
        else:
            intent = intent_from_kw
            confidence = conf_from_kw
        difficulty = _detect_difficulty(query_lower)
        if difficulty == "default" and quant_method:
            difficulty = "intermediate"
    elif policy_subject == "reasoning" or logic_method:
        subject = "reasoning"
        if logic_method or intent_from_kw == "solve_question":
            intent = "solve_question"
            confidence = 0.80
        else:
            intent = intent_from_kw
            confidence = conf_from_kw
        difficulty = _detect_difficulty(query_lower)
    elif policy_subject == "english":
        subject = "english"
        intent = intent_from_kw
        confidence = conf_from_kw
        difficulty = _detect_difficulty(query_lower)
    else:
        intent = intent_from_kw
        confidence = conf_from_kw
        subject = _detect_subject(query_lower)
        difficulty = _detect_difficulty(query_lower)
        if subject in {"unknown", "general"} and quant_method:
            subject = "math"
            intent = "solve_question"
            confidence = max(confidence, 0.80)
            if difficulty == "default":
                difficulty = "intermediate"
        elif subject in {"unknown", "general"} and logic_method:
            subject = "reasoning"
            intent = "solve_question"
            confidence = max(confidence, 0.78)

    response_style = _detect_style(query_lower, intent)

    from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
        resolve_retrieval_hints,
    )

    hints = resolve_retrieval_hints(query, subject, {})
    pattern_candidate = hints.pattern_topic_key if hints.strength in {"medium", "strong"} else None

    result = QueryClassification(
        intent=intent,  # type: ignore[arg-type]
        subject=subject,
        topic=hints.topic_hint or pattern_candidate,
        topic_confidence=0.80 if pattern_candidate else None,
        pattern_topic_candidate=pattern_candidate,
        pattern_family_candidate=hints.pattern_family_key,
        retrieval_tags=hints.retrieval_tags or hints.matched_signals[:10],
        difficulty=difficulty,  # type: ignore[arg-type]
        response_style=response_style,  # type: ignore[arg-type]
        confidence=confidence,
        classification_source="deterministic",
        requires_recent_conversation=requires_recent_conversation(query),
    )
    logger.debug(
        "deterministic_classifier  intent=%s  subject=%s  difficulty=%s  confidence=%.2f",
        result.intent,
        result.subject,
        result.difficulty,
        result.confidence,
    )
    return result


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def _load_classifier_prompt() -> str:
    """Return the composed shared text-classifier prompt."""
    from services.prompt_loader import load_prompt  # noqa: PLC0415

    return "\n\n".join(
        (
            load_prompt("classification_semantics"),
            load_prompt("query_classifier_text"),
            load_prompt("query_classifier"),
        )
    )


def _classifier_prompt_character_budget(
    *,
    query: str,
    conversation_candidates: str | None,
    context_gate: str = "CONTEXT_NOT_NEEDED",
) -> dict[str, int]:
    """Return content-free section sizes for development prompt audits."""
    prompt = _load_classifier_prompt()
    text_marker = "# Text Classification Mode"
    output_marker = "# Strong Classifier JSON Output"
    text_start = prompt.index(text_marker)
    output_start = prompt.index(output_marker)
    shared = prompt[:text_start]
    text_overlay = prompt[text_start:output_start]
    output_contract = prompt[output_start:]
    budget = {
        "base_instructions_chars": len(shared),
        "schema_chars": len(output_contract),
        "classification_chars": len(shared),
        "retrieval_chars": 0,
        "conversation_chars": 0,
        "web_search_chars": 0,
        "method_rules_chars": 0,
        "examples_chars": 0,
        "text_overlay_chars": len(text_overlay),
    }
    user_input = _build_classifier_input(
        query,
        conversation_candidates=conversation_candidates,
        context_gate=context_gate,
    )
    budget.update(
        {
            "query_chars": len(query),
            "candidate_context_chars": len(conversation_candidates or ""),
            "other_chars": len(user_input) - len(query) - len(conversation_candidates or ""),
            "total_chars": len(prompt) + len(user_input),
        }
    )
    return budget


def _log_classifier_prompt_budget(
    *,
    task_role: str,
    query: str,
    conversation_candidates: str | None,
    context_gate: str,
    provider_input_tokens: int | None,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        budget = _classifier_prompt_character_budget(
            query=query,
            conversation_candidates=conversation_candidates,
            context_gate=context_gate,
        )
        logger.debug(
            "CLASSIFIER PROMPT BUDGET role=%s base_instructions_chars=%d schema_chars=%d "
            "classification_chars=%d retrieval_chars=%d conversation_chars=%d "
            "web_search_chars=%d method_rules_chars=%d examples_chars=%d query_chars=%d "
            "candidate_context_chars=%d other_chars=%d total_chars=%d "
            "provider_input_tokens=%s",
            task_role,
            budget["base_instructions_chars"],
            budget["schema_chars"],
            budget["classification_chars"],
            budget["retrieval_chars"],
            budget["conversation_chars"],
            budget["web_search_chars"],
            budget["method_rules_chars"],
            budget["examples_chars"],
            budget["query_chars"],
            budget["candidate_context_chars"],
            budget["other_chars"],
            budget["total_chars"],
            provider_input_tokens if provider_input_tokens is not None else "unavailable",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "CLASSIFIER PROMPT BUDGET unavailable error_type=%s",
            type(exc).__name__,
        )


_RELATION_ACTIONS: dict[str, frozenset[str]] = {
    "NEW_QUESTION": frozenset({"ANSWER_CURRENT"}),
    "FOLLOW_UP": frozenset(
        {
            "ANSWER_WITH_CONTEXT",
            "EXPLAIN_PREVIOUS",
            "GENERATE_SIMILAR",
            "TRANSFORM_PREVIOUS",
        }
    ),
    "CONTINUATION": frozenset({"CONTINUE_PREVIOUS"}),
    "CORRECTION": frozenset({"VERIFY_AND_CORRECT"}),
    "RESOLVE_AGAIN": frozenset({"RESOLVE_FROM_SCRATCH"}),
    "AMBIGUOUS": frozenset({"ASK_CLARIFICATION"}),
}
_GENERIC_LATEST_TURN_REFERENCE = re.compile(
    r"^(?:(?:your|ur|this|previous|last|the(?:\s+(?:previous|last))?)\s+)?"
    r"(?:(?:question|problem)\s+)?(?:answer|solution)\s+"
    r"(?:is\s+)?(?:wrong|worng|incorrect|galat)\b|"
    r"^(?:it|this|that)\s+is\s+(?:a\s+)?(?:wrong|incorrect)\s+(?:answer|solution)\b|"
    r"^(?:again\s+wrong|solve\s+it\s+again|solve\s+it\s+from\s+scratch|"
    r"again\s+wrong\s+solve\s+it\s+from\s+scratch)\b"
)


def _is_generic_latest_turn_reference(query: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", query.casefold())
    return bool(_GENERIC_LATEST_TURN_REFERENCE.search(" ".join(normalized.split())))


def _build_classifier_input(
    query: str,
    *,
    conversation_candidates: str | None,
    context_gate: str,
    primary_result: QueryClassification | None = None,
    conflict_reason: str | None = None,
) -> str:
    sections = [
        "[CURRENT_QUERY]",
        query,
        "[/CURRENT_QUERY]",
        f"[CONTEXT_GATE]{context_gate}[/CONTEXT_GATE]",
    ]
    if conversation_candidates:
        sections.append(conversation_candidates)
    if primary_result is not None:
        sections.extend(
            (
                "[PRIMARY_CLASSIFIER_RESULT_REJECTED]",
                primary_result.model_dump_json(
                    include={
                        "intent",
                        "subject",
                        "pattern_topic_candidate",
                        "pattern_family_candidate",
                        "confidence",
                        "need_web_search",
                        "web_search_reason",
                        "relation",
                        "selected_turn_id",
                        "requested_action",
                    }
                ),
                f"validation_reason: {conflict_reason or 'low_confidence'}",
                "[/PRIMARY_CLASSIFIER_RESULT_REJECTED]",
            )
        )
    return "\n".join(sections)


def _validate_conversation_contract(
    classification: QueryClassification,
    *,
    query: str,
    candidate_turn_ids: tuple[str, ...],
    context_gate: str,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
) -> QueryClassification:
    if (
        is_short_incomplete_topic_phrase(query)
        and classification.intent == "solve_question"
        and classification.relation == "NEW_QUESTION"
    ):
        raise ConversationClassificationConflict(
            "incomplete_topic_routed_as_solve",
            classification,
        )
    if context_gate == "CONTEXT_NOT_NEEDED":
        return classification.model_copy(
            update={
                "relation": "NEW_QUESTION",
                "selected_turn_id": None,
                "requested_action": "ANSWER_CURRENT",
                "requires_recent_conversation": False,
            }
        )
    allowed_actions = _RELATION_ACTIONS.get(classification.relation, frozenset())
    if classification.requested_action not in allowed_actions:
        raise ConversationClassificationConflict(
            "relation_action_incompatible",
            classification,
        )
    reference = analyze_reference(query)
    compatibility = assess_candidate_compatibility(query, candidate_cards)
    compatible_ids = tuple(item.turn_id for item in compatibility if item.compatible)
    unambiguous_id = unambiguous_compatible_turn_id(query, candidate_cards)
    incompatible_ids = definitively_incompatible_turn_ids(query, candidate_cards)
    logger.info(
        "reference_analysis local_reference=%s external_reference_detected=%s "
        "reference_types=%s",
        str(reference.local_reference).lower(),
        str(reference.external_reference_detected).lower(),
        ",".join(reference.reference_types) or "none",
    )
    for item in compatibility:
        logger.info(
            "candidate_compatibility turn_id=%s compatible=%s reason=%s recency_rank=%d",
            item.turn_id,
            str(item.compatible).lower(),
            item.reason,
            item.recency_rank,
        )
    if classification.relation == "NEW_QUESTION":
        if classification.selected_turn_id is not None:
            raise ConversationClassificationConflict(
                "new_question_selected_turn_present",
                classification,
            )
        if context_gate == "CONTEXT_REQUIRED" or (
            reference.external_reference_detected and compatible_ids
        ):
            raise ConversationClassificationConflict(
                "explicit_context_reference_unresolved",
                classification,
            )
        if reference.external_reference_detected and not compatible_ids:
            raise ConversationClassificationConflict(
                "external_reference_without_compatible_candidate",
                classification,
            )
        return classification.model_copy(
            update={"requires_recent_conversation": False}
        )
    if classification.relation == "AMBIGUOUS":
        if classification.selected_turn_id is not None:
            raise ConversationClassificationConflict(
                "ambiguous_selected_turn_present",
                classification,
            )
        if reference.external_reference_detected and unambiguous_id is not None:
            raise ConversationClassificationConflict(
                "unambiguous_external_reference_marked_ambiguous",
                classification,
            )
        return classification.model_copy(
            update={"requires_recent_conversation": True}
        )
    if (
        classification.selected_turn_id is None
        or classification.selected_turn_id not in candidate_turn_ids
    ):
        raise ConversationClassificationConflict(
            "selected_turn_unknown",
            classification,
        )
    if (
        reference.external_reference_detected
        and has_multiple_compatible_entities(query, candidate_cards)
    ):
        raise ConversationClassificationConflict(
            "multiple_compatible_reference_entities",
            classification,
        )
    if (
        reference.external_reference_detected
        and classification.selected_turn_id in incompatible_ids
        and not (
            classification.relation in {"CORRECTION", "RESOLVE_AGAIN"}
            and _is_generic_latest_turn_reference(query)
            and classification.selected_turn_id == candidate_turn_ids[-1]
        )
    ):
        raise ConversationClassificationConflict(
            "selected_turn_semantically_incompatible",
            classification,
        )
    if (
        reference.external_reference_detected
        and classification.relation == "FOLLOW_UP"
        and classification.requested_action == "EXPLAIN_PREVIOUS"
        and "person" in reference.reference_types
    ):
        raise ConversationClassificationConflict(
            "factual_person_follow_up_mapped_to_explanation",
            classification,
        )
    if (
        classification.relation in {"CORRECTION", "RESOLVE_AGAIN"}
        and _is_generic_latest_turn_reference(query)
        and classification.selected_turn_id != candidate_turn_ids[-1]
    ):
        raise ConversationClassificationConflict(
            "generic_latest_reference_selected_older_turn",
            classification,
        )
    return classification.model_copy(update={"requires_recent_conversation": True})


def _apply_deterministic_conversation_contract(
    classification: QueryClassification,
    *,
    query: str,
    candidate_cards: tuple[ConversationCandidateCard, ...],
    candidate_turn_ids: tuple[str, ...],
    context_gate: str,
) -> QueryClassification:
    if context_gate == "CONTEXT_NOT_NEEDED":
        return classification.model_copy(
            update={
                "relation": "NEW_QUESTION",
                "selected_turn_id": None,
                "requested_action": "ANSWER_CURRENT",
                "requires_recent_conversation": False,
            }
        )
    normalized = " ".join(query.casefold().split())
    reference = analyze_reference(query)
    unambiguous_id = unambiguous_compatible_turn_id(query, candidate_cards)
    if re.search(r"\b(?:from scratch|solve it again|again wrong|dobara solve)\b", normalized):
        relation = "RESOLVE_AGAIN"
        action = "RESOLVE_FROM_SCRATCH"
        selected = candidate_turn_ids[-1] if candidate_turn_ids else None
    elif _is_generic_latest_turn_reference(query) or re.search(
        r"\b(?:wrong|worng|incorrect|galat)\b", normalized
    ):
        relation = "CORRECTION"
        action = "VERIFY_AND_CORRECT"
        selected = candidate_turn_ids[-1] if candidate_turn_ids else None
    elif (
        reference.external_reference_detected
        and unambiguous_id is not None
    ):
        relation = "FOLLOW_UP"
        action = "ANSWER_WITH_CONTEXT"
        selected = unambiguous_id
    elif reference.external_reference_detected:
        relation = "AMBIGUOUS"
        action = "ASK_CLARIFICATION"
        selected = None
    elif not candidate_turn_ids:
        relation = "AMBIGUOUS"
        action = "ASK_CLARIFICATION"
        selected = None
    elif re.search(r"\b(?:continue|go on|aage)\b", normalized):
        relation = "CONTINUATION"
        action = "CONTINUE_PREVIOUS"
        selected = candidate_turn_ids[-1]
    elif re.search(r"\b(?:similar|same type|another question|aise aur)\b", normalized):
        relation = "FOLLOW_UP"
        action = "GENERATE_SIMILAR"
        selected = candidate_turn_ids[-1]
    elif re.search(r"\b(?:make it|convert it|change the values|previous question)\b", normalized):
        relation = "FOLLOW_UP"
        action = "TRANSFORM_PREVIOUS"
        selected = candidate_turn_ids[-1]
    elif context_gate == "CONTEXT_REQUIRED":
        relation = "FOLLOW_UP"
        action = "EXPLAIN_PREVIOUS"
        selected = candidate_turn_ids[-1]
    else:
        relation = "AMBIGUOUS"
        action = "ASK_CLARIFICATION"
        selected = None
    intent = {
        "ANSWER_WITH_CONTEXT": classification.intent,
        "EXPLAIN_PREVIOUS": "explain_concept",
        "CONTINUE_PREVIOUS": classification.intent,
        "GENERATE_SIMILAR": "practice_question",
        "TRANSFORM_PREVIOUS": "practice_question",
        "VERIFY_AND_CORRECT": "solve_question",
        "RESOLVE_FROM_SCRATCH": "solve_question",
        "ASK_CLARIFICATION": "general_doubt",
    }.get(action, classification.intent)
    academic_updates: dict[str, object] = {}
    if (
        relation in {"CORRECTION", "RESOLVE_AGAIN"}
        and _is_generic_latest_turn_reference(query)
        and selected is not None
    ):
        selected_card = next(
            (card for card in candidate_cards if card.turn_id == selected),
            None,
        )
        if selected_card is not None:
            if selected_card.subject not in {"", "unknown"}:
                academic_updates["subject"] = selected_card.subject
            if selected_card.topic:
                academic_updates["topic"] = selected_card.topic
            if selected_card.difficulty in {
                "default",
                "basic",
                "intermediate",
                "advanced",
            }:
                academic_updates["difficulty"] = selected_card.difficulty
            academic_updates["pattern_topic_candidate"] = None
            academic_updates["pattern_family_candidate"] = None
    return classification.model_copy(
        update={
            "intent": intent,
            "relation": relation,
            "selected_turn_id": selected,
            "requested_action": action,
            "requires_recent_conversation": relation != "NEW_QUESTION",
            **academic_updates,
        }
    )


def _latest_candidate_question(
    conversation_candidates: str | None,
) -> str | None:
    if not conversation_candidates:
        return None
    questions = re.findall(
        r"^question_preview:\s*(.+)$",
        conversation_candidates,
        flags=re.MULTILINE,
    )
    return questions[-1].strip() if questions else None


def _normalize_verification_relation(
    classification: QueryClassification,
) -> QueryClassification:
    """Re-label the one relation both classifier models get wrong on "are you sure?".

    VERIFY_AND_CORRECT is legal only under CORRECTION, yet both models pair it with
    FOLLOW_UP while selecting the right turn.  No consumer reads the relation for this
    action — selected_context_builder and answer_correctness both key on the action —
    so re-labelling restores a shape the contract already permits and changes nothing
    downstream.  Only a fully resolved pair is re-labelled; an unresolved one keeps
    falling through to clarification.
    """
    if (
        classification.relation == "FOLLOW_UP"
        and classification.requested_action == "VERIFY_AND_CORRECT"
        and classification.selected_turn_id is not None
    ):
        return classification.model_copy(update={"relation": "CORRECTION"})
    return classification


def _parse_classifier_orchestrated_content(
    content: str,
    *,
    route_id: str,
) -> QueryClassification:
    """Strict-parse classifier JSON and validate against QueryClassification."""
    from services.doubt_solver.classifier_json import (  # noqa: PLC0415
        parse_classifier_json_strict,
    )

    raw_dict, recovered = parse_classifier_json_strict(content)
    if recovered:
        logger.info(
            "classifier_json_recovered  route_id=%s  classifier_json_recovered=true",
            route_id,
        )
    classification = QueryClassification.model_validate(raw_dict)
    return _normalize_verification_relation(
        _build_query_classification(classification, classification_source="llm")
    )


def _classify_with_llm_orchestrated(
    query: str,
    request_id: str | None = None,
    *,
    task_role: str = "classifier",
    conversation_candidates: str | None = None,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
    context_gate: str = "CONTEXT_NOT_NEEDED",
    primary_result: QueryClassification | None = None,
    conflict_reason: str | None = None,
) -> QueryClassification:
    """Run classification via the orchestrated Azure-first path.

    Resolves route with subject="general", task_role (classifier or
    classifier_strong), difficulty="default".

    Args:
        query:      The student's question (already validated by caller).
        request_id: Trace ID propagated from the graph state.
        task_role:  Orchestration task role — ``classifier`` (primary) or
                    ``classifier_strong`` (low-confidence fallback).

    Raises:
        Any exception from orchestrator, JSON parsing, or validation.
    """
    from schemas.llm_routing import RouteRequest  # noqa: PLC0415

    _request_id = request_id or str(uuid.uuid4())
    route_id = f"general.{task_role}.default"

    orchestrator = _get_classifier_orchestrator()
    route_request = RouteRequest(
        request_id=_request_id,
        subject="general",
        task_role=task_role,  # type: ignore[arg-type]
        difficulty="default",
        intent="classify",
    )

    classifier_input = _build_classifier_input(
        query,
        conversation_candidates=conversation_candidates,
        context_gate=context_gate,
        primary_result=primary_result,
        conflict_reason=conflict_reason,
    )
    from observability import bind_llm_attempt_type  # noqa: PLC0415

    attempt_type = "strong" if task_role == _CLASSIFIER_STRONG_TASK_ROLE else "primary"
    with bind_llm_attempt_type(attempt_type):
        result = orchestrator.generate(  # type: ignore[union-attr]
            route_request=route_request,
            query=classifier_input,
            classification=None,
            context=None,
        )
    _log_classifier_prompt_budget(
        task_role=task_role,
        query=query,
        conversation_candidates=conversation_candidates,
        context_gate=context_gate,
        provider_input_tokens=getattr(result, "input_tokens", None),
    )

    classification = _parse_classifier_orchestrated_content(
        result.content, route_id=route_id
    )
    if _is_generic_latest_turn_reference(query):
        classification = _apply_deterministic_conversation_contract(
            classification,
            query=query,
            candidate_cards=candidate_cards,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )
    return _validate_conversation_contract(
        classification,
        query=query,
        candidate_cards=candidate_cards,
        candidate_turn_ids=candidate_turn_ids,
        context_gate=context_gate,
    )


def classify_query_with_strong_model(
    query: str,
    *,
    request_id: str,
    primary_result: QueryClassification,
    conflict_reason: str,
) -> QueryClassification:
    """Run the existing bounded strong route without invoking another primary."""
    return _classify_with_llm_orchestrated(
        query,
        request_id=request_id,
        task_role=_CLASSIFIER_STRONG_TASK_ROLE,
        primary_result=primary_result,
        conflict_reason=conflict_reason,
    )


def _deterministic_classifier_fallback(
    query: str,
    *,
    strong_classifier_used: bool = False,
    primary_decision: PrimaryClassificationDecision | None = None,
    primary_classification: QueryClassification | None = None,
    context_gate: str = "CONTEXT_NOT_NEEDED",
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
) -> ClassifierRunResult:
    """Hardened deterministic fallback when both classifier models fail."""
    fallback = _classify_deterministic(query)
    classification = _build_query_classification(
            fallback,
            classification_source="fallback",
            confidence_override=max(fallback.confidence, 0.55),
        )
    classification = _apply_deterministic_conversation_contract(
        classification,
        query=query,
        candidate_cards=candidate_cards,
        candidate_turn_ids=candidate_turn_ids,
        context_gate=context_gate,
    )
    if (
        context_gate == "UNCERTAIN"
        and classification.relation
        not in {"FOLLOW_UP", "CORRECTION", "RESOLVE_AGAIN"}
    ):
        classification = classification.model_copy(
            update={
                "relation": "AMBIGUOUS",
                "selected_turn_id": None,
                "requested_action": "ASK_CLARIFICATION",
                "requires_recent_conversation": True,
            }
        )
    return ClassifierRunResult(
        classification=classification,
        strong_classifier_used=strong_classifier_used,
        primary_decision=primary_decision,
        primary_classification=primary_classification,
    )


def _classify_with_llm_orchestrated_or_fallback(
    query: str,
    request_id: str | None = None,
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation_candidates: str | None = None,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
    context_gate: str = "CONTEXT_NOT_NEEDED",
) -> ClassifierRunResult:
    """Run orchestrated classifier with strong-model fallback on low confidence."""
    t_start = time.perf_counter()
    _rid = request_id or "unknown"
    primary_route_id = "general.classifier.default"
    threshold = get_classifier_confidence_threshold()
    conversation_kwargs: dict[str, object] = {}
    if conversation_candidates is not None or context_gate != "CONTEXT_NOT_NEEDED":
        conversation_kwargs = {
            "conversation_candidates": conversation_candidates,
            "candidate_turn_ids": candidate_turn_ids,
            "context_gate": context_gate,
        }
        if candidate_cards:
            conversation_kwargs["candidate_cards"] = candidate_cards

    try:
        primary = _classify_with_llm_orchestrated(
            query,
            request_id=request_id,
            task_role="classifier",
            **conversation_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        decision = _decision_for_primary_exception(exc)
        _log_classifier_json_error(exc, route_id=primary_route_id)
        _log_classifier_strong_decision(decision, primary_confidence=None)
        _log_classifier_fallback_decision(
            request_id=_rid,
            reason=decision.primary_reason.value,
            fallback_used=True,
            strong_classifier_used=True,
        )
        if on_before_strong_classifier is not None:
            on_before_strong_classifier()
        try:
            strong = _classify_with_llm_orchestrated(
                query,
                request_id=request_id,
                task_role=_CLASSIFIER_STRONG_TASK_ROLE,
                primary_result=(
                    exc.classification
                    if isinstance(exc, ConversationClassificationConflict)
                    else None
                ),
                conflict_reason=(
                    exc.reason
                    if isinstance(exc, ConversationClassificationConflict)
                    else decision.primary_reason.value
                ),
                **conversation_kwargs,
            )
            _log_classifier_primary_result(
                strong,
                valid_json=True,
                route_id="general.classifier_strong.default",
            )
            duration_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "orchestrated_classifier  request_id=%s  primary_failed=true  "
                "confidence_threshold=%.2f  fallback_used=true  "
                "final_source=llm_strong  duration_ms=%.2f",
                _rid,
                threshold,
                duration_ms,
            )
            return ClassifierRunResult(
                classification=strong,
                strong_classifier_used=True,
                primary_decision=decision,
                primary_classification=(
                    exc.classification
                    if isinstance(exc, ConversationClassificationConflict)
                    else None
                ),
            )
        except Exception as strong_exc:  # noqa: BLE001
            _log_classifier_json_error(
                strong_exc, route_id="general.classifier_strong.default"
            )
            _log_classifier_fallback_decision(
                request_id=_rid,
                reason="strong_invalid_json",
                fallback_used=True,
                strong_classifier_used=True,
            )
            logger.warning(
                "Strong classifier failed — using deterministic fallback: %s  request_id=%s",
                type(strong_exc).__name__,
                _rid,
            )
            return _deterministic_classifier_fallback(
                query,
                strong_classifier_used=True,
                primary_decision=decision,
                primary_classification=(
                    exc.classification
                    if isinstance(exc, ConversationClassificationConflict)
                    else None
                ),
                context_gate=context_gate,
                candidate_turn_ids=candidate_turn_ids,
                candidate_cards=candidate_cards,
            )

    decision = _evaluate_primary_classification(
        query,
        primary,
        context_gate=context_gate,
        candidate_cards=candidate_cards,
        candidate_turn_ids=candidate_turn_ids,
    )
    _log_classifier_primary_result(
        primary,
        valid_json=True,
        route_id=primary_route_id,
        decision=decision,
    )
    _log_classifier_strong_decision(
        decision,
        primary_confidence=primary.confidence,
    )
    if decision.accepted:
        duration_ms = (time.perf_counter() - t_start) * 1000
        _log_classifier_fallback_decision(
            request_id=_rid,
            reason=decision.primary_reason.value,
            fallback_used=False,
            strong_classifier_used=False,
        )
        logger.info(
            "orchestrated_classifier  request_id=%s  primary_confidence=%.2f  "
            "confidence_threshold=%.2f  fallback_used=false  final_source=llm  "
            "duration_ms=%.2f",
            _rid,
            primary.confidence,
            threshold,
            duration_ms,
        )
        return ClassifierRunResult(
            classification=primary,
            strong_classifier_used=False,
            primary_decision=decision,
            primary_classification=primary,
        )

    reason = decision.primary_reason.value
    _log_classifier_fallback_decision(
        request_id=_rid,
        reason=reason,
        fallback_used=True,
        strong_classifier_used=True,
    )

    if on_before_strong_classifier is not None:
        on_before_strong_classifier()

    try:
        strong = _classify_with_llm_orchestrated(
            query,
            request_id=request_id,
            task_role=_CLASSIFIER_STRONG_TASK_ROLE,
            primary_result=primary,
            conflict_reason=reason,
            **conversation_kwargs,
        )
        _log_classifier_primary_result(
            strong,
            valid_json=True,
            route_id="general.classifier_strong.default",
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "orchestrated_classifier  request_id=%s  primary_confidence=%.2f  "
            "confidence_threshold=%.2f  fallback_used=true  final_source=llm_strong  "
            "duration_ms=%.2f",
            _rid,
            primary.confidence,
            threshold,
            duration_ms,
        )
        return ClassifierRunResult(
            classification=strong,
            strong_classifier_used=True,
            primary_decision=decision,
            primary_classification=primary,
        )
    except Exception as exc:  # noqa: BLE001
        _log_classifier_json_error(exc, route_id="general.classifier_strong.default")
        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.warning(
            "Strong classifier failed — using deterministic fallback: %s  request_id=%s  "
            "primary_confidence=%.2f  duration_ms=%.2f",
            type(exc).__name__,
            _rid,
            primary.confidence,
            duration_ms,
        )
        return _deterministic_classifier_fallback(
            query,
            strong_classifier_used=True,
            primary_decision=decision,
            primary_classification=primary,
            context_gate=context_gate,
            candidate_turn_ids=candidate_turn_ids,
            candidate_cards=candidate_cards,
        )


def _classify_with_llm(query: str) -> QueryClassification:
    """Call model_router and parse the structured JSON response.

    Raises:
        Any exception from model_router or JSON parsing — caller must handle.
    """
    # Deferred imports: only loaded when LLM path is active.
    from schemas.llm import LlmMessage  # noqa: PLC0415
    from services import model_router  # noqa: PLC0415

    system_prompt = _load_classifier_prompt()
    messages = [
        LlmMessage(role="system", content=system_prompt),
        LlmMessage(role="user", content=query),
    ]

    response = model_router.generate(_CLASSIFIER_ROLE, messages)

    # [AI RISK] LLM output is untrusted — parse and validate before use.
    from services.doubt_solver.classifier_json import parse_classifier_json_strict  # noqa: PLC0415

    raw_dict, _recovered = parse_classifier_json_strict(response.content)
    classification = QueryClassification.model_validate(raw_dict)

    result = _build_query_classification(classification, classification_source="llm")
    if requires_recent_conversation(query) and not result.requires_recent_conversation:
        result = result.model_copy(update={"requires_recent_conversation": True})
    return result


def _classify_with_llm_or_fallback(query: str) -> QueryClassification:
    """Try LLM classification; fall back to deterministic on any failure."""
    t_start = time.perf_counter()
    try:
        result = _classify_with_llm(query)
        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "llm_classifier  intent=%s  confidence=%.2f  source=llm  duration_ms=%.2f",
            result.intent,
            result.confidence,
            duration_ms,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        # Log safe warning — exc may contain partial model output; never log query.
        logger.warning("LLM classifier failed — falling back to deterministic: %s", exc)
        fallback = _classify_deterministic(query)
        return _build_query_classification(
            fallback,
            classification_source="fallback",
            confidence_override=min(fallback.confidence, 0.55),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_query(
    query: str,
    request_id: str | None = None,
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation_candidates: str | None = None,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
    context_gate: str = "CONTEXT_NOT_NEEDED",
) -> QueryClassification:
    """Classify a student query, dispatching to LLM or deterministic based on config.

    Args:
        query:      The student's question or doubt (already validated by entrypoint).
        request_id: Trace ID from the originating request.  Pass
                    ``state["request_id"]`` from graph nodes.  Optional for
                    legacy callers — a new UUID is generated at the boundary
                    when not provided.
        on_before_strong_classifier: Optional hook invoked once before the strong
                    classifier runs (streaming UX only).

    Returns:
        A validated QueryClassification instance.
    """
    # Deferred import so dotenv has loaded before config is read.
    from config import get_settings  # noqa: PLC0415

    t_start = time.perf_counter()
    settings = get_settings()

    if not settings.enable_real_llm:
        academic_query = (
            _latest_candidate_question(conversation_candidates)
            if context_gate != "CONTEXT_NOT_NEEDED"
            else None
        ) or query
        result = _classify_deterministic(academic_query)
        result = _apply_deterministic_conversation_contract(
            result,
            query=query,
            candidate_cards=candidate_cards,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            "classify_query  source=deterministic  intent=%s  duration_ms=%.2f",
            result.intent,
            duration_ms,
        )
        return result

    # Orchestrated path — Azure-first via model registry when
    # ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.
    if settings.enable_orchestrated_doubt_solver:
        kwargs: dict[str, object] = {}
        if conversation_candidates is not None or context_gate != "CONTEXT_NOT_NEEDED":
            kwargs = {
                "conversation_candidates": conversation_candidates,
                "candidate_cards": candidate_cards,
                "candidate_turn_ids": candidate_turn_ids,
                "context_gate": context_gate,
            }
        return _classify_with_llm_orchestrated_or_fallback(
            query,
            request_id=request_id,
            on_before_strong_classifier=on_before_strong_classifier,
            **kwargs,
        ).classification

    # Legacy model_router path — ENABLE_ORCHESTRATED_DOUBT_SOLVER=false.
    # Check whether the classifier role is explicitly configured.
    # If ENABLE_REAL_LLM=true but the role is missing, fall back gracefully
    # rather than raising a hard error — classification is non-critical.
    try:
        role_map: dict = json.loads(settings.llm_role_config_json)
    except Exception:  # noqa: BLE001
        # [SECURITY] Do not log the raw config value — it may contain partial keys.
        logger.warning(
            "classify_query: LLM_ROLE_CONFIG_JSON is not valid JSON — "
            "ENABLE_REAL_LLM=true but config cannot be parsed; returning fallback classification"
        )
        _det = _classify_deterministic(query)
        fallback = QueryClassification(
            intent=_det.intent,
            subject=_det.subject,
            topic=_det.topic,
            difficulty=_det.difficulty,
            response_style=_det.response_style,
            confidence=min(_det.confidence, 0.55),
            retrieval_need=_det.retrieval_need,
            classification_source="fallback",
        )
        return _apply_deterministic_conversation_contract(
            fallback,
            query=query,
            candidate_cards=candidate_cards,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )

    if _CLASSIFIER_ROLE not in role_map:
        logger.debug(
            "classify_query: role %r not in LLM_ROLE_CONFIG_JSON — using deterministic",
            _CLASSIFIER_ROLE,
        )
        return _apply_deterministic_conversation_contract(
            _classify_deterministic(query),
            query=query,
            candidate_cards=candidate_cards,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )

    legacy = _classify_with_llm_or_fallback(query)
    return _apply_deterministic_conversation_contract(
        legacy,
        query=query,
        candidate_cards=candidate_cards,
        candidate_turn_ids=candidate_turn_ids,
        context_gate=context_gate,
    )
