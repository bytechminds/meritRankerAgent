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
from typing import Any

from schemas.doubt_solver import QueryClassification

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
    ],
    "english": ["grammar", "vocabulary", "sentence", "word", "synonym", "antonym", "tense"],
    "reasoning": ["series", "pattern", "sequence", "analogy", "odd one out", "logical"],
    "science": ["physics", "chemistry", "biology", "force", "energy", "atom", "molecule"],
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
    "synonym",
    "antonym",
    "sentence correction",
    "fill in the blank",
    "cloze test",
    "reading comprehension",
    "para jumble",
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


def _classifier_conflicts_with_signals(query: str, classification: QueryClassification) -> bool:
    """True when LLM labels conflict with strong deterministic subject signals."""
    query_lower = query.lower()
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
) -> None:
    logger.info(
        "classifier_primary_result  route_id=%s  valid_json=%s  confidence=%.2f  "
        "subject=%s  intent=%s  difficulty=%s",
        route_id,
        str(valid_json).lower(),
        classification.confidence,
        classification.subject,
        classification.intent,
        classification.difficulty,
    )


def _log_classifier_json_error(exc: BaseException, *, route_id: str) -> None:
    from services.doubt_solver.classifier_json import ClassifierJsonError  # noqa: PLC0415

    if isinstance(exc, ClassifierJsonError):
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
    ):
        for signal in signals:
            if signal in query_lower and len(signal) > best_len:
                best_len = len(signal)
                best_subject = subject
    return best_subject


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


def _requires_logical_inference_method(query_lower: str) -> bool:
    """True when solving is primarily inference/navigation, not numeric calculation."""
    if _requires_quantitative_solving_method(query_lower):
        return False
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
        _POLICY_REASONING_SIGNALS + _POLICY_MATH_SIGNALS + _POLICY_ENGLISH_SIGNALS,
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
            _POLICY_REASONING_SIGNALS + _POLICY_MATH_SIGNALS + _POLICY_ENGLISH_SIGNALS,
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
    """Return the classifier system prompt (cached after first read)."""
    from services.prompt_loader import load_prompt  # noqa: PLC0415

    return load_prompt("query_classifier")


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
    return _build_query_classification(classification, classification_source="llm")


def _classify_with_llm_orchestrated(
    query: str,
    request_id: str | None = None,
    *,
    task_role: str = "classifier",
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

    result = orchestrator.generate(  # type: ignore[union-attr]
        route_request=route_request,
        query=query,
        classification=None,
        context=None,
    )

    classification = _parse_classifier_orchestrated_content(
        result.content, route_id=route_id
    )
    if requires_recent_conversation(query) and not classification.requires_recent_conversation:
        classification = classification.model_copy(
            update={"requires_recent_conversation": True}
        )
    return classification


def _deterministic_classifier_fallback(
    query: str,
    *,
    strong_classifier_used: bool = False,
) -> ClassifierRunResult:
    """Hardened deterministic fallback when both classifier models fail."""
    fallback = _classify_deterministic(query)
    return ClassifierRunResult(
        classification=_build_query_classification(
            fallback,
            classification_source="fallback",
            confidence_override=max(fallback.confidence, 0.55),
        ),
        strong_classifier_used=strong_classifier_used,
    )


def _classify_with_llm_orchestrated_or_fallback(
    query: str,
    request_id: str | None = None,
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
) -> ClassifierRunResult:
    """Run orchestrated classifier with strong-model fallback on low confidence."""
    t_start = time.perf_counter()
    _rid = request_id or "unknown"
    primary_route_id = "general.classifier.default"
    threshold = get_classifier_confidence_threshold()

    try:
        primary = _classify_with_llm_orchestrated(
            query, request_id=request_id, task_role="classifier"
        )
        _log_classifier_primary_result(
            primary, valid_json=True, route_id=primary_route_id
        )
    except Exception as exc:  # noqa: BLE001
        _log_classifier_json_error(exc, route_id=primary_route_id)
        _log_classifier_fallback_decision(
            request_id=_rid,
            reason="primary_invalid_json",
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
            return ClassifierRunResult(classification=strong, strong_classifier_used=True)
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
            return _deterministic_classifier_fallback(query, strong_classifier_used=True)

    if primary.requires_recent_conversation:
        duration_ms = (time.perf_counter() - t_start) * 1000
        _log_classifier_fallback_decision(
            request_id=_rid,
            reason="follow_up_context_required",
            fallback_used=False,
            strong_classifier_used=False,
        )
        logger.info(
            "orchestrated_classifier request_id=%s follow_up_context_required=true "
            "fallback_used=false strong_classifier_used=false duration_ms=%.2f",
            _rid,
            duration_ms,
        )
        return ClassifierRunResult(
            classification=primary,
            strong_classifier_used=False,
        )

    needs_strong = (
        primary.confidence < threshold
        or _classifier_conflicts_with_signals(query, primary)
    )
    if not needs_strong:
        duration_ms = (time.perf_counter() - t_start) * 1000
        _log_classifier_fallback_decision(
            request_id=_rid,
            reason="primary_accepted",
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
        return ClassifierRunResult(classification=primary, strong_classifier_used=False)

    reason = (
        "primary_low_confidence"
        if primary.confidence < threshold
        else "primary_signal_conflict"
    )
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
        return ClassifierRunResult(classification=strong, strong_classifier_used=True)
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
        return _deterministic_classifier_fallback(query, strong_classifier_used=True)


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
        result = _classify_deterministic(query)
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
        return _classify_with_llm_orchestrated_or_fallback(
            query,
            request_id=request_id,
            on_before_strong_classifier=on_before_strong_classifier,
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
        return QueryClassification(
            intent=_det.intent,
            subject=_det.subject,
            topic=_det.topic,
            difficulty=_det.difficulty,
            response_style=_det.response_style,
            confidence=min(_det.confidence, 0.55),
            retrieval_need=_det.retrieval_need,
            classification_source="fallback",
        )

    if _CLASSIFIER_ROLE not in role_map:
        logger.debug(
            "classify_query: role %r not in LLM_ROLE_CONFIG_JSON — using deterministic",
            _CLASSIFIER_ROLE,
        )
        return _classify_deterministic(query)

    return _classify_with_llm_or_fallback(query)
