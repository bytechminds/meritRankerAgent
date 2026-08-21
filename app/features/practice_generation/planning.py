"""Request resolution, blueprint policy, validation, and bounded planning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import ValidationError

from features.practice_generation.schemas import (
    Complexity,
    Difficulty,
    PlannerFamily,
    PlannerSlot,
    PracticeBlueprint,
    PracticeFreshnessRequirement,
    PracticeGenerationRequest,
    PracticeLaunchDecision,
    PracticeType,
    QuestionType,
    VerificationPolicy,
)
from practice_limits import MAX_PRACTICE_QUESTIONS
from schemas.doubt_solver import CanonicalLanguage, normalize_question_language
from services.classification.web_search_demand import is_freshness_sensitive_query
from services.llm.orchestration.errors import ProviderExecutionError
from tools.web_search.models import FreshEvidenceBundle

_GENERATION_TARGET = (
    r"(?:questions?|problems?|items?|sawaals?|sawals?|prashn|practice\s+(?:set|test)|"
    r"quiz|topic\s+test|sectional\s+test|mini\s+mocks?|mock\s+test|full\s+mock)"
)
_CREATION_SIGNAL = re.compile(
    rf"\b(?P<verb>generate|create|give|provide|make|build|prepare|banao|"
    rf"bana\s+do|taiyar\s+karo|tayyar\s+karo)\b"
    rf"(?P<object>.{{0,100}}?)\b(?P<artifact>{_GENERATION_TARGET})\b",
    re.IGNORECASE,
)
_HINDI_CREATION_SIGNAL = re.compile(
    r"(?:बनाओ|बना\s+दो|तैयार\s+करो|दीजिए|दे\s+दो).{0,100}"
    r"(?:प्रश्न|सवाल|क्विज|मॉक\s+टेस्ट)|"
    r"(?:प्रश्न|सवाल|क्विज|मॉक\s+टेस्ट).{0,100}"
    r"(?:बनाओ|बना\s+दो|तैयार\s+करो|दीजिए|दे\s+दो)",
    re.IGNORECASE,
)
_COUNT_SIGNAL = re.compile(
    r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"twenty|fifty|hundred|ek|do|teen|char|chaar|paanch|das|bees|pachas|sau)\b"
    r"(?:\s+\S+){0,3}\s+(?:questions?|problems?|items?|sawaals?|sawals?|prashn)\b",
    re.IGNORECASE,
)
_HINDI_COUNT_SIGNAL = re.compile(
    r"(?:एक|दो|तीन|चार|पांच|पाँच|दस|बीस|पचास|सौ)"
    r"(?:\s+\S+){0,3}\s+(?:प्रश्न|सवाल)",
)
_ADVICE_SIGNAL = re.compile(
    r"\b(?:how\s+should|how\s+can|explain|which\s+topics?|"
    r"what\s+(?:is|are)|is\s+this|batao|samjhao|kya\s+hai)\b|"
    r"(?:के\s+बारे|क्या\s+है|समझाओ)",
    re.IGNORECASE,
)
_NON_ARTIFACT_OBJECT = re.compile(
    r"\b(?:tips?|strategy|strategies|study\s+plan|preparation|advice|benefits?|"
    r"importance|schedule)\b",
    re.IGNORECASE,
)
_COUNT_UNIT = (
    r"(?:questions?|problems?|items?|sawaals?|sawals?|prashn|quiz|practice|mock|test)"
)
_COUNT_PATTERN = re.compile(
    r"(?<!\w)(-?\d{1,3})\b(?:-" + _COUNT_UNIT + r"\b|(?:\s+\S+){0,3}\s+" + _COUNT_UNIT + r"\b)",
    re.IGNORECASE,
)
_MIXED_DIFFICULTY_SIGNAL = re.compile(r"\b(?:mixed|mix)\b", re.IGNORECASE)
_EXPLICIT_DIFFICULTY_SIGNAL = re.compile(
    r"\b(?:basic|easy|beginner|intermediate|medium|moderate|advanced|hard|"
    r"difficult|expert)\b",
    re.IGNORECASE,
)
_EXPLICIT_DELIVERY_LANGUAGE = re.compile(
    r"(?:\b(?:in|into|using)\s+(?P<english>english|hindi|hinglish)\b|"
    r"\b(?P<roman>english|hindi|hinglish)\s+(?:me|mein)\b|"
    r"(?P<hindi>हिंदी|हिन्दी)\s*(?:में|मे))",
    re.IGNORECASE,
)
_WORD_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "fifty": 50,
    "hundred": 100,
    "ek": 1,
    "do": 2,
    "teen": 3,
    "char": 4,
    "chaar": 4,
    "paanch": 5,
    "das": 10,
    "bees": 20,
    "pachas": 50,
    "sau": 100,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "दस": 10,
    "बीस": 20,
    "पचास": 50,
    "सौ": 100,
}
_DEFAULT_COUNTS = {
    PracticeType.SIMILAR_QUESTION: 1,
    PracticeType.QUICK_PRACTICE: 5,
    PracticeType.QUIZ: 10,
    PracticeType.TOPIC_TEST: 20,
    PracticeType.SECTIONAL_TEST: 50,
    PracticeType.FULL_MOCK: 100,
    PracticeType.CURRENT_AFFAIRS_SET: 10,
}

PRACTICE_ASYNC_NOT_CONFIGURED = "PRACTICE_AGENTCORE_ASYNC_NOT_CONFIGURED"


class PracticeRequestCountError(ValueError):
    """Raised when an explicit Practice request is outside the supported range."""

    reason_code = "PRACTICE_REQUEST_COUNT_OUT_OF_RANGE"


def practice_async_unavailable_message(language: str) -> str:
    if language == "hindi":
        return (
            "Practice generation abhi temporary unavailable hai. "
            "AgentCore background execution enable hone ke baad dobara try karein."
        )
    return (
        "Practice generation is temporarily unavailable. "
        "Please try again after AgentCore background execution is enabled."
    )


def practice_route_enabled() -> bool:
    """Read only the safe route flag; resource loading belongs to future orchestration."""
    return os.getenv("PRACTICE_GENERATION_ENABLED", "false").casefold() == "true"


class PlannerProvider(Protocol):
    def plan(
        self,
        request: PracticeGenerationRequest,
        *,
        tier: str,
        repair_feedback: str | None = None,
    ) -> str: ...


class BlueprintPlanningError(RuntimeError):
    """Raised when complex planning cannot be repaired safely."""

    def __init__(
        self,
        reason_code: str,
        *,
        diagnostics: tuple[PlannerValidationDiagnostic, ...] = (),
        fallback_invoked: bool = False,
        fallback_result: str | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.diagnostics = diagnostics
        self.fallback_invoked = fallback_invoked
        self.fallback_result = fallback_result


def decide_practice_launch(
    query: str,
    classification: dict[str, object],
) -> PracticeLaunchDecision:
    if str(classification.get("intent") or "") != "practice":
        return PracticeLaunchDecision(
            eligible=False,
            reason_code="CLASSIFIER_INTENT_NOT_PRACTICE",
        )
    normalized = re.sub(r"\s+", " ", query).strip()
    creation = _CREATION_SIGNAL.search(normalized)
    hindi_creation = _HINDI_CREATION_SIGNAL.search(normalized)
    requested_artifact: str | None = None
    explicit_creation = False
    creation_context = (
        creation.group("object") + normalized[creation.end() : creation.end() + 60]
        if creation is not None
        else ""
    )
    if creation is not None and not _NON_ARTIFACT_OBJECT.search(creation_context):
        requested_artifact = creation.group("artifact").casefold()
        explicit_creation = True
    if hindi_creation is not None:
        requested_artifact = "questions"
        explicit_creation = True
    count_creation = _COUNT_SIGNAL.search(normalized) or _HINDI_COUNT_SIGNAL.search(normalized)
    if count_creation is not None and not _ADVICE_SIGNAL.search(normalized):
        requested_artifact = "questions"
        explicit_creation = True
    if re.search(
        r"\b(?:another|similar)\s+question\b",
        normalized,
        re.IGNORECASE,
    ):
        requested_artifact = "similar_question"
        explicit_creation = True
    if re.search(r"\bquiz\s+me\b", normalized, re.IGNORECASE):
        requested_artifact = "quiz"
        explicit_creation = True
    if _ADVICE_SIGNAL.search(normalized) and not explicit_creation:
        return PracticeLaunchDecision(
            eligible=False,
            reason_code="PRACTICE_ADVICE_REQUEST",
            requested_artifact=None,
        )
    if explicit_creation:
        return PracticeLaunchDecision(
            eligible=True,
            reason_code="EXPLICIT_PRACTICE_CREATION_REQUEST",
            requested_artifact=requested_artifact,
        )
    return PracticeLaunchDecision(
        eligible=False,
        reason_code="PRACTICE_CREATION_SIGNAL_MISSING",
        requested_artifact=None,
    )


def resolve_practice_freshness_requirement(
    query: str,
    classification: dict[str, object],
) -> PracticeFreshnessRequirement:
    """Require evidence only for explicit dynamic-fact Practice requests."""
    if str(classification.get("intent") or "") != "practice":
        return PracticeFreshnessRequirement()
    if not is_freshness_sensitive_query(query):
        return PracticeFreshnessRequirement()
    reason = str(classification.get("web_search_reason") or "current_event").strip()
    return PracticeFreshnessRequirement(
        requires_fresh_evidence=True,
        freshness_reason=reason or "current_event",
    )


@dataclass(frozen=True)
class BlueprintPlanResult:
    blueprint: PracticeBlueprint
    planner_calls: int
    repaired: bool
    deterministic_fallback: bool
    tier: str
    validation_reason_code: str | None = None
    validation_actual_slot_count: int | None = None
    validation_duration_ms: int | None = None
    validation_diagnostics: tuple[PlannerValidationDiagnostic, ...] = ()


@dataclass(frozen=True)
class PlannerValidationDiagnostic:
    """Safe validation metadata retained across bounded planner recovery."""

    attempt: int
    phase: str
    schema_name: str
    error_count: int
    field_paths: tuple[str, ...]
    error_types: tuple[str, ...]
    reason_code: str
    # Which layer rejected the response: json parse, Pydantic schema, or the
    # semantic blueprint contract applied after model validation.
    validation_stage: str = "schema"
    actual_slot_count: int | None = None
    duration_ms: int | None = None


def _planner_validation_reason(
    error: Exception,
    *,
    raw: str,
) -> tuple[str, int | None]:
    """Return a stable validation reason without retaining planner content."""
    actual_slot_count: int | None = None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            blueprint = payload.get("blueprint", payload)
            if isinstance(blueprint, dict) and isinstance(blueprint.get("slots"), list):
                actual_slot_count = len(blueprint["slots"])
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    message = str(error).casefold()
    if "slot count" in message or "accepted_count" in message:
        return "PLANNER_SLOT_COUNT_MISMATCH", actual_slot_count
    if "slot ids must be unique" in message or "slot id" in message and "duplicate" in message:
        return "PLANNER_DUPLICATE_SLOT", actual_slot_count
    if "topic coverage" in message or "topic_coverage" in message:
        return "PLANNER_TOPIC_COVERAGE_INVALID", actual_slot_count
    if "canonical identifiers" in message:
        return "PLANNER_SLOT_IDENTIFIER_INVALID", actual_slot_count
    if "difficulty" in message:
        return "PLANNER_INVALID_DIFFICULTY", actual_slot_count
    if "category" in message:
        return "PLANNER_MISSING_CATEGORY", actual_slot_count
    if "accepted" in message or "required_count" in message:
        return "PLANNER_INVALID_TOTAL", actual_slot_count
    return "PLANNER_SCHEMA_INVALID", actual_slot_count


def _planner_validation_diagnostic(
    error: Exception,
    *,
    raw: str,
    attempt: int,
    phase: str,
    duration_ms: int,
) -> PlannerValidationDiagnostic:
    """Extract allowlisted validation metadata without retaining model content."""
    reason_code, actual_slot_count = _planner_validation_reason(error, raw=raw)
    schema_name = "PracticeBlueprint"
    field_paths: tuple[str, ...] = ()
    error_types: tuple[str, ...] = ()
    error_count = 1
    if isinstance(error, ValidationError):
        errors = error.errors(include_url=False)
        error_count = len(errors)
        field_paths = tuple(
            sorted(
                {
                    ".".join(str(part) for part in item.get("loc", ())) or "$"
                    for item in errors
                }
            )
        )
        error_types = tuple(
            sorted({str(item.get("type") or "validation_error") for item in errors})
        )
    elif isinstance(error, json.JSONDecodeError):
        schema_name = "PlannerResponse"
        field_paths = ("$",)
        error_types = ("json_invalid",)
    elif isinstance(error, (TypeError, ValueError)):
        field_paths = ("$",)
        error_types = (type(error).__name__,)
    if isinstance(error, ValidationError):
        validation_stage = "schema"
    elif isinstance(error, json.JSONDecodeError):
        validation_stage = "parse"
    else:
        validation_stage = "contract"
    return PlannerValidationDiagnostic(
        attempt=attempt,
        phase=phase,
        schema_name=schema_name,
        validation_stage=validation_stage,
        error_count=error_count,
        field_paths=field_paths,
        error_types=error_types,
        reason_code=reason_code,
        actual_slot_count=actual_slot_count,
        duration_ms=duration_ms,
    )


def _planner_repair_feedback(diagnostic: PlannerValidationDiagnostic) -> str:
    """Provide bounded, non-sensitive correction data to the sole repair attempt."""
    fields = ",".join(diagnostic.field_paths) or "$"
    error_types = ",".join(diagnostic.error_types) or "validation_error"
    return (
        f"reason={diagnostic.reason_code};fields={fields};"
        f"types={error_types};schema={diagnostic.schema_name}"
    )


def _deterministic_slot_difficulties(
    request: PracticeGenerationRequest,
) -> tuple[Difficulty, ...]:
    """Use a balanced generic fallback only for an explicitly mixed request."""
    requested_mixed = request.mixed_difficulty_requested or bool(
        _MIXED_DIFFICULTY_SIGNAL.search(request.original_query)
    )
    requested_difficulty = request.explicit_difficulty_requested or bool(
        _EXPLICIT_DIFFICULTY_SIGNAL.search(request.original_query)
    )
    if not requested_mixed or requested_difficulty:
        return (request.difficulty,) * request.accepted_count
    levels = (Difficulty.BASIC, Difficulty.INTERMEDIATE, Difficulty.ADVANCED)
    return tuple(levels[index % len(levels)] for index in range(request.accepted_count))


def resolve_practice_type(query: str) -> PracticeType:
    normalized = query.casefold()
    if "similar" in normalized:
        return PracticeType.SIMILAR_QUESTION
    if "full mock" in normalized:
        return PracticeType.FULL_MOCK
    if "sectional" in normalized:
        return PracticeType.SECTIONAL_TEST
    if "topic test" in normalized:
        return PracticeType.TOPIC_TEST
    if "current affair" in normalized:
        return PracticeType.CURRENT_AFFAIRS_SET
    if "quiz" in normalized:
        return PracticeType.QUIZ
    return PracticeType.QUICK_PRACTICE


def resolve_requested_count(query: str, practice_type: PracticeType) -> int:
    numeric = _COUNT_PATTERN.search(query)
    if numeric is not None:
        return int(numeric.group(1))
    normalized = query.casefold()
    for word, count in _WORD_COUNTS.items():
        artifact = (
            r"(?:प्रश्न|सवाल)"
            if any("\u0900" <= character <= "\u097f" for character in word)
            else (
                r"(?:questions?|problems?|items?|sawaals?|sawals?|prashn|"
                r"quiz|practice|mock|test)\b"
            )
        )
        if re.search(
            rf"(?<!\w){re.escape(word)}(?!\w)(?:\s+\S+){{0,3}}\s+" + artifact,
            normalized,
        ):
            return count
    return _DEFAULT_COUNTS[practice_type]


def required_fresh_evidence_count(query: str) -> int:
    """Return the exact number of independently grounded fresh facts required."""
    practice_type = resolve_practice_type(query)
    return resolve_requested_count(query, practice_type)


def is_fresh_evidence_request_supported(query: str) -> bool:
    """Only a valid Practice count may enter a freshness-required retrieval path."""
    practice_type = resolve_practice_type(query)
    count = resolve_requested_count(query, practice_type)
    return 1 <= count <= MAX_PRACTICE_QUESTIONS


def validate_practice_requested_count(query: str) -> int:
    """Resolve and validate the single user-requested Practice count authority."""
    practice_type = resolve_practice_type(query)
    requested_count = resolve_requested_count(query, practice_type)
    if not 1 <= requested_count <= MAX_PRACTICE_QUESTIONS:
        raise PracticeRequestCountError(PracticeRequestCountError.reason_code)
    return requested_count


def resolve_practice_request(
    *,
    request_id: str,
    user_id: str,
    conversation_id: str,
    turn_id: str,
    query: str,
    subject: str,
    topic: str | None,
    difficulty: str,
    language: str,
    exam_id: str | None,
    exam_stage: str | None,
    exam_profile_id: str | None = None,
    source_question_reference: str | None = None,
    freshness_requirement: PracticeFreshnessRequirement | None = None,
    fresh_evidence: FreshEvidenceBundle | None = None,
) -> PracticeGenerationRequest:
    practice_type = resolve_practice_type(query)
    requested_count = validate_practice_requested_count(query)
    accepted_count = requested_count
    normalized_difficulty = {
        "basic": Difficulty.BASIC,
        "intermediate": Difficulty.INTERMEDIATE,
        "advanced": Difficulty.ADVANCED,
    }.get(difficulty, Difficulty.INTERMEDIATE)
    display_topic = topic or subject.replace("_", " ").title()
    title = f"{display_topic} {practice_type.value.replace('_', ' ').title()}"
    resolved_language, language_source = resolve_practice_delivery_language(
        query,
        language,
    )
    freshness_requirement = freshness_requirement or PracticeFreshnessRequirement()
    return PracticeGenerationRequest(
        request_id=request_id,
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        original_query=query,
        practice_type=practice_type,
        requested_count=requested_count,
        accepted_count=accepted_count,
        subject=subject,
        topic=topic,
        difficulty=normalized_difficulty,
        mixed_difficulty_requested=bool(_MIXED_DIFFICULTY_SIGNAL.search(query)),
        explicit_difficulty_requested=bool(
            _EXPLICIT_DIFFICULTY_SIGNAL.search(query)
        ),
        language=resolved_language,
        language_source=language_source,
        exam_id=exam_id,
        exam_stage=exam_stage,
        exam_profile_id=exam_profile_id,
        source_question_reference=source_question_reference,
        requires_fresh_evidence=freshness_requirement.requires_fresh_evidence,
        freshness_reason=freshness_requirement.freshness_reason,
        fresh_evidence=fresh_evidence,
        assessment_title=title,
    )


def resolve_practice_delivery_language(
    query: str,
    requested_language: str,
) -> tuple[CanonicalLanguage, Literal["REQUEST", "EXPLICIT_QUERY"]]:
    """Give an unambiguous current practice-language command precedence."""
    requested = normalize_question_language(requested_language)
    explicit: CanonicalLanguage | None = None
    for match in _EXPLICIT_DELIVERY_LANGUAGE.finditer(query):
        raw = match.group("english") or match.group("roman") or match.group("hindi")
        if raw:
            explicit = normalize_question_language(raw)
    if explicit is not None:
        return explicit, "EXPLICIT_QUERY"
    return requested, "REQUEST"


def request_idempotency_key(request: PracticeGenerationRequest) -> str:
    normalized = {
        "userId": request.user_id,
        "conversationId": request.conversation_id,
        "turnId": request.turn_id,
        "practiceType": request.practice_type.value,
        "requestedCount": request.requested_count,
        "acceptedCount": request.accepted_count,
        "subject": request.subject.casefold(),
        "topic": (request.topic or "").casefold(),
        "difficulty": request.difficulty.value,
        "language": request.language,
        "examId": request.exam_id or "",
        "examStage": request.exam_stage or "",
        "examProfileId": request.exam_profile_id or "",
        "includeSolutions": request.include_solutions,
        "sourceQuestionReference": request.source_question_reference or "",
        "normalizedQuery": re.sub(
            r"\s+",
            " ",
            request.original_query.casefold(),
        ).strip(),
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deterministic_test_id(idempotency_key: str) -> str:
    return f"practice-{idempotency_key[:32]}"


def planner_tier(request: PracticeGenerationRequest) -> str:
    if (
        request.practice_type in {PracticeType.SECTIONAL_TEST, PracticeType.FULL_MOCK}
        or request.accepted_count >= 31
    ):
        return "strong"
    if request.accepted_count >= 11:
        return "standard"
    return "light"


def select_planner_family(request: PracticeGenerationRequest) -> PlannerFamily:
    normalized = request.subject.casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"math", "reasoning"}:
        return PlannerFamily.QUANT_REASONING
    if normalized == "english":
        return PlannerFamily.ENGLISH
    return PlannerFamily.FACTUAL


def _generator_route_hint(subject: str, difficulty: Difficulty) -> str:
    normalized = subject.casefold().replace("-", "_").replace(" ", "_")
    route_subject = normalized if normalized in {"math", "reasoning", "english"} else "general"
    return f"{route_subject}.generator.{difficulty.value}"


def _slot_complexity(difficulty: Difficulty) -> Complexity:
    return {
        Difficulty.BASIC: Complexity.LOW,
        Difficulty.INTERMEDIATE: Complexity.MEDIUM,
        Difficulty.ADVANCED: Complexity.HIGH,
    }[difficulty]


_FALLBACK_TOPIC_SEPARATOR = re.compile(r"\s*(?:,|;|\||\+|/)\s*")
_FALLBACK_IDENTIFIER_INVALID = re.compile(r"[^a-z0-9]+")


def _canonical_fallback_topic(value: str) -> str:
    normalized = _FALLBACK_IDENTIFIER_INVALID.sub(
        "_",
        value.casefold().replace("&", " and "),
    ).strip("_")
    if not normalized:
        raise ValueError("PRACTICE_FALLBACK_TOPIC_INVALID")
    return normalized


_LIST_CONJUNCTION_PREFIX = re.compile(r"^(?:and|or)\s+", re.IGNORECASE)


def _strip_list_conjunction(value: str) -> str:
    """Drop a leading list conjunction from an already-separated topic.

    Applies only after the accepted delimiters have split the list, so an Oxford
    comma ("percentage, ratio, and average") stops producing the topic
    "and_average". Compound academic names keep their internal conjunction —
    "profit and loss" has no leading "and" and is untouched — and a value that is
    nothing but a conjunction is left alone rather than emptied.
    """
    stripped = _LIST_CONJUNCTION_PREFIX.sub("", value.strip(), count=1).strip()
    return stripped or value


def _requested_topic_ids(request: PracticeGenerationRequest) -> tuple[str, ...]:
    raw_topic = (request.topic or request.subject).strip()
    values = [
        _strip_list_conjunction(value)
        for value in _FALLBACK_TOPIC_SEPARATOR.split(raw_topic)
    ]
    topics = tuple(
        dict.fromkeys(
            _canonical_fallback_topic(value)
            for value in values
            if value.strip()
        )
    )
    if not topics:
        raise ValueError("PRACTICE_FALLBACK_TOPIC_INVALID")
    return topics


def _verification_policy(
    subject: str,
    difficulty: Difficulty,
    question_type: QuestionType,
    *,
    topic: str,
    question_intent: str,
) -> VerificationPolicy:
    normalized_subject = subject.casefold()
    complex_text = f"{topic} {question_intent}".casefold()
    if normalized_subject in {"math", "reasoning"} and difficulty in {
        Difficulty.INTERMEDIATE,
        Difficulty.ADVANCED,
    }:
        return VerificationPolicy.MANDATORY
    if normalized_subject in {"math", "reasoning"} and (
        "puzzle" in complex_text
        or question_type
        in {
            QuestionType.NUMERICAL,
            QuestionType.MSQ,
        }
    ):
        return VerificationPolicy.MANDATORY
    if normalized_subject in {"math", "reasoning"}:
        return VerificationPolicy.SELECTIVE
    return VerificationPolicy.NONE


def apply_system_bucket_policy(
    blueprint: PracticeBlueprint,
    request: PracticeGenerationRequest,
) -> PracticeBlueprint:
    if any(bucket.question_type is not QuestionType.MCQ for bucket in blueprint.buckets):
        raise ValueError("PLAYER_UNSUPPORTED_QUESTION_TYPE")
    if blueprint.schema_version == "2":
        requested_exam = (
            request.exam_id.strip().upper().replace("-", "_").replace(" ", "_")
            if request.exam_id
            else None
        )
        requested_subject = request.subject.casefold().replace("-", "_").replace(" ", "_")
        requested_topics = set(_requested_topic_ids(request))
        planned_topics = {slot.topic_id for slot in blueprint.slots}
        # Feasibility-aware coverage, not weaker coverage. One slot carries one
        # topic, so a request for fewer questions than topics cannot cover them
        # all; demanding it rejected every such blueprint outright. Above the
        # boundary the original requirement is unchanged; below it the plan must
        # still draw only from the requested topics, so no topic is ever invented.
        if request.accepted_count >= len(requested_topics):
            if not requested_topics.issubset(planned_topics):
                raise ValueError("PLANNER_SLOT_TOPIC_COVERAGE_INVALID")
        elif not planned_topics.issubset(requested_topics):
            raise ValueError("PLANNER_SLOT_TOPIC_COVERAGE_INVALID")
        for slot in blueprint.slots:
            if slot.question_type is not QuestionType.MCQ:
                raise ValueError("PLAYER_UNSUPPORTED_QUESTION_TYPE")
            if (
                requested_subject not in {"general", "other"}
                and slot.subject_id != requested_subject
            ):
                raise ValueError("PLANNER_SLOT_SUBJECT_MISMATCH")
            if requested_exam and slot.exam_ids != [requested_exam]:
                raise ValueError("PLANNER_SLOT_EXAM_MISMATCH")
            if slot.generator_route_hint != _generator_route_hint(
                slot.subject_id,
                slot.difficulty,
            ):
                raise ValueError("PLANNER_SLOT_ROUTE_HINT_INVALID")
    return blueprint.model_copy(
        update={
            "buckets": [
                bucket.model_copy(
                    update={
                        "verification_policy": _verification_policy(
                            bucket.subject,
                            bucket.difficulty,
                            bucket.question_type,
                            topic=bucket.topic,
                            question_intent=bucket.question_intent,
                        ),
                        "solution_required": request.include_solutions,
                    }
                )
                for bucket in blueprint.buckets
            ]
        }
    )


def deterministic_blueprint(
    request: PracticeGenerationRequest,
) -> PracticeBlueprint:
    topics = _requested_topic_ids(request)
    question_type = QuestionType.MCQ
    exam_ids = (
        [request.exam_id.strip().upper().replace("-", "_").replace(" ", "_")]
        if request.exam_id
        else []
    )
    slot_difficulties = _deterministic_slot_difficulties(request)
    blueprint = PracticeBlueprint(
        schema_version="2",
        practice_type=request.practice_type,
        accepted_count=request.accepted_count,
        planner_family=select_planner_family(request),
        slots=[
            PlannerSlot(
                slot_id=f"slot-{index:03d}",
                subject_id=request.subject,
                topic_id=topics[(index - 1) % len(topics)],
                category_id=topics[(index - 1) % len(topics)],
                difficulty=slot_difficulties[index - 1],
                complexity=_slot_complexity(slot_difficulties[index - 1]),
                exam_ids=exam_ids,
                question_type=question_type,
                target_skill=f"solve_{topics[(index - 1) % len(topics)]}",
                variation_hint=f"variant_{index:03d}",
                generator_route_hint=_generator_route_hint(
                    request.subject,
                    slot_difficulties[index - 1],
                ),
                generation_group_hint=min(request.accepted_count, 5),
            )
            for index in range(1, request.accepted_count + 1)
        ],
    )
    return apply_system_bucket_policy(blueprint, request)


def parse_blueprint(raw: str, request: PracticeGenerationRequest) -> PracticeBlueprint:
    payload = json.loads(raw)
    if "blueprint" in payload:
        payload = payload["blueprint"]
    payload["schema_version"] = "2"
    payload["practice_type"] = request.practice_type.value
    payload["accepted_count"] = request.accepted_count
    payload["planner_family"] = select_planner_family(request).value
    return apply_system_bucket_policy(
        PracticeBlueprint.model_validate(payload),
        request,
    )


def _quick_practice_is_deterministic(request: PracticeGenerationRequest) -> bool:
    """Quick Practice plans mechanically, so the planner LLM adds nothing.

    Scoped to QUICK_PRACTICE only: every other practice type keeps the existing
    LLM planner, and flipping PRACTICE_PLANNER_MODE back to "llm" restores the
    previous behaviour without touching code.
    """
    if os.getenv("PRACTICE_PLANNER_MODE", "deterministic").strip().lower() != "deterministic":
        return False
    return request.practice_type is PracticeType.QUICK_PRACTICE


class BlueprintManager:
    def __init__(self, planner: PlannerProvider, *, repair_limit: int = 1) -> None:
        self._planner = planner
        self._repair_limit = min(max(repair_limit, 0), 1)

    @staticmethod
    def _deterministic_result(
        request: PracticeGenerationRequest,
        *,
        planner_calls: int,
        repaired: bool,
        deterministic_fallback: bool,
        tier: str,
        diagnostics: tuple[PlannerValidationDiagnostic, ...],
    ) -> BlueprintPlanResult:
        try:
            blueprint = deterministic_blueprint(request)
        except (TypeError, ValueError, ValidationError) as exc:
            diagnostic = _planner_validation_diagnostic(
                exc,
                raw="",
                attempt=planner_calls,
                phase="fallback",
                duration_ms=0,
            )
            raise BlueprintPlanningError(
                "PRACTICE_PLANNER_FALLBACK_INVALID",
                diagnostics=(*diagnostics, diagnostic),
                fallback_invoked=True,
                fallback_result="invalid",
            ) from exc
        first_diagnostic = diagnostics[0] if diagnostics else None
        return BlueprintPlanResult(
            blueprint=blueprint,
            planner_calls=planner_calls,
            repaired=repaired,
            deterministic_fallback=deterministic_fallback,
            tier=tier,
            validation_reason_code=(
                first_diagnostic.reason_code if first_diagnostic is not None else None
            ),
            validation_actual_slot_count=(
                first_diagnostic.actual_slot_count if first_diagnostic is not None else None
            ),
            validation_duration_ms=(
                first_diagnostic.duration_ms if first_diagnostic is not None else None
            ),
            validation_diagnostics=diagnostics,
        )

    def build(self, request: PracticeGenerationRequest) -> BlueprintPlanResult:
        tier = planner_tier(request)
        if request.accepted_count <= 2 or _quick_practice_is_deterministic(request):
            result = self._deterministic_result(
                request,
                planner_calls=0,
                repaired=False,
                deterministic_fallback=False,
                tier=tier,
                diagnostics=(),
            )
            return result

        calls = 0
        feedback: str | None = None
        diagnostics: list[PlannerValidationDiagnostic] = []
        for attempt in range(self._repair_limit + 1):
            calls += 1
            started_at = time.monotonic()
            raw = ""
            try:
                raw = self._planner.plan(
                    request,
                    tier=tier,
                    repair_feedback=feedback,
                )
                return BlueprintPlanResult(
                    blueprint=parse_blueprint(raw, request),
                    planner_calls=calls,
                    repaired=attempt > 0,
                    deterministic_fallback=False,
                    tier=tier,
                    validation_reason_code=(
                        diagnostics[0].reason_code if diagnostics else None
                    ),
                    validation_actual_slot_count=(
                        diagnostics[0].actual_slot_count if diagnostics else None
                    ),
                    validation_duration_ms=(
                        diagnostics[0].duration_ms if diagnostics else None
                    ),
                    validation_diagnostics=tuple(diagnostics),
                )
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                diagnostic = _planner_validation_diagnostic(
                    exc,
                    raw=raw,
                    attempt=calls,
                    phase="initial" if attempt == 0 else "repair",
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                )
                diagnostics.append(diagnostic)
                feedback = _planner_repair_feedback(diagnostic)
            except ProviderExecutionError as exc:  # provider boundary owns retries/fallbacks
                if request.practice_type in {
                    PracticeType.SECTIONAL_TEST,
                    PracticeType.FULL_MOCK,
                }:
                    raise BlueprintPlanningError(
                        "PRACTICE_PLANNER_PROVIDER_UNAVAILABLE",
                        diagnostics=tuple(diagnostics),
                    ) from exc
                return self._deterministic_result(
                    request,
                    planner_calls=calls,
                    repaired=False,
                    deterministic_fallback=True,
                    tier=tier,
                    diagnostics=tuple(diagnostics),
                )

        if request.practice_type in {
            PracticeType.SECTIONAL_TEST,
            PracticeType.FULL_MOCK,
        }:
            raise BlueprintPlanningError(
                "PRACTICE_PLANNER_REPAIR_FAILED",
                diagnostics=tuple(diagnostics),
            )
        return self._deterministic_result(
            request,
            planner_calls=calls,
            repaired=self._repair_limit == 1,
            deterministic_fallback=True,
            tier=tier,
            diagnostics=tuple(diagnostics),
        )
