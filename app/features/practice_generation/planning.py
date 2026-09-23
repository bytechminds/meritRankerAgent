"""Request resolution, blueprint policy, validation, and bounded planning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from features.practice_generation.metadata_normalization import (
    normalize_subject as normalize_practice_subject,
)
from features.practice_generation.request_intelligence import (
    PracticeRequestInterpreter,
    interpret_practice_request,
    interpreted_total_count,
)
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
    RequestedPracticeConstraint,
    VerificationPolicy,
)
from observability.events import log_event
from practice_limits import (
    MAX_REQUESTED_PRACTICE_QUESTIONS,
    effective_practice_question_count,
)
from schemas.doubt_solver import CanonicalLanguage, normalize_question_language
from schemas.practice_limit import PracticeLimitation
from schemas.practice_request_intelligence import (
    CustomDifficulty,
    MixedDifficulty,
    PracticeRequestIntelligence,
    SingleDifficulty,
)
from services.classification.web_search_demand import is_freshness_sensitive_query
from services.llm.orchestration.errors import ProviderExecutionError
from services.llm.providers.finish_reasons import normalize_completion_outcome
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
# The unit a student writes the number against. Devanagari forms and the common
# clipped/misspelled English ones belong here for the same reason "sawal" already
# does: the digit is explicit and unambiguous, and only the unit word varies. Longer
# alternatives precede their prefixes so "questions" is never consumed as "ques".
_COUNT_UNIT = (
    r"(?:questions?|quesitons?|questin|ques|problems?|items?|sawaals?|sawals?|prashn|"
    r"सवाल|प्रश्न|quiz|practice|mock|test)"
)


def _safe_count_filler(candidate: str) -> str:
    """Return the filler allowed between a count candidate and its count unit.

    Filler may not cross another count candidate or a hard clause boundary, so an
    earlier unrelated number cannot reach past the real request to claim the unit:
    without this, "in 2 weeks. Create 10 questions" resolved to 2.
    """
    return rf"(?:\s+(?!{candidate})[^\s.?!;:]+)"


_COUNT_FILLER = _safe_count_filler(r"-?\d{1,3}\b")
_COUNT_PATTERN = re.compile(
    r"(?<!\w)(-?\d{1,3})\b(?:-"
    + _COUNT_UNIT
    + r"\b|"
    + _COUNT_FILLER
    + r"{0,3}\s+"
    + _COUNT_UNIT
    + r"\b)",
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
_WORD_COUNT_FILLER = _safe_count_filler(
    r"(?:" + "|".join(re.escape(word) for word in _WORD_COUNTS) + r")(?!\w)"
)
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


class PracticeRequestConstraintCountError(PracticeRequestCountError):
    """Raised when one question per accepted constraint cannot fit the request."""

    reason_code = "PRACTICE_REQUEST_CONSTRAINT_COUNT_INFEASIBLE"


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
    # The classifier owns "does this student want practice content created?". Requiring
    # the raw text to prove creation again re-ran that semantic decision with exact
    # keywords and rejected legitimate misspelled requests. The advice guard above is
    # retained as containment only; it never fires once a creation signal is present.
    return PracticeLaunchDecision(
        eligible=True,
        reason_code="CLASSIFIER_PRACTICE_CREATION_INTENT",
        requested_artifact=requested_artifact,
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
    # A safe, bounded summary of the violated constraint ("max_length=160,
    # actual_length=213") built only from Pydantic's own ``ctx`` and a length computed
    # from (never retaining) the offending value. Empty when the error carries none.
    constraint: str = ""
    # Verbatim text of the failing invariant, populated ONLY when the message is one
    # of our own literals. Model and student text can never appear here.
    owned_message: str = ""
    # "module.py:line" of the raise inside our own source tree. Code location only —
    # it carries no model, student or field content, and is empty for anything raised
    # outside this application.
    owned_origin: str = ""


@lru_cache(maxsize=1)
def _owned_value_error_messages() -> frozenset[str]:
    """Literal ValueError messages raised by this package's own contract code.

    Only these may be logged verbatim. Anything else — including any message that
    interpolates model output, student text or field values — stays hidden behind the
    reason code. Built by reading this package's own source; if that is unavailable the
    set is empty, so the safe default is to log nothing.
    """
    package_root = Path(__file__).resolve().parent
    literal = re.compile(r'raise ValueError\(\s*"([^"\\]*)"\s*\)')
    messages: set[str] = set()
    try:
        for module_path in package_root.rglob("*.py"):
            messages.update(literal.findall(module_path.read_text(encoding="utf-8")))
    except OSError:
        return frozenset()
    return frozenset(messages)


def _owned_error_origin(error: Exception) -> str:
    """Return "module.py:line" for the deepest frame inside this application.

    A dynamic ValueError (an enum coercion, a stdlib call) carries a message we must
    not log. Its raise site is our own code metadata and identifies the invariant
    exactly, so it is safe to disclose and is what the message allowlist cannot cover.
    """
    app_root = Path(__file__).resolve().parents[2]
    origin = ""
    for frame in traceback.extract_tb(error.__traceback__):
        try:
            path = Path(frame.filename).resolve()
        except OSError:
            continue
        if app_root in path.parents and "site-packages" not in str(path):
            origin = f"{path.name}:{frame.lineno}"
    return origin


def _owned_error_message(error: Exception) -> str:
    """Return the message only when it is one of our own literals."""
    message = str(error)
    return message if message in _owned_value_error_messages() else ""


# Pydantic's own machine-readable ``errors()[i]["type"]``, grouped into the same small
# stable category set the business-rule checks below already return. Consulted only as
# a fallback, after every specific PLANNER_* message pattern has had a chance to match,
# so a named business-rule code is never coarsened just because this set also covers it.
_STRUCTURAL_MISSING_TYPES = frozenset({"missing"})
_STRUCTURAL_TYPE_ERROR_TYPES = frozenset(
    {
        "int_parsing",
        "int_type",
        "float_parsing",
        "float_type",
        "string_type",
        "bool_parsing",
        "bool_type",
        "list_type",
        "dict_type",
        "enum",
        "literal_error",
        "uuid_parsing",
        "date_parsing",
        "datetime_parsing",
        "json_type",
    }
)
_STRUCTURAL_CONSTRAINT_TYPES = frozenset(
    {
        "string_too_long",
        "string_too_short",
        "too_long",
        "too_short",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "string_pattern_mismatch",
        "multiple_of",
    }
)
# A raise site that names its own invariant (e.g. ``raise ValueError("PLANNER_SLOT_
# SUBJECT_MISMATCH")``) already IS a reason code; recognizing the shape generalizes to
# every such constant without hardcoding each one here, current or future.
_CONSTANT_SHAPED_MESSAGE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")


def _structural_error_bucket(error: Exception) -> str | None:
    """Classify by Pydantic's/the parser's own typed error, not a free-text guess.

    Returns one of ``"parse_failure"``/``"missing"``/``"constraint"``/``"type"``, or
    None when the error carries no machine-readable type this module knows how to
    bucket. Generic across every schema-v2 boundary (planner, generator, ...): each
    caller maps these buckets to its own prefixed reason codes, so the classification
    logic — and the Pydantic error-type sets it reads — is written once and reused,
    never re-derived per boundary.
    """
    if isinstance(error, json.JSONDecodeError):
        return "parse_failure"
    if not isinstance(error, ValidationError):
        return None
    errors = error.errors(include_url=False)
    if not errors:
        return None
    # One response can fail several fields at once; the first is the same one
    # ``field_paths``/``error_types`` already lead with, so the reason code and the
    # diagnostic stay describing the same failure.
    error_type = str(errors[0].get("type") or "")
    if error_type in _STRUCTURAL_MISSING_TYPES:
        return "missing"
    if error_type in _STRUCTURAL_CONSTRAINT_TYPES:
        return "constraint"
    if error_type in _STRUCTURAL_TYPE_ERROR_TYPES:
        return "type"
    return None


_PLANNER_STRUCTURAL_REASON_CODES = {
    "parse_failure": "PLANNER_PARSE_FAILURE",
    "missing": "PLANNER_MISSING_FIELD",
    "constraint": "PLANNER_SCHEMA_CONSTRAINT_INVALID",
    "type": "PLANNER_SCHEMA_TYPE_INVALID",
}


def _structural_reason_from_pydantic_type(error: Exception) -> str | None:
    """Map the generic structural-error bucket to a planner-prefixed reason code."""
    bucket = _structural_error_bucket(error)
    return _PLANNER_STRUCTURAL_REASON_CODES.get(bucket) if bucket else None


# Bound metadata Pydantic attaches to a constraint failure — never the value itself.
_SAFE_CONSTRAINT_CTX_KEYS = (
    "max_length",
    "min_length",
    "le",
    "ge",
    "lt",
    "gt",
    "max_items",
    "min_items",
    "multiple_of",
)


def _constraint_hint(error: Exception) -> str:
    """A safe, bounded ``bound=N,actual_length=M`` summary for a repair attempt.

    Built only from Pydantic's own ``ctx`` (which carries the rule, never the value
    that broke it) plus a length computed from ``input`` without ever retaining that
    input. Empty when the error is not a constraint-shaped ValidationError.
    """
    if not isinstance(error, ValidationError):
        return ""
    errors = error.errors(include_url=False)
    if not errors:
        return ""
    first = errors[0]
    ctx = first.get("ctx") or {}
    parts = [
        f"{key}={ctx[key]}"
        for key in _SAFE_CONSTRAINT_CTX_KEYS
        if key in ctx and isinstance(ctx[key], (int, float))
    ]
    if not parts:
        return ""
    actual_length = ctx.get("actual_length")
    if not isinstance(actual_length, (int, float)):
        raw_input = first.get("input")
        if isinstance(raw_input, (str, list, tuple, set, dict)):
            actual_length = len(raw_input)
    if isinstance(actual_length, (int, float)):
        parts.append(f"actual_length={actual_length}")
    return ",".join(parts)


def _planner_validation_reason(
    error: Exception,
    *,
    raw: str,
    finish_reason: str | None = None,
) -> tuple[str, int | None]:
    """Return a stable validation reason without retaining planner content."""
    # A truncated response is a token-budget/provider outcome, not a schema defect —
    # the model may have been about to produce a perfectly valid blueprint. Checked
    # first, and only for a parse failure, since a truncated response's own defect IS
    # an incomplete JSON parse; a validation error on an otherwise complete, parseable
    # response is a real schema/content issue whatever the finish reason was.
    if isinstance(error, json.JSONDecodeError) and finish_reason is not None:
        if normalize_completion_outcome(finish_reason) == "output_token_exhausted":
            return "PLANNER_OUTPUT_TRUNCATED", None
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
    # Contract invariants that previously collapsed into the catch-all, which made a
    # production planner failure impossible to identify from logs alone.
    if "topic_evidence_ungrounded" in message:
        return "PLANNER_TOPIC_EVIDENCE_UNGROUNDED", actual_slot_count
    if "topic_evidence_duplicate" in message:
        return "PLANNER_TOPIC_EVIDENCE_DUPLICATE", actual_slot_count
    if "intentionally distinct" in message:
        return "PLANNER_SLOTS_NOT_DISTINCT", actual_slot_count
    if "planner_family" in message:
        return "PLANNER_FAMILY_MISSING", actual_slot_count
    if "unsupported" in message and "subject" in message:
        return "PLANNER_UNSUPPORTED_SUBJECT", actual_slot_count
    if "exam ids must be unique" in message:
        return "PLANNER_DUPLICATE_EXAM_ID", actual_slot_count
    if "variation constraints must be unique" in message:
        return "PLANNER_DUPLICATE_VARIATION_CONSTRAINT", actual_slot_count
    if "schema-v1 blueprint cannot contain" in message:
        return "PLANNER_SCHEMA_VERSION_MISMATCH", actual_slot_count
    if "compatibility buckets" in message:
        return "PLANNER_BUCKETS_MISSING", actual_slot_count
    if "bucket ids must be unique" in message:
        return "PLANNER_DUPLICATE_BUCKET", actual_slot_count
    if _CONSTANT_SHAPED_MESSAGE.match(str(error)):
        return str(error), actual_slot_count
    structural_reason = _structural_reason_from_pydantic_type(error)
    if structural_reason is not None:
        return structural_reason, actual_slot_count
    return "PLANNER_SCHEMA_INVALID", actual_slot_count


def _planner_validation_diagnostic(
    error: Exception,
    *,
    raw: str,
    attempt: int,
    phase: str,
    duration_ms: int,
    finish_reason: str | None = None,
) -> PlannerValidationDiagnostic:
    """Extract allowlisted validation metadata without retaining model content."""
    reason_code, actual_slot_count = _planner_validation_reason(
        error, raw=raw, finish_reason=finish_reason
    )
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
        constraint=_constraint_hint(error),
        owned_message=_owned_error_message(error),
        owned_origin=_owned_error_origin(error),
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
    feedback = (
        f"reason={diagnostic.reason_code};fields={fields};"
        f"types={error_types};schema={diagnostic.schema_name}"
    )
    if diagnostic.constraint:
        # e.g. "max_length=160,actual_length=213" — the numeric bound and how far the
        # previous attempt was from it, so the one repair attempt has an actual target
        # instead of only the field name and error class.
        feedback += f";constraint={diagnostic.constraint}"
    return feedback


_DIFFICULTY_ORDER: tuple[Difficulty, ...] = (
    Difficulty.BASIC,
    Difficulty.INTERMEDIATE,
    Difficulty.ADVANCED,
)


def _deterministic_slot_difficulties(
    request: PracticeGenerationRequest,
) -> tuple[Difficulty, ...]:
    """Use a balanced generic fallback only for an explicitly mixed request."""
    if request.difficulty_distribution:
        # Explicit per-level counts are the most specific instruction there is, and
        # the request contract already guarantees they sum to accepted_count.
        return tuple(
            level
            for level in _DIFFICULTY_ORDER
            for _ in range(request.difficulty_distribution.get(level, 0))
        )
    requested_mixed = request.mixed_difficulty_requested or bool(
        _MIXED_DIFFICULTY_SIGNAL.search(request.original_query)
    )
    requested_difficulty = request.explicit_difficulty_requested or bool(
        _EXPLICIT_DIFFICULTY_SIGNAL.search(request.original_query)
    )
    if not requested_mixed or requested_difficulty:
        return (request.difficulty,) * request.accepted_count
    return tuple(
        _DIFFICULTY_ORDER[index % len(_DIFFICULTY_ORDER)]
        for index in range(request.accepted_count)
    )


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


def explicit_requested_count(query: str) -> int | None:
    """Return the count the student demonstrably wrote, or None when they wrote none.

    Separating "no count found" from the practice-type default matters: a default is
    not evidence of intent, so it must never be weighed against an interpretation of
    the student's own words. The patterns themselves are unchanged — no new
    language-specific parsing is added here.
    """
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
            rf"(?<!\w){re.escape(word)}(?!\w)"
            + _WORD_COUNT_FILLER
            + r"{0,3}\s+"
            + artifact,
            normalized,
        ):
            return count
    return None


def resolve_requested_count(query: str, practice_type: PracticeType) -> int:
    """Return the explicit count, or the practice-type default when none was written."""
    explicit = explicit_requested_count(query)
    if explicit is not None:
        return explicit
    return _DEFAULT_COUNTS[practice_type]


def required_fresh_evidence_count(query: str) -> int:
    """Return the bounded number of independently grounded fresh facts required."""
    practice_type = resolve_practice_type(query)
    return effective_practice_question_count(
        _validated_practice_count(resolve_requested_count(query, practice_type))
    )


def is_fresh_evidence_request_supported(query: str) -> bool:
    """Only a valid Practice count may enter a freshness-required retrieval path."""
    practice_type = resolve_practice_type(query)
    try:
        _validated_practice_count(resolve_requested_count(query, practice_type))
    except PracticeRequestCountError:
        return False
    return True


def validate_practice_requested_count(query: str) -> int:
    """Resolve, validate, and normalize the server-owned Practice count contract."""
    practice_type = resolve_practice_type(query)
    requested_count = _validated_practice_count(resolve_requested_count(query, practice_type))
    return effective_practice_question_count(requested_count)


# Closed set of factual families Practice may adopt from the classifier's
# `pattern_family_candidate`. The classifier is instructed to label every factual
# subject as `general` (classification_semantics.md), which the Practice Authority guard
# refuses, so the qualified factual family was unreachable. The candidate field is
# model-authored and unconstrained in its own schema, so Practice validates it against
# this allowlist before trusting it and otherwise keeps the existing fail-closed path.
# Scoped to the Practice routing boundary: the shared classifier contract is unchanged.
# This exact value set is also the only vocabulary the classifier prompt
# (prompts/classification_semantics.md, "Retrieval hints") is told it may use for
# `pattern_family_candidate` — keep both lists identical.
_FACTUAL_FAMILY_CANDIDATES: frozenset[str] = frozenset(
    {
        "HISTORY",
        "GEOGRAPHY",
        "POLITY",
        "ECONOMICS",
        "SCIENCE",
        "PHYSICS",
        "CHEMISTRY",
        "BIOLOGY",
        "COMPUTER_SCIENCE",
    }
)


def canonical_practice_subject(
    subject: object,
    pattern_family_candidate: object = None,
) -> str:
    """Resolve the Practice subject, adopting a factual family only when allowlisted.

    An explicit subject always wins: a candidate can never override math, english or any
    other classified subject. Only the `general` catch-all may be narrowed, and only to
    one of the nine approved factual families.
    """
    resolved = str(subject or "general").strip().casefold() or "general"
    if resolved != "general":
        return resolved
    candidate = str(pattern_family_candidate or "").strip().upper()
    if candidate in _FACTUAL_FAMILY_CANDIDATES:
        return candidate.casefold()
    return resolved


def resolve_practice_request(
    *,
    request_id: str,
    user_id: str,
    conversation_id: str,
    turn_id: str,
    query: str,
    subject: str,
    topic: str | None,
    topics: list[str] | None = None,
    requested_count: int | None = None,
    difficulty: str,
    language: str,
    exam_id: str | None,
    exam_stage: str | None,
    exam_profile_id: str | None = None,
    source_question_reference: str | None = None,
    freshness_requirement: PracticeFreshnessRequirement | None = None,
    fresh_evidence: FreshEvidenceBundle | None = None,
    request_interpreter: PracticeRequestInterpreter | None = None,
) -> PracticeGenerationRequest:
    """Resolve one Practice request into deterministic, planner-ready constraints.

    ``topics`` and ``requested_count`` carry constraints a caller already resolved
    and trusts. Supplying either bypasses Request Intelligence entirely, so a
    structured request never pays for interpretation it does not need; free text
    reaches the interpreter instead. Language, practice type and every bound stay
    deterministic either way — interpretation only supplies constraints the
    deterministic layer cannot see.
    """
    practice_type = resolve_practice_type(query)
    explicit_count = explicit_requested_count(query)
    normalized_difficulty = {
        "basic": Difficulty.BASIC,
        "intermediate": Difficulty.INTERMEDIATE,
        "advanced": Difficulty.ADVANCED,
    }.get(difficulty, Difficulty.INTERMEDIATE)
    resolved_language, language_source = resolve_practice_delivery_language(
        query,
        language,
    )
    freshness_requirement = freshness_requirement or PracticeFreshnessRequirement()
    intelligence = (
        None
        if topics is not None or requested_count is not None
        else interpret_practice_request(
            request_interpreter,
            request_id=request_id,
            query=query,
            subject=subject,
            language=resolved_language,
            exam_id=exam_id,
            exam_stage=exam_stage,
            explicit_count=explicit_count,
        )
    )
    requested_count = _resolve_practice_count(
        practice_type,
        structured_count=requested_count,
        explicit_count=explicit_count,
        intelligence=intelligence,
        freshness_required=freshness_requirement.requires_fresh_evidence,
    )
    accepted_count = effective_practice_question_count(requested_count)
    limitation = (
        PracticeLimitation(
            type="QUESTION_COUNT_LIMIT",
            requested_value=requested_count,
            effective_value=accepted_count,
            maximum_value=accepted_count,
            message_key="PRACTICE_MAX_QUESTIONS_LIMITED",
        )
        if requested_count != accepted_count
        else None
    )
    mixed_difficulty_requested = bool(_MIXED_DIFFICULTY_SIGNAL.search(query))
    explicit_difficulty_requested = bool(_EXPLICIT_DIFFICULTY_SIGNAL.search(query))
    difficulty_distribution: dict[Difficulty, int] | None = None
    if intelligence is not None:
        topics = _interpreted_topics(intelligence) or topics
        difficulty_spec = intelligence.difficulty
        if isinstance(difficulty_spec, MixedDifficulty):
            mixed_difficulty_requested = True
        elif isinstance(difficulty_spec, SingleDifficulty):
            explicit_difficulty_requested = True
            normalized_difficulty = Difficulty(difficulty_spec.level.lower())
        elif isinstance(difficulty_spec, CustomDifficulty):
            candidate = _interpreted_distribution(difficulty_spec)
            # Preserve an explicit distribution through the V1 count cap. The
            # total is deterministic and must still match the raw request; scale
            # proportionally only after that authority check.
            if sum(candidate.values()) == requested_count:
                difficulty_distribution = _scale_distribution(
                    candidate,
                    target_count=accepted_count,
                )
    trusted_constraints = _interpreted_constraints(intelligence)
    if trusted_constraints and accepted_count < len(trusted_constraints):
        raise PracticeRequestConstraintCountError(
            PracticeRequestConstraintCountError.reason_code
        )
    display_topic = topic or subject.replace("_", " ").title()
    title = f"{display_topic} {practice_type.value.replace('_', ' ').title()}"
    return PracticeGenerationRequest(
        request_id=request_id,
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        original_query=query,
        practice_type=practice_type,
        requested_count=requested_count,
        accepted_count=accepted_count,
        limitation=limitation,
        subject=subject,
        topic=topic,
        topics=topics,
        trusted_constraints=trusted_constraints,
        difficulty=normalized_difficulty,
        mixed_difficulty_requested=mixed_difficulty_requested,
        explicit_difficulty_requested=explicit_difficulty_requested,
        difficulty_distribution=difficulty_distribution,
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


def _resolve_practice_count(
    practice_type: PracticeType,
    *,
    structured_count: int | None,
    explicit_count: int | None,
    intelligence: PracticeRequestIntelligence | None,
    freshness_required: bool,
) -> int:
    """Resolve the single authoritative Practice count, in evidence order.

    A trusted structured count wins; then an interpretation of the student's own
    words; then a count the deterministic parser demonstrably found; then the
    practice-type default. Interpretation is skipped when fresh evidence has already
    been retrieved against the deterministic count, because generation must never be
    allowed to outrun the evidence that was actually gathered.
    """
    if structured_count is not None:
        return _validated_practice_count(structured_count)
    if intelligence is not None and not freshness_required:
        interpreted = interpreted_total_count(intelligence)
        if interpreted is not None:
            return _validated_practice_count(interpreted)
    if explicit_count is not None:
        return _validated_practice_count(explicit_count)
    return _validated_practice_count(_DEFAULT_COUNTS[practice_type])


def _validated_practice_count(count: int) -> int:
    if not 1 <= count <= MAX_REQUESTED_PRACTICE_QUESTIONS:
        raise PracticeRequestCountError(PracticeRequestCountError.reason_code)
    return count


def _interpreted_topics(
    intelligence: PracticeRequestIntelligence,
) -> list[str] | None:
    """Return the explicitly requested topic names, or None for a broad request."""
    names = [topic.normalized_name for topic in intelligence.topics]
    return names or None


def _interpreted_constraints(
    intelligence: PracticeRequestIntelligence | None,
) -> tuple[RequestedPracticeConstraint, ...]:
    """Keep only accepted explicit subject identities, never provider internals."""
    if intelligence is None:
        return ()
    return tuple(
        RequestedPracticeConstraint(
            subject_id=topic.subject_id,
            topic_id=_canonical_fallback_topic(topic.normalized_name),
            source_text=topic.source_text,
        )
        for topic in intelligence.topics
        if topic.subject_id is not None
    )


def _interpreted_distribution(
    difficulty: CustomDifficulty,
) -> dict[Difficulty, int]:
    """Return validated per-level counts as the planner's difficulty vocabulary."""
    distribution = difficulty.distribution
    return {
        Difficulty.BASIC: distribution.basic,
        Difficulty.INTERMEDIATE: distribution.intermediate,
        Difficulty.ADVANCED: distribution.advanced,
    }


def _scale_distribution(
    distribution: dict[Difficulty, int],
    *,
    target_count: int,
) -> dict[Difficulty, int]:
    """Proportionally fit an explicit distribution into the effective count.

    Integer quotas use the largest-remainder method with the stable canonical
    difficulty order as the sole tie-breaker. No semantic interpretation or model
    call is introduced by the count cap.
    """
    source_total = sum(distribution.values())
    if source_total == target_count:
        return distribution

    quotas: dict[Difficulty, int] = {}
    remainders: list[tuple[int, int, Difficulty]] = []
    for index, difficulty in enumerate(_DIFFICULTY_ORDER):
        quota, remainder = divmod(distribution.get(difficulty, 0) * target_count, source_total)
        quotas[difficulty] = quota
        remainders.append((remainder, index, difficulty))

    for _, _, difficulty in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : target_count - sum(quotas.values())
    ]:
        quotas[difficulty] += 1
    return quotas


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


def practice_generator_route_subject(subject: str, difficulty: Difficulty | str) -> str:
    """Project a Practice slot onto its scoped generator route subject.

    ``practice_math`` is an internal route distinction, not a student-facing
    subject. It confines the qualified Terra promotion to Practice Math's
    default/intermediate authoring path while preserving the shared Math routes
    used by Doubt Solver, basic Practice, and advanced Practice.
    """
    normalized = normalize_practice_subject(subject) or "general"
    difficulty_value = difficulty.value if isinstance(difficulty, Difficulty) else difficulty
    if normalized == "math" and difficulty_value in {"default", "intermediate"}:
        return "practice_math"
    # All non-promoted subjects stay on the existing resolver path. In particular,
    # Science/Polity/etc. must reach the resolver's factual-family aliases rather
    # than silently collapsing to general.
    return normalized


def _generator_route_hint(subject: str, difficulty: Difficulty) -> str:
    route_subject = practice_generator_route_subject(subject, difficulty)
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


def _query_word_set(text: str) -> set[str]:
    """Unicode-safe word set for the student's own text, script-independent."""
    words: set[str] = set()
    current: list[str] = []
    for character in text.casefold():
        if character.isalnum() or unicodedata.category(character).startswith("M"):
            current.append(character)
        elif current:
            words.add("".join(current))
            current = []
    if current:
        words.add("".join(current))
    return words


def _topics_grounded_in_query(topic_ids: set[str], query: str) -> bool:
    """True when every one of these topics is spelled out in the student's own words.

    No topic list, no translation, and no splitting on "and" — a single named concept
    such as "Profit and Loss" is matched whole because every one of its words must be
    present. Works in any script.
    """
    if not topic_ids:
        return False
    haystack = _query_word_set(query)
    return all(
        set(part for part in topic_id.split("_") if part).issubset(haystack)
        for topic_id in topic_ids
    )


def _normalized_for_grounding(text: str) -> str:
    """Casefold and collapse whitespace without stripping any script."""
    return " ".join(text.casefold().split())


def _grounded_topic_ids(
    blueprint: PracticeBlueprint, request: PracticeGenerationRequest
) -> set[str]:
    """Return the planned topics the student demonstrably asked for.

    The planner supplies the semantic normalization; this verifies only the source.
    A span is accepted when it literally occurs in the original query, so a topic the
    planner invented has nothing to stand on and is excluded. The span is compared
    whole, which is what keeps "Profit and Loss" from being read as two topics.
    """
    evidence = blueprint.requested_topic_evidence
    if not evidence:
        return set()
    haystack = _normalized_for_grounding(request.original_query)
    planned = {slot.topic_id for slot in blueprint.slots}
    grounded: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item.source_text.casefold(), item.topic_id)
        if key in seen:
            raise ValueError("PLANNER_TOPIC_EVIDENCE_DUPLICATE")
        seen.add(key)
        normalized_span = _normalized_for_grounding(item.source_text)
        if normalized_span not in haystack:
            # Content-free shape of the mismatch: lengths and two booleans only. This
            # distinguishes a planner that did not return a verbatim span from a
            # normalizer that rejects one it should accept, without logging the span
            # or the student's words.
            log_event(
                "planner_topic_evidence_ungrounded",
                component="practice.planning",
                stage="plan",
                status="failed",
                details={
                    "sourceEmpty": not item.source_text.strip(),
                    "sourceLength": len(item.source_text),
                    "groundingTextLength": len(request.original_query),
                    # What providers.py hands the planner. Equal to groundingTextLength
                    # means planner and validator saw the same text; different means they
                    # did not, which is a plumbing defect rather than a planner defect.
                    "constraintsLength": len(request.original_query[:1000]),
                    "exactSubstring": item.source_text in request.original_query,
                    "normalizedSubstring": normalized_span in haystack,
                },
            )
            raise ValueError("PLANNER_TOPIC_EVIDENCE_UNGROUNDED")
        if item.topic_id in planned:
            grounded.add(item.topic_id)
    return grounded


def _trusted_topic_constraints(request: PracticeGenerationRequest) -> tuple[str, ...]:
    """Explicit topic constraints that did not come from the intelligence planner.

    Only a structured topic list supplied with the request qualifies. The classifier's
    ``topic`` is deliberately excluded: it is one broad label, and on this path the
    student's actual composition lives only inside the planner response. Treating the
    broad label as a constraint is what silently widened a specific request into a
    subject-wide one.
    """
    if request.trusted_constraints:
        return tuple(constraint.topic_id for constraint in request.trusted_constraints)
    return tuple(value for value in (request.topics or ()) if value.strip())


def _requested_topic_ids(request: PracticeGenerationRequest) -> tuple[str, ...]:
    if request.trusted_constraints:
        return tuple(constraint.topic_id for constraint in request.trusted_constraints)
    if request.topics:
        # An explicitly requested topic set is already separated, so it is used as
        # given rather than re-split out of a single delimited label.
        values = [value.strip() for value in request.topics if value.strip()]
    else:
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


def _trusted_subject_by_topic(request: PracticeGenerationRequest) -> dict[str, str]:
    return {
        constraint.topic_id: constraint.subject_id
        for constraint in request.trusted_constraints
    }


def trusted_constraint_references(
    request: PracticeGenerationRequest,
) -> tuple[tuple[str, RequestedPracticeConstraint], ...]:
    """Return request-scoped, deterministic identities for trusted constraints."""
    references = tuple(
        (f"tc-{index:03d}", constraint)
        for index, constraint in enumerate(request.trusted_constraints, start=1)
    )
    identities = {
        (constraint.subject_id, constraint.topic_id)
        for _reference, constraint in references
    }
    if len(identities) != len(references):
        raise ValueError("PRACTICE_TRUSTED_CONSTRAINT_DUPLICATE")
    return references


def _validate_trusted_slot_composition(
    blueprint: PracticeBlueprint,
    request: PracticeGenerationRequest,
) -> None:
    references = dict(trusted_constraint_references(request))
    referenced: set[str] = set()
    for slot in blueprint.slots:
        reference = slot.constraint_ref
        constraint = references.get(reference or "")
        if constraint is None:
            raise ValueError("PLANNER_SLOT_CONSTRAINT_REF_INVALID")
        referenced.add(reference)
        if slot.subject_id != constraint.subject_id:
            raise ValueError("PLANNER_SLOT_SUBJECT_MISMATCH")
        if slot.topic_id != constraint.topic_id:
            raise ValueError("PLANNER_SLOT_TOPIC_MISMATCH")
    if request.accepted_count >= len(references) and referenced != set(references):
        raise ValueError("PLANNER_SLOT_CONSTRAINT_COVERAGE_INVALID")


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
    if blueprint.schema_version == "2":
        # Route hints are server-owned execution metadata. Recompute them from the
        # immutable slot subject/difficulty rather than trusting a planner response
        # or a persisted blueprint created before a route-only promotion.
        canonical_slots = [
            slot.model_copy(
                update={
                    "generator_route_hint": _generator_route_hint(
                        slot.subject_id,
                        slot.difficulty,
                    )
                }
            )
            for slot in blueprint.slots
        ]
        if canonical_slots != blueprint.slots:
            payload = blueprint.model_dump(mode="json")
            payload["slots"] = [slot.model_dump(mode="json") for slot in canonical_slots]
            blueprint = PracticeBlueprint.model_validate(payload)
    if any(bucket.question_type is not QuestionType.MCQ for bucket in blueprint.buckets):
        raise ValueError("PLAYER_UNSUPPORTED_QUESTION_TYPE")
    if blueprint.schema_version == "2":
        requested_exam = (
            request.exam_id.strip().upper().replace("-", "_").replace(" ", "_")
            if request.exam_id
            else None
        )
        requested_subject = request.subject.casefold().replace("-", "_").replace(" ", "_")
        trusted_constraints = bool(request.trusted_constraints)
        trusted_subjects = _trusted_subject_by_topic(request)
        requested_topics = set(_requested_topic_ids(request))
        planned_topics = {slot.topic_id for slot in blueprint.slots}
        # Feasibility-aware coverage, not weaker coverage. One slot carries one
        # topic, so a request for fewer questions than topics cannot cover them
        # all; demanding it rejected every such blueprint outright. Above the
        # boundary the original requirement is unchanged; below it the plan must
        # still draw only from the requested topics, so no topic is ever invented.
        if trusted_constraints:
            _validate_trusted_slot_composition(blueprint, request)
        elif request.accepted_count >= len(requested_topics):
            # The classifier may answer one broad label for a request that named
            # several specific topics; requiring that label as a slot topic would
            # erase the student's composition. Grounded evidence is what replaces it:
            # when every planned topic is tied to words the student actually wrote,
            # that composition is authoritative and the broad label is context only.
            # Grounding only decides this when the requested topics are not already
            # covered. Evaluating it first made an invalid evidence span fatal for
            # plans whose coverage was complete, where its result is never read.
            if not requested_topics.issubset(planned_topics):
                grounded = _grounded_topic_ids(blueprint, request)
                if not (grounded and grounded == planned_topics):
                    raise ValueError("PLANNER_SLOT_TOPIC_COVERAGE_INVALID")
        elif not planned_topics.issubset(requested_topics):
            raise ValueError("PLANNER_SLOT_TOPIC_COVERAGE_INVALID")
        for slot in blueprint.slots:
            if slot.question_type is not QuestionType.MCQ:
                raise ValueError("PLAYER_UNSUPPORTED_QUESTION_TYPE")
            if (
                not trusted_constraints
                and slot.topic_id in trusted_subjects
                and slot.subject_id != trusted_subjects[slot.topic_id]
            ):
                raise ValueError("PLANNER_SLOT_SUBJECT_MISMATCH")
            if (
                not trusted_constraints
                and not trusted_subjects
                and requested_subject not in {"general", "other"}
                and slot.subject_id != requested_subject
            ):
                raise ValueError("PLANNER_SLOT_SUBJECT_MISMATCH")
            if requested_exam and slot.exam_ids != [requested_exam]:
                raise ValueError("PLANNER_SLOT_EXAM_MISMATCH")
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
    trusted_references = trusted_constraint_references(request)
    topics = _requested_topic_ids(request)
    trusted_subjects = _trusted_subject_by_topic(request)
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
                constraint_ref=(
                    trusted_references[(index - 1) % len(trusted_references)][0]
                    if trusted_references
                    else None
                ),
                subject_id=(
                    trusted_references[(index - 1) % len(trusted_references)][1].subject_id
                    if trusted_references
                    else trusted_subjects.get(
                        topics[(index - 1) % len(topics)], request.subject
                    )
                ),
                topic_id=(
                    trusted_references[(index - 1) % len(trusted_references)][1].topic_id
                    if trusted_references
                    else topics[(index - 1) % len(topics)]
                ),
                category_id=(
                    trusted_references[(index - 1) % len(trusted_references)][1].topic_id
                    if trusted_references
                    else topics[(index - 1) % len(topics)]
                ),
                difficulty=slot_difficulties[index - 1],
                complexity=_slot_complexity(slot_difficulties[index - 1]),
                exam_ids=exam_ids,
                question_type=question_type,
                target_skill=f"solve_{topics[(index - 1) % len(topics)]}",
                variation_hint=f"variant_{index:03d}",
                generator_route_hint=_generator_route_hint(
                    (
                        trusted_references[(index - 1) % len(trusted_references)][1].subject_id
                        if trusted_references
                        else request.subject
                    ),
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
    if request.trusted_constraints:
        # Source evidence was validated before this immutable request was created.
        # Planner-provided evidence is therefore neither accepted nor needed here.
        payload = dict(payload)
        payload["requestedTopicEvidence"] = []
    # Server-owned slot fields are not planner output: the route hint is recomputed by
    # apply_system_bucket_policy and the exam is the one the request supplied.
    exam_ids = (
        [request.exam_id.strip().upper().replace("-", "_").replace(" ", "_")]
        if request.exam_id
        else []
    )
    if isinstance(payload.get("slots"), list):
        payload["slots"] = [
            {
                "exam_ids": exam_ids,
                "generator_route_hint": (
                    f"{slot.get('subject_id')}.generator.{slot.get('difficulty')}"
                ),
                **slot,
            }
            if isinstance(slot, dict)
            else slot
            for slot in payload["slots"]
        ]
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


# Above this count a request is worth one planning inference: the deterministic plan
# distributes slots mechanically, which stops representing what a large multi-topic or
# multilingual request actually asked for. Exclusive bound — 20 stays deterministic.
INTELLIGENCE_PLANNING_COUNT_THRESHOLD = 20

PlanningMode = Literal["DETERMINISTIC", "INTELLIGENCE"]


def select_planning_mode(
    request: PracticeGenerationRequest,
) -> tuple[PlanningMode, str]:
    """Return the single authoritative planning route for one Practice request.

    The count is the one already resolved by ``resolve_practice_request``; nothing is
    re-parsed here. Below the threshold the pre-existing routing rules are preserved
    exactly, so cheap Practice keeps costing no planning inference.
    """
    # Precedence is explicit so the escalation below cannot capture a path that must
    # stay deterministic. Each rule is evaluated in order and the first one wins.
    # 1. Above the threshold, planning is already worth an inference.
    if request.accepted_count > INTELLIGENCE_PLANNING_COUNT_THRESHOLD:
        return "INTELLIGENCE", "COUNT_OVER_THRESHOLD"
    # 2. Forced-deterministic cases keep their existing route untouched.
    if request.accepted_count <= 2:
        return "DETERMINISTIC", "EXISTING_DETERMINISTIC_RULE"
    # 3. A structured topic list is trusted composition; nothing needs interpreting.
    # A similar-question request still does: only the planner can derive the
    # reference question's pattern.
    if (
        _trusted_topic_constraints(request)
        and request.practice_type is not PracticeType.SIMILAR_QUESTION
    ):
        return "DETERMINISTIC", "TRUSTED_TOPIC_CONSTRAINTS"
    if _quick_practice_is_deterministic(request):
        # 4. The deterministic plan can only draw topics from the classifier label.
        # That is safe when the student's own words contain it, and silently widens
        # the request when they do not — so an ungrounded label escalates to the
        # planner, where C1 fails closed if semantics still cannot be preserved.
        try:
            derived = set(_requested_topic_ids(request))
        except ValueError:
            # A classifier label that will not canonicalize cannot ground anything,
            # and routing must stay total: escalate rather than raise here, so the
            # existing deterministic validation still owns the malformed-topic error.
            derived = set()
        if derived and _topics_grounded_in_query(derived, request.original_query):
            return "DETERMINISTIC", "EXISTING_DETERMINISTIC_RULE"
        return "INTELLIGENCE", "CLASSIFIER_TOPIC_UNGROUNDED"
    # 5. Every remaining practice type keeps the existing planner route.
    return "INTELLIGENCE", "EXISTING_PRACTICE_TYPE_RULE"


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
        mode, reason = select_planning_mode(request)
        log_event(
            "practice_planning_route_selected",
            component="practice.planning",
            stage="route",
            status="selected",
            details={
                "planningMode": mode,
                "planningReason": reason,
                "requestedCount": request.accepted_count,
                "practiceType": request.practice_type.value,
            },
        )
        if mode == "DETERMINISTIC":
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
                    # Optional: only RoutedPlannerProvider sets this (see its
                    # last_finish_reason docstring); a Protocol implementer that does
                    # not is unaffected and this is simply None for it, exactly as
                    # before this attribute existed.
                    finish_reason=getattr(self._planner, "last_finish_reason", None),
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
                if not _trusted_topic_constraints(request):
                    # A planner failure may cost availability; it must never broaden
                    # what the student asked for. Without trustworthy explicit topics
                    # the only honest deterministic plan is the classifier's broad
                    # label, which would admit subject-wide reuse the student never
                    # requested. Fail closed instead.
                    raise BlueprintPlanningError(
                        "PRACTICE_PLANNER_SEMANTIC_FALLBACK_UNSAFE",
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
        if not _trusted_topic_constraints(request):
            raise BlueprintPlanningError(
                "PRACTICE_PLANNER_SEMANTIC_FALLBACK_UNSAFE",
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
