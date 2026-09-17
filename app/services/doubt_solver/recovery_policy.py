"""Typed failure diagnosis and deterministic recovery decisions for the Doubt Solver.

A failure is described by what failed (kind), where (source), and why (reason code);
kind and source are always derived here from the reason code, never taken from a model.
One pure function per node maps that diagnosis plus request-local budget use to exactly
one recovery action. No model is named: retries re-resolve the same route, and provider
fallback stays inside the model executor.

With recovery disabled the decision is only recorded (``RECOVERY_DECISION_SHADOW``) and
every request ends as it always did. With recovery enabled the controller carries the
action out and logs ``RECOVERY_DECISION``; each budget below is spent at most once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from observability import log_event, snapshot_llm_usage_records
from services.llm.orchestration.errors import (
    LlmConfigLoadError,
    LlmConfigValidationError,
    LlmRouteNotFoundError,
    LlmRouteResolutionError,
    ModelConfigResolutionError,
    ModelExecutionConfigError,
    PromptResolverError,
)

logger = logging.getLogger(__name__)

DIAGNOSIS_SCHEMA_VERSION = 1

FailureKind = Literal["NONE", "SEMANTIC", "STRUCTURAL", "INFRASTRUCTURE", "UNKNOWN"]
FailureSource = Literal["NONE", "QUESTION", "CANDIDATE", "VERIFIER", "GENERATOR", "UNKNOWN"]
TechnicalReasonCode = Literal[
    "OUTPUT_TOKEN_EXHAUSTED",
    "TIMEOUT",
    "PROVIDER_FAILURE",
    "CONFIGURATION_FAILURE",
    "TECHNICAL_UNCLASSIFIED",
]
VerificationReasonCode = Literal[
    "NONE",
    "WRONG_FINAL_ANSWER",
    "CANDIDATE_CONFLICT",
    "INCOMPLETE_QUESTION",
    "MULTIPLE_DEFENSIBLE_ANSWERS",
    "VERIFIER_LOW_CONFIDENCE",
    "VERIFIER_SELF_CONTRADICTION",
    # A verdict arrived but its semantic cause is missing or invalid: fail safe, never
    # retry, never regenerate. Distinct from TECHNICAL_UNCLASSIFIED below.
    "SEMANTIC_UNCLASSIFIED",
    "PARSE_FAILURE",
    "SCHEMA_FAILURE",
    "INPUT_TOO_LARGE",
    "OUTPUT_TOKEN_EXHAUSTED",
    "TIMEOUT",
    "PROVIDER_FAILURE",
    "CONFIGURATION_FAILURE",
    "TECHNICAL_UNCLASSIFIED",
]
GenerationReasonCode = Literal[
    "QUALITY_FAILED",
    "TRUNCATED_QUALITY_FAILED",
    "OUTPUT_TOKEN_EXHAUSTED",
    "TIMEOUT",
    "PROVIDER_FAILURE",
    "CONFIGURATION_FAILURE",
    "TECHNICAL_UNCLASSIFIED",
]
RecoveryAction = Literal[
    "COMPLETE",
    "RETRY_SAME_NODE",
    "REPAIR_CANDIDATE",
    "REGENERATE_CANDIDATE",
    "ASK_CLARIFICATION",
    "FAIL_TEMPORARY",
    "FAIL_SAFE",
]

SERVICE_TEMPORARILY_UNAVAILABLE = "SERVICE_TEMPORARILY_UNAVAILABLE"
# What a regenerated candidate is told. It learns only that an independent check rejected
# the previous attempt: never the verifier's own answer, its reasoning, or a correction,
# because the point is a second independent derivation, not copying the verifier.
CANDIDATE_RECOVERY_INSTRUCTION = (
    "An independent check rejected your previous answer to this question. Solve it again "
    "from the beginning and derive the result independently. Do not assume the previous "
    "answer was right, and do not defend or restate it."
)
# Sources the diagnostic call may attribute a semantic failure to. It can never claim a
# technical source, and it is ignored entirely for technical failures.
_DIAGNOSABLE_SOURCES = frozenset({"QUESTION", "CANDIDATE", "VERIFIER"})
# Only these diagnosed causes may spend the candidate slot: Phase 0b proved exactly these
# two, and nothing is generalised beyond that evidence.
CANDIDATE_RECOVERY_REASONS = frozenset({"WRONG_FINAL_ANSWER", "CANDIDATE_CONFLICT"})
# A question fault the student can act on; the clarification asks for what is missing.
CLARIFICATION_REASONS = frozenset({"INCOMPLETE_QUESTION", "MULTIPLE_DEFENSIBLE_ANSWERS"})
# Repeating the identical verifier call cannot clear these: the request is deterministically
# rejected, misconfigured, or already at the route's token ceiling.
_UNRETRYABLE_TECHNICAL_REASONS = frozenset(
    {"CONFIGURATION_FAILURE", "OUTPUT_TOKEN_EXHAUSTED", "INPUT_TOO_LARGE"}
)

_VERIFICATION_DIAGNOSIS: dict[str, tuple[FailureKind, FailureSource]] = {
    "NONE": ("NONE", "NONE"),
    "WRONG_FINAL_ANSWER": ("SEMANTIC", "CANDIDATE"),
    "CANDIDATE_CONFLICT": ("SEMANTIC", "CANDIDATE"),
    "INCOMPLETE_QUESTION": ("SEMANTIC", "QUESTION"),
    "MULTIPLE_DEFENSIBLE_ANSWERS": ("SEMANTIC", "QUESTION"),
    "VERIFIER_LOW_CONFIDENCE": ("SEMANTIC", "VERIFIER"),
    "VERIFIER_SELF_CONTRADICTION": ("SEMANTIC", "VERIFIER"),
    "SEMANTIC_UNCLASSIFIED": ("SEMANTIC", "UNKNOWN"),
    "PARSE_FAILURE": ("STRUCTURAL", "VERIFIER"),
    "SCHEMA_FAILURE": ("STRUCTURAL", "VERIFIER"),
    # The route's own ceiling was reached: structural, and repeating the identical call
    # with the identical budget would hit it again.
    "OUTPUT_TOKEN_EXHAUSTED": ("STRUCTURAL", "VERIFIER"),
    # The composed input could not be brought under the query contract: deterministic,
    # so the identical call would be refused again.
    "INPUT_TOO_LARGE": ("STRUCTURAL", "VERIFIER"),
    "TIMEOUT": ("INFRASTRUCTURE", "VERIFIER"),
    "PROVIDER_FAILURE": ("INFRASTRUCTURE", "VERIFIER"),
    "CONFIGURATION_FAILURE": ("UNKNOWN", "VERIFIER"),
    "TECHNICAL_UNCLASSIFIED": ("UNKNOWN", "VERIFIER"),
}
_GENERATION_DIAGNOSIS: dict[str, tuple[FailureKind, FailureSource]] = {
    "QUALITY_FAILED": ("STRUCTURAL", "CANDIDATE"),
    "TRUNCATED_QUALITY_FAILED": ("STRUCTURAL", "CANDIDATE"),
    "OUTPUT_TOKEN_EXHAUSTED": ("STRUCTURAL", "GENERATOR"),
    "TIMEOUT": ("INFRASTRUCTURE", "GENERATOR"),
    "PROVIDER_FAILURE": ("INFRASTRUCTURE", "GENERATOR"),
    "CONFIGURATION_FAILURE": ("UNKNOWN", "GENERATOR"),
    "TECHNICAL_UNCLASSIFIED": ("UNKNOWN", "GENERATOR"),
}
# Deterministic misconfiguration: a retry cannot succeed.
_CONFIGURATION_ERRORS = (
    LlmConfigLoadError,
    LlmConfigValidationError,
    LlmRouteNotFoundError,
    LlmRouteResolutionError,
    ModelConfigResolutionError,
    ModelExecutionConfigError,
    PromptResolverError,
)
# Provider failure kinds (services/llm/providers/errors.py) by what a later attempt can do:
# transient ones may clear, misconfiguration cannot, and the rest (invalid_request,
# safety_blocked, output_token_exhausted) are not infrastructure.
_PROVIDER_FAILURE_KIND_CODES: dict[str, TechnicalReasonCode] = {
    "timeout": "TIMEOUT",
    "rate_limited": "PROVIDER_FAILURE",
    "insufficient_quota": "PROVIDER_FAILURE",
    "provider_unavailable": "PROVIDER_FAILURE",
    "empty_stream": "PROVIDER_FAILURE",
    "empty_answer": "PROVIDER_FAILURE",
    "output_token_exhausted": "OUTPUT_TOKEN_EXHAUSTED",
    "authentication_failed": "CONFIGURATION_FAILURE",
    "provider_not_configured": "CONFIGURATION_FAILURE",
    "model_not_configured": "CONFIGURATION_FAILURE",
    "model_not_found": "CONFIGURATION_FAILURE",
    "unsupported_parameter": "CONFIGURATION_FAILURE",
}
# Quality reasons that mean the draft ran out of room rather than being badly laid out.
_TRUNCATION_REASON_CODES = frozenset({"missing_final_answer", "math_unbalanced_inline"})


def verification_diagnosis(reason_code: str) -> tuple[FailureKind, FailureSource]:
    return _VERIFICATION_DIAGNOSIS.get(reason_code, ("UNKNOWN", "UNKNOWN"))


def generation_diagnosis(reason_code: str) -> tuple[FailureKind, FailureSource]:
    return _GENERATION_DIAGNOSIS.get(reason_code, ("UNKNOWN", "UNKNOWN"))


def technical_reason_code(exc: BaseException) -> TechnicalReasonCode:
    """Classify a failed call from its explicit exception chain.

    Only provider failure kinds that a later attempt can plausibly clear count as
    infrastructure. ``unknown_provider_error`` is what the executor stamps on any
    unexpected exception, including code bugs, so it defers to its cause. Implicit
    ``__context__`` is ignored: an unrelated error raised while handling a timeout is
    not a timeout.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _CONFIGURATION_ERRORS):
            return "CONFIGURATION_FAILURE"
        if isinstance(current, TimeoutError):
            return "TIMEOUT"
        failure_kind = getattr(current, "failure_kind", None)
        if isinstance(failure_kind, str) and failure_kind != "unknown_provider_error":
            return _PROVIDER_FAILURE_KIND_CODES.get(failure_kind, "TECHNICAL_UNCLASSIFIED")
        current = current.__cause__
    return "TECHNICAL_UNCLASSIFIED"


def quality_reason_code(
    reason_codes: list[str] | tuple[str, ...], *, continuation_used: bool
) -> GenerationReasonCode:
    if continuation_used or _TRUNCATION_REASON_CODES.intersection(reason_codes):
        return "TRUNCATED_QUALITY_FAILED"
    return "QUALITY_FAILED"


@dataclass(frozen=True)
class RecoveryBudgetUse:
    """What this request has already spent. Every limit is one, and independent.

    Four responsibilities, four separate limits: a continuation completing a truncated
    draft, a presentation rewrite fixing layout, the single candidate slot (a hard
    structural repair, a truncation regeneration, or a verifier-driven regeneration),
    and one verifier-local retry. No recovery may reset another, and provider fallback
    inside the executor spends none of them.
    """

    continuation_used: bool = False
    presentation_rewrite_used: bool = False
    candidate_recovery_used: bool = False
    verifier_technical_retry_used: bool = False

    @classmethod
    def from_request_usage(cls) -> RecoveryBudgetUse:
        """Read the generator-side spend from this request's recorded model calls.

        The executor suffixes an attempt it had to send to a fallback model
        ("repair_fallback"), and that is still the same attempt for budget purposes.
        The two verifier-side limits are never derived here: they belong to the
        controller's ledger, so nothing the generator or a provider does hands one back.
        """
        attempt_types = {
            record.attempt_type.removesuffix("_fallback") for record in snapshot_llm_usage_records()
        }
        return cls(
            continuation_used="continuation" in attempt_types,
            presentation_rewrite_used="rewrite" in attempt_types,
            candidate_recovery_used="repair" in attempt_types,
        )


class RecoveryBudget:
    """The request-local ledger: each slot can be taken once and never given back."""

    def __init__(self) -> None:
        self._candidate_recovery_used = False
        self._verifier_technical_retry_used = False

    def take_candidate_recovery(self) -> bool:
        """Claim the candidate slot before acting. False when it is already spent."""
        if self.snapshot().candidate_recovery_used:
            return False
        self._candidate_recovery_used = True
        return True

    def take_verifier_technical_retry(self) -> bool:
        if self._verifier_technical_retry_used:
            return False
        self._verifier_technical_retry_used = True
        return True

    def snapshot(self) -> RecoveryBudgetUse:
        """Generator spend comes from recorded calls; verifier spend from this ledger."""
        recorded = RecoveryBudgetUse.from_request_usage()
        return RecoveryBudgetUse(
            continuation_used=recorded.continuation_used,
            presentation_rewrite_used=recorded.presentation_rewrite_used,
            candidate_recovery_used=(
                recorded.candidate_recovery_used or self._candidate_recovery_used
            ),
            verifier_technical_retry_used=self._verifier_technical_retry_used,
        )


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    failure_kind: FailureKind
    failure_source: FailureSource
    reason_code: str
    terminal_code: str | None


def decide_verification_recovery(
    *,
    approved: bool,
    reason_code: str,
    budget: RecoveryBudgetUse,
    failure_source: str | None = None,
) -> RecoveryDecision:
    """Map one verdict to one action.

    ``failure_source`` comes from the separate diagnostic call when it ran. It may only
    redirect an already-failing verdict; approval belongs to the verifier alone, so an
    approved verdict completes whatever the diagnosis says about the question.
    """
    kind, source = verification_diagnosis(reason_code)
    if failure_source is not None and failure_source in _DIAGNOSABLE_SOURCES:
        source = failure_source  # type: ignore[assignment]

    def decision(action: RecoveryAction, terminal_code: str | None) -> RecoveryDecision:
        return RecoveryDecision(action, kind, source, reason_code, terminal_code)

    if approved:
        return decision("COMPLETE", None)
    if kind == "INFRASTRUCTURE":
        # The executor has already walked the configured provider fallback chain.
        return decision("FAIL_TEMPORARY", SERVICE_TEMPORARILY_UNAVAILABLE)
    if kind in ("STRUCTURAL", "UNKNOWN") and source == "VERIFIER":
        if (
            reason_code not in _UNRETRYABLE_TECHNICAL_REASONS
            and not budget.verifier_technical_retry_used
        ):
            return decision("RETRY_SAME_NODE", None)
        return decision("FAIL_TEMPORARY", "ANSWER_VERIFICATION_UNAVAILABLE")
    if source == "QUESTION" and reason_code in CLARIFICATION_REASONS:
        return decision("ASK_CLARIFICATION", "QUESTION_NEEDS_CLARIFICATION")
    if (
        source == "CANDIDATE"
        and reason_code in CANDIDATE_RECOVERY_REASONS
        and not budget.candidate_recovery_used
    ):
        return decision("REGENERATE_CANDIDATE", None)
    return decision("FAIL_SAFE", "ANSWER_VERIFICATION_FAILED")


def decide_generation_recovery(*, reason_code: str, budget: RecoveryBudgetUse) -> RecoveryDecision:
    kind, source = generation_diagnosis(reason_code)

    def decision(action: RecoveryAction, terminal_code: str | None) -> RecoveryDecision:
        return RecoveryDecision(action, kind, source, reason_code, terminal_code)

    if kind == "INFRASTRUCTURE":
        return decision("FAIL_TEMPORARY", SERVICE_TEMPORARILY_UNAVAILABLE)
    if source == "GENERATOR":
        # The approved budget has no generator retry: provider fallback stays inside
        # the executor, and any other technical generator failure is terminal.
        return decision("FAIL_TEMPORARY", "ANSWER_PROVIDER_FAILED")
    if source == "CANDIDATE" and not budget.candidate_recovery_used:
        return decision(
            "REGENERATE_CANDIDATE"
            if reason_code == "TRUNCATED_QUALITY_FAILED"
            else "REPAIR_CANDIDATE",
            None,
        )
    return decision("FAIL_SAFE", "ANSWER_QUALITY_FAILED")


def diagnosis_can_be_used(verification: Any) -> bool:
    """Whether a diagnosis of this verdict could change anything, so is worth paying for.

    A verifier timeout or parse failure keeps its own technical reason: the diagnostic
    call cannot know the provider was unreachable, and letting it relabel one would
    recommend regenerating candidates during an outage. Deterministic recomputation and
    the self-contradiction guard are this system's own findings, not opinions, so they
    are not reopened either.
    """
    if getattr(verification, "method", None) == "deterministic":
        return False
    reason_code = str(getattr(verification, "reason_code", ""))
    if reason_code == "VERIFIER_SELF_CONTRADICTION":
        return False
    return verification_diagnosis(reason_code)[0] == "SEMANTIC"


def _diagnosed(verification: Any, diagnosis: Any | None) -> bool:
    if diagnosis is None or not diagnosis.classified:
        return False
    return diagnosis_can_be_used(verification)


def recovery_reason_code(verification: Any, diagnosis: Any | None) -> str:
    """The reason the controller acts on: the diagnosis only when it may speak."""
    if _diagnosed(verification, diagnosis):
        return str(diagnosis.reason_code)
    return str(verification.reason_code)


def recovery_failure_source(verification: Any, diagnosis: Any | None) -> str | None:
    if _diagnosed(verification, diagnosis):
        return str(diagnosis.failure_source)
    return None


def log_recovery_decision(
    decision: RecoveryDecision,
    *,
    applied: bool,
    budget: RecoveryBudgetUse,
    verifier_status: str,
    attempt_number: int,
    node: Literal["verifier", "generator"],
    subject: str,
    difficulty: str,
    outcome: str,
) -> None:
    """Record a recovery the controller actually carried out. Codes and counts only."""
    log_event(
        "RECOVERY_DECISION",
        component="doubt_solver.recovery",
        stage="verify_correctness",
        status=outcome,
        details={
            "schema_version": DIAGNOSIS_SCHEMA_VERSION,
            "applied": applied,
            "verifier_status": verifier_status,
            "failure_kind": decision.failure_kind,
            "failure_source": decision.failure_source,
            "reason_code": decision.reason_code,
            "recovery_action": decision.action,
            "terminal_code": decision.terminal_code,
            "attempt_number": attempt_number,
            "node": node,
            # The role and its inputs, never a model name: the route is resolved from these.
            "subject": subject,
            "difficulty": difficulty,
            "continuation_used": budget.continuation_used,
            "presentation_rewrite_used": budget.presentation_rewrite_used,
            "candidate_recovery_used": budget.candidate_recovery_used,
            "verifier_technical_retry_used": budget.verifier_technical_retry_used,
        },
    )


def shadow_verification_recovery(
    verification: Any, *, actual_terminal_code: str, diagnosis: Any | None = None
) -> None:
    _shadow(
        "verification",
        lambda budget: decide_verification_recovery(
            approved=bool(verification.approved),
            reason_code=recovery_reason_code(verification, diagnosis),
            budget=budget,
            failure_source=recovery_failure_source(verification, diagnosis),
        ),
        actual_terminal_code,
    )


def shadow_generation_failure(exc: BaseException, *, actual_terminal_code: str) -> None:
    _shadow(
        "generation",
        lambda budget: decide_generation_recovery(
            reason_code=technical_reason_code(exc), budget=budget
        ),
        actual_terminal_code,
    )


def shadow_quality_recovery(quality: Any, *, actual_terminal_code: str) -> None:
    _shadow(
        "generation",
        lambda budget: decide_generation_recovery(
            reason_code=quality_reason_code(
                tuple(quality.reason_codes),
                continuation_used=budget.continuation_used,
            ),
            budget=budget,
        ),
        actual_terminal_code,
    )


def _shadow(
    node: Literal["generation", "verification"],
    decide: Callable[[RecoveryBudgetUse], RecoveryDecision],
    actual_terminal_code: str,
) -> None:
    """Record what the controller would do. Phase 0 never acts on it, and a failure
    here must never change how the request ends."""
    try:
        budget = RecoveryBudgetUse.from_request_usage()
        _log_shadow_decision(
            decide(budget),
            node=node,
            budget=budget,
            actual_terminal_code=actual_terminal_code,
        )
    except Exception:  # noqa: BLE001 - shadow telemetry only
        logger.warning("recovery_decision_shadow unavailable node=%s", node, exc_info=True)


def _log_shadow_decision(
    decision: RecoveryDecision,
    *,
    node: Literal["generation", "verification"],
    budget: RecoveryBudgetUse,
    actual_terminal_code: str,
) -> None:
    log_event(
        "RECOVERY_DECISION_SHADOW",
        component="doubt_solver.recovery",
        stage=node,
        status="shadow",
        details={
            "schema_version": DIAGNOSIS_SCHEMA_VERSION,
            "action": decision.action,
            "failure_kind": decision.failure_kind,
            "failure_source": decision.failure_source,
            "reason_code": decision.reason_code,
            "shadow_terminal_code": decision.terminal_code,
            "actual_terminal_code": actual_terminal_code,
            "continuation_used": budget.continuation_used,
            "presentation_rewrite_used": budget.presentation_rewrite_used,
            "candidate_recovery_used": budget.candidate_recovery_used,
            "verifier_technical_retry_used": budget.verifier_technical_retry_used,
        },
        level=logging.INFO,
    )
