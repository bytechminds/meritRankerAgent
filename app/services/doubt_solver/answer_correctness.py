"""Selective independent correctness verification for high-risk answers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from observability import log_event
from schemas.doubt_solver import CanonicalLanguage
from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_quality import _scalar_answer
from services.doubt_solver.recovery_policy import (
    FailureKind,
    FailureSource,
    VerificationReasonCode,
    technical_reason_code,
    verification_diagnosis,
)
from services.llm.orchestration.orchestrator import MAX_QUERY_CHARS, LlmOrchestrator
from services.llm.structured_output import (
    INTERNAL_FAILURE,
    PROVIDER_FAILURE,
    CanonicalScalarString,
    StructuredOutputDiagnostic,
    StructuredOutputError,
    parse_structured_output,
)

logger = logging.getLogger(__name__)

VerificationStatus = Literal["match", "mismatch", "ambiguous", "unavailable"]
_STRUCTURED_OUTPUT_REASON_CODES: dict[str, VerificationReasonCode] = {
    "parse": "PARSE_FAILURE",
    "schema": "SCHEMA_FAILURE",
    "contract": "SCHEMA_FAILURE",
    "internal": "TECHNICAL_UNCLASSIFIED",
}


class _VerifierOutput(BaseModel):
    status: Literal["MATCH", "MISMATCH", "AMBIGUOUS"]
    # A defensible answer is as often a bare number as it is text with a unit or an
    # option label, and both spellings carry the same verdict. Only this field opts
    # into that equivalence; every other field below stays strict.
    independent_answer: CanonicalScalarString = Field(min_length=1)
    single_defensible_answer: bool
    reason: str = Field(min_length=1)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


@dataclass(frozen=True)
class CorrectnessVerification:
    status: VerificationStatus
    independent_answer: str | None = None
    single_defensible_answer: bool = False
    reason: str = "verification_unavailable"
    method: Literal["deterministic", "model", "unavailable"] = "unavailable"
    # Set only by this module's runtime, never read from the model's output.
    reason_code: VerificationReasonCode = "TECHNICAL_UNCLASSIFIED"

    @property
    def approved(self) -> bool:
        return self.status == "match" and self.single_defensible_answer

    @property
    def failure_kind(self) -> FailureKind:
        return verification_diagnosis(self.reason_code)[0]

    @property
    def failure_source(self) -> FailureSource:
        return verification_diagnosis(self.reason_code)[1]


# The verifier's composed input has to satisfy the orchestrator's own query contract,
# which is smaller than the per-field caps this module applied before. Compaction is
# deterministic, uses no model, and never touches the answer shown to the student: it
# only shortens the copy sent for verification.
_VERIFIER_INPUT_TEMPLATE = (
    "[QUESTION]\n{question}\n[/QUESTION]\n[CANDIDATE_ANSWER]\n{candidate}\n[/CANDIDATE_ANSWER]"
)
_COMPACTION_MARKER = "\n[... working compacted for verification ...]\n"
# A solution states its result at the end, so the tail keeps the larger share; the head
# keeps enough of the setup for the verifier to see what was being solved.
_TAIL_SHARE = 0.6
# Floor that keeps the candidate meaningful when the question is very long.
_MIN_CANDIDATE_CHARS = 600


class VerifierInputTooLarge(RuntimeError):
    """The composed input cannot be brought under the orchestrator's query contract."""


def _head_and_tail(text: str, budget: int) -> str:
    """Keep the start and the end of `text` within `budget`, dropping the middle."""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    usable = budget - len(_COMPACTION_MARKER)
    if usable <= 0:
        return text[:budget]
    tail = int(usable * _TAIL_SHARE)
    head = usable - tail
    return text[:head] + _COMPACTION_MARKER + text[len(text) - tail :]


def build_verifier_input(query: str, candidate_answer: str) -> str:
    """Compose the verifier payload, guaranteed to fit the orchestrator's query limit.

    A short question and answer compose exactly as before. When they do not fit, the
    question is preserved first, and whatever the candidate may still spend is split
    between its opening working and its ending, where the final answer lives.
    """
    composed = _VERIFIER_INPUT_TEMPLATE.format(
        question=query[:4000], candidate=candidate_answer[:8000]
    )
    if len(composed) <= MAX_QUERY_CHARS:
        return composed
    framing = len(_VERIFIER_INPUT_TEMPLATE.format(question="", candidate=""))
    available = MAX_QUERY_CHARS - framing
    question = _head_and_tail(query, available - _MIN_CANDIDATE_CHARS)
    candidate = _head_and_tail(candidate_answer, available - len(question))
    return _VERIFIER_INPUT_TEMPLATE.format(question=question, candidate=candidate)


def requires_independent_correctness_verification(
    *,
    subject: str,
    difficulty: str,
    intent: str,
    requested_action: str | None = None,
    context_need: str | None = None,
) -> bool:
    """Decide whether the answer is checked before it reaches the student.

    `context_need` is this request's ContextNeedGate decision, or None when no gate ran
    (an image request, or a caller with no conversation). `default` difficulty means the
    classifier had insufficient evidence, not that it found the work simple — and a turn
    whose premises may live in an earlier turn is graded on what little it states. So a
    default solve is checked unless the gate certified the turn as standalone.
    """
    if requested_action in {"VERIFY_AND_CORRECT", "RESOLVE_FROM_SCRATCH"}:
        return True
    if intent in {"practice", "practice_question"}:
        return True
    if subject not in {"math", "reasoning"}:
        return False
    if difficulty in {"intermediate", "advanced"}:
        return True
    return (
        difficulty == "default"
        and intent in {"solve", "solve_question"}
        and context_need is not None
        and context_need != "CONTEXT_NOT_NEEDED"
    )


def _numbers_in(text: str) -> set[float]:
    values: set[float] = set()
    for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        try:
            values.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _single_declared_number(text: str) -> float | None:
    """A compact numeric scalar, or text stating exactly one number; otherwise None."""
    scalar = _scalar_answer(text)
    if scalar is not None:
        kind, number = scalar
        if kind != "number":
            return None
        try:
            return float(number)
        except ValueError:
            return None
    stated = _numbers_in(text)
    return next(iter(stated)) if len(stated) == 1 else None


def _candidate_number(candidate_answer: str, *, query: str = "") -> float | None:
    """Return the number the answer line declares, or None when that is not provable.

    A compact scalar ("25", "25 m", "15%", "\\(12\\)") is read the same way the answer
    quality gate reads it, and a line stating a single number declares that number. An
    answer written as an expression declares the result after its last "=". Otherwise
    a number the line repeats from the question ("base 22", "remainders 22, 23 and 24",
    "divided by 7") is context rather than the answer and is set aside, and exactly one
    number must remain. The last number on a prose line is never assumed to be the
    answer: it was routinely the base, divisor or remainder, which made the
    self-consistency guard reject correct answers.

    None keeps that guard silent so the verifier's own verdict stands. The guard must
    never invent a candidate.
    """
    answer_line = re.search(
        r"(?im)^\s*\*{0,2}(?:final\s+)?answer\s*:\s*\*{0,2}\s*([^\n]{1,300})",
        candidate_answer,
    )
    if answer_line is None:
        return None
    value = answer_line.group(1)
    number = _single_declared_number(value)
    if number is not None:
        return number
    if "=" in value:
        number = _single_declared_number(value.rsplit("=", 1)[1])
        if number is not None:
            return number
    declared = _numbers_in(value) - _numbers_in(query)
    return next(iter(declared)) if len(declared) == 1 else None


def _deterministic_expected_answer(query: str) -> float | None:
    normalized = " ".join(query.casefold().replace(",", "").split())
    rank = re.search(
        r"\b(\d+)(?:st|nd|rd|th)\s+from\s+(?:the\s+)?left\b.{0,80}"
        r"\b(\d+)(?:st|nd|rd|th)\s+from\s+(?:the\s+)?right\b",
        normalized,
    )
    if rank is not None:
        return float(int(rank.group(1)) + int(rank.group(2)) - 1)

    percentage = re.search(
        r"\bwhat\s+is\s+(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if percentage is not None:
        return float(percentage.group(1)) * float(percentage.group(2)) / 100

    interest = re.search(
        r"amounts?\s+to\s+(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s+in\s+"
        r"(\d+(?:\.\d+)?)\s+years?.{0,80}?"
        r"(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s+in\s+"
        r"(\d+)\s+years?\s+(?:and\s+)?(\d+)\s+months?.{0,160}?"
        r"compounded\s+every\s+6\s+months?",
        normalized,
    )
    if interest is not None:
        first_amount = float(interest.group(1))
        first_years = float(interest.group(2))
        second_amount = float(interest.group(3))
        second_years = float(interest.group(4)) + float(interest.group(5)) / 12
        elapsed = second_years - first_years
        if elapsed <= 0:
            return None
        annual_interest = (second_amount - first_amount) / elapsed
        principal = first_amount - annual_interest * first_years
        if principal <= 0:
            return None
        half_year_rate = annual_interest / principal / 2
        compound_interest = principal * ((1 + half_year_rate) ** 4 - 1)
        options_match = re.findall(r"\b[A-D][).:\s-]+(\d+(?:\.\d+)?)\b", query)
        if options_match:
            options = [float(value) for value in options_match]
            return min(options, key=lambda value: abs(value - compound_interest))
        return compound_interest
    return None


def _independent_number(independent_answer: str) -> float | None:
    """Return the value of a concise independent answer only when unambiguous."""
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", independent_answer)
    if len(numbers) != 1:
        return None
    try:
        return float(numbers[0].replace(",", ""))
    except ValueError:
        return None


def _enforce_match_self_consistency(
    verification: CorrectnessVerification,
    *,
    candidate_answer: str,
    query: str = "",
) -> CorrectnessVerification:
    """Reject a MATCH that contradicts the verifier's own independent answer.

    The verifier contract defines MATCH as the independently derived result agreeing
    with the candidate, so a MATCH carrying a different answer is self-contradictory
    and must not reach the student. Stays silent unless both sides reduce to a single
    unambiguous number, leaving fractions, ratios, options, symbolic and text answers
    to the model's own judgement.
    """
    if verification.status != "match" or verification.independent_answer is None:
        return verification
    independent = _independent_number(verification.independent_answer)
    candidate = _candidate_number(candidate_answer, query=query)
    if independent is None or candidate is None:
        return verification
    if abs(candidate - independent) <= max(0.01, abs(independent) * 0.0001):
        return verification
    return replace(
        verification,
        status="mismatch",
        reason="verifier_self_contradiction",
        reason_code="VERIFIER_SELF_CONTRADICTION",
    )


def deterministic_verify(
    *, query: str, candidate_answer: str
) -> CorrectnessVerification | None:
    expected = _deterministic_expected_answer(query)
    candidate = _candidate_number(candidate_answer, query=query)
    if expected is None or candidate is None:
        return None
    tolerance = max(0.01, abs(expected) * 0.0001)
    matched = abs(candidate - expected) <= tolerance
    return CorrectnessVerification(
        status="match" if matched else "mismatch",
        independent_answer=f"{expected:g}",
        single_defensible_answer=True,
        reason="deterministic_recomputation",
        method="deterministic",
        reason_code="NONE" if matched else "WRONG_FINAL_ANSWER",
    )


class AnswerCorrectnessVerifier:
    """Use the existing verifier role only for routes selected by deterministic policy."""

    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def verify(
        self,
        *,
        request_id: str,
        query: str,
        candidate_answer: str,
        subject: str,
        difficulty: str,
        language: CanonicalLanguage,
    ) -> CorrectnessVerification:
        deterministic = deterministic_verify(
            query=query,
            candidate_answer=candidate_answer,
        )
        if deterministic is not None:
            self._log_verification(
                deterministic,
                subject=subject,
                difficulty=difficulty,
            )
            return deterministic
        verifier_input = build_verifier_input(query, candidate_answer)
        if len(verifier_input) > MAX_QUERY_CHARS:
            # Unreachable while the compactor holds. Kept so an oversize input is refused
            # deterministically here instead of being retried identically downstream.
            verification = self._unavailable(
                VerifierInputTooLarge(f"composed {len(verifier_input)} characters"),
                INTERNAL_FAILURE,
                "INPUT_TOO_LARGE",
            )
            self._log_verification(
                verification,
                subject=subject,
                difficulty=difficulty,
                failure_kind=INTERNAL_FAILURE,
            )
            return verification
        # Each stage fails with its own kind so an unusable verdict is never
        # reported as an unreachable provider. The student-facing outcome is
        # unchanged: all of them remain ANSWER_VERIFICATION_UNAVAILABLE.
        diagnostic: StructuredOutputDiagnostic | None = None
        failure_kind: str | None = None
        try:
            result = self._orchestrator.generate(
                route_request=RouteRequest(
                    request_id=request_id,
                    subject="general",
                    task_role="verifier",
                    difficulty="default",
                    intent="solve",
                    language=language,
                ),
                query=verifier_input,
            )
        except Exception as exc:  # noqa: BLE001
            failure_kind = PROVIDER_FAILURE
            verification = self._unavailable(exc, failure_kind, technical_reason_code(exc))
        else:
            try:
                parsed = parse_structured_output(result.content, _VerifierOutput)
            except StructuredOutputError as exc:
                diagnostic = exc.diagnostic
                failure_kind = diagnostic.failure_kind
                verification = self._unavailable(
                    exc, failure_kind, _STRUCTURED_OUTPUT_REASON_CODES[diagnostic.stage]
                )
            except Exception as exc:  # noqa: BLE001 - residual guard, fails closed
                failure_kind = INTERNAL_FAILURE
                verification = self._unavailable(exc, failure_kind, "TECHNICAL_UNCLASSIFIED")
            else:
                approved = parsed.status == "MATCH" and parsed.single_defensible_answer
                verification = CorrectnessVerification(
                    status=parsed.status.casefold(),  # type: ignore[arg-type]
                    independent_answer=parsed.independent_answer,
                    single_defensible_answer=parsed.single_defensible_answer,
                    reason=parsed.reason,
                    method="model",
                    # The verifier states no semantic cause yet, so a rejection is
                    # recorded as unclassified rather than guessed.
                    reason_code="NONE" if approved else "SEMANTIC_UNCLASSIFIED",
                )
                verification = _enforce_match_self_consistency(
                    verification,
                    candidate_answer=candidate_answer,
                    query=query,
                )
        self._log_verification(
            verification,
            subject=subject,
            difficulty=difficulty,
            failure_kind=failure_kind,
            diagnostic=diagnostic,
        )
        return verification

    @staticmethod
    def _unavailable(
        exc: Exception, failure_kind: str, reason_code: VerificationReasonCode
    ) -> CorrectnessVerification:
        """Fail closed with the same external reason code every stage has used."""
        logger.warning(
            "answer_correctness_verification unavailable failure_kind=%s error_type=%s",
            failure_kind,
            type(exc).__name__,
        )
        logger.debug(
            "answer_correctness_verification original_exception",
            exc_info=True,
        )
        return CorrectnessVerification(
            status="unavailable",
            reason="ANSWER_VERIFICATION_UNAVAILABLE",
            reason_code=reason_code,
        )

    @staticmethod
    def _log_verification(
        verification: CorrectnessVerification,
        *,
        subject: str,
        difficulty: str,
        failure_kind: str | None = None,
        diagnostic: StructuredOutputDiagnostic | None = None,
    ) -> None:
        log_event(
            "correctness_verification_completed",
            component="doubt_solver.correctness",
            stage="verify_correctness",
            status=verification.status,
            error_code=(
                "ANSWER_VERIFICATION_UNAVAILABLE"
                if verification.status == "unavailable"
                else None
            ),
            details={
                "verification_status": verification.status,
                "single_defensible_answer": verification.single_defensible_answer,
                "approved": verification.approved,
                "subject": subject,
                "difficulty": difficulty,
                "method": verification.method,
                "diagnosis_reason_code": verification.reason_code,
                "diagnosis_failure_kind": verification.failure_kind,
                "diagnosis_failure_source": verification.failure_source,
                **({"failureKind": failure_kind} if failure_kind else {}),
                **(diagnostic.as_event_details() if diagnostic is not None else {}),
            },
        )
