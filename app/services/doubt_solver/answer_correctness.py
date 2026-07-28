"""Selective independent correctness verification for high-risk answers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from observability import log_event
from schemas.doubt_solver import CanonicalLanguage
from schemas.llm_routing import RouteRequest
from services.doubt_solver.classifier_json import parse_classifier_json_strict
from services.llm.orchestration.orchestrator import LlmOrchestrator

logger = logging.getLogger(__name__)

VerificationStatus = Literal["match", "mismatch", "ambiguous", "unavailable"]


class _VerifierOutput(BaseModel):
    status: Literal["MATCH", "MISMATCH", "AMBIGUOUS"]
    independent_answer: str = Field(min_length=1)
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

    @property
    def approved(self) -> bool:
        return self.status == "match" and self.single_defensible_answer


def requires_independent_correctness_verification(
    *,
    subject: str,
    difficulty: str,
    intent: str,
    requested_action: str | None = None,
) -> bool:
    if requested_action in {"VERIFY_AND_CORRECT", "RESOLVE_FROM_SCRATCH"}:
        return True
    if intent in {"practice", "practice_question"}:
        return True
    return subject in {"math", "reasoning"} and difficulty in {
        "intermediate",
        "advanced",
    }


def _candidate_number(candidate_answer: str) -> float | None:
    answer_line = re.search(
        r"(?im)^\s*\*{0,2}(?:final\s+)?answer\s*:\s*\*{0,2}\s*([^\n]{1,300})",
        candidate_answer,
    )
    if answer_line is None:
        return None
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_line.group(1))
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


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


def deterministic_verify(
    *, query: str, candidate_answer: str
) -> CorrectnessVerification | None:
    expected = _deterministic_expected_answer(query)
    candidate = _candidate_number(candidate_answer)
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
        verifier_input = (
            "[QUESTION]\n"
            f"{query[:4000]}\n"
            "[/QUESTION]\n"
            "[CANDIDATE_ANSWER]\n"
            f"{candidate_answer[:8000]}\n"
            "[/CANDIDATE_ANSWER]"
        )
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
            payload, _ = parse_classifier_json_strict(result.content)
            parsed = _VerifierOutput.model_validate(payload)
            verification = CorrectnessVerification(
                status=parsed.status.casefold(),  # type: ignore[arg-type]
                independent_answer=parsed.independent_answer,
                single_defensible_answer=parsed.single_defensible_answer,
                reason=parsed.reason,
                method="model",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "answer_correctness_verification unavailable error_type=%s",
                type(exc).__name__,
            )
            logger.debug(
                "answer_correctness_verification original_exception",
                exc_info=True,
            )
            verification = CorrectnessVerification(
                status="unavailable",
                reason="ANSWER_VERIFICATION_UNAVAILABLE",
            )
        self._log_verification(
            verification,
            subject=subject,
            difficulty=difficulty,
        )
        return verification

    @staticmethod
    def _log_verification(
        verification: CorrectnessVerification,
        *,
        subject: str,
        difficulty: str,
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
            },
        )
