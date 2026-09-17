"""Shadow semantic diagnosis of an already-decided verification verdict.

The primary verifier owns MATCH / MISMATCH / AMBIGUOUS and approval; this bounded second
call only labels where a problem lies (`failure_source`) and why (`reason_code`). It never
sees the approval decision, never feeds back into it, and a failure here is recorded as
unclassified rather than raised. Status and source are orthogonal: a MATCH whose candidate
correctly explains that the question cannot be answered uniquely is still QUESTION /
INCOMPLETE_QUESTION.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from config import get_settings
from observability import log_event
from schemas.doubt_solver import CanonicalLanguage
from schemas.llm_routing import RouteRequest
from services.doubt_solver.recovery_policy import (
    FailureSource,
    VerificationReasonCode,
    diagnosis_can_be_used,
)
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.structured_output import parse_structured_output

logger = logging.getLogger(__name__)

_DIAGNOSIS_PROMPT = "answer_diagnosis.md"
# Each reason belongs to exactly one source; a pair that disagrees is not trusted.
_REASON_SOURCES: dict[str, FailureSource] = {
    "NONE": "NONE",
    "WRONG_FINAL_ANSWER": "CANDIDATE",
    "CANDIDATE_CONFLICT": "CANDIDATE",
    "INCOMPLETE_QUESTION": "QUESTION",
    "MULTIPLE_DEFENSIBLE_ANSWERS": "QUESTION",
    "VERIFIER_LOW_CONFIDENCE": "VERIFIER",
}


class _DiagnosisOutput(BaseModel):
    failure_source: str | None = Field(default=None)
    reason_code: str | None = Field(default=None)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @field_validator("failure_source", "reason_code", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> str | None:
        """Anything that is not a plain string counts as unstated, never as a bad schema."""
        return value.strip().upper() if isinstance(value, str) else None


@dataclass(frozen=True)
class SemanticDiagnosis:
    """What the diagnostic call could establish. Defaults say "nothing"."""

    failure_source: FailureSource = "UNKNOWN"
    reason_code: VerificationReasonCode = "SEMANTIC_UNCLASSIFIED"
    stated: bool = False

    @property
    def classified(self) -> bool:
        return self.reason_code != "SEMANTIC_UNCLASSIFIED"


def _validated(stated_source: str | None, stated_reason: str | None) -> SemanticDiagnosis:
    expected = _REASON_SOURCES.get(stated_reason or "")
    if expected is None or stated_source != expected:
        return SemanticDiagnosis(stated=stated_reason is not None or stated_source is not None)
    return SemanticDiagnosis(
        failure_source=expected,
        reason_code=stated_reason,  # type: ignore[arg-type]
        stated=True,
    )


def diagnose_verification_failure(
    adapter: object,
    *,
    request_id: str,
    query: str,
    candidate_answer: str,
    verdict: str,
    verification: object | None = None,
    method: str = "model",
    language: CanonicalLanguage = "english",
) -> SemanticDiagnosis | None:
    """Label a failed verification, or return None when no diagnosis should run.

    Skipped for a technical verdict (there is no semantic cause to find, and the prompt
    has no verdict to label) and for deterministic recomputation (this system's own
    arithmetic finding is not reopened). Nothing here may change how the request ends,
    so every failure is swallowed and reported as unclassified.
    """
    try:
        if verdict.casefold() == "unavailable" or method == "deterministic":
            return None
        if verification is not None and not diagnosis_can_be_used(verification):
            # Nothing this call returns could be used, so it is not worth paying for.
            return None
        settings = get_settings()
        if not (settings.answer_diagnosis_shadow_enabled or settings.answer_recovery_enabled):
            return None
        diagnoser = getattr(adapter, "semantic_diagnoser", None)
        if diagnoser is None:
            return None
        return diagnoser.diagnose(
            request_id=request_id,
            query=query,
            candidate_answer=candidate_answer,
            verdict=verdict,
            language=language,
        )
    except Exception:  # noqa: BLE001 - a diagnosis failure is unclassified, never fatal
        logger.warning("semantic_diagnosis unavailable", exc_info=True)
        return None


class AnswerDiagnoser:
    """One bounded structured call on the existing verifier route."""

    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def diagnose(
        self,
        *,
        request_id: str,
        query: str,
        candidate_answer: str,
        verdict: str,
        language: CanonicalLanguage = "english",
    ) -> SemanticDiagnosis:
        started = time.monotonic()
        failure: str | None = None
        diagnosis = SemanticDiagnosis()
        try:
            result = self._orchestrator.generate_structured(
                route_request=RouteRequest(
                    request_id=request_id,
                    subject="general",
                    task_role="verifier",
                    difficulty="default",
                    intent="solve",
                    language=language,
                ),
                user_content=(
                    "[QUESTION]\n"
                    f"{query[:4000]}\n"
                    "[/QUESTION]\n"
                    "[CANDIDATE_ANSWER]\n"
                    f"{candidate_answer[:8000]}\n"
                    "[/CANDIDATE_ANSWER]\n"
                    "[VERDICT]\n"
                    f"{verdict.upper()}\n"
                    "[/VERDICT]"
                ),
                prompt=_DIAGNOSIS_PROMPT,
            )
            parsed = parse_structured_output(result.content, _DiagnosisOutput)
        except Exception as exc:  # noqa: BLE001 - diagnosis never blocks a request
            failure = type(exc).__name__
            logger.debug("semantic_diagnosis unavailable", exc_info=True)
        else:
            diagnosis = _validated(parsed.failure_source, parsed.reason_code)
        log_event(
            "SEMANTIC_DIAGNOSIS_COMPLETED",
            component="doubt_solver.diagnosis",
            stage="verify_correctness",
            status="classified" if diagnosis.classified else "unclassified",
            duration_ms=int((time.monotonic() - started) * 1000),
            details={
                "verdict": verdict,
                "diagnosis_failure_source": diagnosis.failure_source,
                "diagnosis_reason_code": diagnosis.reason_code,
                "diagnosis_stated": diagnosis.stated,
                "failure": failure,
            },
        )
        return diagnosis
