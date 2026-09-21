"""Phase 1: bounded local recovery for a verified answer.

Repair the responsibility that failed and reuse every upstream node that already
succeeded. A candidate fault regenerates once, a question fault becomes a clarification,
an eligible technical verifier failure retries the verifier alone, and an outage says so.
Every budget is single-use, and the verifier keeps sole authority over approval.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import config as cfg_module
import services.doubt_solver.recovery_policy as recovery_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from observability.llm_usage import (
    begin_llm_usage_collection,
    record_llm_call,
    reset_llm_usage_collection,
)
from schemas.llm_usage import ProviderTokenUsage
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.answer_diagnosis import SemanticDiagnosis
from services.doubt_solver.question_integrity import ambiguous_question_message
from services.doubt_solver.recovery_policy import (
    CANDIDATE_RECOVERY_INSTRUCTION,
    RecoveryBudget,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.errors import ProviderExecutionError

_QUERY = "Solve for x: 2x + 3 = 11."
_ANSWER_A = "2x + 3 = 11\n\n2x = 8\n\nx = 4\n\n**Answer:** 4"
_ANSWER_B = "2x + 3 = 11\n\n2x = 8\n\nx = 4 exactly\n\n**Answer:** 4"
_MALFORMED = "2x = 8 so \\( x = 8 \\div 2"
_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "intermediate",
    "need_web_search": False,
    "classifier_confidence": 0.99,
    "classification_source": "llm",
}


def _verdict(
    status: str,
    *,
    single: bool = True,
    method: str = "model",
    reason_code: str = "SEMANTIC_UNCLASSIFIED",
) -> CorrectnessVerification:
    return CorrectnessVerification(
        status=status,  # type: ignore[arg-type]
        independent_answer="4",
        single_defensible_answer=single,
        reason="r",
        method=method,  # type: ignore[arg-type]
        reason_code=reason_code,  # type: ignore[arg-type]
    )


_MATCH = _verdict("match", reason_code="NONE")
_MISMATCH = _verdict("mismatch")
_AMBIGUOUS = _verdict("ambiguous", single=False)
_TIMEOUT = CorrectnessVerification(
    status="unavailable", reason="ANSWER_VERIFICATION_UNAVAILABLE", reason_code="TIMEOUT"
)
_PARSE_FAILURE = CorrectnessVerification(
    status="unavailable",
    reason="ANSWER_VERIFICATION_UNAVAILABLE",
    reason_code="PARSE_FAILURE",
)
_CANDIDATE_FAULT = SemanticDiagnosis("CANDIDATE", "WRONG_FINAL_ANSWER", True)
_CANDIDATE_CONFLICT = SemanticDiagnosis("CANDIDATE", "CANDIDATE_CONFLICT", True)
_INCOMPLETE = SemanticDiagnosis("QUESTION", "INCOMPLETE_QUESTION", True)
_MULTIPLE = SemanticDiagnosis("QUESTION", "MULTIPLE_DEFENSIBLE_ANSWERS", True)
_UNCLASSIFIED = SemanticDiagnosis()


class _Verifier:
    """Returns one verdict per call, repeating the last one."""

    def __init__(self, *verdicts: CorrectnessVerification) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[str] = []

    def verify(self, **kwargs: object) -> CorrectnessVerification:
        self.calls.append(str(kwargs["candidate_answer"]))
        index = min(len(self.calls) - 1, len(self._verdicts) - 1)
        return self._verdicts[index]


class _Diagnoser:
    def __init__(self, *diagnoses: SemanticDiagnosis | Exception) -> None:
        self._diagnoses = list(diagnoses)
        self.calls = 0

    def diagnose(self, **_: object) -> SemanticDiagnosis:
        self.calls += 1
        result = self._diagnoses[min(self.calls - 1, len(self._diagnoses) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


class _Adapter:
    """Returns a fresh candidate per generate call and records its attempt types."""

    def __init__(
        self,
        *candidates: str,
        verifier: _Verifier,
        diagnoser: _Diagnoser | None = None,
        attempts: tuple[str, ...] = ("primary",),
        error: Exception | None = None,
    ) -> None:
        self._candidates = list(candidates)
        self._attempts = attempts
        self._error = error
        self.generate_calls: list[dict] = []
        self.correctness_verifier = verifier
        if diagnoser is not None:
            self.semantic_diagnoser = diagnoser

    def generate(self, **kwargs: object) -> str:
        self.generate_calls.append(dict(kwargs))
        if self._error is not None and len(self.generate_calls) > 1:
            raise self._error
        for attempt in self._attempts if len(self.generate_calls) == 1 else ("repair",):
            record_llm_call(
                request_id=str(kwargs["request_id"]),
                role="math.generator.intermediate",
                provider="mock",
                model="mock",
                deployment=None,
                attempt_type=attempt,
                streaming=False,
                usage=ProviderTokenUsage(),
                duration_ms=1,
                status="succeeded",
            )
        return self._candidates[min(len(self.generate_calls) - 1, len(self._candidates) - 1)]


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "true")
    monkeypatch.setenv("ANSWER_RECOVERY_ENABLED", "true")
    cfg_module._settings = None
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    yield
    cfg_module._settings = None


@pytest.fixture
def recovery_events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    events: list[dict] = []
    original = recovery_module.log_event

    def _record(name: str, **kwargs: object) -> None:
        if name == "RECOVERY_DECISION":
            events.append(dict(kwargs.get("details") or {}))
        original(name, **kwargs)

    monkeypatch.setattr(recovery_module, "log_event", _record)
    return events


def _stream(adapter: _Adapter) -> list:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="phase1",
                query=_QUERY,
                language="english",
                classification=dict(_CLASSIFICATION),
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


def _terminal(events: list):
    return events[-1].type, events[-1].metadata


def _delivered(events: list) -> str:
    return "".join(event.content or "" for event in events if event.type == "chunk")


# ---------------------------------------------------------------------------
# A. Candidate recovery succeeds
# ---------------------------------------------------------------------------


def test_a_wrong_candidate_is_regenerated_once_and_then_delivered(
    recovery_events: list[dict],
) -> None:
    verifier = _Verifier(_MISMATCH, _MATCH)
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    events = _stream(adapter)

    assert events[-1].type == "complete"
    assert _delivered(events) == _ANSWER_B
    # Exactly one regeneration and exactly two verifications, on different candidates.
    assert len(adapter.generate_calls) == 2
    assert verifier.calls == [_ANSWER_A, _ANSWER_B]
    # Upstream work is reused, not repeated: the second call carries the same inputs.
    first, second = adapter.generate_calls
    assert {k: v for k, v in second.items() if k != "recovery_instruction"} == first
    assert second["recovery_instruction"] == CANDIDATE_RECOVERY_INSTRUCTION
    actions = [event["recovery_action"] for event in recovery_events]
    assert actions == ["REGENERATE_CANDIDATE"]
    assert recovery_events[0]["candidate_recovery_used"] is True


def test_the_regeneration_never_receives_the_verifier_answer_or_reasoning() -> None:
    verifier = _Verifier(
        CorrectnessVerification(
            status="mismatch",
            independent_answer="SECRET-VERIFIER-ANSWER",
            single_defensible_answer=True,
            reason="SECRET-VERIFIER-REASONING",
            method="model",
            reason_code="SEMANTIC_UNCLASSIFIED",
        ),
        _MATCH,
    )
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    _stream(adapter)

    sent = repr(adapter.generate_calls[1])
    assert "SECRET-VERIFIER-ANSWER" not in sent
    assert "SECRET-VERIFIER-REASONING" not in sent
    assert "independently" in CANDIDATE_RECOVERY_INSTRUCTION


# ---------------------------------------------------------------------------
# B / C. Candidate recovery is bounded
# ---------------------------------------------------------------------------


def test_b_a_second_wrong_candidate_is_terminal(recovery_events: list[dict]) -> None:
    verifier = _Verifier(_MISMATCH)
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {"retryable": False, "code": "ANSWER_VERIFICATION_FAILED", "user_retryable": True},
    )
    assert len(adapter.generate_calls) == 2  # no third candidate
    assert len(verifier.calls) == 2  # exactly one verification after regeneration
    assert [event["recovery_action"] for event in recovery_events] == [
        "REGENERATE_CANDIDATE",
        "FAIL_SAFE",
    ]


def test_c_a_self_contradicting_candidate_also_regenerates_once() -> None:
    verifier = _Verifier(_AMBIGUOUS, _MATCH)
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_CONFLICT)
    )

    events = _stream(adapter)

    assert events[-1].type == "complete"
    assert len(adapter.generate_calls) == 2


# ---------------------------------------------------------------------------
# D / E. Question faults become a clarification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagnosis", [_INCOMPLETE, _MULTIPLE])
def test_d_a_question_fault_asks_the_student_for_what_is_missing(
    diagnosis: SemanticDiagnosis,
) -> None:
    """A question the deterministic gates admitted is blamed only after a fresh candidate.

    The approved policy changed here: one ambiguous verdict is no longer enough to end the
    request, because the same question family is answered successfully on other turns. The
    single candidate slot is spent first, and a still-ambiguous second verdict is what asks
    the student for what is missing.
    """
    verifier = _Verifier(_AMBIGUOUS, _AMBIGUOUS)
    diagnoser = _Diagnoser(diagnosis)
    adapter = _Adapter(_ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    terminal = events[-1]
    assert terminal.metadata == {
        "retryable": False,
        "code": "QUESTION_NEEDS_CLARIFICATION",
        "user_retryable": False,
    }
    assert len(adapter.generate_calls) == 2  # exactly one regeneration, never a second
    assert len(verifier.calls) == 2
    assert diagnoser.calls == 1  # the post-regeneration verdict pays for no diagnosis
    # The wording still comes from the diagnosis, even though it was paid for once.
    assert terminal.label == ambiguous_question_message(
        "english", multiple_answers=diagnosis is _MULTIPLE
    )
    # The student is told what to send, and never an internal code.
    assert terminal.label is not None and len(terminal.label) > 20
    for code in ("INCOMPLETE_QUESTION", "MULTIPLE_DEFENSIBLE_ANSWERS", "QUESTION"):
        assert code not in terminal.label


# ---------------------------------------------------------------------------
# F. MATCH stays authoritative
# ---------------------------------------------------------------------------


def test_f_a_match_completes_even_when_the_question_is_diagnosed_incomplete() -> None:
    verifier = _Verifier(_MATCH)
    diagnoser = _Diagnoser(_INCOMPLETE)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert events[-1].type == "complete"
    assert _delivered(events) == _ANSWER_A
    # A healthy answer never pays for a diagnosis.
    assert diagnoser.calls == 0
    assert len(adapter.generate_calls) == 1


# ---------------------------------------------------------------------------
# G / H. Verifier-local technical retry
# ---------------------------------------------------------------------------


def test_g_a_parse_failure_retries_the_verifier_with_the_same_candidate(
    recovery_events: list[dict],
) -> None:
    verifier = _Verifier(_PARSE_FAILURE, _MATCH)
    diagnoser = _Diagnoser(_CANDIDATE_FAULT)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert events[-1].type == "complete"
    assert verifier.calls == [_ANSWER_A, _ANSWER_A]  # identical candidate
    assert len(adapter.generate_calls) == 1  # no generator rerun
    assert diagnoser.calls == 0  # a technical failure is never diagnosed
    assert [event["recovery_action"] for event in recovery_events] == ["RETRY_SAME_NODE"]


def test_h_a_second_parse_failure_is_terminal(recovery_events: list[dict]) -> None:
    verifier = _Verifier(_PARSE_FAILURE)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(_UNCLASSIFIED))

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {"retryable": False, "code": "ANSWER_VERIFICATION_UNAVAILABLE"},
    )
    assert len(verifier.calls) == 2  # no third verification
    assert [event["recovery_action"] for event in recovery_events] == [
        "RETRY_SAME_NODE",
        "FAIL_TEMPORARY",
    ]


# ---------------------------------------------------------------------------
# I / J. Technical terminals
# ---------------------------------------------------------------------------


def test_i_provider_exhaustion_is_reported_as_a_temporary_outage() -> None:
    verifier = _Verifier(_TIMEOUT)
    diagnoser = _Diagnoser(_CANDIDATE_FAULT)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {
            "retryable": True,
            "code": "SERVICE_TEMPORARILY_UNAVAILABLE",
            "user_retryable": True,
        },
    )
    assert len(verifier.calls) == 1  # the executor owns provider fallback; no outer retry
    assert len(adapter.generate_calls) == 1
    assert diagnoser.calls == 0


def test_j_a_misconfiguration_is_not_retried(recovery_events: list[dict]) -> None:
    configuration_failure = CorrectnessVerification(
        status="unavailable",
        reason="ANSWER_VERIFICATION_UNAVAILABLE",
        reason_code="CONFIGURATION_FAILURE",
    )
    verifier = _Verifier(configuration_failure)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(_UNCLASSIFIED))

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_UNAVAILABLE"  # type: ignore[index]
    assert len(verifier.calls) == 1
    assert [event["recovery_action"] for event in recovery_events] == ["FAIL_TEMPORARY"]


def test_j_an_exhausted_token_budget_is_not_retried_with_the_same_budget() -> None:
    exhausted = CorrectnessVerification(
        status="unavailable",
        reason="ANSWER_VERIFICATION_UNAVAILABLE",
        reason_code="OUTPUT_TOKEN_EXHAUSTED",
    )
    verifier = _Verifier(exhausted)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(_UNCLASSIFIED))

    _stream(adapter)

    assert len(verifier.calls) == 1


# ---------------------------------------------------------------------------
# K / L. Unknown semantics and a broken diagnoser both fail safe
# ---------------------------------------------------------------------------


def test_k_an_unclassified_semantic_failure_never_regenerates() -> None:
    verifier = _Verifier(_MISMATCH)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(_UNCLASSIFIED))

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert len(adapter.generate_calls) == 1
    assert len(verifier.calls) == 1


def test_l_a_diagnosis_failure_fails_safe_without_escaping() -> None:
    verifier = _Verifier(_MISMATCH)
    adapter = _Adapter(
        _ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(RuntimeError("diagnoser down"))
    )

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert len(adapter.generate_calls) == 1


def test_a_regenerated_candidate_that_fails_quality_is_not_delivered() -> None:
    verifier = _Verifier(_MISMATCH, _MATCH)
    adapter = _Adapter(
        _ANSWER_A, _MALFORMED, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_QUALITY_FAILED"  # type: ignore[index]
    assert _MALFORMED not in _delivered(events)
    assert len(verifier.calls) == 1  # the malformed candidate never reaches verification


def test_a_failed_regeneration_call_is_terminal() -> None:
    verifier = _Verifier(_MISMATCH)
    adapter = _Adapter(
        _ANSWER_A,
        verifier=verifier,
        diagnoser=_Diagnoser(_CANDIDATE_FAULT),
        error=ProviderExecutionError("down", failure_kind="provider_unavailable"),
    )

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_REPAIR_FAILED"  # type: ignore[index]


def test_no_diagnosis_is_paid_for_after_a_regeneration() -> None:
    """One verification follows a regeneration and ends the request either way."""
    verifier = _Verifier(_MISMATCH)
    diagnoser = _Diagnoser(_CANDIDATE_FAULT)
    adapter = _Adapter(_ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert len(verifier.calls) == 2
    assert diagnoser.calls == 1  # only the first failure was diagnosed


def test_the_loop_fails_closed_if_a_budget_ever_stopped_being_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence in depth: exhausting the attempts must never deliver an unapproved answer.

    Forces the verifier-retry slot to stay reusable, which is the only shape that can run
    the loop to exhaustion and reach the guard after it.
    """
    monkeypatch.setattr(
        recovery_module.RecoveryBudget, "take_verifier_technical_retry", lambda self: True
    )
    verifier = _Verifier(_PARSE_FAILURE)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=_Diagnoser(_UNCLASSIFIED))

    events = _stream(adapter)

    assert len(verifier.calls) == 3  # the loop ran out of attempts
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert _delivered(events) == ""


def test_an_outage_after_a_regeneration_still_reads_as_an_outage() -> None:
    """The second verification keeps its own reason: an outage is never a wrong answer."""
    verifier = _Verifier(_MISMATCH, _TIMEOUT)
    diagnoser = _Diagnoser(_CANDIDATE_FAULT)
    adapter = _Adapter(_ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {
            "retryable": True,
            "code": "SERVICE_TEMPORARILY_UNAVAILABLE",
            "user_retryable": True,
        },
    )
    assert diagnoser.calls == 1  # the second failure is still not diagnosed


def test_a_technical_failure_after_a_regeneration_may_use_the_unused_verifier_retry() -> None:
    verifier = _Verifier(_MISMATCH, _PARSE_FAILURE, _MATCH)
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    events = _stream(adapter)

    assert events[-1].type == "complete"
    assert len(verifier.calls) == 3  # regeneration, its verification, one verifier retry
    assert len(adapter.generate_calls) == 2  # still only one regeneration


# ---------------------------------------------------------------------------
# M. Budgets cannot be handed back
# ---------------------------------------------------------------------------


def test_m_each_slot_can_be_taken_only_once() -> None:
    budget = RecoveryBudget()

    assert budget.take_candidate_recovery() is True
    assert budget.take_candidate_recovery() is False
    assert budget.take_verifier_technical_retry() is True
    assert budget.take_verifier_technical_retry() is False
    snapshot = budget.snapshot()
    assert snapshot.candidate_recovery_used is True
    assert snapshot.verifier_technical_retry_used is True


@pytest.mark.parametrize(
    "attempt_type", ["primary", "continuation", "rewrite", "primary_fallback", "repair_fallback"]
)
def test_m_no_generator_attempt_hands_a_verifier_retry_back(attempt_type: str) -> None:
    budget = RecoveryBudget()
    budget.take_verifier_technical_retry()
    token = begin_llm_usage_collection()

    record_llm_call(
        request_id="budget",
        role="math.generator.intermediate",
        provider="mock",
        model="mock",
        deployment=None,
        attempt_type=attempt_type,
        streaming=False,
        usage=ProviderTokenUsage(),
        duration_ms=1,
        status="succeeded",
    )

    try:
        assert budget.snapshot().verifier_technical_retry_used is True
        assert budget.take_verifier_technical_retry() is False
    finally:
        reset_llm_usage_collection(token)


def test_m_a_quality_repair_already_spent_blocks_verifier_driven_regeneration() -> None:
    """The presentation rewrite is separate, but a candidate repair is the same slot."""
    budget = RecoveryBudget()
    token = begin_llm_usage_collection()
    record_llm_call(
        request_id="budget",
        role="math.generator.intermediate",
        provider="mock",
        model="mock",
        deployment=None,
        attempt_type="repair",
        streaming=False,
        usage=ProviderTokenUsage(),
        duration_ms=1,
        status="succeeded",
    )

    try:
        assert budget.snapshot().candidate_recovery_used is True
        assert budget.take_candidate_recovery() is False
    finally:
        reset_llm_usage_collection(token)


def test_m_a_presentation_rewrite_leaves_the_candidate_slot_free() -> None:
    budget = RecoveryBudget()
    token = begin_llm_usage_collection()
    for attempt in ("primary", "continuation", "rewrite"):
        record_llm_call(
            request_id="budget",
            role="math.generator.intermediate",
            provider="mock",
            model="mock",
            deployment=None,
            attempt_type=attempt,
            streaming=False,
            usage=ProviderTokenUsage(),
            duration_ms=1,
            status="succeeded",
        )

    try:
        snapshot = budget.snapshot()
        assert (snapshot.continuation_used, snapshot.presentation_rewrite_used) == (True, True)
        assert snapshot.candidate_recovery_used is False
        assert budget.take_candidate_recovery() is True
    finally:
        reset_llm_usage_collection(token)


# ---------------------------------------------------------------------------
# Rollout and lifecycle
# ---------------------------------------------------------------------------


def test_recovery_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANSWER_RECOVERY_ENABLED", raising=False)
    monkeypatch.delenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", raising=False)
    cfg_module._settings = None
    verifier = _Verifier(_MISMATCH)
    diagnoser = _Diagnoser(_CANDIDATE_FAULT)
    adapter = _Adapter(_ANSWER_A, verifier=verifier, diagnoser=diagnoser)

    events = _stream(adapter)

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert len(adapter.generate_calls) == 1
    assert diagnoser.calls == 0


def test_recovery_implies_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery acts on a diagnosis, so it can never run without one."""
    monkeypatch.delenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", raising=False)
    monkeypatch.setenv("ANSWER_RECOVERY_ENABLED", "true")
    cfg_module._settings = None
    settings = cfg_module.get_settings()

    assert settings.answer_recovery_enabled is True
    assert settings.answer_diagnosis_shadow_enabled is True


def test_every_recovery_keeps_one_terminal_event_and_no_heartbeat_after_it() -> None:
    verifier = _Verifier(_MISMATCH, _MATCH)
    adapter = _Adapter(
        _ANSWER_A, _ANSWER_B, verifier=verifier, diagnoser=_Diagnoser(_CANDIDATE_FAULT)
    )

    events = _stream(adapter)

    terminals = [index for index, event in enumerate(events) if event.type in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0] == len(events) - 1


def test_recovery_never_names_a_model() -> None:
    """The policy may read subject, difficulty and role; never a model or provider."""
    source = (
        recovery_module.__file__,
        "services/doubt_solver/answer_diagnosis.py",
    )
    from pathlib import Path

    for path in source:
        text = Path(path).read_text().casefold()
        for name in ("terra", "deepseek", "gpt-", "gemini", "claude", "azure", "openai"):
            assert name not in text, f"{name} appears in {path}"
