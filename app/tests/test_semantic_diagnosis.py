"""Phase 0b: a separate shadow diagnosis that labels a verdict it can never change.

The primary verifier keeps its prompt and its authority: it alone decides MATCH / MISMATCH /
AMBIGUOUS and approval. This second bounded call only reports where a problem lies and why,
it is off by default, and nothing it returns reaches the student.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import config as cfg_module
import services.doubt_solver.answer_diagnosis as diagnosis_module
import services.doubt_solver.recovery_policy as recovery_module
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.answer_diagnosis import (
    AnswerDiagnoser,
    SemanticDiagnosis,
    diagnose_verification_failure,
)
from services.doubt_solver.recovery_policy import (
    RecoveryBudgetUse,
    decide_verification_recovery,
)

_VERIFIER_PROMPT = Path("prompts/answer_correctness_verifier.md")
_DIAGNOSIS_PROMPT = Path("prompts/answer_diagnosis.md")


class _Orchestrator:
    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls: list[dict] = []

    def generate_structured(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=self._content)


def _diagnose(content: str = "", error: Exception | None = None, *, verdict: str = "mismatch"):
    orchestrator = _Orchestrator(content, error)
    diagnosis = AnswerDiagnoser(orchestrator=orchestrator).diagnose(  # type: ignore[arg-type]
        request_id="diagnosis-test",
        query="Solve for x: 2x + 3 = 11.",
        candidate_answer="**Answer:** 5",
        verdict=verdict,
    )
    return diagnosis, orchestrator


def _output(source: str, reason: str) -> str:
    return f'{{"failure_source":{source},"reason_code":{reason}}}'


# ---------------------------------------------------------------------------
# The primary verifier is untouched
# ---------------------------------------------------------------------------


def test_the_primary_verifier_prompt_never_mentions_the_diagnosis() -> None:
    verifier_prompt = _VERIFIER_PROMPT.read_text()

    assert "reason_code" not in verifier_prompt
    assert "failure_source" not in verifier_prompt
    # The verdict vocabulary stays where it always was.
    for verdict in ("`MATCH`", "`MISMATCH`", "`AMBIGUOUS`"):
        assert verdict in verifier_prompt


def test_the_diagnosis_prompt_asks_only_for_the_two_labels() -> None:
    prompt = _DIAGNOSIS_PROMPT.read_text()

    assert "`failure_source` and `reason_code`" in prompt
    for code in (
        "NONE",
        "WRONG_FINAL_ANSWER",
        "CANDIDATE_CONFLICT",
        "INCOMPLETE_QUESTION",
        "MULTIPLE_DEFENSIBLE_ANSWERS",
        "VERIFIER_LOW_CONFIDENCE",
    ):
        assert f"`{code}`" in prompt
    assert set(diagnosis_module._REASON_SOURCES) == {
        "NONE",
        "WRONG_FINAL_ANSWER",
        "CANDIDATE_CONFLICT",
        "INCOMPLETE_QUESTION",
        "MULTIPLE_DEFENSIBLE_ANSWERS",
        "VERIFIER_LOW_CONFIDENCE",
    }


def test_the_diagnosis_runs_on_the_existing_verifier_route() -> None:
    _, orchestrator = _diagnose(_output('"CANDIDATE"', '"WRONG_FINAL_ANSWER"'))

    call = orchestrator.calls[0]
    assert call["prompt"] == "answer_diagnosis.md"
    route = call["route_request"]
    assert (route.subject, route.task_role, route.difficulty) == ("general", "verifier", "default")


# ---------------------------------------------------------------------------
# Validation: a pair is trusted only when the source matches the reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reason", "expected_source", "expected_reason"),
    [
        ('"CANDIDATE"', '"WRONG_FINAL_ANSWER"', "CANDIDATE", "WRONG_FINAL_ANSWER"),
        ('"CANDIDATE"', '"CANDIDATE_CONFLICT"', "CANDIDATE", "CANDIDATE_CONFLICT"),
        ('"QUESTION"', '"INCOMPLETE_QUESTION"', "QUESTION", "INCOMPLETE_QUESTION"),
        ('"QUESTION"', '"MULTIPLE_DEFENSIBLE_ANSWERS"', "QUESTION", "MULTIPLE_DEFENSIBLE_ANSWERS"),
        ('"VERIFIER"', '"VERIFIER_LOW_CONFIDENCE"', "VERIFIER", "VERIFIER_LOW_CONFIDENCE"),
        ('"NONE"', '"NONE"', "NONE", "NONE"),
        ('"candidate"', '"wrong_final_answer"', "CANDIDATE", "WRONG_FINAL_ANSWER"),
        # A pair that disagrees, an unknown value, or a missing half is not trusted.
        ('"QUESTION"', '"WRONG_FINAL_ANSWER"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('"CANDIDATE"', '"INCOMPLETE_QUESTION"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('"GENERATOR"', '"WRONG_FINAL_ANSWER"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('"CANDIDATE"', '"SOMETHING_ELSE"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('"CANDIDATE"', "null", "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ("null", '"WRONG_FINAL_ANSWER"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ("null", "null", "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        # Runtime-only codes can never be claimed by the model.
        ('"VERIFIER"', '"TIMEOUT"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('"VERIFIER"', '"SEMANTIC_UNCLASSIFIED"', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        # Non-string values are unstated, not schema failures.
        ("12", "true", "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
        ('["CANDIDATE"]', '{"a":1}', "UNKNOWN", "SEMANTIC_UNCLASSIFIED"),
    ],
)
def test_only_an_agreeing_pair_is_trusted(
    source: str, reason: str, expected_source: str, expected_reason: str
) -> None:
    diagnosis, _ = _diagnose(_output(source, reason))

    assert (diagnosis.failure_source, diagnosis.reason_code) == (expected_source, expected_reason)
    assert diagnosis.classified is (expected_reason != "SEMANTIC_UNCLASSIFIED")


@pytest.mark.parametrize(
    "content", ["", "not json", '{"failure_source":"CANDIDATE"}, extra', '{"unexpected":"x"}']
)
def test_unusable_output_is_unclassified(content: str) -> None:
    diagnosis, _ = _diagnose(content)

    assert diagnosis == SemanticDiagnosis()


def test_a_provider_failure_is_swallowed() -> None:
    diagnosis, _ = _diagnose(error=RuntimeError("provider down"))

    assert diagnosis.classified is False


def test_the_event_records_codes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        diagnosis_module,
        "log_event",
        lambda _name, **kwargs: captured.append(dict(kwargs.get("details") or {})),
    )

    _diagnose(_output('"QUESTION"', '"INCOMPLETE_QUESTION"'), verdict="ambiguous")

    details = captured[-1]
    assert details["diagnosis_failure_source"] == "QUESTION"
    assert details["diagnosis_reason_code"] == "INCOMPLETE_QUESTION"
    assert details["diagnosis_stated"] is True
    assert details["verdict"] == "ambiguous"
    assert "2x + 3" not in repr(details)


# ---------------------------------------------------------------------------
# Status and source are orthogonal, and the diagnosis never decides
# ---------------------------------------------------------------------------


def test_an_approved_verdict_completes_whatever_the_diagnosis_says() -> None:
    """MATCH + QUESTION + INCOMPLETE_QUESTION is a valid pair and still completes."""
    decision = decide_verification_recovery(
        approved=True,
        reason_code="INCOMPLETE_QUESTION",
        failure_source="QUESTION",
        budget=RecoveryBudgetUse(),
    )

    assert decision.action == "COMPLETE"
    assert decision.terminal_code is None


@pytest.mark.parametrize(
    ("source", "reason", "action"),
    [
        ("QUESTION", "INCOMPLETE_QUESTION", "ASK_CLARIFICATION"),
        ("QUESTION", "MULTIPLE_DEFENSIBLE_ANSWERS", "ASK_CLARIFICATION"),
        ("CANDIDATE", "WRONG_FINAL_ANSWER", "REGENERATE_CANDIDATE"),
        ("CANDIDATE", "CANDIDATE_CONFLICT", "REGENERATE_CANDIDATE"),
        ("VERIFIER", "VERIFIER_LOW_CONFIDENCE", "FAIL_SAFE"),
    ],
)
def test_a_failed_verdict_takes_the_diagnosed_route(source: str, reason: str, action: str) -> None:
    decision = decide_verification_recovery(
        approved=False, reason_code=reason, failure_source=source, budget=RecoveryBudgetUse()
    )

    assert (decision.action, decision.failure_source) == (action, source)


def test_a_diagnosis_cannot_relabel_a_technical_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real wiring: a verifier timeout keeps its technical action.

    Asserting on the decision function alone would pass vacuously, because the wiring is
    what decides whether the diagnosis is consulted at all.
    """
    decisions: list[dict] = []
    monkeypatch.setattr(
        recovery_module,
        "log_event",
        lambda name, **kwargs: (
            decisions.append(dict(kwargs.get("details") or {}))
            if name == "RECOVERY_DECISION_SHADOW"
            else None
        ),
    )
    timeout = CorrectnessVerification(
        status="unavailable", reason="ANSWER_VERIFICATION_UNAVAILABLE", reason_code="TIMEOUT"
    )

    recovery_module.shadow_verification_recovery(
        timeout,
        actual_terminal_code="ANSWER_VERIFICATION_UNAVAILABLE",
        diagnosis=SemanticDiagnosis("CANDIDATE", "WRONG_FINAL_ANSWER", True),
    )

    assert decisions[-1]["action"] == "FAIL_TEMPORARY"
    assert decisions[-1]["reason_code"] == "TIMEOUT"
    assert decisions[-1]["shadow_terminal_code"] == "SERVICE_TEMPORARILY_UNAVAILABLE"


def test_a_semantic_failure_does_take_the_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions: list[dict] = []
    monkeypatch.setattr(
        recovery_module,
        "log_event",
        lambda name, **kwargs: (
            decisions.append(dict(kwargs.get("details") or {}))
            if name == "RECOVERY_DECISION_SHADOW"
            else None
        ),
    )
    mismatch = CorrectnessVerification(
        status="mismatch",
        single_defensible_answer=True,
        reason="differs",
        method="model",
        reason_code="SEMANTIC_UNCLASSIFIED",
    )

    recovery_module.shadow_verification_recovery(
        mismatch,
        actual_terminal_code="ANSWER_VERIFICATION_FAILED",
        diagnosis=SemanticDiagnosis("QUESTION", "INCOMPLETE_QUESTION", True),
    )

    assert decisions[-1]["action"] == "ASK_CLARIFICATION"
    assert decisions[-1]["failure_source"] == "QUESTION"


def test_a_technical_verdict_is_never_sent_for_diagnosis(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    """No paid call during a provider outage, and the prompt never sees UNAVAILABLE."""
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "true")
    adapter = _Adapter(SemanticDiagnosis("CANDIDATE", "WRONG_FINAL_ANSWER", True))

    result = diagnose_verification_failure(
        adapter,
        request_id="wiring",
        query="Solve for x.",
        candidate_answer="**Answer:** 4",
        verdict="unavailable",
    )

    assert result is None
    assert adapter.calls == 0


@pytest.mark.parametrize(
    ("verification", "expected_action", "expected_reason"),
    [
        (
            # Deterministic recomputation is arithmetic ground truth.
            CorrectnessVerification(
                status="mismatch",
                independent_answer="10",
                single_defensible_answer=True,
                reason="deterministic_recomputation",
                method="deterministic",
                reason_code="WRONG_FINAL_ANSWER",
            ),
            "REGENERATE_CANDIDATE",
            "WRONG_FINAL_ANSWER",
        ),
        (
            # The self-contradiction guard is this system's own finding.
            CorrectnessVerification(
                status="mismatch",
                independent_answer="12",
                single_defensible_answer=True,
                reason="verifier_self_contradiction",
                method="model",
                reason_code="VERIFIER_SELF_CONTRADICTION",
            ),
            "FAIL_SAFE",
            "VERIFIER_SELF_CONTRADICTION",
        ),
    ],
)
def test_a_deterministic_finding_is_not_reopened(
    monkeypatch: pytest.MonkeyPatch,
    verification: CorrectnessVerification,
    expected_action: str,
    expected_reason: str,
) -> None:
    decisions: list[dict] = []
    monkeypatch.setattr(
        recovery_module,
        "log_event",
        lambda name, **kwargs: (
            decisions.append(dict(kwargs.get("details") or {}))
            if name == "RECOVERY_DECISION_SHADOW"
            else None
        ),
    )

    recovery_module.shadow_verification_recovery(
        verification,
        actual_terminal_code="ANSWER_VERIFICATION_FAILED",
        diagnosis=SemanticDiagnosis("QUESTION", "INCOMPLETE_QUESTION", True),
    )

    assert decisions[-1]["action"] == expected_action
    assert decisions[-1]["reason_code"] == expected_reason


def test_an_unclassified_diagnosis_leaves_the_verdict_alone() -> None:
    decision = decide_verification_recovery(
        approved=False,
        reason_code="SEMANTIC_UNCLASSIFIED",
        failure_source=None,
        budget=RecoveryBudgetUse(),
    )

    assert decision.action == "FAIL_SAFE"
    assert decision.terminal_code == "ANSWER_VERIFICATION_FAILED"


# ---------------------------------------------------------------------------
# Wiring: off by default, and never able to break a request
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_settings() -> Iterator[None]:
    cfg_module._settings = None
    yield
    cfg_module._settings = None


class _Adapter:
    def __init__(self, diagnosis: SemanticDiagnosis | Exception) -> None:
        self._diagnosis = diagnosis
        self.calls = 0
        self.semantic_diagnoser = self

    def diagnose(self, **_: object) -> SemanticDiagnosis:
        self.calls += 1
        if isinstance(self._diagnosis, Exception):
            raise self._diagnosis
        return self._diagnosis


def _shadow(adapter: object) -> SemanticDiagnosis | None:
    return diagnose_verification_failure(
        adapter,
        request_id="wiring",
        query="Solve for x.",
        candidate_answer="**Answer:** 4",
        verdict="mismatch",
    )


def test_the_diagnosis_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "false")
    monkeypatch.setenv("ANSWER_RECOVERY_ENABLED", "false")
    adapter = _Adapter(SemanticDiagnosis("CANDIDATE", "WRONG_FINAL_ANSWER", True))

    assert _shadow(adapter) is None
    assert adapter.calls == 0


def test_the_diagnosis_runs_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "true")
    adapter = _Adapter(SemanticDiagnosis("QUESTION", "INCOMPLETE_QUESTION", True))

    diagnosis = _shadow(adapter)

    assert adapter.calls == 1
    assert diagnosis is not None and diagnosis.reason_code == "INCOMPLETE_QUESTION"


def test_a_broken_diagnoser_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "true")
    adapter = _Adapter(RuntimeError("diagnoser bug"))

    assert _shadow(adapter) is None


def test_an_adapter_without_a_diagnoser_is_fine(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setenv("ANSWER_DIAGNOSIS_SHADOW_ENABLED", "true")

    assert _shadow(object()) is None


def test_approval_is_never_read_by_the_diagnoser() -> None:
    """The diagnostic call receives the verdict, never the approval decision."""
    verification = CorrectnessVerification(
        status="mismatch", single_defensible_answer=True, reason="differs", method="model"
    )
    _, orchestrator = _diagnose(
        _output('"CANDIDATE"', '"WRONG_FINAL_ANSWER"'), verdict=verification.status
    )

    sent = str(orchestrator.calls[0]["user_content"])
    assert "MISMATCH" in sent
    assert "approved" not in sent.casefold()
