"""Phase 0 of verification recovery: typed diagnosis and shadow recovery decisions.

The verifier result now carries a runtime-derived reason code, failure kind and failure
source, and a deterministic controller records the action it would take. Nothing acts
on that decision yet: every request must end exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

import config as cfg_module
import services.doubt_solver.recovery_policy as recovery_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
from observability.llm_usage import (
    begin_llm_usage_collection,
    record_llm_call,
    reset_llm_usage_collection,
)
from schemas.llm_usage import ProviderTokenUsage
from services.doubt_solver.answer_correctness import (
    AnswerCorrectnessVerifier,
    CorrectnessVerification,
    deterministic_verify,
)
from services.doubt_solver.answer_quality import GENERATION_FAILURE_MESSAGE
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.recovery_policy import (
    RecoveryBudgetUse,
    decide_generation_recovery,
    decide_verification_recovery,
    quality_reason_code,
    technical_reason_code,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.errors import (
    LlmExecutionError,
    LlmRouteNotFoundError,
    PromptNotFoundError,
    ProviderExecutionError,
)

_FRESH = RecoveryBudgetUse()

# ---------------------------------------------------------------------------
# Verification decision matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason_code", "budget", "action", "kind", "source", "terminal"),
    [
        ("WRONG_FINAL_ANSWER", _FRESH, "REGENERATE_CANDIDATE", "SEMANTIC", "CANDIDATE", None),
        ("CANDIDATE_CONFLICT", _FRESH, "REGENERATE_CANDIDATE", "SEMANTIC", "CANDIDATE", None),
        (
            "WRONG_FINAL_ANSWER",
            RecoveryBudgetUse(candidate_recovery_used=True),
            "FAIL_SAFE",
            "SEMANTIC",
            "CANDIDATE",
            "ANSWER_VERIFICATION_FAILED",
        ),
        (
            "INCOMPLETE_QUESTION",
            _FRESH,
            "ASK_CLARIFICATION",
            "SEMANTIC",
            "QUESTION",
            "QUESTION_NEEDS_CLARIFICATION",
        ),
        (
            "MULTIPLE_DEFENSIBLE_ANSWERS",
            _FRESH,
            "ASK_CLARIFICATION",
            "SEMANTIC",
            "QUESTION",
            "QUESTION_NEEDS_CLARIFICATION",
        ),
        (
            "VERIFIER_LOW_CONFIDENCE",
            _FRESH,
            "FAIL_SAFE",
            "SEMANTIC",
            "VERIFIER",
            "ANSWER_VERIFICATION_FAILED",
        ),
        (
            "VERIFIER_SELF_CONTRADICTION",
            _FRESH,
            "FAIL_SAFE",
            "SEMANTIC",
            "VERIFIER",
            "ANSWER_VERIFICATION_FAILED",
        ),
        (
            "SEMANTIC_UNCLASSIFIED",
            _FRESH,
            "FAIL_SAFE",
            "SEMANTIC",
            "UNKNOWN",
            "ANSWER_VERIFICATION_FAILED",
        ),
        ("PARSE_FAILURE", _FRESH, "RETRY_SAME_NODE", "STRUCTURAL", "VERIFIER", None),
        ("SCHEMA_FAILURE", _FRESH, "RETRY_SAME_NODE", "STRUCTURAL", "VERIFIER", None),
        (
            "SCHEMA_FAILURE",
            RecoveryBudgetUse(verifier_technical_retry_used=True),
            "FAIL_TEMPORARY",
            "STRUCTURAL",
            "VERIFIER",
            "ANSWER_VERIFICATION_UNAVAILABLE",
        ),
        ("TECHNICAL_UNCLASSIFIED", _FRESH, "RETRY_SAME_NODE", "UNKNOWN", "VERIFIER", None),
        (
            "TECHNICAL_UNCLASSIFIED",
            RecoveryBudgetUse(verifier_technical_retry_used=True),
            "FAIL_TEMPORARY",
            "UNKNOWN",
            "VERIFIER",
            "ANSWER_VERIFICATION_UNAVAILABLE",
        ),
        (
            "CONFIGURATION_FAILURE",
            _FRESH,
            "FAIL_TEMPORARY",
            "UNKNOWN",
            "VERIFIER",
            "ANSWER_VERIFICATION_UNAVAILABLE",
        ),
        (
            "TIMEOUT",
            _FRESH,
            "FAIL_TEMPORARY",
            "INFRASTRUCTURE",
            "VERIFIER",
            "SERVICE_TEMPORARILY_UNAVAILABLE",
        ),
        (
            "PROVIDER_FAILURE",
            _FRESH,
            "FAIL_TEMPORARY",
            "INFRASTRUCTURE",
            "VERIFIER",
            "SERVICE_TEMPORARILY_UNAVAILABLE",
        ),
        # A code outside the contract is never trusted: fail safe, no retry, no regeneration.
        (
            "NOT_A_REAL_CODE",
            _FRESH,
            "FAIL_SAFE",
            "UNKNOWN",
            "UNKNOWN",
            "ANSWER_VERIFICATION_FAILED",
        ),
    ],
)
def test_verification_decision_matrix(
    reason_code: str,
    budget: RecoveryBudgetUse,
    action: str,
    kind: str,
    source: str,
    terminal: str | None,
) -> None:
    decision = decide_verification_recovery(approved=False, reason_code=reason_code, budget=budget)

    assert (decision.action, decision.failure_kind, decision.failure_source) == (
        action,
        kind,
        source,
    )
    assert decision.terminal_code == terminal


def test_approved_verification_completes() -> None:
    decision = decide_verification_recovery(approved=True, reason_code="NONE", budget=_FRESH)

    assert decision.action == "COMPLETE"
    assert decision.terminal_code is None


def test_exhausted_budgets_never_loop() -> None:
    spent = RecoveryBudgetUse(
        continuation_used=True,
        candidate_recovery_used=True,
        verifier_technical_retry_used=True,
    )
    for reason_code in recovery_module._VERIFICATION_DIAGNOSIS:
        action = decide_verification_recovery(
            approved=False, reason_code=reason_code, budget=spent
        ).action
        assert action not in {"RETRY_SAME_NODE", "REGENERATE_CANDIDATE", "REPAIR_CANDIDATE"}
    for reason_code in recovery_module._GENERATION_DIAGNOSIS:
        action = decide_generation_recovery(reason_code=reason_code, budget=spent).action
        assert action not in {"RETRY_SAME_NODE", "REGENERATE_CANDIDATE", "REPAIR_CANDIDATE"}


def test_only_a_candidate_repair_consumes_the_single_candidate_slot() -> None:
    """A presentation rewrite is bounded elsewhere and must leave the slot free."""
    token = begin_llm_usage_collection()
    try:
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
        after_rewrite = RecoveryBudgetUse.from_request_usage()
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
        after_repair = RecoveryBudgetUse.from_request_usage()
    finally:
        reset_llm_usage_collection(token)

    assert (after_rewrite.continuation_used, after_rewrite.candidate_recovery_used) == (
        True,
        False,
    )
    assert after_repair.candidate_recovery_used is True


def test_an_attempt_that_fell_back_to_another_model_still_counts() -> None:
    """The executor suffixes the attempt type when it uses a fallback model."""
    token = begin_llm_usage_collection()
    try:
        for attempt in ("primary", "continuation_fallback", "repair_fallback"):
            record_llm_call(
                request_id="budget-fallback",
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
        budget = RecoveryBudgetUse.from_request_usage()
    finally:
        reset_llm_usage_collection(token)

    assert (budget.continuation_used, budget.candidate_recovery_used) == (True, True)


def test_continuation_does_not_consume_generation_recovery() -> None:
    decision = decide_generation_recovery(
        reason_code="TRUNCATED_QUALITY_FAILED",
        budget=RecoveryBudgetUse(continuation_used=True),
    )

    assert decision.action == "REGENERATE_CANDIDATE"


# ---------------------------------------------------------------------------
# Generation decision matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason_code", "budget", "action", "terminal"),
    [
        ("QUALITY_FAILED", _FRESH, "REPAIR_CANDIDATE", None),
        ("TRUNCATED_QUALITY_FAILED", _FRESH, "REGENERATE_CANDIDATE", None),
        (
            "QUALITY_FAILED",
            RecoveryBudgetUse(candidate_recovery_used=True),
            "FAIL_SAFE",
            "ANSWER_QUALITY_FAILED",
        ),
        ("TIMEOUT", _FRESH, "FAIL_TEMPORARY", "SERVICE_TEMPORARILY_UNAVAILABLE"),
        ("PROVIDER_FAILURE", _FRESH, "FAIL_TEMPORARY", "SERVICE_TEMPORARILY_UNAVAILABLE"),
        ("TECHNICAL_UNCLASSIFIED", _FRESH, "FAIL_TEMPORARY", "ANSWER_PROVIDER_FAILED"),
        ("CONFIGURATION_FAILURE", _FRESH, "FAIL_TEMPORARY", "ANSWER_PROVIDER_FAILED"),
    ],
)
def test_generation_decision_matrix(
    reason_code: str, budget: RecoveryBudgetUse, action: str, terminal: str | None
) -> None:
    decision = decide_generation_recovery(reason_code=reason_code, budget=budget)

    assert decision.action == action
    assert decision.terminal_code == terminal


@pytest.mark.parametrize(
    ("reason_codes", "continuation_used", "expected"),
    [
        (["missing_final_answer"], False, "TRUNCATED_QUALITY_FAILED"),
        (["math_unbalanced_inline"], False, "TRUNCATED_QUALITY_FAILED"),
        (["too_many_display_math_blocks"], True, "TRUNCATED_QUALITY_FAILED"),
        (["conflicting_answer_values"], False, "QUALITY_FAILED"),
    ],
)
def test_quality_reason_code(
    reason_codes: list[str], continuation_used: bool, expected: str
) -> None:
    assert quality_reason_code(reason_codes, continuation_used=continuation_used) == expected


def _wrapped(cause: Exception, **kwargs: object) -> ProviderExecutionError:
    try:
        raise cause
    except Exception as exc:  # noqa: BLE001
        try:
            raise ProviderExecutionError("executor failed", **kwargs) from exc  # type: ignore[arg-type]
        except ProviderExecutionError as wrapped:
            return wrapped


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderExecutionError("x", failure_kind="timeout"), "TIMEOUT"),
        (ProviderExecutionError("x", failure_kind="provider_unavailable"), "PROVIDER_FAILURE"),
        (ProviderExecutionError("x", failure_kind="rate_limited"), "PROVIDER_FAILURE"),
        (ProviderExecutionError("x", failure_kind="model_not_found"), "CONFIGURATION_FAILURE"),
        (ProviderExecutionError("x", failure_kind="invalid_request"), "TECHNICAL_UNCLASSIFIED"),
        (ProviderExecutionError("x", failure_kind="safety_blocked"), "TECHNICAL_UNCLASSIFIED"),
        # The executor stamps unknown_provider_error on any unexpected exception.
        (_wrapped(TimeoutError("slow")), "TIMEOUT"),
        (_wrapped(TypeError("code bug")), "TECHNICAL_UNCLASSIFIED"),
        (TimeoutError("slow"), "TIMEOUT"),
        (LlmRouteNotFoundError("missing route"), "CONFIGURATION_FAILURE"),
        (PromptNotFoundError("missing prompt"), "CONFIGURATION_FAILURE"),
        (RuntimeError("unexpected"), "TECHNICAL_UNCLASSIFIED"),
    ],
)
def test_technical_reason_code(error: Exception, expected: str) -> None:
    assert technical_reason_code(error) == expected


def test_technical_reason_code_reads_the_wrapped_cause() -> None:
    try:
        try:
            raise ProviderExecutionError("x", failure_kind="timeout")
        except ProviderExecutionError as cause:
            raise LlmExecutionError("wrapped") from cause
    except LlmExecutionError as wrapped:
        assert technical_reason_code(wrapped) == "TIMEOUT"


def test_an_error_raised_while_handling_a_timeout_is_not_a_timeout() -> None:
    try:
        try:
            raise TimeoutError("slow")
        except TimeoutError:
            raise KeyError("unrelated bug")  # noqa: B904 - implicit context on purpose
    except KeyError as unrelated:
        assert technical_reason_code(unrelated) == "TECHNICAL_UNCLASSIFIED"


# ---------------------------------------------------------------------------
# Verifier envelope: every field is runtime-derived
# ---------------------------------------------------------------------------


class _Orchestrator:
    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self._content = content
        self._error = error

    def generate(self, **_: object) -> object:
        if self._error is not None:
            raise self._error
        return type("Result", (), {"content": self._content})()


def _verify(content: str = "", error: Exception | None = None, *, query: str = "Solve it."):
    return AnswerCorrectnessVerifier(orchestrator=_Orchestrator(content, error)).verify(  # type: ignore[arg-type]
        request_id="phase0",
        query=query,
        candidate_answer="**Answer:** Option A",
        subject="math",
        difficulty="intermediate",
        language="english",
    )


def _verdict(status: str, single: str = "true", answer: str = "Option A") -> str:
    return (
        f'{{"status":"{status}","independent_answer":"{answer}",'
        f'"single_defensible_answer":{single},"reason":"r"}}'
    )


@pytest.mark.parametrize(
    ("content", "error", "status", "reason_code", "kind", "source"),
    [
        (_verdict("MATCH"), None, "match", "NONE", "NONE", "NONE"),
        (_verdict("MISMATCH"), None, "mismatch", "SEMANTIC_UNCLASSIFIED", "SEMANTIC", "UNKNOWN"),
        (
            _verdict("AMBIGUOUS", "false"),
            None,
            "ambiguous",
            "SEMANTIC_UNCLASSIFIED",
            "SEMANTIC",
            "UNKNOWN",
        ),
        (_verdict("MATCH", "false"), None, "match", "SEMANTIC_UNCLASSIFIED", "SEMANTIC", "UNKNOWN"),
        ("not json", None, "unavailable", "PARSE_FAILURE", "STRUCTURAL", "VERIFIER"),
        ('{"status":"MATCH"}', None, "unavailable", "SCHEMA_FAILURE", "STRUCTURAL", "VERIFIER"),
        (
            "",
            ProviderExecutionError("x", failure_kind="timeout"),
            "unavailable",
            "TIMEOUT",
            "INFRASTRUCTURE",
            "VERIFIER",
        ),
        (
            "",
            ProviderExecutionError("x", failure_kind="rate_limited"),
            "unavailable",
            "PROVIDER_FAILURE",
            "INFRASTRUCTURE",
            "VERIFIER",
        ),
        ("", RuntimeError("boom"), "unavailable", "TECHNICAL_UNCLASSIFIED", "UNKNOWN", "VERIFIER"),
    ],
)
def test_verifier_result_carries_a_runtime_diagnosis(
    content: str,
    error: Exception | None,
    status: str,
    reason_code: str,
    kind: str,
    source: str,
) -> None:
    result = _verify(content, error)

    assert result.status == status
    assert (result.reason_code, result.failure_kind, result.failure_source) == (
        reason_code,
        kind,
        source,
    )


def test_self_contradiction_is_attributed_to_the_verifier() -> None:
    content = (
        '{"status":"MATCH","independent_answer":"12","single_defensible_answer":true,'
        '"reason":"agrees"}'
    )
    result = AnswerCorrectnessVerifier(orchestrator=_Orchestrator(content)).verify(  # type: ignore[arg-type]
        request_id="phase0",
        query="Find x.",
        candidate_answer="**Answer:** 15",
        subject="math",
        difficulty="intermediate",
        language="english",
    )

    assert (result.status, result.reason) == ("mismatch", "verifier_self_contradiction")
    assert (result.reason_code, result.failure_source) == (
        "VERIFIER_SELF_CONTRADICTION",
        "VERIFIER",
    )


def test_deterministic_recomputation_names_the_wrong_candidate() -> None:
    wrong = deterministic_verify(query="What is 20% of 50?", candidate_answer="**Answer:** 12")
    right = deterministic_verify(query="What is 20% of 50?", candidate_answer="**Answer:** 10")

    assert wrong is not None and right is not None
    assert (wrong.reason_code, wrong.failure_source) == ("WRONG_FINAL_ANSWER", "CANDIDATE")
    assert (right.reason_code, right.approved) == ("NONE", True)


def test_verification_event_logs_codes_but_never_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        "services.doubt_solver.answer_correctness.log_event",
        lambda _name, **kwargs: captured.append(dict(kwargs.get("details") or {})),
    )

    _verify(
        '{"status":"MISMATCH","independent_answer":"B","single_defensible_answer":true,'
        '"reason":"SECRET-FREE-TEXT"}'
    )

    details = captured[-1]
    assert details["diagnosis_reason_code"] == "SEMANTIC_UNCLASSIFIED"
    assert details["diagnosis_failure_kind"] == "SEMANTIC"
    assert details["diagnosis_failure_source"] == "UNKNOWN"
    assert "SECRET-FREE-TEXT" not in repr(details)


# ---------------------------------------------------------------------------
# Streaming path: shadow decisions change nothing
# ---------------------------------------------------------------------------

_QUERY = "Solve for x: 2x + 3 = 11."
_ANSWER = "2x + 3 = 11\n\n2x = 8\n\nx = 4\n\n**Answer:** 4"
_TRUNCATED = "2x = 8 so \\( x = 8 \\div 2"
_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "intermediate",
    "need_web_search": False,
    "classifier_confidence": 0.99,
    "classification_source": "llm",
}
_MISMATCH = CorrectnessVerification(
    status="mismatch",
    independent_answer="5",
    single_defensible_answer=True,
    reason="differs",
    method="model",
    reason_code="SEMANTIC_UNCLASSIFIED",
)


class _Verifier:
    def __init__(self, verdict: CorrectnessVerification) -> None:
        self.verdict = verdict
        self.calls = 0

    def verify(self, **_: object) -> CorrectnessVerification:
        self.calls += 1
        return self.verdict


class _Adapter:
    def __init__(
        self,
        content: str,
        verdict: CorrectnessVerification,
        *,
        attempts: tuple[str, ...] = ("primary",),
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._attempts = attempts
        self._error = error
        self.generate_calls = 0
        self.correctness_verifier = _Verifier(verdict)

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        if self._error is not None:
            raise self._error
        for attempt in self._attempts:
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
        return self._content


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
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
def shadow_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str | None, dict]]:
    events: list[tuple[str | None, dict]] = []
    original = recovery_module.log_event

    def _record(name: str, **kwargs: object) -> None:
        if name == "RECOVERY_DECISION_SHADOW":
            events.append((kwargs.get("stage"), dict(kwargs.get("details") or {})))  # type: ignore[arg-type]
        original(name, **kwargs)

    monkeypatch.setattr(recovery_module, "log_event", _record)
    return events


def _stream(adapter: _Adapter) -> list:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="phase0-stream",
                query=_QUERY,
                language="english",
                classification=dict(_CLASSIFICATION),
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


def _terminal(events: list) -> tuple[str, dict | None]:
    terminal = events[-1]
    return terminal.type, terminal.metadata


def test_semantic_mismatch_is_shadowed_without_changing_the_terminal(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    adapter = _Adapter(_ANSWER, _MISMATCH)

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {"retryable": False, "code": "ANSWER_VERIFICATION_FAILED", "user_retryable": True},
    )
    assert (adapter.generate_calls, adapter.correctness_verifier.calls) == (1, 1)
    assert len(shadow_events) == 1
    node, details = shadow_events[0]
    assert node == "verification"
    assert details["action"] == "FAIL_SAFE"
    assert details["reason_code"] == "SEMANTIC_UNCLASSIFIED"
    assert details["actual_terminal_code"] == "ANSWER_VERIFICATION_FAILED"


def test_wrong_candidate_regeneration_is_only_recorded(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    adapter = _Adapter(_ANSWER, replace(_MISMATCH, reason_code="WRONG_FINAL_ANSWER"))

    events = _stream(adapter)

    assert _terminal(events)[1]["code"] == "ANSWER_VERIFICATION_FAILED"  # type: ignore[index]
    assert adapter.generate_calls == 1  # the shadow REGENERATE_CANDIDATE was not executed
    assert shadow_events[0][1]["action"] == "REGENERATE_CANDIDATE"


def test_verifier_timeout_keeps_its_existing_terminal(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    timeout = CorrectnessVerification(
        status="unavailable", reason="ANSWER_VERIFICATION_UNAVAILABLE", reason_code="TIMEOUT"
    )

    events = _stream(_Adapter(_ANSWER, timeout))

    assert _terminal(events) == (
        "error",
        {"retryable": False, "code": "ANSWER_VERIFICATION_UNAVAILABLE"},
    )
    details = shadow_events[0][1]
    assert details["action"] == "FAIL_TEMPORARY"
    assert details["shadow_terminal_code"] == "SERVICE_TEMPORARILY_UNAVAILABLE"


def test_truncated_draft_after_continuation_shadows_one_regeneration(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    """The advanced-Math shape: continuation used the shared budget, so no repair ran."""
    adapter = _Adapter(_TRUNCATED, _MISMATCH, attempts=("primary", "continuation"))

    events = _stream(adapter)

    assert _terminal(events)[1]["code"] == "ANSWER_QUALITY_FAILED"  # type: ignore[index]
    assert adapter.generate_calls == 1
    assert adapter.correctness_verifier.calls == 0
    node, details = shadow_events[0]
    assert node == "generation"
    assert details["reason_code"] == "TRUNCATED_QUALITY_FAILED"
    assert details["action"] == "REGENERATE_CANDIDATE"
    assert details["continuation_used"] is True
    assert details["candidate_recovery_used"] is False


def test_generator_provider_failure_keeps_its_existing_terminal(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    adapter = _Adapter(
        _ANSWER,
        _MISMATCH,
        error=ProviderExecutionError("down", failure_kind="provider_unavailable"),
    )

    events = _stream(adapter)

    assert _terminal(events) == (
        "error",
        {"retryable": True, "code": "ANSWER_PROVIDER_FAILED", "user_retryable": True},
    )
    details = shadow_events[0][1]
    assert (details["reason_code"], details["action"]) == ("PROVIDER_FAILURE", "FAIL_TEMPORARY")


def test_approved_answer_records_no_shadow_decision(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    match = CorrectnessVerification(
        status="match",
        independent_answer="4",
        single_defensible_answer=True,
        reason="agrees",
        method="model",
        reason_code="NONE",
    )

    events = _stream(_Adapter(_ANSWER, match))

    assert events[-1].type == "complete"
    assert shadow_events == []


def test_a_failing_shadow_decision_never_changes_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _stream(_Adapter(_ANSWER, _MISMATCH))

    def _explode(**_: object) -> None:
        raise RuntimeError("shadow bug")

    monkeypatch.setattr(recovery_module, "decide_verification_recovery", _explode)
    shadowed = _stream(_Adapter(_ANSWER, _MISMATCH))

    assert [(event.type, event.metadata) for event in shadowed if event.type == "error"] == [
        (event.type, event.metadata) for event in baseline if event.type == "error"
    ]


# ---------------------------------------------------------------------------
# Non-streaming graph path
# ---------------------------------------------------------------------------


class _GraphAdapter:
    def __init__(self, verdict: CorrectnessVerification) -> None:
        self.correctness_verifier = _Verifier(verdict)

    def generate_final(self, **kwargs: object):  # noqa: ANN201
        record_llm_call(
            request_id=str(kwargs["request_id"]),
            role="math.generator.intermediate",
            provider="mock",
            model="mock",
            deployment=None,
            attempt_type="primary",
            streaming=False,
            usage=ProviderTokenUsage(),
            duration_ms=1,
            status="succeeded",
        )
        return build_final_answer_result(
            content=_ANSWER, language="english", quality_status="passed_quality_gate"
        )


def _graph_state() -> dict:
    return {
        "request_id": "phase0-graph",
        "actor_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": _QUERY,
        "original_query": _QUERY,
        "language": "english",
        "exam_id": "CAT",
        "exam_stage": None,
        "classification": {**_CLASSIFICATION, "topic": "linear", "retrieval_required": False},
        "query_classification": None,
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
        "final_answer": None,
        "conversation_context": "",
        "conversation_relation": None,
        "conversation_preparation": None,
        "source_modality": "text",
    }


def test_graph_verification_failure_is_shadowed_without_changing_the_result(
    shadow_events: list[tuple[str | None, dict]],
) -> None:
    adapter = _GraphAdapter(_MISMATCH)
    token = begin_llm_usage_collection()
    try:
        result = build_orchestrated_doubt_solver_graph(adapter).invoke(_graph_state())
    finally:
        reset_llm_usage_collection(token)

    assert adapter.correctness_verifier.calls == 1
    assert result["answer"] == GENERATION_FAILURE_MESSAGE
    assert result["final_answer"]["quality_status"] == "failed_quality_gate"
    assert [details["action"] for _, details in shadow_events] == ["FAIL_SAFE"]
