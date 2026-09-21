"""A solvable question gets one fresh candidate before the question is blamed.

An AMBIGUOUS verdict used to terminate the request immediately: the diagnoser said
QUESTION, the policy asked for clarification, and a question family the system answers
successfully on other turns received zero regeneration opportunity. One fresh candidate is
now spent from the same single slot first. Nothing else about the budget changes — candidate
recovery stays capped at one for the whole request.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from observability.llm_usage import record_llm_call
from schemas.llm_usage import ProviderTokenUsage
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.answer_diagnosis import SemanticDiagnosis
from services.doubt_solver.recovery_policy import (
    RecoveryBudgetUse,
    decide_verification_recovery,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)

_QUERY = (
    "An investor puts a sum into a scheme paying 12% per annum compounded annually and "
    "withdraws the interest each year. Find the effective yield after three years."
)
_ANSWER = (
    "**Approach:** Apply the compound interest relation.\n\n"
    "Step 1: The principal earns 12% each year and the interest is withdrawn.\n"
    "Step 2: So the yield stays simple across the three years.\n\n"
    "**Answer:** 36%"
)
_FREE = RecoveryBudgetUse()
_USED = RecoveryBudgetUse(candidate_recovery_used=True)


def _verdict(status: str, reason_code: str = "SEMANTIC_UNCLASSIFIED") -> CorrectnessVerification:
    return CorrectnessVerification(
        status=status,  # type: ignore[arg-type]
        single_defensible_answer=status == "match",
        method="model",
        reason_code="NONE" if status == "match" else reason_code,  # type: ignore[arg-type]
    )


class _Verifier:
    """Returns the scripted verdicts in order and counts real verification calls."""

    def __init__(self, *verdicts: CorrectnessVerification) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def verify(self, **_: object) -> CorrectnessVerification:
        verdict = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return verdict


class _Diagnoser:
    def __init__(self, source: str = "QUESTION", reason: str = "INCOMPLETE_QUESTION") -> None:
        self.calls = 0
        self._source = source
        self._reason = reason

    def diagnose(self, **_: object) -> SemanticDiagnosis:
        self.calls += 1
        return SemanticDiagnosis(self._source, self._reason, True)  # type: ignore[arg-type]


class _Adapter:
    def __init__(self, verifier: _Verifier, diagnoser: _Diagnoser | None = None) -> None:
        self.correctness_verifier = verifier
        self.semantic_diagnoser = diagnoser or _Diagnoser()
        self.generate_calls = 0
        self.recovery_instructions: list[object] = []

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        self.recovery_instructions.append(kwargs.get("recovery_instruction"))
        return _ANSWER


@pytest.fixture
def _recovery_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
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


def _stream(adapter: _Adapter) -> list[object]:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="ambiguous-recovery",
                query=_QUERY,
                language="english",
                classification={
                    "subject": "math",
                    "intent": "solve",
                    "difficulty": "advanced",
                    "need_web_search": False,
                    "classifier_confidence": 0.98,
                    "classification_source": "llm",
                },
                classifier_confidence=0.98,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


def _terminal(events: list[object]) -> object:
    terminals = [index for index, e in enumerate(events) if e.type in ("complete", "error")]
    assert len(terminals) == 1, "exactly one terminal event"
    assert terminals[0] == len(events) - 1, "terminal is last"
    return events[-1]


# --- 1. AMBIGUOUS -> regeneration -> MATCH ---------------------------------------------


def test_ambiguous_then_regeneration_then_match_completes(_recovery_on: None) -> None:
    verifier = _Verifier(_verdict("ambiguous"), _verdict("match"))
    adapter = _Adapter(verifier)

    events = _stream(adapter)

    assert adapter.generate_calls == 2
    assert verifier.calls == 2
    assert adapter.recovery_instructions[1] is not None  # the existing recovery instruction
    assert _terminal(events).type == "complete"


# --- 2. AMBIGUOUS -> regeneration -> AMBIGUOUS -----------------------------------------


def test_a_second_ambiguous_verdict_asks_for_clarification(_recovery_on: None) -> None:
    verifier = _Verifier(_verdict("ambiguous"), _verdict("ambiguous"))
    adapter = _Adapter(verifier)

    events = _stream(adapter)

    assert adapter.generate_calls == 2  # no third generation
    assert verifier.calls == 2
    terminal = _terminal(events)
    assert terminal.type == "error"
    assert terminal.metadata["code"] == "QUESTION_NEEDS_CLARIFICATION"


# --- 3. AMBIGUOUS -> regeneration -> MISMATCH ------------------------------------------


def test_a_mismatch_after_regeneration_fails_safe(_recovery_on: None) -> None:
    verifier = _Verifier(_verdict("ambiguous"), _verdict("mismatch"))
    adapter = _Adapter(verifier)

    events = _stream(adapter)

    assert adapter.generate_calls == 2  # no second regeneration
    assert verifier.calls == 2
    terminal = _terminal(events)
    assert terminal.type == "error"
    assert terminal.metadata["code"] == "ANSWER_VERIFICATION_FAILED"


# --- 6. The existing MISMATCH recovery is untouched -------------------------------------


def test_mismatch_candidate_recovery_is_unchanged(_recovery_on: None) -> None:
    verifier = _Verifier(_verdict("mismatch"), _verdict("match"))
    adapter = _Adapter(verifier, _Diagnoser("CANDIDATE", "WRONG_FINAL_ANSWER"))

    events = _stream(adapter)

    assert adapter.generate_calls == 2
    assert verifier.calls == 2
    assert _terminal(events).type == "complete"


# --- Healthy path costs nothing extra ---------------------------------------------------


def test_a_match_costs_no_extra_call(_recovery_on: None) -> None:
    verifier = _Verifier(_verdict("match"))
    adapter = _Adapter(verifier)

    events = _stream(adapter)

    assert adapter.generate_calls == 1
    assert verifier.calls == 1
    assert adapter.semantic_diagnoser.calls == 0  # an approved verdict pays for no diagnosis
    assert _terminal(events).type == "complete"


# --- Policy matrix (4, 5, 7, 8 and the approved table) ----------------------------------


@pytest.mark.parametrize(
    ("name", "status", "reason", "source", "budget", "eligible", "action", "terminal"),
    [
        ("match", "match", "NONE", "NONE", _FREE, True, "COMPLETE", None),
        (
            "mismatch candidate, slot free",
            "mismatch", "WRONG_FINAL_ANSWER", "CANDIDATE", _FREE, True,
            "REGENERATE_CANDIDATE", None,
        ),
        (
            "mismatch candidate, slot used",
            "mismatch", "WRONG_FINAL_ANSWER", "CANDIDATE", _USED, True,
            "FAIL_SAFE", "ANSWER_VERIFICATION_FAILED",
        ),
        (
            "ambiguous solvable, slot free",
            "ambiguous", "INCOMPLETE_QUESTION", "QUESTION", _FREE, True,
            "REGENERATE_CANDIDATE", None,
        ),
        (
            "ambiguous solvable, slot used",
            "ambiguous", "INCOMPLETE_QUESTION", "QUESTION", _USED, True,
            "ASK_CLARIFICATION", "QUESTION_NEEDS_CLARIFICATION",
        ),
        (
            "ambiguous multiple answers, slot free",
            "ambiguous", "MULTIPLE_DEFENSIBLE_ANSWERS", "QUESTION", _FREE, True,
            "REGENERATE_CANDIDATE", None,
        ),
        (
            "ambiguous proven incomplete",
            "ambiguous", "INCOMPLETE_QUESTION", "QUESTION", _FREE, False,
            "ASK_CLARIFICATION", "QUESTION_NEEDS_CLARIFICATION",
        ),
        (
            "mismatch blamed on the question",
            "mismatch", "INCOMPLETE_QUESTION", "QUESTION", _FREE, True,
            "ASK_CLARIFICATION", "QUESTION_NEEDS_CLARIFICATION",
        ),
        (
            "technical parse failure",
            "unavailable", "PARSE_FAILURE", None, _FREE, True, "RETRY_SAME_NODE", None,
        ),
        (
            "technical, unretryable",
            "unavailable", "INPUT_TOO_LARGE", None, _FREE, True,
            "FAIL_TEMPORARY", "ANSWER_VERIFICATION_UNAVAILABLE",
        ),
        (
            "provider outage",
            "unavailable", "PROVIDER_FAILURE", None, _FREE, True,
            "FAIL_TEMPORARY", "SERVICE_TEMPORARILY_UNAVAILABLE",
        ),
    ],
)
def test_the_approved_policy_matrix(
    name: str,
    status: str,
    reason: str,
    source: str | None,
    budget: RecoveryBudgetUse,
    eligible: bool,
    action: str,
    terminal: str | None,
) -> None:
    decision = decide_verification_recovery(
        approved=status == "match",
        reason_code=reason,
        budget=budget,
        failure_source=source,
        verifier_status=status,
        question_eligible_to_solve=eligible,
    )

    assert decision.action == action, name
    assert decision.terminal_code == terminal, name


def test_a_caller_that_cannot_vouch_for_the_question_keeps_the_old_behaviour() -> None:
    """The default is conservative: no caller silently gains regeneration."""
    decision = decide_verification_recovery(
        approved=False,
        reason_code="INCOMPLETE_QUESTION",
        budget=_FREE,
        failure_source="QUESTION",
        verifier_status="ambiguous",
    )

    assert decision.action == "ASK_CLARIFICATION"


# --- 4 & 5. The deterministic question gates still run before any candidate exists -------


def test_a_question_needing_context_clarifies_before_any_candidate_is_generated(
    _recovery_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No candidate, so no AMBIGUOUS verdict and no regeneration can arise from this turn.

    Mandatory case 4 — a question the integrity normalizer flags as missing information or
    ambiguous — is covered upstream of this service by
    `test_question_integrity.py::test_flagged_question_without_a_trusted_repair_stops_before_any_solver_call`,
    which asserts the request never reaches a solver call at all.
    """

    class _Selected:
        clarification_required = True
        relation = "FOLLOW_UP"
        requested_action = "ANSWER_WITH_CONTEXT"
        selected_turn_id = None
        context_policy = "current_only"
        context_characters = 0
        conversation_context = ""
        resolved_query = _QUERY

    class _Understanding:
        def understand(self, **_: object) -> object:
            from schemas.conversation import (
                ContextNeedAssessment,
                ConversationPreparation,
                RecentContextLoadResult,
            )

            return ConversationPreparation(
                gate=ContextNeedAssessment(decision="UNCERTAIN"),
                context_load=RecentContextLoadResult(),
            )

    monkeypatch.setattr(
        streaming_module, "build_selected_generation_context", lambda **_: _Selected()
    )
    monkeypatch.setattr(
        streaming_module,
        "orchestrated_classify_query_stage",
        lambda **_: _classification_stage(),
    )
    verifier = _Verifier(_verdict("ambiguous"))
    adapter = _Adapter(verifier)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="ambiguous-context",
                query=_QUERY,
                language="english",
                actor_id="student",
                conversation_id="conv",
            ),
            adapter=adapter,  # type: ignore[arg-type]
            conversation_understanding=_Understanding(),  # type: ignore[arg-type]
        )
    )

    assert adapter.generate_calls == 0
    assert verifier.calls == 0
    assert _terminal(events).type in ("error", "complete")


def _classification_stage():  # noqa: ANN202
    from schemas.doubt_solver import QueryClassification
    from services.classification.contracts import ClassificationStageResult

    raw = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.98,
        difficulty="advanced",
        classification_source="llm",
        relation="FOLLOW_UP",
        requested_action="ANSWER_WITH_CONTEXT",
        requires_recent_conversation=True,
    )
    return ClassificationStageResult(
        raw=raw,
        classification={
            "subject": "math",
            "intent": "solve",
            "difficulty": "advanced",
            "need_web_search": False,
            "classifier_confidence": 0.98,
            "classification_source": "llm",
        },
        modality="text",
        status="accepted",
        classifier_confidence=0.98,
        classifier_fallback=False,
    )


# --- Image questions never passed the text gates, so they keep clarifying ---------------


@pytest.mark.parametrize(
    ("modality", "uncertain"), [("image", False), ("image", True), ("text", True)]
)
def test_a_question_that_skipped_the_text_gates_is_not_regenerated(
    _recovery_on: None, modality: str, uncertain: bool
) -> None:
    """The integrity normalizer and context gate run only for confidently read text."""
    verifier = _Verifier(_verdict("ambiguous"), _verdict("match"))
    adapter = _Adapter(verifier)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="ambiguous-image",
                query=_QUERY,
                language="english",
                source_modality=modality,  # type: ignore[arg-type]
                image_uncertain=uncertain,
                classification={
                    "subject": "math",
                    "intent": "solve",
                    "difficulty": "advanced",
                    "need_web_search": False,
                    "classifier_confidence": 0.98,
                    "classification_source": "llm",
                },
                classifier_confidence=0.98,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )

    assert adapter.generate_calls == 1  # no regeneration
    assert verifier.calls == 1
    terminal = _terminal(events)
    assert terminal.type == "error"
    assert terminal.metadata["code"] == "QUESTION_NEEDS_CLARIFICATION"


# --- The candidate cap does not depend on a telemetry record surviving ------------------


def test_a_spent_structural_repair_is_known_to_the_ledger_without_its_usage_record(
    _recovery_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the repair's usage record is lost, the slot must still read as spent."""
    from services.doubt_solver import recovery_policy

    monkeypatch.setattr(
        recovery_policy.RecoveryBudgetUse,
        "from_request_usage",
        classmethod(lambda cls: cls()),  # telemetry says nothing was ever spent
    )
    malformed = "Step 1: work \\( unclosed\n\n**Answer:** 36%"
    verifier = _Verifier(_verdict("ambiguous"), _verdict("ambiguous"))

    class _RepairingAdapter(_Adapter):
        def generate(self, **kwargs: object) -> str:
            self.generate_calls += 1
            self.recovery_instructions.append(kwargs.get("recovery_instruction"))
            return malformed if self.generate_calls == 1 else _ANSWER

    adapter = _RepairingAdapter(verifier)

    _stream(adapter)

    # One primary draft, one structural repair, and no third candidate from the ambiguity.
    assert adapter.generate_calls == 2
    assert all(instruction is None for instruction in adapter.recovery_instructions)


# --- A repeated ambiguous verdict is read with the diagnosis that earned the candidate ---


@pytest.mark.parametrize(
    ("verdicts", "diagnosis", "terminal"),
    [
        # A verifier fault stays a verifier fault: it does not become the question's fault.
        (("ambiguous", "ambiguous"), ("VERIFIER", "VERIFIER_LOW_CONFIDENCE"),
         "ANSWER_VERIFICATION_FAILED"),
        # The existing mismatch recovery keeps its own terminal whatever follows it.
        (("mismatch", "ambiguous"), ("CANDIDATE", "WRONG_FINAL_ANSWER"),
         "ANSWER_VERIFICATION_FAILED"),
        (("mismatch", "mismatch"), ("CANDIDATE", "WRONG_FINAL_ANSWER"),
         "ANSWER_VERIFICATION_FAILED"),
    ],
)
def test_the_post_regeneration_terminal_follows_what_earned_the_regeneration(
    _recovery_on: None,
    verdicts: tuple[str, str],
    diagnosis: tuple[str, str],
    terminal: str,
) -> None:
    verifier = _Verifier(*(_verdict(v) for v in verdicts))
    adapter = _Adapter(verifier, _Diagnoser(*diagnosis))

    events = _stream(adapter)

    assert adapter.generate_calls == 2
    assert verifier.calls == 2
    assert adapter.semantic_diagnoser.calls == 1  # never a second paid diagnosis
    last = _terminal(events)
    assert last.type == "error"
    assert last.metadata["code"] == terminal


@pytest.mark.parametrize("multiple", [False, True])
def test_the_clarification_keeps_the_wording_its_diagnosis_chose(
    _recovery_on: None, multiple: bool
) -> None:
    from services.doubt_solver.question_integrity import ambiguous_question_message

    reason = "MULTIPLE_DEFENSIBLE_ANSWERS" if multiple else "INCOMPLETE_QUESTION"
    verifier = _Verifier(_verdict("ambiguous"), _verdict("ambiguous"))
    adapter = _Adapter(verifier, _Diagnoser("QUESTION", reason))

    events = _stream(adapter)

    last = _terminal(events)
    assert last.metadata["code"] == "QUESTION_NEEDS_CLARIFICATION"
    assert last.label == ambiguous_question_message("english", multiple_answers=multiple)


# --- Recovery authority is independent of usage/telemetry records -----------------------


class _StructurallyRepairedAdapter(_Adapter):
    """The first draft is malformed, so the existing structural repair runs before the loop.

    Each generation writes its usage record the way the orchestrator does in production — the
    draft as "primary", the structural repair as "repair" — so a test can remove that record
    deliberately rather than never having written it.
    """

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        self.recovery_instructions.append(kwargs.get("recovery_instruction"))
        record_llm_call(
            request_id=str(kwargs["request_id"]),
            role="math.generator.advanced",
            provider="mock",
            model="mock",
            deployment=None,
            attempt_type="primary" if self.generate_calls == 1 else "repair",
            streaming=False,
            usage=ProviderTokenUsage(),
            duration_ms=1,
            status="succeeded",
        )
        if self.generate_calls == 1:
            return "Step 1: work \\( unclosed\n\n**Answer:** 36%"
        return _ANSWER


@pytest.mark.parametrize("record_lost", [False, True])
def test_a_spent_structural_repair_blocks_mismatch_regeneration_whatever_telemetry_says(
    _recovery_on: None, monkeypatch: pytest.MonkeyPatch, record_lost: bool
) -> None:
    """The candidate slot is spent by the structural repair itself, not by its usage record.

    With the record written, the ledger already saw the repair through it: this case is the
    pre-existing behaviour and passes against the pre-change controller too. With the record
    lost, the previous code read the slot as free and regenerated a second time — a latent
    violation of the one-candidate-recovery invariant. Both now end the same way.
    """
    if record_lost:
        from services.doubt_solver import recovery_policy

        monkeypatch.setattr(
            recovery_policy.RecoveryBudgetUse,
            "from_request_usage",
            classmethod(lambda cls: cls()),
        )
    verifier = _Verifier(_verdict("mismatch"), _verdict("match"))
    adapter = _StructurallyRepairedAdapter(
        verifier, _Diagnoser("CANDIDATE", "WRONG_FINAL_ANSWER")
    )

    events = _stream(adapter)

    assert adapter.generate_calls == 2  # the draft and its structural repair, nothing more
    assert all(instruction is None for instruction in adapter.recovery_instructions)
    assert verifier.calls == 1
    last = _terminal(events)
    assert last.type == "error"
    assert last.metadata["code"] == "ANSWER_VERIFICATION_FAILED"
