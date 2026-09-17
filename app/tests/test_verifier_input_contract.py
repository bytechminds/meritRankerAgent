"""The verifier's composed input must satisfy the orchestrator's query contract.

A long candidate used to pass generation and quality and then be refused locally, before
any provider call, because the verifier composed up to ~12,100 characters while
`LlmOrchestrator.generate` accepts 4,000. Compaction is deterministic, uses no model, and
never changes the answer the student sees.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from observability.llm_usage import record_llm_call
from schemas.llm_usage import ProviderTokenUsage
from services.doubt_solver.answer_correctness import (
    AnswerCorrectnessVerifier,
    build_verifier_input,
)
from services.doubt_solver.answer_diagnosis import SemanticDiagnosis
from services.doubt_solver.recovery_policy import (
    RecoveryBudgetUse,
    decide_verification_recovery,
    verification_diagnosis,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.orchestrator import MAX_QUERY_CHARS

_QUESTION = (
    "A and B are two diametrically opposite points on a circular track. Two friends start "
    "together from A and B respectively and run in opposite directions. Find the number of "
    "distinct meeting points on the track."
)
_ANSWER_LINE = "**Answer:** 7 distinct meeting points"
_HEAD_MARKER = "Step 1: let the speeds be in the ratio 4:3."


def _candidate(length: int) -> str:
    """A realistic long solution: setup at the top, final answer at the very end."""
    filler = "Step: the runners meet after covering a combined distance of one lap. "
    body = (filler * (length // len(filler) + 1))[:length]
    return f"{_HEAD_MARKER}\n{body}\n\n{_ANSWER_LINE}"


class _Orchestrator:
    """Records what reached the provider boundary, and enforces the real query contract."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[str] = []

    def generate(self, **kwargs: object) -> object:
        query = str(kwargs["query"])
        if len(query) > MAX_QUERY_CHARS:
            raise AssertionError(f"composed input of {len(query)} chars reached the provider")
        self.calls.append(query)
        return SimpleNamespace(content=self._content)


_MATCH = (
    '{"status":"MATCH","independent_answer":"7","single_defensible_answer":true,"reason":"agrees"}'
)


def _verify(candidate: str, question: str = _QUESTION, content: str = _MATCH):
    orchestrator = _Orchestrator(content)
    result = AnswerCorrectnessVerifier(orchestrator=orchestrator).verify(  # type: ignore[arg-type]
        request_id="contract",
        query=question,
        candidate_answer=candidate,
        subject="math",
        difficulty="advanced",
        language="english",
    )
    return result, orchestrator


# ---------------------------------------------------------------------------
# 1. A short candidate composes exactly as before
# ---------------------------------------------------------------------------


def test_a_short_candidate_is_unchanged_and_reaches_the_provider_once() -> None:
    candidate = f"{_HEAD_MARKER}\n2x = 8 so x = 4.\n\n{_ANSWER_LINE}"
    legacy = (
        "[QUESTION]\n"
        f"{_QUESTION[:4000]}\n"
        "[/QUESTION]\n"
        "[CANDIDATE_ANSWER]\n"
        f"{candidate[:8000]}\n"
        "[/CANDIDATE_ANSWER]"
    )

    assert build_verifier_input(_QUESTION, candidate) == legacy

    result, orchestrator = _verify(candidate)

    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0] == legacy
    assert result.status == "match"


# ---------------------------------------------------------------------------
# 2. The a3a08c11 shape: a long regenerated candidate now reaches the verifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [4_000, 6_500, 12_000, 60_000])
def test_a_long_candidate_is_compacted_and_verified_exactly_once(length: int) -> None:
    result, orchestrator = _verify(_candidate(length))

    assert len(orchestrator.calls) == 1, "the provider must be invoked exactly once"
    assert len(orchestrator.calls[0]) <= MAX_QUERY_CHARS
    assert result.status == "match"
    assert result.reason_code == "NONE"


# ---------------------------------------------------------------------------
# 3 / 4. Compaction keeps the ending (the final answer) and the beginning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [4_000, 6_500, 12_000, 60_000])
def test_compaction_keeps_the_final_answer_and_the_opening_context(length: int) -> None:
    composed = build_verifier_input(_QUESTION, _candidate(length))

    assert composed.endswith("\n[/CANDIDATE_ANSWER]")
    assert _ANSWER_LINE in composed, "the final answer sits at the end and must survive"
    assert _HEAD_MARKER in composed, "the opening context must survive"
    assert "compacted" in composed, "the gap must be marked, never silently joined"
    assert _QUESTION in composed, "the question keeps priority"


def test_compaction_is_deterministic() -> None:
    candidate = _candidate(20_000)

    assert build_verifier_input(_QUESTION, candidate) == build_verifier_input(_QUESTION, candidate)


def test_compaction_never_alters_the_candidate_itself() -> None:
    candidate = _candidate(20_000)
    before = candidate

    build_verifier_input(_QUESTION, candidate)

    assert candidate == before


# ---------------------------------------------------------------------------
# 5. Extreme combinations still compose within the limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question_chars", "candidate_chars"),
    [(4_000, 8_000), (12_000, 60_000), (60_000, 60_000), (200_000, 200_000)],
)
def test_any_combination_composes_within_the_contract(
    question_chars: int, candidate_chars: int
) -> None:
    question = ("The question states a long premise. " * (question_chars // 36))[:question_chars]
    composed = build_verifier_input(question, _candidate(candidate_chars))

    assert len(composed) <= MAX_QUERY_CHARS
    assert _ANSWER_LINE in composed


def test_a_very_long_question_still_leaves_room_for_the_answer() -> None:
    question = ("Premise sentence number one. " * 4_000)[:100_000]
    composed = build_verifier_input(question, _candidate(30_000))

    assert len(composed) <= MAX_QUERY_CHARS
    assert _ANSWER_LINE in composed
    assert question[:200] in composed


def test_an_oversize_input_is_structural_and_never_retried() -> None:
    """The safety net: deterministic oversize must not spend a verifier retry."""
    kind, source = verification_diagnosis("INPUT_TOO_LARGE")
    decision = decide_verification_recovery(
        approved=False, reason_code="INPUT_TOO_LARGE", budget=RecoveryBudgetUse()
    )

    assert (kind, source) == ("STRUCTURAL", "VERIFIER")
    assert decision.action == "FAIL_TEMPORARY"
    assert decision.terminal_code == "ANSWER_VERIFICATION_UNAVAILABLE"


def test_an_oversize_input_is_not_reported_as_a_provider_failure() -> None:
    kind, _ = verification_diagnosis("INPUT_TOO_LARGE")

    assert kind != "INFRASTRUCTURE"


# ---------------------------------------------------------------------------
# 6 / 7. The existing verdicts and the Phase 1 invariant still hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (_MATCH, "match"),
        (
            '{"status":"MISMATCH","independent_answer":"9","single_defensible_answer":true,'
            '"reason":"differs"}',
            "mismatch",
        ),
        (
            '{"status":"AMBIGUOUS","independent_answer":"9","single_defensible_answer":false,'
            '"reason":"two readings"}',
            "ambiguous",
        ),
    ],
)
def test_every_verdict_still_works_on_a_long_candidate(verdict: str, expected: str) -> None:
    result, orchestrator = _verify(_candidate(9_000), content=verdict)

    assert result.status == expected
    assert len(orchestrator.calls) == 1


def test_a_parse_failure_on_a_long_candidate_is_still_a_parse_failure() -> None:
    result, orchestrator = _verify(_candidate(9_000), content="not json")

    assert (result.status, result.reason_code) == ("unavailable", "PARSE_FAILURE")
    assert len(orchestrator.calls) == 1, "the provider was reached; only its reply was unusable"


def test_the_phase_1_invariant_a_regenerated_candidate_is_verified_exactly_once() -> None:
    """A regenerated candidate is longer than the first; both must reach the verifier."""
    orchestrator = _Orchestrator(_MATCH)
    verifier = AnswerCorrectnessVerifier(orchestrator=orchestrator)  # type: ignore[arg-type]
    first_candidate = _candidate(3_000)
    regenerated_candidate = _candidate(9_000)

    for candidate in (first_candidate, regenerated_candidate):
        verifier.verify(
            request_id="contract",
            query=_QUESTION,
            candidate_answer=candidate,
            subject="math",
            difficulty="advanced",
            language="english",
        )

    assert len(orchestrator.calls) == 2
    assert all(len(sent) <= MAX_QUERY_CHARS for sent in orchestrator.calls)
    assert _ANSWER_LINE in orchestrator.calls[1]


# ---------------------------------------------------------------------------
# 8. End to end: the controller regenerates, and the long candidate is verified once
# ---------------------------------------------------------------------------

_MISMATCH = (
    '{"status":"MISMATCH","independent_answer":"9","single_defensible_answer":true,'
    '"reason":"differs"}'
)


class _SequencedOrchestrator(_Orchestrator):
    """Answers each verifier call in turn, still enforcing the query contract."""

    def __init__(self, *replies: str) -> None:
        super().__init__("")
        self._replies = list(replies)

    def generate(self, **kwargs: object) -> object:
        index = min(len(self.calls), len(self._replies) - 1)
        self._content = self._replies[index]
        return super().generate(**kwargs)


class _Diagnoser:
    def __init__(self) -> None:
        self.calls = 0

    def diagnose(self, **_: object) -> SemanticDiagnosis:
        self.calls += 1
        return SemanticDiagnosis("CANDIDATE", "WRONG_FINAL_ANSWER", True)


class _Adapter:
    """First candidate short, regenerated candidate long — the a3a08c11 shape."""

    def __init__(self, orchestrator: _SequencedOrchestrator) -> None:
        self.correctness_verifier = AnswerCorrectnessVerifier(orchestrator=orchestrator)  # type: ignore[arg-type]
        self.semantic_diagnoser = _Diagnoser()
        self.generate_calls = 0

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
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
        return _SHORT_SOLUTION if self.generate_calls == 1 else _LONG_SOLUTION


_SHORT_SOLUTION = (
    "Let the speeds be in the ratio 4:3.\n\nCombined they cover one lap per meeting.\n\n"
    "So the runners meet at 7 distinct points.\n\n**Answer:** 7 distinct meeting points"
)
_LONG_SOLUTION = (
    "Let the speeds be in the ratio 4:3.\n\n"
    # Longer than the verifier's query contract, and inside the 8,000-char limit that
    # applies to the answer the student receives.
    + "Each meeting happens after a combined lap, so the positions repeat. " * 60
    + "\n\n**Answer:** 7 distinct meeting points"
)


@pytest.fixture
def _recovery_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
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


def test_a_regenerated_long_candidate_reaches_the_verifier_exactly_once(
    _recovery_settings: None,
) -> None:
    orchestrator = _SequencedOrchestrator(_MISMATCH, _MATCH)
    adapter = _Adapter(orchestrator)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="contract-stream",
                query=_QUESTION,
                language="english",
                classification={
                    "subject": "math",
                    "intent": "solve",
                    "difficulty": "advanced",
                    "need_web_search": False,
                    "classifier_confidence": 0.99,
                    "classification_source": "llm",
                },
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )

    # The regenerated candidate is long enough that the old composition was refused.
    assert len(_LONG_SOLUTION) > MAX_QUERY_CHARS
    assert adapter.generate_calls == 2
    # Both verifications reached the provider, the second carrying the regenerated answer.
    assert len(orchestrator.calls) == 2
    assert all(len(sent) <= MAX_QUERY_CHARS for sent in orchestrator.calls)
    assert _ANSWER_LINE in orchestrator.calls[1]
    assert adapter.semantic_diagnoser.calls == 1
    # Lifecycle: delivered, exactly one terminal, and it is the last event.
    terminals = [index for index, e in enumerate(events) if e.type in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0] == len(events) - 1
    assert events[-1].type == "complete"
