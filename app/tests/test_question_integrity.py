"""Conditional normalization of structurally malformed questions.

Clean questions never reach the normalizer. A flagged question is restructured by one
bounded call whose output a deterministic guard rejects if it changes a number, option,
or unit, or drops a negation. AMBIGUOUS / MISSING_INFORMATION stop the request before
any solver call with QUESTION_NEEDS_CLARIFICATION (not retryable by system or student).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

import main
from schemas.doubt_solver import DoubtSolverFinalResponse, DoubtSolverStreamEvent, ResponseContent
from services.doubt_solver.question_integrity import (
    QUESTION_NEEDS_CLARIFICATION,
    QuestionNormalization,
    QuestionNormalizer,
    assess_question_integrity,
    compose_normalized_query,
)

CLEAN_QUESTIONS = [
    "A shopkeeper marks goods 40% above cost and gives 20% discount. What is the profit percent?",
    "If CAT is coded as DBU, how is DOG coded?",
    "Find the next number: 2, 6, 12, 20, 30, ?",
    "Solve for x: \\( 2x + 3 = 11 \\)",
    "Evaluate \\[ \\int_0^1 x^2 \\, dx \\]",
    "Which of the following is a prime number? (a) 21 (b) 27 (c) 29 (d) 33",
    "Which of the following is a prime number?\n(a) 21\n(b) 27\n(c) 29\n(d) 33",
    "2 x + 3 y = 12 and 4 x - y = 10, find x and y",
    "x + y = 10\nx - y = 2\nFind x.",
    "एक ट्रेन 240 मीटर लंबी है और 16 सेकंड में एक खंभे को पार करती है। उसकी गति km/h में क्या है?",
    "Train ki speed kya hogi agar 240 m train 16 sec me pole cross kare?",
    "Create a practice quiz of 5 questions on coding decoding for SSC CGL",
    "If f(x) = x^2 - 3x + 2, find f(2).",
    "The ratio of A to B is 3 : 4 and B to C is 5 : 6. Find A : C.",
    "Why is option B correct?",
    # Shapes that once flagged: labels out of order, vertical options, letter series,
    # LaTeX row spacing. None of them is formatting damage.
    "Given P(B) = 0.3 and P(A) = 0.6, find P(A|B) if A and B are independent.",
    "If f(b) - f(a) = 10 and b - a = 2, find f'(c) by LMV theorem.",
    "Consider the following statements: (a) Statement one is about rivers. (b) Statement two. "
    "Which of the above is correct? (a) Only 1 (b) Only 2 (c) Both (d) Neither",
    "Match List-I with List-II: (A) Kaziranga (B) Gir (C) Sundarbans (D) Jim Corbett. "
    "Codes: (a) A-1 B-2 (b) A-2 B-1 (c) A-3 B-4 (d) A-4 B-3",
    "Why is (b) correct and not (a)?",
    "Compare options (d) and (b).",
    "n(B) = 20, n(A) = 30, n(A∪B)=40. Find n(A∩B).",
    "Which of these is an allotrope of carbon (C)? (a) Diamond (b) Ozone (c) Quartz (d) Mica",
    "Assertion (A): Ozone layer absorbs UV rays. Reason (R): Ozone is a triatomic molecule. "
    "(a) Both A and R are true (b) A true R false (c) A false R true (d) Both false",
    "What is the next number in the series 2, 4, 8, 16, ?\n24\n32\n30\n28",
    "Find the odd one out:\n121\n144\n169\n180",
    "Find the average of the marks:\nMath\n85\nScience\n90\nEnglish\n78\nHindi\n88",
    "Solve the series:\n3\n9\n27\n?",
    "Find the missing term: A C E G I K M O Q S U W Y ? B D F H J L N P R T V X",
    "How many 5s are followed by 3 but not preceded by 7? 5 3 7 5 3 8 5 3 2 7 5 3 9 5 3 1 5 3",
    "Evaluate the determinant \\[ \\begin{vmatrix} 1 & 2 \\\\[2pt] 3 & 4 \\end{vmatrix} \\]",
    "Agar P(B)=0.2 aur P(A)=0.5 ho to P(A∩B) kya hoga agar independent hain?",
    "Which symbol should replace ? in 12 ? 4 = 3?\n+\n-\n×\n÷",
]

MALFORMED = {
    "ocr_symbol_corruption": (
        "Given the system of equations ⎧ ⎪ ⎨ ⎪ ⎩ 2 x + y + 2 z = 4 x + 2 y + 3 z = − 1 "
        "3 x + 2 y + z = 9 , find x + y + z."
    ),
    "unbalanced_math_delimiters": "Solve \\( 3x + 4 = 19 for x.",
    "fragmented_equation_lines": "Solve:\n2x\n+\n3\n=\n11\nfind x",
    "excessive_isolated_fragments": (
        "2 x + y + 2 z = 4 x + 2 y + 3 z = - 1 3 x + 2 y + z = 9 find x + y + z"
    ),
}


@pytest.mark.parametrize("question", CLEAN_QUESTIONS)
def test_clean_questions_raise_no_integrity_signal(question: str) -> None:
    assert assess_question_integrity(question) == ()


@pytest.mark.parametrize(("signal", "question"), list(MALFORMED.items()))
def test_malformed_questions_raise_the_expected_signal(signal: str, question: str) -> None:
    assert signal in assess_question_integrity(question)


# ---------------------------------------------------------------------------
# Normalizer outcomes and the no-new-facts guard
# ---------------------------------------------------------------------------

_SYSTEM = MALFORMED["ocr_symbol_corruption"]
_SYSTEM_REPAIRED = (
    "Given the system of equations 2x + y + 2z = 4\nx + 2y + 3z = -1\n"
    "3x + 2y + z = 9, find x + y + z."
)


class _FakeOrchestrator:
    def __init__(self, content: str | Exception) -> None:
        self._content = content
        self.calls: list[dict] = []

    def generate_structured(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if isinstance(self._content, Exception):
            raise self._content
        return SimpleNamespace(content=self._content)


def _normalize(
    content: str | Exception, query: str = _SYSTEM
) -> tuple[QuestionNormalization, _FakeOrchestrator]:
    orchestrator = _FakeOrchestrator(content)
    result = QuestionNormalizer(orchestrator=orchestrator).normalize(  # type: ignore[arg-type]
        request_id="normalize",
        query=query,
        language="english",
        signals=assess_question_integrity(query),
    )
    return result, orchestrator


def test_recoverable_formatting_is_normalized_once_on_the_existing_strong_classifier_route() -> (
    None
):
    result, orchestrator = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": _SYSTEM_REPAIRED})
    )

    assert result == QuestionNormalization(
        outcome="normalized", normalized_question=_SYSTEM_REPAIRED
    )
    assert len(orchestrator.calls) == 1
    call = orchestrator.calls[0]
    assert call["route_request"].task_role == "classifier_strong"
    assert call["prompt"] == "question_normalizer.md"
    assert call["user_content"] == _SYSTEM


@pytest.mark.parametrize(
    ("original", "normalized"),
    [
        (_SYSTEM, _SYSTEM_REPAIRED.replace("= 9", "= 19")),  # changed number
        (_SYSTEM, _SYSTEM_REPAIRED + " Assume z = 2."),  # new fact / number
        (_SYSTEM, _SYSTEM_REPAIRED.replace("= -1", "= 1")),  # sign flip
        (_SYSTEM, _SYSTEM_REPAIRED.replace("x + 2y", "x - 2y")),  # operator flip
        (
            "Find the LCM of 12 and 18 ⎪ (b) 36 (a) 24 (d) 72 (c) 48",
            "Find the LCM of 12 and 18. (a) 24 (b) 36 (c) 48 72",  # dropped option label
        ),
        (
            "A is 5 years older than B ⎪ . Find B.",
            "A is 5 months older than B. Find B.",  # time unit changed
        ),
        (
            "x is 5 more than y ⎪ . Find x.",
            "x is 5 less than y. Find x.",  # comparison changed
        ),
        (
            "Which number ⎪ is a prime? (a) 2 (b) 3 (c) 4 (d) 5",
            "Which number is not a prime? (a) 2 (b) 3 (c) 4 (d) 5",  # added negation
        ),
        (
            "A car covers 120 km in 3 hours ⎪ at constant speed. Speed in km/h?",
            "A car covers 120 m in 3 hours at constant speed. Speed in km/h?",  # unit changed
        ),
        (
            "Which number ⎪ is not a prime? (a) 2 (b) 3 (c) 4 (d) 5",
            "Which number is a prime? (a) 2 (b) 3 (c) 4 (d) 5",  # dropped negation
        ),
    ],
)
def test_guard_rejects_normalized_text_that_changes_facts(original: str, normalized: str) -> None:
    result, _ = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": normalized}),
        query=original,
    )
    assert result.outcome == "guard_rejected"
    assert result.normalized_question is None


@pytest.mark.parametrize("status", ["AMBIGUOUS", "MISSING_INFORMATION"])
def test_ambiguous_or_missing_information_needs_clarification(status: str) -> None:
    result, _ = _normalize(json.dumps({"status": status, "normalized_question": None}))
    assert result == QuestionNormalization(outcome="needs_clarification")


@pytest.mark.parametrize("content", ["not json", RuntimeError("provider down")])
def test_normalizer_failure_falls_back_to_the_existing_path(content: str | Exception) -> None:
    result, _ = _normalize(content)
    assert result == QuestionNormalization(outcome="unavailable")


def test_guard_accepts_the_live_faithful_repair() -> None:
    """The normalization observed live for the garbled system (answer 2) passes."""
    live = (
        "Given the system of equations ⎧ ⎪ ⎨ ⎪ ⎩ 2 x + y + 2 z = 4 x + 2 y + 3 z = − 1 "
        "3 x + 2 y + z = 9 , find the value of x + y + z."
    )
    repaired = (
        "Given the system of equations:\n2x + y + 2z = 4\nx + 2y + 3z = -1\n"
        "3x + 2y + z = 9\nfind the value of x + y + z."
    )
    result, _ = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": repaired}),
        query=live,
    )
    assert result == QuestionNormalization(outcome="normalized", normalized_question=repaired)


def test_composed_query_keeps_the_original_verbatim() -> None:
    composed = compose_normalized_query(_SYSTEM, _SYSTEM_REPAIRED)
    assert composed.startswith(_SYSTEM)
    assert composed.endswith(_SYSTEM_REPAIRED)


# ---------------------------------------------------------------------------
# Request boundary wiring (main.invoke)
# ---------------------------------------------------------------------------


class _SpyNormalizer:
    def __init__(self, outcome: QuestionNormalization) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    def normalize(self, **kwargs: object) -> QuestionNormalization:
        self.calls.append(kwargs)
        return self.outcome


@pytest.fixture
def stream_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[list]:
    captured: list = []

    def stream_source(input, **_kwargs):  # noqa: ANN001, ANN202
        captured.append(input)
        response = DoubtSolverFinalResponse(
            request_id=input.request_id, content=ResponseContent(value="ok"), answer="ok"
        )
        yield DoubtSolverStreamEvent(type="chunk", request_id=input.request_id, content="ok")
        yield DoubtSolverStreamEvent(
            type="complete",
            request_id=input.request_id,
            stage="complete",
            label="Done",
            metadata={
                "request_id": input.request_id,
                "conversation_id": input.conversation_id,
                "turn_id": input.turn_id,
                "persisted": True,
            },
            response=response,
        )

    monkeypatch.setattr(main, "orchestrated_doubt_solver_graph", object())
    monkeypatch.setattr(main, "orchestrated_adapter", object())
    monkeypatch.setattr(main, "stream_doubt_solver", stream_source)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            enable_orchestrated_doubt_solver=True, answer_stream_heartbeat_interval_seconds=0.01
        ),
    )
    yield captured


def _post(query: str, *, stream: bool = True):  # noqa: ANN202
    with TestClient(main.app) as client:
        return client.post(
            "/invocations",
            json={
                "mode": "doubt_solver",
                "query": query,
                "user_id": "local-user",
                "conversation_id": "conversation-integrity",
                "turn_id": "turn-integrity",
                "stream": stream,
            },
        )


def _frames(response) -> list[dict]:  # noqa: ANN001
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_clean_question_never_calls_the_normalizer(
    monkeypatch: pytest.MonkeyPatch, stream_capture: list
) -> None:
    spy = _SpyNormalizer(QuestionNormalization(outcome="unavailable"))
    monkeypatch.setattr(main, "question_normalizer", spy)

    response = _post(CLEAN_QUESTIONS[0])

    assert spy.calls == []
    assert [frame["type"] for frame in _frames(response)] == ["chunk", "complete"]
    assert stream_capture[0].query == CLEAN_QUESTIONS[0]


def test_recoverable_question_reaches_the_solver_composed_with_the_original_preserved(
    monkeypatch: pytest.MonkeyPatch, stream_capture: list
) -> None:
    spy = _SpyNormalizer(
        QuestionNormalization(outcome="normalized", normalized_question=_SYSTEM_REPAIRED)
    )
    monkeypatch.setattr(main, "question_normalizer", spy)

    _post(_SYSTEM)

    assert len(spy.calls) == 1
    assert stream_capture[0].query == compose_normalized_query(_SYSTEM, _SYSTEM_REPAIRED)
    assert stream_capture[0].original_query == _SYSTEM


@pytest.mark.parametrize("outcome", ["needs_clarification", "guard_rejected", "unavailable"])
def test_flagged_question_without_a_trusted_repair_stops_before_any_solver_call(
    outcome: str, monkeypatch: pytest.MonkeyPatch, stream_capture: list
) -> None:
    """The damaged text is never solved: a misread of it can pass a verifier that reads
    the same damaged text (observed live: a garbled system answered 23 instead of 2)."""
    monkeypatch.setattr(
        main,
        "question_normalizer",
        _SpyNormalizer(QuestionNormalization(outcome=outcome)),  # type: ignore[arg-type]
    )

    frames = _frames(_post(_SYSTEM))

    assert stream_capture == []
    assert [frame["type"] for frame in frames] == ["error"]
    assert frames[0]["metadata"] == {
        "retryable": False,
        "user_retryable": False,
        "code": QUESTION_NEEDS_CLARIFICATION,
    }
    assert "re-type" in frames[0]["label"]


def test_unreadable_question_non_stream_returns_the_clarification_without_solving(
    monkeypatch: pytest.MonkeyPatch, stream_capture: list
) -> None:
    class _NeverInvokedGraph:
        def invoke(self, _state):  # noqa: ANN001, ANN202
            raise AssertionError("solver must not run")

    monkeypatch.setattr(main, "orchestrated_doubt_solver_graph", _NeverInvokedGraph())
    monkeypatch.setattr(
        main,
        "question_normalizer",
        _SpyNormalizer(QuestionNormalization(outcome="needs_clarification")),
    )

    body = _post(_SYSTEM, stream=False).json()

    assert body["success"] is False
    assert body["error"] == QUESTION_NEEDS_CLARIFICATION
    assert "re-type" in body["answer"]


# ---------------------------------------------------------------------------
# The normalized text must reach generation and verification, not just the input
# ---------------------------------------------------------------------------


def _solver_classification() -> dict:
    return {
        "subject": "math",
        "intent": "solve",
        "difficulty": "intermediate",
        "retrieval_required": False,
        "requires_recent_conversation": False,
        "need_web_search": False,
        "confidence": 0.99,
        "classification_source": "llm",
    }


class _RecordingSolver:
    """Records the query the generator and the correctness verifier actually receive."""

    def __init__(self) -> None:
        self.generated_queries: list[str] = []
        self.verified_queries: list[str] = []
        solver = self

        class _Verifier:
            def verify(self, **kwargs: object):  # noqa: ANN202
                from services.doubt_solver.answer_correctness import CorrectnessVerification

                solver.verified_queries.append(str(kwargs["query"]))
                return CorrectnessVerification(
                    status="match",
                    independent_answer="2",
                    single_defensible_answer=True,
                    reason="ok",
                    method="model",
                )

        self.correctness_verifier = _Verifier()

    def generate(self, **kwargs: object) -> str:
        self.generated_queries.append(str(kwargs["query"]))
        return (
            "Adding the three equations gives 6x + 5y + 6z = 12; "
            "solving gives x = 3.5, y = 0, z = -1.5.\n\n**Answer:** 2"
        )


def test_streaming_generation_and_verification_receive_the_normalized_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import config as cfg_module
    import graphs.doubt_solver_graph as graph_module
    import services.doubt_solver.streaming_doubt_solver_service as streaming_module
    from services.conversation.conversation_understanding import ConversationUnderstandingService
    from services.doubt_solver.streaming_doubt_solver_service import (
        StreamDoubtSolverInput,
        stream_doubt_solver,
    )

    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    cfg_module._settings = None
    monkeypatch.setattr(
        graph_module,
        "orchestrated_classify_query",
        MagicMock(return_value=_solver_classification()),
    )
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    # This test is about which text reaches the verifier, so the verifier always runs.
    monkeypatch.setattr(
        streaming_module, "requires_independent_correctness_verification", lambda **_: True
    )
    composed = compose_normalized_query(_SYSTEM, _SYSTEM_REPAIRED)
    solver = _RecordingSolver()

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="normalized-stream",
                query=composed,
                original_query=_SYSTEM,
                language="english",
                actor_id="student-1",
                conversation_id="conversation-1",
                turn_id="turn-1",
            ),
            adapter=solver,  # type: ignore[arg-type]
            conversation_understanding=ConversationUnderstandingService(persistence=MagicMock()),
        )
    )
    cfg_module._settings = None

    assert events[-1].type == "complete"
    assert solver.generated_queries == [composed]
    assert solver.verified_queries == [composed]


def test_graph_generation_receives_the_normalized_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    import graphs.doubt_solver_graph as graph_module
    from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
    from services.conversation.conversation_understanding import ConversationUnderstandingService

    monkeypatch.setattr(
        graph_module,
        "orchestrated_classify_query",
        MagicMock(return_value=_solver_classification()),
    )
    composed = compose_normalized_query(_SYSTEM, _SYSTEM_REPAIRED)
    solver = _RecordingSolver()
    graph = build_orchestrated_doubt_solver_graph(
        solver, conversation_understanding=ConversationUnderstandingService(persistence=MagicMock())
    )

    graph.invoke(
        {
            "request_id": "normalized-graph",
            "actor_id": "student-1",
            "conversation_id": "conversation-1",
            "turn_id": "turn-1",
            "query": composed,
            "original_query": _SYSTEM,
            "language": "english",
            "exam_id": None,
            "exam_stage": None,
            "classification": None,
            "retrieval_context": {},
            "context_text": "",
            "conversation_context": "",
            "answer": None,
            "final_answer": None,
        }
    )

    assert solver.generated_queries == [composed]


@pytest.mark.parametrize(
    "incomplete_repair",
    [
        # Observed live: the model called this RECOVERABLE and added nothing, so the fact
        # guard passed, yet equations and the question itself are missing.
        "Given the system of equations:\n2x + y =\n3x - =\nfind",
        "Solve 2x + 3 = 11 and 4x -",
        "A train crosses a pole in 16 seconds. Calculate",
    ],
)
def test_structurally_incomplete_repair_needs_clarification(incomplete_repair: str) -> None:
    result, _ = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": incomplete_repair}),
        query=incomplete_repair.replace("\n", " ⎪\n"),
    )
    assert result == QuestionNormalization(outcome="needs_clarification")


# ---------------------------------------------------------------------------
# Semantic safety: a repair may change representation, never interpretation
# ---------------------------------------------------------------------------

_DEBRIS = " ⎪"  # formatting damage, so the original is a realistic flagged input

SEMANTIC_MUTATIONS = [
    (
        "increase by 20% → decrease",
        "The price increases by 20%{d}. Find the new price.",
        "The price decreases by 20%. Find the new price.",
    ),
    (
        "twice → thrice",
        "A number is twice{d} 12. Find the number.",
        "A number is thrice 12. Find the number.",
    ),
    ("x + y → x - y", "x + y = 10{d}", "x - y = 10"),
    (
        "not greater → greater",
        "A is not greater than B{d}. Is A = 5?",
        "A is greater than B. Is A = 5?",
    ),
    (
        "before 2010 → after",
        "He was born before 2010{d}. Find his age in 2020.",
        "He was born after 2010. Find his age in 2020.",
    ),
    ("at least 5 → at most 5", "At least 5 students{d} passed.", "At most 5 students passed."),
    (
        "20 kg / 30 kg rebound",
        "A weighs 20 kg and B weighs 30 kg{d}. Find the total.",
        "A weighs 30 kg and B weighs 20 kg. Find the total.",
    ),
    ("20 kg → 20 m", "The mass is 20 kg{d}.", "The mass is 20 m."),
    (
        "profit → loss",
        "He sells at a profit of 20%{d}. Find the cost price.",
        "He sells at a loss of 20%. Find the cost price.",
    ),
    (
        "doubles → triples",
        "The sum doubles in 5 years{d}. Find the rate.",
        "The sum triples in 5 years. Find the rate.",
    ),
    ("≤ → ≥", "x ≤ 5 and y = 2{d}", "x ≥ 5 and y = 2"),
    (
        "option contents swapped",
        "Which is prime?{d} (a) 21 (b) 29",
        "Which is prime? (a) 29 (b) 21",
    ),
    ("fact invented", "x + y = 10{d}. Find x.", "x + y = 10 where x is positive. Find x."),
    (
        "Hindi vowel sign: son → daughter",
        "A, B का पुत्र है{d}। A, C का क्या है?",
        "A, B का पुत्री है। A, C का क्या है?",
    ),
    ("grouping parentheses dropped", "Evaluate (2 + 3) × 4{d}", "Evaluate 2 + 3 × 4"),
    ("root scope changed", "Find √(16 + 9){d}.", "Find √16 + 9."),
    ("root invented", "Find 16 + 9{d}.", "Find √16 + 9."),
    ("!= → =", "If x != 5{d}, is x prime?", "If x = 5, is x prime?"),
    ("union → intersection", "Find n(A ∪ B){d}.", "Find n(A ∩ B)."),
    ("parallel → perpendicular", "AB ∥ CD{d}. Find the angle.", "AB ⊥ CD. Find the angle."),
    ("factorial dropped", "Evaluate 5! + 3{d}", "Evaluate 5 + 3"),
    ("absolute value dropped", "Solve |x| = 3{d}", "Solve x = 3"),
    ("letter case: molar → molal", "Find the pH of 0.1 M HCl{d}.", "Find the pH of 0.1 m HCl."),
    ("ratio colon dropped", "A:B = 2:3{d}. Find A.", "A B = 2 3. Find A."),
    ("digit grouping split", "The price is 1,500{d} rupees.", "The price is 1, 500 rupees."),
    ("bracket debris dropped", "Evaluate ⎛2 + 3⎞ × 4{d}", "Evaluate 2 + 3 × 4"),
    ("leading decimal point dropped", "If x = .5{d}, find 2x.", "If x = 5, find 2x."),
    (
        "comma splits an expression into a list",
        "Find the product of 5 - 3 and 2{d}.",
        "Find the product of 5, -3 and 2.",
    ),
    (
        "spaced ratio colon dropped",
        "If a: b = 2:3{d}, find a when b = 6.",
        "If a b = 2:3, find a when b = 6.",
    ),
    (
        "separator deleted: two equations chained",
        "Given x = 2, y = 3{d}. Find x + y.",
        "Given x = 2y = 3. Find x + y.",
    ),
    (
        "line break deleted: two equations chained",
        "Given x = 2\ny = 3{d}. Find x + y.",
        "Given x = 2y = 3. Find x + y.",
    ),
    (
        "list comma dropped before a letter",
        "Find the sum of 3, x and 5{d}.",
        "Find the sum of 3 x and 5.",
    ),
    (
        "list comma invented before a letter",
        "Find the sum of 3 x and 5{d}.",
        "Find the sum of 3, x and 5.",
    ),
    (
        "ratio colon at a line break dropped",
        "If a:\nb = 2:3{d}, find a when b = 6.",
        "If a b = 2:3, find a when b = 6.",
    ),
    (
        "line break before a sign-led equation deleted",
        "Solve 2x + y = 5{d}\n-x + y = 2",
        "Solve 2x + y = 5 -x + y = 2",
    ),
    (
        "line break before a negative value deleted",
        "Find the mean of the values{d}\n5\n-3\n7",
        "Find the mean of the values\n5 - 3\n7",
    ),
    ("unicode line separator deleted", "Given x = 2\u2028y = 3{d}", "Given x = 2y = 3"),
    (
        "two-letter ratio colon dropped",
        "If AB:\nCD = 2:3{d}, find CD when AB = 4.",
        "If AB\nCD = 2:3, find CD when AB = 4.",
    ),
    ("line break splits a coefficient", "Solve 3x + 2y = 5{d}", "Solve 3\nx + 2y = 5"),
    ("Hindi word split after a vowel sign", "इसकी कीमत 20 है{d}", "इसकी की\nमत 20 है"),
    (
        "space moved across a decimal point",
        "A pen costs Rs .50{d} each.",
        "A pen costs Rs. 50 each.",
    ),
    ("not-equal turned into factorial", "If 5 != x{d}, find x.", "If 5! = x, find x."),
    (
        "bulleted equations joined",
        "Given{d}\n* x = 2\n* y = 3\nFind x + y.",
        "Given\n* x = 2 * y = 3\nFind x + y.",
    ),
    (
        "ratio colon after a name dropped",
        "The ratio of ages Ram:\nShyam = 3:4{d}",
        "The ratio of ages Ram\nShyam = 3:4",
    ),
]


@pytest.mark.parametrize(
    ("original", "mutated"),
    [(original.format(d=_DEBRIS), mutated) for _, original, mutated in SEMANTIC_MUTATIONS],
    ids=[name for name, _, _ in SEMANTIC_MUTATIONS],
)
def test_semantic_mutation_is_rejected(original: str, mutated: str) -> None:
    result, _ = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": mutated}),
        query=original,
    )
    assert result == QuestionNormalization(outcome="guard_rejected")


VALID_REPAIRS = [
    (
        "equation split across lines",
        "2x+\n y = 10" + _DEBRIS + "\nFind x if y = 4.",
        "2x + y = 10\nFind x if y = 4.",
    ),
    (
        "vertical options",
        "Which is prime?" + _DEBRIS + "\n(a)\n21\n(b)\n27",
        "Which is prime?\n(a) 21\n(b) 27",
    ),
    (
        "symbol spacing and unicode operators",
        "Evaluate 3×4 − 2" + _DEBRIS + ".",
        "Evaluate 3 * 4 - 2.",
    ),
    (
        "formula debris removed",
        "Given the system of equations ⎧ ⎪ ⎨ ⎪ ⎩ 2 x + y + 2 z = 4 x + 2 y + 3 z = − 1 "
        "3 x + 2 y + z = 9 , find the value of x + y + z.",
        "Given the system of equations:\n2x + y + 2z = 4\nx + 2y + 3z = -1\n"
        "3x + 2y + z = 9\nfind the value of x + y + z.",
    ),
    (
        "Hindi layout",
        "यदि 2x +\n3 = 11" + _DEBRIS + " है, तो x का मान ज्ञात कीजिए।",
        "यदि 2x + 3 = 11 है, तो x का मान ज्ञात कीजिए।",
    ),
]


@pytest.mark.parametrize(
    ("original", "repaired"),
    [(original, repaired) for _, original, repaired in VALID_REPAIRS],
    ids=[name for name, _, _ in VALID_REPAIRS],
)
def test_formatting_only_repair_is_accepted(original: str, repaired: str) -> None:
    result, orchestrator = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": repaired}),
        query=original,
    )
    assert result == QuestionNormalization(outcome="normalized", normalized_question=repaired)
    assert len(orchestrator.calls) == 1


@pytest.mark.parametrize(
    "incomplete_repair",
    [
        "Which is prime? (a) 21 (b) (c) 29",  # missing option text
        "Solve 2x + * 3 = 11",  # missing operand
        "Find the value of x if 3x = ...",  # unrecoverable placeholder
    ],
)
def test_incomplete_repair_is_not_made_solvable(incomplete_repair: str) -> None:
    result, _ = _normalize(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": incomplete_repair}),
        query=incomplete_repair + _DEBRIS,
    )
    assert result.outcome in {"needs_clarification", "guard_rejected"}
    assert result.normalized_question is None


def test_semantic_mutation_reaches_the_student_only_as_a_clarification(
    monkeypatch: pytest.MonkeyPatch, stream_capture: list
) -> None:
    original = "A weighs 20 kg and B weighs 30 kg" + _DEBRIS + ". Find the total."
    mutated = "A weighs 30 kg and B weighs 20 kg. Find the total."
    orchestrator = _FakeOrchestrator(
        json.dumps({"status": "RECOVERABLE_FORMATTING", "normalized_question": mutated})
    )
    monkeypatch.setattr(main, "question_normalizer", QuestionNormalizer(orchestrator=orchestrator))  # type: ignore[arg-type]

    frames = _frames(_post(original))

    assert len(orchestrator.calls) == 1
    assert stream_capture == []  # no generator, no verifier
    assert [frame["type"] for frame in frames] == ["error"]
    assert frames[0]["metadata"] == {
        "retryable": False,
        "user_retryable": False,
        "code": QUESTION_NEEDS_CLARIFICATION,
    }
