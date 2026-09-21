"""Focused practice routing tests for the temporary safe gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from features.practice_generation.planning import (
    PRACTICE_ASYNC_NOT_CONFIGURED,
    decide_practice_launch,
    resolve_practice_delivery_language,
    resolve_practice_request,
)
from features.practice_generation.schemas import PracticeLaunchResult
from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)


class Adapter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_final(self, **_kwargs):
        self.calls += 1
        raise AssertionError("normal answer generation must not run")


def graph_state(query: str = "Create five algebra questions") -> dict:
    return {
        "request_id": "request-1",
        "actor_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": query,
        "original_query": query,
        "language": "english",
        "exam_id": "CAT",
        "exam_stage": None,
        "classification": {
            "intent": "practice",
            "subject": "math",
            "topic": "algebra",
            "difficulty": "intermediate",
            "retrieval_required": False,
        },
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


def test_explicit_practice_language_command_overrides_the_selected_language() -> None:
    assert resolve_practice_delivery_language(
        "Create five current affairs questions in Hindi",
        "english",
    ) == ("hindi", "EXPLICIT_QUERY")
    assert resolve_practice_delivery_language(
        "Create five questions",
        "hinglish",
    ) == ("hinglish", "REQUEST")


def test_non_stream_creation_request_returns_controlled_disabled_result(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    adapter = Adapter()
    result = build_orchestrated_doubt_solver_graph(adapter).invoke(graph_state())

    assert adapter.calls == 0
    assert result["final_answer"]["quality_status"] == "failed_quality_gate"
    assert "temporarily unavailable" in result["answer"]
    assert result.get("practice_test_id") is None


def test_stream_creation_request_returns_typed_disabled_error(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    adapter = Adapter()
    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="request-1",
                actor_id="user-1",
                conversation_id="conversation-1",
                turn_id="turn-1",
                query="Create five algebra questions",
                original_query="Create five algebra questions",
                language="english",
                exam_id="CAT",
                classification={
                    "subject": "math",
                    "topic": "algebra",
                    "intent": "practice",
                    "difficulty": "intermediate",
                    "retrieval_required": False,
                },
                classifier_confidence=0.99,
            ),
            adapter=adapter,
        )
    )

    assert adapter.calls == 0
    assert events[-1].type == "error"
    assert events[-1].metadata == {
        "retryable": False,
        "code": PRACTICE_ASYNC_NOT_CONFIGURED,
    }


def test_stream_creation_request_returns_existing_practice_completion(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    adapter = Adapter()
    captured = []
    ordering = []

    class Launcher:
        def launch(self, request):
            captured.append(request)
            return PracticeLaunchResult(
                test_id="practice-123",
                status="GENERATING",
                requested_count=5,
                accepted_count=5,
                count_clamped=False,
                progress_percent=0,
                playable=False,
                message="Your practice test is being prepared.",
            )

        def start(self, test_id, delivery_id=None):
            assert test_id == "practice-123"
            assert delivery_id == "request-1"
            ordering.append("started")
            return True

        def abort(self, _test_id, _code, _delivery_id=None):
            raise AssertionError("successful linkage must not abort")

    class Persistence:
        def persist_completed_turn(self, *_args, **_kwargs):
            ordering.append("persisted")
            return SimpleNamespace(history_write_status="succeeded")

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="request-1",
                actor_id="user-1",
                conversation_id="conversation-1",
                turn_id="turn-1",
                query="Create five algebra questions",
                original_query="Create five algebra questions",
                language="english",
                exam_id="CAT",
                classification={
                    "subject": "math",
                    "topic": "algebra",
                    "intent": "practice",
                    "difficulty": "intermediate",
                    "retrieval_required": False,
                },
                classifier_confidence=0.99,
            ),
            adapter=adapter,
            conversation_persistence=Persistence(),
            practice_launcher=Launcher(),
        )
    )

    assert adapter.calls == 0
    assert len(captured) == 1
    assert captured[0].user_id == "user-1"
    assert events[-1].type == "practice_generation_started"
    assert events[-1].data is not None
    assert events[-1].data.response_type == "practice_generation"
    assert events[-1].data.practice_test_id == "practice-123"
    assert events[-1].data.status == "GENERATING"
    assert ordering == ["persisted", "started"]
    serialized = events[-1].model_dump(mode="json", by_alias=True)
    assert serialized["data"] == {
        "responseType": "practice_generation",
        "practiceTestId": "practice-123",
        "status": "GENERATING",
        "message": "Your practice test is being prepared.",
        "requestedCount": 5,
        "effectiveCount": 5,
    }
    assert "taskId" not in str(serialized)


def test_non_stream_creation_request_uses_same_practice_launcher(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    adapter = Adapter()
    result = build_orchestrated_doubt_solver_graph(
        adapter,
        practice_launcher=lambda _request: PracticeLaunchResult(
            test_id="practice-123",
            status="GENERATING",
            requested_count=5,
            accepted_count=5,
            count_clamped=False,
            progress_percent=0,
            playable=False,
            message="Your practice test is being prepared.",
        ),
    ).invoke(graph_state())

    assert adapter.calls == 0
    assert result["response_type"] == "practice_generation"
    assert result["practice_test_id"] == "practice-123"


def test_duration_based_mini_mock_uses_practice_launcher(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    adapter = Adapter()
    result = build_orchestrated_doubt_solver_graph(
        adapter,
        practice_launcher=lambda _request: PracticeLaunchResult(
            test_id="practice-mini-mock",
            status="GENERATING",
            requested_count=5,
            accepted_count=5,
            count_clamped=False,
            progress_percent=0,
            playable=False,
            message="Your mini mock is being prepared.",
        ),
    ).invoke(graph_state("Create a 10-minute Reasoning mini mock for CAT."))

    assert adapter.calls == 0
    assert result["response_type"] == "practice_generation"
    assert result["practice_test_id"] == "practice-mini-mock"


def test_hindi_and_hinglish_explicit_creation_preserve_language() -> None:
    cases = [
        ("पाँच प्रतिशत के सवाल बनाओ", "hindi", 5),
        ("Paanch percentage sawaal banao", "hinglish", 5),
    ]
    for query, language, expected_count in cases:
        decision = decide_practice_launch(query, {"intent": "practice"})
        request = resolve_practice_request(
            request_id="request-1",
            user_id="user-1",
            conversation_id="conversation-1",
            turn_id="turn-1",
            query=query,
            subject="math",
            topic="percentage",
            difficulty="intermediate",
            language=language,
            exam_id="CAT",
            exam_stage=None,
        )
        assert decision.eligible is True
        assert request.language == language
        assert request.accepted_count == expected_count


def test_request_resolver_preserves_selected_source_reference() -> None:
    request = resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-2",
        query="Give me one similar question",
        subject="math",
        topic="percentage",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
        source_question_reference="turn-1",
    )

    assert request.source_question_reference == "turn-1"


def test_practice_advice_remains_on_normal_doubt_solver_path() -> None:
    class AdviceAdapter:
        def generate_final(self, **_kwargs):
            return build_final_answer_result(
                content="Use a focused practice schedule.",
                language="english",
                quality_status="checked",
            )

    result = build_orchestrated_doubt_solver_graph(AdviceAdapter()).invoke(
        graph_state("How should I practise algebra?")
    )

    assert result.get("persistence_suppressed") is not True
    assert result["answer"] == "Use a focused practice schedule."


@pytest.mark.parametrize(
    ("query", "language"),
    (
        ("Create five percentage practice questions", "english"),
        ("पाँच प्रतिशत के सवाल बनाओ", "hindi"),
        ("Paanch percentage sawaal banao", "hinglish"),
    ),
)
def test_practice_handoff_preserves_language_exam_and_stage(
    query: str,
    language: str,
) -> None:
    """No selected signal may disappear across the Doubt Solver → Practice handoff."""
    request = resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query=query,
        subject="math",
        topic="percentage",
        difficulty="intermediate",
        language=language,
        exam_id="SSC_CGL",
        exam_stage="TIER_2",
    )

    assert request.language == language
    assert request.exam_id == "SSC_CGL"
    assert request.exam_stage == "TIER_2"
    assert request.accepted_count == 5


@pytest.mark.parametrize(
    "query",
    (
        "Create 5 advanced percentage questions",
        "give me some english practice",
        "cretae three percetnage practice qestions",
    ),
)
def test_practice_creation_never_falls_through_to_the_inline_answer_path(query: str) -> None:
    """Reliability lock for the observed quality-gate terminal failures.

    Every real local quality-gate termination in the retained traces was a
    practice-intent request rejected with PRACTICE_CREATION_SIGNAL_MISSING, answered
    inline by practice.generator.default, and then discarded by the doubt-solver
    answer-quality contract that a multi-question practice set cannot satisfy. The
    inline fallthrough is the failure's antecedent, so it must stay unreachable.
    """
    decision = decide_practice_launch(query, {"intent": "practice"})

    assert decision.eligible is True
    assert decision.reason_code != "PRACTICE_CREATION_SIGNAL_MISSING"
