"""Regression coverage for evidence-grounded current-affairs Practice."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from features.practice_generation.orchestration import _request
from features.practice_generation.planning import (
    is_fresh_evidence_request_supported,
    resolve_practice_freshness_requirement,
    resolve_practice_request,
)
from features.practice_generation.providers import (
    RoutedQuestionGenerator,
    RoutedQuestionVerifier,
)
from features.practice_generation.schemas import (
    Complexity,
    DemandBucket,
    Difficulty,
    GeneratedQuestion,
    GenerationGroup,
    PlannerSlot,
    PracticeLaunchResult,
    QuestionType,
    VerificationDecision,
    VerificationPolicy,
)
from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
from services.classification.web_search_demand import is_freshness_sensitive_query
from tools.web_search.models import FreshEvidenceBundle, FreshnessWindow, WebSearchItem
from tools.web_search.source_policy import WebSourcePolicyResolver
from tools.web_search.web_search_tool import _build_fresh_evidence_bundle


def _evidence(count: int = 2) -> FreshEvidenceBundle:
    return FreshEvidenceBundle(
        requested_window=FreshnessWindow(
            start_date="2026-07-15",
            end_date="2026-08-14",
            source="default",
            label="August 2026",
        ),
        retrieved_at="2026-08-14T08:00:00+00:00",
        search_query="current affairs SSC GD",
        items=[
            WebSearchItem(
                title=f"Verified event {index}",
                url=f"https://pib.gov.in/release-{index}",
                snippet=f"Verified current-affairs event number {index} for SSC preparation.",
                source="pib.gov.in",
                published_at="2026-08-01",
                source_quality="trusted",
            )
            for index in range(1, count + 1)
        ],
    )


def _current_request() -> object:
    query = "Create 2 current affairs questions for SSC GD"
    requirement = resolve_practice_freshness_requirement(
        query,
        {"intent": "practice", "web_search_reason": "current_affairs"},
    )
    return resolve_practice_request(
        request_id="request-fresh",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query=query,
        subject="general",
        topic="current_affairs",
        difficulty="intermediate",
        language="english",
        exam_id="SSC_GD",
        exam_stage=None,
        freshness_requirement=requirement,
        fresh_evidence=_evidence(),
    )


def _graph_state(query: str, classification: dict[str, object]) -> dict[str, object]:
    return {
        "request_id": "request-fresh",
        "actor_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": query,
        "original_query": query,
        "language": "english",
        "exam_id": "SSC_GD",
        "exam_stage": None,
        "exam_profile_id": None,
        "classification": classification,
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


class _NoAnswerAdapter:
    def generate_final(self, **_kwargs):
        raise AssertionError("fresh Practice must not run normal answer generation")


def _launch_result() -> PracticeLaunchResult:
    return PracticeLaunchResult(
        test_id="practice-fresh",
        status="GENERATING",
        requested_count=2,
        accepted_count=2,
        count_clamped=False,
        progress_percent=0,
        playable=False,
        message="Your practice test is being prepared.",
    )


class _FixedDate(date):
    @classmethod
    def today(cls) -> _FixedDate:
        return cls(2026, 8, 14)


def test_freshness_policy_covers_dynamic_facts_without_static_subject_regressions() -> None:
    assert all(
        is_freshness_sensitive_query(query)
        for query in (
            "Create 10 current affairs questions",
            "Create 5 latest current affairs questions",
            "Create 5 questions on recent national events",
            "Create questions about current office holders",
            "Who is the current Prime Minister of India?",
        )
    )
    assert not any(
        is_freshness_sensitive_query(query)
        for query in (
            "Create 5 Time and Work questions",
            "Create 5 syllogism questions",
            "Create questions on the Revolt of 1857",
        )
    )


def test_default_and_explicit_month_windows_are_deterministic(monkeypatch) -> None:
    monkeypatch.setattr("tools.web_search.source_policy.date", _FixedDate)
    resolver = WebSourcePolicyResolver()
    current = resolver.resolve(
        query="Create 5 latest current affairs questions",
        web_search_query=None,
        subject="general",
        topic="current_affairs",
        retrieval_tags=[],
        web_search_reason="current_affairs",
        source_strictness="authoritative_first",
        default_recent_days=30,
    )
    august = resolver.resolve(
        query="Create August 2026 current affairs questions",
        web_search_query=None,
        subject="general",
        topic="current_affairs",
        retrieval_tags=[],
        web_search_reason="current_affairs",
        source_strictness="authoritative_first",
        default_recent_days=30,
    )

    assert (current.start_date, current.end_date, current.freshness_source) == (
        "2026-07-15",
        "2026-08-14",
        "default",
    )
    assert current.temporal_mode == "LATEST"
    assert (august.start_date, august.end_date, august.freshness_source) == (
        "2026-08-01",
        "2026-08-31",
        "user_explicit",
    )
    assert august.temporal_mode == "EXPLICIT_MONTH"


def test_current_affairs_graph_attaches_fresh_evidence_before_async_launch(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    captured: list[object] = []

    def collect(_state, **_kwargs):
        return {
            "context_text": "",
            "retrieval_context": {},
            "fresh_evidence": _evidence().model_dump(),
        }

    monkeypatch.setattr("graphs.doubt_solver_graph._orchestrated_collect_context_node", collect)
    graph = build_orchestrated_doubt_solver_graph(
        _NoAnswerAdapter(),
        practice_launcher=lambda request: captured.append(request) or _launch_result(),
    )
    result = graph.invoke(
        _graph_state(
            "Create 2 current affairs questions for SSC GD",
            {
                "intent": "practice",
                "subject": "general",
                "topic": "current_affairs",
                "difficulty": "intermediate",
                "need_web_search": True,
                "web_search_reason": "current_affairs",
            },
        )
    )

    request = captured[0]
    assert request.requires_fresh_evidence is True
    assert request.fresh_evidence is not None
    assert len(request.fresh_evidence.items) == 2
    assert result["practice_test_id"] == "practice-fresh"


def test_fresh_practice_without_evidence_does_not_launch_or_use_model_memory(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    launched = []
    monkeypatch.setattr(
        "graphs.doubt_solver_graph._orchestrated_collect_context_node",
        lambda _state, **_kwargs: {"context_text": "", "retrieval_context": {}},
    )
    result = build_orchestrated_doubt_solver_graph(
        _NoAnswerAdapter(),
        practice_launcher=lambda request: launched.append(request) or _launch_result(),
    ).invoke(
        _graph_state(
            "Create 2 current affairs questions for SSC GD",
            {
                "intent": "practice",
                "subject": "general",
                "need_web_search": True,
                "web_search_reason": "current_affairs",
            },
        )
    )

    assert launched == []
    assert "could not be started" in result["answer"]


def test_static_practice_bypasses_fresh_evidence_collection(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    monkeypatch.setattr(
        "services.context_retrieval.context_retrieval_service.ContextRequestBuilder.from_query_and_classification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected web search")
        ),
    )
    captured: list[object] = []
    result = build_orchestrated_doubt_solver_graph(
        _NoAnswerAdapter(),
        practice_launcher=lambda request: captured.append(request) or _launch_result(),
    ).invoke(
        _graph_state(
            "Create 2 Time and Work questions",
            {"intent": "practice", "subject": "math", "difficulty": "intermediate"},
        )
    )

    assert captured[0].requires_fresh_evidence is False
    assert captured[0].fresh_evidence is None
    assert result["practice_test_id"] == "practice-fresh"


def test_invalid_fresh_practice_count_does_not_search_or_launch(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    monkeypatch.setattr(
        "services.context_retrieval.context_retrieval_service.ContextRequestBuilder.from_query_and_classification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected web search")
        ),
    )
    launched: list[object] = []
    query = "Create 101 current affairs questions for SSC GD"

    result = build_orchestrated_doubt_solver_graph(
        _NoAnswerAdapter(),
        practice_launcher=lambda request: launched.append(request) or _launch_result(),
    ).invoke(
        _graph_state(
            query,
            {
                "intent": "practice",
                "subject": "general",
                "need_web_search": True,
                "web_search_reason": "current_affairs",
            },
        )
    )

    assert is_fresh_evidence_request_supported(query) is False
    assert launched == []
    assert "could not be started" in result["answer"]


def test_historical_current_affairs_window_and_evidence_filter_are_preserved() -> None:
    policy = WebSourcePolicyResolver().resolve(
        query="Create 2 current affairs questions from 2024",
        web_search_query="current affairs 2024",
        subject="general",
        topic="current_affairs",
        retrieval_tags=[],
        web_search_reason="current_affairs",
        source_strictness="authoritative_first",
        default_recent_days=30,
    )
    bundle = _build_fresh_evidence_bundle(
        policy=policy,
        search_query="current affairs 2024",
        items=[
            WebSearchItem(
                title="In-window event",
                url="https://pib.gov.in/2024-event",
                snippet="A usable factual event from 2024 with enough source detail.",
                published_at="2024-06-01",
            ),
            WebSearchItem(
                title="Stale event",
                url="https://pib.gov.in/2023-event",
                snippet="A usable but out-of-window event from 2023 with source detail.",
                published_at="2023-12-31",
            ),
            WebSearchItem(
                title="Duplicate event",
                url="https://pib.gov.in/2024-event",
                snippet="A duplicate source URL that must not be counted twice for evidence.",
                published_at="2024-06-01",
            ),
            WebSearchItem(
                title="Undated event",
                url="https://pib.gov.in/undated-event",
                snippet="A result without a source publication date cannot prove freshness.",
            ),
        ],
    )

    assert bundle.requested_window.start_date == "2024-01-01"
    assert bundle.requested_window.end_date == "2024-12-31"
    assert [item.url for item in bundle.items] == ["https://pib.gov.in/2024-event"]


def test_hundred_current_affairs_slots_accept_a_complete_mock_evidence_bundle() -> None:
    query = "Create 100 latest current affairs questions for SSC GD"
    request = resolve_practice_request(
        request_id="request-fresh-100",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query=query,
        subject="general",
        topic="current_affairs",
        difficulty="intermediate",
        language="english",
        exam_id="SSC_GD",
        exam_stage=None,
        freshness_requirement=resolve_practice_freshness_requirement(
            query,
            {"intent": "practice", "web_search_reason": "current_affairs"},
        ),
        fresh_evidence=_evidence(100),
    )

    assert request.requested_count == 100
    assert request.accepted_count == 100
    assert request.fresh_evidence is not None
    assert len(request.fresh_evidence.items) == 100


def test_fresh_evidence_survives_practice_request_reconstruction() -> None:
    request = _current_request()
    assessment = {
        "userId": request.user_id,
        "name": request.assessment_title,
        "meta": {
            "practiceRequest": {
                "requestId": request.request_id,
                "conversationId": request.conversation_id,
                "turnId": request.turn_id,
                "practiceType": request.practice_type.value,
                "requestedCount": request.requested_count,
                "acceptedCount": request.accepted_count,
                "subject": request.subject,
                "topic": request.topic,
                "difficulty": request.difficulty.value,
                "language": request.language,
                "examId": request.exam_id,
                "examStage": request.exam_stage,
                "examProfileId": request.exam_profile_id,
                "includeSolutions": request.include_solutions,
                "assessmentTitle": request.assessment_title,
                "requiresFreshEvidence": True,
                "freshnessReason": request.freshness_reason,
                "freshEvidence": request.fresh_evidence.model_dump(mode="json"),
            }
        },
    }

    rebuilt = _request(assessment)

    assert rebuilt.requires_fresh_evidence is True
    assert rebuilt.fresh_evidence == request.fresh_evidence


class _StructuredCapture:
    def __init__(self, content: str) -> None:
        self.content = content
        self.payloads: list[dict[str, object]] = []

    def generate_structured(self, **kwargs):
        self.payloads.append(json.loads(kwargs["user_content"]))
        return SimpleNamespace(
            content=self.content,
            route_decision=SimpleNamespace(route_id="general.verifier.default"),
            model="mock",
        )


def test_verifier_rejects_a_fresh_fact_without_a_selected_evidence_url() -> None:
    request = _current_request()
    bucket = DemandBucket(
        bucket_id="slot-bucket-001",
        subject="general",
        topic="current_affairs",
        difficulty=Difficulty.INTERMEDIATE,
        question_type=QuestionType.MCQ,
        required_count=2,
        question_intent="Assess current events",
        verification_policy=VerificationPolicy.MANDATORY,
    )
    slot = PlannerSlot(
        slot_id="slot-001",
        subject_id="general",
        topic_id="current_affairs",
        category_id="current_affairs",
        difficulty=Difficulty.INTERMEDIATE,
        complexity=Complexity.LOW,
        target_skill="recall",
        variation_hint="event_one",
        generator_route_hint="general.generator.intermediate",
    )
    question = GeneratedQuestion(
        schema_version="2",
        generation_item_id="item-1",
        bucket_id=bucket.bucket_id,
        slot_id=slot.slot_id,
        question="Which verified event happened in the supplied evidence?",
        question_type=QuestionType.MCQ,
        options=["A", "B", "C", "D"],
        canonical_options=[
            {"option_id": "0", "value": "A"},
            {"option_id": "1", "value": "B"},
            {"option_id": "2", "value": "C"},
            {"option_id": "3", "value": "D"},
        ],
        correct_option_id="0",
        correct_answer="A",
        answer_explanation="The supplied evidence supports option A.",
        solution="The supplied evidence supports option A.",
        subject="general",
        topic="current_affairs",
        difficulty=Difficulty.INTERMEDIATE,
    )
    orchestrator = _StructuredCapture(
        json.dumps(
            {
                "schema_version": "2",
                "generation_item_id": "item-1",
                "slot_id": "slot-001",
                "decision": "ACCEPT",
                "independently_solved_option_id": "0",
                "reason_codes": ["INDEPENDENT_SOLUTION_MATCH"],
            }
        )
    )

    result = RoutedQuestionVerifier(orchestrator).verify_slot(
        request=request,
        bucket=bucket,
        slot=slot,
        question=question,
    )

    assert result.decision is VerificationDecision.REGENERATE
    assert result.reason_codes == ["UNSUPPORTED_FACT"]
    assert orchestrator.payloads[0]["fresh_evidence"]["items"][0]["url"] == (
        "https://pib.gov.in/release-1"
    )


def test_generator_uses_only_slot_relevant_fresh_evidence() -> None:
    request = _current_request()
    bucket = DemandBucket(
        bucket_id="slot-bucket-001",
        subject="general",
        topic="current_affairs",
        difficulty=Difficulty.INTERMEDIATE,
        question_type=QuestionType.MCQ,
        required_count=2,
        question_intent="Assess current events",
        verification_policy=VerificationPolicy.MANDATORY,
    )
    slots = tuple(
        PlannerSlot(
            slot_id=f"slot-{index:03d}",
            subject_id="general",
            topic_id="current_affairs",
            category_id="current_affairs",
            difficulty=Difficulty.INTERMEDIATE,
            complexity=Complexity.LOW,
            target_skill=f"recall_{index}",
            variation_hint=f"event_{index}",
            generator_route_hint="general.generator.intermediate",
        )
        for index in (1, 2)
    )
    orchestrator = _StructuredCapture('{"questions":[]}')

    payload = RoutedQuestionGenerator(orchestrator)._slot_payload(
        request=request,
        bucket=bucket,
        group=GenerationGroup(
            group_id="slot-group-001",
            bucket_id=bucket.bucket_id,
            required_count=2,
            slot_ids=[slot.slot_id for slot in slots],
        ),
        slots=slots,
        exclude_normalized_texts=(),
        repair_feedback=(),
        repair_candidates_by_slot=None,
        repair_reason_codes_by_slot=None,
        replacement_wave=0,
        guidance_by_slot={},
    )

    assert [item["url"] for item in payload["fresh_evidence"]["items"]] == [
        "https://pib.gov.in/release-1",
        "https://pib.gov.in/release-2",
    ]
    assert [
        (entry["slot_id"], entry["items"][0]["url"])
        for entry in payload["fresh_evidence"]["evidence_by_slot"]
    ] == [
        ("slot-001", "https://pib.gov.in/release-1"),
        ("slot-002", "https://pib.gov.in/release-2"),
    ]
