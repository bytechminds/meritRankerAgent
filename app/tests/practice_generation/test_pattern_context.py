"""Practice-only tests for the feature-gated canonical Pattern context provider."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from features.practice_generation.pattern_context import (
    NoOpPatternContextProvider,
    RuntimePatternContextProvider,
)
from features.practice_generation.schemas import PlannerSlot, PracticeGenerationRequest
from observability.context import bind_execution_context
from retrieval.pattern_intelligence import (
    PatternRuntimeService,
    VectorPatternCandidate,
)


class _Finder:
    def __init__(self, candidates: Sequence[VectorPatternCandidate]) -> None:
        self._candidates = candidates
        self.calls: list[tuple[str, str | None, int]] = []

    def find_candidates(
        self,
        *,
        query: str,
        subject: str | None,
        limit: int,
    ) -> Sequence[VectorPatternCandidate]:
        self.calls.append((query, subject, limit))
        return self._candidates


class _UnavailableFinder(_Finder):
    def find_candidates(
        self,
        *,
        query: str,
        subject: str | None,
        limit: int,
    ) -> Sequence[VectorPatternCandidate]:
        del query, subject, limit
        raise RuntimeError("vector unavailable")


class _PatternStore:
    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self._records = records
        self.calls: list[tuple[str, ...]] = []

    def batch_fetch(self, *, pattern_ids: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        ids = tuple(pattern_ids)
        self.calls.append(ids)
        return [self._records[pattern_id] for pattern_id in ids if pattern_id in self._records]


def _request() -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="practice-pattern-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        original_query="Create two algebra questions",
        practice_type="QUIZ",
        requested_count=2,
        accepted_count=2,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra practice",
    )


def _slot(slot_id: str, *, target_skill: str) -> PlannerSlot:
    return PlannerSlot(
        slot_id=slot_id,
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty="intermediate",
        complexity="medium",
        exam_ids=["CAT"],
        question_type="mcq",
        target_skill=target_skill,
        variation_hint="new_values",
        generator_route_hint="quant.generator",
        reasoning_target="isolate_variable",
    )


def _pattern() -> dict[str, Any]:
    return {
        "patternId": "pattern-algebra-1",
        "status": "active",
        "subject": "quant",
        "topic": "algebra",
        "coreConcept": "linear equation",
        "complexityLevel": "medium",
        "exams": ["CAT"],
        "category": "algebra",
        "questionType": "mcq",
        "kbCurrentVersionHash": "version-1",
        "patternGraph": {
            "given": [{"identity": "single_variable_equation"}],
            "condition": [{"identity": "preserve_equality"}],
            "find": {"identity": "variable_value"},
            "operations": [{"identity": "isolate_variable"}],
            "not_same_when": ["nonlinear equations"],
        },
    }


def test_provider_groups_compatible_slots_and_projects_only_guidance() -> None:
    finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1",
                score=0.9,
                versionHash="version-1",
            )
        ]
    )
    store = _PatternStore({"pattern-algebra-1": _pattern()})
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=finder, pattern_store=store),
        max_candidates=12,
    )
    slots = (
        _slot("slot-001", target_skill="solve_linear_equation"),
        _slot("slot-002", target_skill="solve_linear_equation"),
    )

    result = provider.resolve_slots(request=_request(), slots=slots)

    assert result.retrieval_group_count == 1
    assert result.vector_query_count == 1
    assert set(result.selections_by_slot) == {"slot-001", "slot-002"}
    assert set(result.guidance_by_slot) == {"slot-001", "slot-002"}
    assert finder.calls and len(finder.calls) == 1
    assert store.calls == [("pattern-algebra-1",)]
    context = result.guidance_by_slot["slot-001"].model_dump(by_alias=True)
    assert context["patternId"] == "pattern-algebra-1"
    assert "correctAnswer" not in context
    assert "solution" not in context


def test_provider_separates_slots_with_distinct_exam_or_reasoning_requirements() -> None:
    finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1",
                score=0.9,
                versionHash="version-1",
            )
        ]
    )
    store = _PatternStore({"pattern-algebra-1": _pattern()})
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=finder, pattern_store=store),
        max_candidates=12,
    )
    cat_slot = _slot("slot-001", target_skill="solve_linear_equation")
    ssc_slot = _slot("slot-002", target_skill="check_equation_solution").model_copy(
        update={"exam_ids": ["SSC"]}
    )

    result = provider.resolve_slots(request=_request(), slots=(cat_slot, ssc_slot))

    assert result.retrieval_group_count == 2
    assert result.vector_query_count == 2
    assert len(finder.calls) == 2


def test_provider_fails_closed_when_pattern_cannot_prove_slot_question_type() -> None:
    finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1",
                score=0.9,
                versionHash="version-1",
            )
        ]
    )
    store = _PatternStore({"pattern-algebra-1": _pattern() | {"questionType": "integer"}})
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=finder, pattern_store=store),
        max_candidates=12,
    )

    result = provider.resolve_slots(
        request=_request(),
        slots=(_slot("slot-001", target_skill="solve_linear_equation"),),
    )

    assert result.guidance_by_slot == {}
    assert result.ignored_pattern_count == 1


def test_unavailable_pattern_vector_returns_empty_optional_context() -> None:
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=_UnavailableFinder(()),
            pattern_store=_PatternStore({}),
        ),
        max_candidates=12,
    )

    result = provider.resolve_slots(
        request=_request(),
        slots=(_slot("slot-001", target_skill="solve_linear_equation"),),
    )

    assert result.guidance_by_slot == {}
    assert result.reuse_questions_by_slot == {}
    assert result.warnings == ("vector_discovery_unavailable",)


def test_provider_keeps_maximum_length_request_id_within_runtime_contract() -> None:
    finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1",
                score=0.9,
                versionHash="version-1",
            )
        ]
    )
    store = _PatternStore({"pattern-algebra-1": _pattern()})
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=finder, pattern_store=store),
        max_candidates=12,
    )
    request = _request().model_copy(update={"request_id": "r" * 128})

    result = provider.resolve_slots(
        request=request,
        slots=(_slot("slot-001", target_skill="solve_linear_equation"),),
    )

    assert result.guidance_by_slot
    assert finder.calls


def test_persisted_server_selection_rehydrates_exact_pattern_without_vector_lookup() -> None:
    class _UnexpectedFinder(_Finder):
        def find_candidates(
            self,
            *,
            query: str,
            subject: str | None,
            limit: int,
        ) -> Sequence[VectorPatternCandidate]:
            raise AssertionError("persisted server selection must skip vector discovery")

    seed_finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1",
                score=0.9,
                versionHash="version-1",
            )
        ]
    )
    store = _PatternStore({"pattern-algebra-1": _pattern()})
    seed_provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=seed_finder, pattern_store=store),
        max_candidates=12,
    )
    slot = _slot("slot-001", target_skill="solve_linear_equation")
    first = seed_provider.resolve_slots(request=_request(), slots=(slot,))

    exact_provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(vector_finder=_UnexpectedFinder(()), pattern_store=store),
        max_candidates=12,
    )
    rehydrated = exact_provider.resolve_slots(
        request=_request(),
        slots=(slot,),
        persisted_selections=first.selections_by_slot,
    )

    assert rehydrated.vector_query_count == 0
    assert rehydrated.guidance_by_slot[slot.slot_id].pattern_id == "pattern-algebra-1"


class _ScopedQuestionStore:
    def __init__(self, questions: Sequence[Mapping[str, Any]]) -> None:
        self._questions = questions
        self.calls: list[str] = []

    def list_by_pattern_id(
        self,
        *,
        pattern_id: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        self.calls.append(pattern_id)
        return [question for question in self._questions if question["patternId"] == pattern_id][
            :limit
        ]


def _difficulty_slot(slot_id: str, *, difficulty: str, complexity: str) -> PlannerSlot:
    return PlannerSlot(
        slot_id=slot_id,
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty=difficulty,
        complexity=complexity,
        exam_ids=["CAT"],
        question_type="mcq",
        target_skill="solve_linear_equation",
        variation_hint="new_values",
        generator_route_hint="quant.generator",
        reasoning_target="isolate_variable",
    )


def _reusable_question(index: int) -> dict[str, Any]:
    return {
        "qbId": f"qb-{index}",
        "patternId": "pattern-algebra-1",
        "patternVersionHash": "version-1",
        "patternLinkEvidence": "VERIFIED_GENERATION",
        "question": f"Solve the linear equation variant {index}.",
        "answers": '{"options": ["1", "2", "3", "4"]}',
        "correctAnswer": "1",
        "explanation": "Isolate the variable.",
        "category": "algebra",
        "difficulty": "MEDIUM",
        "updatedAt": "2026-08-10T00:00:00Z",
        "meta": {
            "status": "ACTIVE",
            "qualityStatus": "VERIFIED",
            "reusable": True,
            "visibility": "PLATFORM",
            "subject": "math",
            "topic": "algebra",
            "questionType": "mcq",
            "language": "english",
            "examIds": ["CAT"],
        },
    }


def _reuse_provider(
    *,
    question_count: int,
) -> tuple[RuntimePatternContextProvider, _Finder, _ScopedQuestionStore]:
    finder = _Finder(
        [VectorPatternCandidate(patternId="pattern-algebra-1", score=0.92, versionHash="version-1")]
    )
    store = _ScopedQuestionStore(
        [_reusable_question(index) for index in range(1, question_count + 1)]
    )
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=finder,
            pattern_store=_PatternStore({"pattern-algebra-1": _pattern()}),
            playable_question_store=store,
            reuse_enabled=True,
        ),
        max_candidates=12,
    )
    return provider, finder, store


def test_grouped_demand_reuses_many_slots_and_leaves_an_exact_deficit() -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 9)
    ]

    result = provider.resolve_slots(
        request=_request(),
        slots=slots,
        student_history_checked=True,
    )

    assert len(result.reuse_questions_by_slot) == 5
    assert len(result.guidance_by_slot) == 3
    assert len(result.reuse_questions_by_slot) + len(result.guidance_by_slot) == len(slots)
    assert len({q.question_id for q in result.reuse_questions_by_slot.values()}) == 5


def test_reuse_never_exceeds_the_planned_slot_count() -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 4)
    ]

    result = provider.resolve_slots(
        request=_request(),
        slots=slots,
        student_history_checked=True,
    )

    assert len(result.reuse_questions_by_slot) == 3
    assert result.guidance_by_slot == {}


def test_attempted_questions_are_excluded_from_practice_direct_reuse() -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 9)
    ]

    result = provider.resolve_slots(
        request=_request(),
        slots=slots,
        seen_question_ids=["qb-2", "qb-4"],
        student_history_checked=True,
    )

    assert {q.question_id for q in result.reuse_questions_by_slot.values()} == {
        "qb-1",
        "qb-3",
        "qb-5",
    }
    assert len(result.guidance_by_slot) == 5


def test_inventory_shortage_never_alters_planned_difficulty_distribution() -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    planned = {"basic": 5, "intermediate": 9, "advanced": 6}
    complexity_by_difficulty = {"basic": "low", "intermediate": "medium", "advanced": "high"}
    slots: list[PlannerSlot] = []
    index = 1
    for difficulty, count in planned.items():
        for _ in range(count):
            slots.append(
                _difficulty_slot(
                    f"slot-{index:03d}",
                    difficulty=difficulty,
                    complexity=complexity_by_difficulty[difficulty],
                )
            )
            index += 1

    result = provider.resolve_slots(
        request=_request(),
        slots=slots,
        student_history_checked=True,
    )

    slots_by_id = {slot.slot_id: slot for slot in slots}
    resolved = {
        difficulty: sum(
            1
            for slot_id in (
                *result.reuse_questions_by_slot,
                *result.guidance_by_slot,
            )
            if slots_by_id[slot_id].difficulty.value == difficulty
        )
        for difficulty in planned
    }
    # Only the medium-complexity Pattern is compatible, so the other bands stay
    # untouched — an inventory shortage never re-balances the planner.
    assert resolved["intermediate"] == planned["intermediate"]
    assert all(
        slots_by_id[slot_id].difficulty.value == "intermediate"
        for slot_id in result.reuse_questions_by_slot
    )


def test_hundred_slot_plan_queries_per_demand_group_not_per_slot() -> None:
    provider, finder, _ = _reuse_provider(question_count=5)
    difficulties = ("basic", "intermediate", "advanced")
    complexity_by_difficulty = {"basic": "low", "intermediate": "medium", "advanced": "high"}
    slots = [
        _difficulty_slot(
            f"slot-{index:03d}",
            difficulty=difficulties[index % 3],
            complexity=complexity_by_difficulty[difficulties[index % 3]],
        )
        for index in range(1, 101)
    ]

    result = provider.resolve_slots(
        request=_request(),
        slots=slots,
        student_history_checked=True,
    )

    assert len(slots) == 100
    assert result.retrieval_group_count == 3
    assert result.vector_query_count <= result.retrieval_group_count
    assert len(finder.calls) < len(slots)
    reused = len(result.reuse_questions_by_slot)
    guided = len(result.guidance_by_slot)
    assert reused + (len(slots) - reused) == 100
    assert reused + guided <= 100


def test_pattern_off_performs_zero_vector_and_hydration_work() -> None:
    from features.practice_generation.pattern_context import NoOpPatternContextProvider

    finder = _UnavailableFinder([])
    store = _PatternStore({"pattern-algebra-1": _pattern()})
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 9)
    ]

    result = NoOpPatternContextProvider().resolve_slots(
        request=_request(),
        slots=slots,
        student_history_checked=True,
    )

    assert result.reuse_questions_by_slot == {}
    assert result.guidance_by_slot == {}
    assert result.vector_query_count == 0
    assert result.expansion_query_count == 0
    assert result.retrieval_group_count == 0
    assert finder.calls == []
    assert store.calls == []


def _pattern_events(records: list, *names: str) -> list[dict[str, Any]]:
    wanted = set(names)
    return [
        record.observability_event
        for record in records
        if getattr(record, "observability_event", {}).get("event") in wanted
    ]


def test_pattern_retrieval_started_and_vector_query_completed_events_fire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 4)
    ]

    with bind_execution_context(activity_id="practice-triangle"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    started = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_STARTED")
    completed = _pattern_events(caplog.records, "PATTERN_VECTOR_QUERY_COMPLETED")
    assert len(started) == 1
    assert started[0]["details"]["subject"] == "math"
    assert started[0]["details"]["topic"] == "algebra"
    assert started[0]["details"]["requiredcount"] == 3
    assert started[0]["details"]["candidatelimit"] == 12
    assert len(completed) == 1
    assert completed[0]["status"] == "completed"
    assert completed[0]["details"]["vectorsearchcount"] >= 1
    assert completed[0]["details"]["candidatecount"] >= 1


def test_multiple_ranked_candidate_decision_events_are_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two compatible Patterns, each holding 1 reusable question for a 2-slot demand.

    A single compatible Pattern stops the scan early when reuse-gathering is off
    (service.py: ``if not gather_reuse: return`` after the first GUIDANCE_SAFE
    match). Ranking beyond the first candidate is only observable when reuse
    demand outlives one Pattern's inventory, which this fixture forces.
    """
    finder = _Finder(
        [
            VectorPatternCandidate(
                patternId="pattern-algebra-1", score=0.95, versionHash="version-1"
            ),
            VectorPatternCandidate(
                patternId="pattern-algebra-2", score=0.90, versionHash="version-1"
            ),
        ]
    )
    second = _pattern()
    second["patternId"] = "pattern-algebra-2"
    store = _ScopedQuestionStore(
        [
            _reusable_question(1),
            _reusable_question(2) | {"patternId": "pattern-algebra-2", "qbId": "qb-2"},
        ]
    )
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=finder,
            pattern_store=_PatternStore(
                {"pattern-algebra-1": _pattern(), "pattern-algebra-2": second}
            ),
            playable_question_store=store,
            reuse_enabled=True,
        ),
        max_candidates=12,
    )
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in (1, 2)
    ]

    with bind_execution_context(activity_id="practice-multi"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    decisions = _pattern_events(caplog.records, "PATTERN_CANDIDATE_DECISION")
    assert [d["details"]["rank"] for d in decisions] == [1, 2]
    assert {d["details"]["patternid"] for d in decisions} == {
        "pattern-algebra-1",
        "pattern-algebra-2",
    }
    assert all(d["details"]["tier"] == "GUIDANCE_SAFE" for d in decisions)


def test_ignore_tier_is_visible_for_an_inactive_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inactive = _pattern()
    inactive["patternId"] = "pattern-inactive"
    inactive["status"] = "retired"
    finder = _Finder(
        [VectorPatternCandidate(patternId="pattern-inactive", score=0.95, versionHash="version-1")]
    )
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=finder,
            pattern_store=_PatternStore({"pattern-inactive": inactive}),
        ),
        max_candidates=12,
    )
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with bind_execution_context(activity_id="practice-ignore"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(request=_request(), slots=slots)

    decisions = _pattern_events(caplog.records, "PATTERN_CANDIDATE_DECISION")
    assert len(decisions) == 1
    assert decisions[0]["details"]["tier"] == "IGNORE"
    assert decisions[0]["details"]["reasoncode"] == "pattern_not_active"
    completed = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_COMPLETED")
    assert completed[-1]["details"]["directreusecount"] == 0
    assert completed[-1]["details"]["generationdeficit"] == 1


def test_direct_reuse_purpose_is_emitted_for_eligible_linked_questions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 4)
    ]

    with bind_execution_context(activity_id="practice-reuse"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    linked = _pattern_events(caplog.records, "PATTERN_LINKED_QUESTION_DECISION")
    assert len(linked) == 3
    assert all(event["details"]["purpose"] == "DIRECT_REUSE" for event in linked)
    assert all(event["details"]["directreuseeligible"] is True for event in linked)
    assert all(event["details"]["studentseen"] is False for event in linked)
    assert len({event["details"]["questionid"] for event in linked}) == 3
    completed = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_COMPLETED")
    assert completed[-1]["details"]["directreusecount"] == 3
    assert completed[-1]["details"]["generationdeficit"] == 0


def test_guidance_safe_never_exposes_a_linked_question_reference_today(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Documents a real gap: Case 2/3 GENERATION_REFERENCE has no data source.

    PatternRuntimeService._resolve defines
    ``tier = REUSE_SAFE if reuse_questions else GUIDANCE_SAFE`` for Practice mode
    (service.py). Reuse_questions is therefore always empty whenever tier is
    GUIDANCE_SAFE, by construction — there is no linked-question reference
    surfaced for Practice guidance today, only PatternGraph structural
    constraints. This is a Pattern Intelligence behavior gap, not an
    observability gap, and is intentionally NOT fixed here.
    """
    finder = _Finder(
        [VectorPatternCandidate(patternId="pattern-algebra-1", score=0.92, versionHash="version-1")]
    )
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=finder,
            pattern_store=_PatternStore({"pattern-algebra-1": _pattern()}),
            # No playable_question_store / reuse_enabled: guidance-only runtime.
        ),
        max_candidates=12,
    )
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with bind_execution_context(activity_id="practice-guidance-only"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(request=_request(), slots=slots)

    assert _pattern_events(caplog.records, "PATTERN_LINKED_QUESTION_DECISION") == []
    completed = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_COMPLETED")
    assert completed[-1]["details"]["linkedquestionsloaded"] == 0
    assert completed[-1]["details"]["directreusecount"] == 0
    assert completed[-1]["details"]["generationdeficit"] == 1


def test_pattern_disabled_emits_zero_pattern_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]
    with caplog.at_level(logging.DEBUG, logger="agent.observability"):
        NoOpPatternContextProvider().resolve_slots(request=_request(), slots=slots)

    assert (
        _pattern_events(
            caplog.records,
            "PATTERN_RETRIEVAL_STARTED",
            "PATTERN_VECTOR_QUERY_COMPLETED",
            "PATTERN_CANDIDATE_DECISION",
            "PATTERN_LINKED_QUESTION_DECISION",
            "PATTERN_RETRIEVAL_COMPLETED",
        )
        == []
    )


def test_production_level_shows_only_the_aggregate_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with bind_execution_context(activity_id="practice-prod"):
        with caplog.at_level(logging.INFO, logger="agent.observability"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    assert (
        _pattern_events(
            caplog.records,
            "PATTERN_RETRIEVAL_STARTED",
            "PATTERN_VECTOR_QUERY_COMPLETED",
            "PATTERN_CANDIDATE_DECISION",
            "PATTERN_LINKED_QUESTION_DECISION",
        )
        == []
    )
    completed = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_COMPLETED")
    assert len(completed) == 1
    assert completed[0]["details"]["directreusecount"] == 1


def test_concurrent_pattern_operations_carry_correct_test_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with caplog.at_level(logging.DEBUG, logger="agent.observability"):
        with bind_execution_context(activity_id="practice-A"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )
        with bind_execution_context(activity_id="practice-B"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    started = _pattern_events(caplog.records, "PATTERN_RETRIEVAL_STARTED")
    assert [event["details"]["testid"] for event in started] == [
        "practice-A",
        "practice-B",
    ]


def test_no_educational_content_leaks_in_pattern_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _, _ = _reuse_provider(question_count=5)
    slots = [
        _difficulty_slot(f"slot-{index:03d}", difficulty="intermediate", complexity="medium")
        for index in range(1, 4)
    ]

    with bind_execution_context(activity_id="practice-privacy"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(
                request=_request(),
                slots=slots,
                student_history_checked=True,
            )

    events = _pattern_events(
        caplog.records,
        "PATTERN_RETRIEVAL_STARTED",
        "PATTERN_VECTOR_QUERY_COMPLETED",
        "PATTERN_CANDIDATE_DECISION",
        "PATTERN_LINKED_QUESTION_DECISION",
        "PATTERN_RETRIEVAL_COMPLETED",
    )
    assert events
    for event in events:
        detail_values = " ".join(str(v) for v in event["details"].values())
        for forbidden in (
            "Solve the linear equation",  # question text from _reusable_question
            "Isolate the variable",  # explanation text
            "given",
            "condition",
            "not_same_when",
        ):
            assert forbidden not in detail_values


def test_vector_result_count_distinguishes_empty_index_from_filtered_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """S3 returned rows, but every score fell below the confidence floor.

    Regression for a real ambiguity found against Dev: candidateCount alone
    cannot distinguish "nothing indexed" from "indexed but below threshold".
    """
    weak = VectorPatternCandidate(
        patternId="pattern-algebra-1", score=0.10, versionHash="version-1"
    )
    finder = _Finder([weak])
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=finder,
            pattern_store=_PatternStore({"pattern-algebra-1": _pattern()}),
        ),
        max_candidates=12,
    )
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with bind_execution_context(activity_id="practice-weak-vector"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(request=_request(), slots=slots)

    completed_vector = _pattern_events(caplog.records, "PATTERN_VECTOR_QUERY_COMPLETED")
    assert completed_vector[0]["details"]["candidatecount"] == 0
    assert completed_vector[0]["details"]["strongcandidatecount"] == 0
    assert completed_vector[0]["details"]["vectorresultcount"] == 1
    assert completed_vector[0]["details"]["topvectorscore"] == pytest.approx(0.10)


def test_empty_vector_index_reports_zero_raw_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = RuntimePatternContextProvider(
        runtime=PatternRuntimeService(
            vector_finder=_Finder([]),
            pattern_store=_PatternStore({}),
        ),
        max_candidates=12,
    )
    slots = [_difficulty_slot("slot-001", difficulty="intermediate", complexity="medium")]

    with bind_execution_context(activity_id="practice-empty-index"):
        with caplog.at_level(logging.DEBUG, logger="agent.observability"):
            provider.resolve_slots(request=_request(), slots=slots)

    completed_vector = _pattern_events(caplog.records, "PATTERN_VECTOR_QUERY_COMPLETED")
    assert completed_vector[0]["details"]["vectorresultcount"] == 0
    assert "topvectorscore" not in completed_vector[0]["details"]
