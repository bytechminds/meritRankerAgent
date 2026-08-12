"""Practice-only tests for the feature-gated canonical Pattern context provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from features.practice_generation.pattern_context import RuntimePatternContextProvider
from features.practice_generation.schemas import PlannerSlot, PracticeGenerationRequest
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
    store = _PatternStore(
        {"pattern-algebra-1": _pattern() | {"questionType": "integer"}}
    )
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
