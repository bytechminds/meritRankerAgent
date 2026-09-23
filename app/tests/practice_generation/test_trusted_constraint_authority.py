"""Trusted Practice composition is immutable after Request Intelligence accepts it."""

from __future__ import annotations

import json
from typing import Any

import pytest

from features.practice_generation.planning import (
    BlueprintManager,
    PracticeRequestConstraintCountError,
    deterministic_blueprint,
    parse_blueprint,
    resolve_practice_request,
    trusted_constraint_references,
)
from features.practice_generation.providers import RoutedPlannerProvider
from features.practice_generation.request_intelligence import token_id, tokenize_query
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor

INCIDENT_QUERY = (
    "create 50 questions of mock test from indian geography, polity, history , science, economy.."
)
_CONSTRAINTS = (
    ("indian geography", "Indian Geography", "geography"),
    ("polity", "Polity", "polity"),
    ("history", "History", "history"),
    ("science", "Science", "science"),
    ("economy", "Economy", "economics"),
)


class _Interpreter:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def interpret(self, **_kwargs: object) -> str:
        return self.raw


def _topic_span(query: str, phrase: str, normalized_name: str, subject_id: str) -> dict[str, Any]:
    tokens = tokenize_query(query)
    needle = [item.text.casefold() for item in tokenize_query(phrase)]
    haystack = [item.text.casefold() for item in tokens]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return {
                "tokenIds": [token_id(index) for index in range(start, start + len(needle))],
                "normalizedName": normalized_name,
                "subjectId": subject_id,
            }
    raise AssertionError(f"missing fixture phrase {phrase!r}")


def _interpretation(query: str, count: int = 50) -> str:
    return json.dumps(
        {
            "interpretationStatus": "RESOLVED",
            "requestedCount": count,
            "topics": [
                _topic_span(query, phrase, normalized_name, subject_id)
                for phrase, normalized_name, subject_id in _CONSTRAINTS
            ],
            "difficulty": {"mode": "UNSPECIFIED"},
        }
    )


def _request(query: str = INCIDENT_QUERY, count: int = 50):
    return resolve_practice_request(
        request_id="incident-request",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query=query,
        subject="general",
        topic=None,
        difficulty="intermediate",
        language="english",
        exam_id=None,
        exam_stage=None,
        request_interpreter=_Interpreter(_interpretation(query, count)),
    )


def _planner_slots(request, *, ref_overrides: dict[int, str | None] | None = None):
    references = trusted_constraint_references(request)
    slots = []
    for index in range(1, request.accepted_count + 1):
        reference, constraint = references[(index - 1) % len(references)]
        if ref_overrides and index in ref_overrides:
            reference = ref_overrides[index]
        slots.append(
            {
                "slot_id": f"slot-{index:03d}",
                "constraint_ref": reference,
                "subject_id": constraint.subject_id,
                "topic_id": constraint.topic_id,
                "category_id": constraint.topic_id,
                "difficulty": "intermediate",
                "complexity": "medium",
                "exam_ids": [],
                "question_type": "mcq",
                "target_skill": f"skill_{index}",
                "variation_hint": f"variation_{index}",
                "pattern_family_id": None,
                "generator_route_hint": "general.generator.intermediate",
                "reasoning_target": None,
                "trap_type": None,
                "not_same_when": [],
                "generation_group_hint": None,
            }
        )
    return slots


def _raw_plan(request, evidence: object = None) -> str:
    payload: dict[str, object] = {"slots": _planner_slots(request)}
    if evidence is not None:
        payload["requestedTopicEvidence"] = evidence
    return json.dumps(payload)


def test_incident_constraints_are_grounded_and_reference_stable() -> None:
    request = _request()

    assert [
        (reference, constraint.subject_id, constraint.topic_id, len(constraint.source_text))
        for reference, constraint in trusted_constraint_references(request)
    ] == [
        ("tc-001", "geography", "indian_geography", len("indian geography")),
        ("tc-002", "polity", "polity", len("polity")),
        ("tc-003", "history", "history", len("history")),
        ("tc-004", "science", "science", len("science")),
        ("tc-005", "economics", "economy", len("economy")),
    ]


@pytest.mark.parametrize(
    "planner_evidence",
    [
        None,
        [],
        [{"sourceText": "Geography", "topicId": "geography"}],
        [{"sourceText": "invented evidence", "topicId": "invented"}],
    ],
)
def test_trusted_plans_do_not_revalidate_planner_retyped_evidence(planner_evidence: object) -> None:
    request = _request()

    blueprint = parse_blueprint(_raw_plan(request, planner_evidence), request)

    assert len(blueprint.slots) == 50
    assert {slot.subject_id for slot in blueprint.slots} == {
        "geography",
        "polity",
        "history",
        "science",
        "economics",
    }
    assert {slot.constraint_ref for slot in blueprint.slots} == {
        "tc-001",
        "tc-002",
        "tc-003",
        "tc-004",
        "tc-005",
    }


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda slot: slot.update(constraint_ref="tc-999"), "CONSTRAINT_REF_INVALID"),
        (lambda slot: slot.update(constraint_ref=None), "CONSTRAINT_REF_INVALID"),
        (lambda slot: slot.update(subject_id="history"), "SUBJECT_MISMATCH"),
        (lambda slot: slot.update(topic_id="geography"), "TOPIC_MISMATCH"),
    ],
)
def test_trusted_plans_reject_invalid_constraint_identity(mutate, reason: str) -> None:
    request = _request()
    payload = {"slots": _planner_slots(request), "requestedTopicEvidence": None}
    mutate(payload["slots"][0])

    with pytest.raises(ValueError, match=reason):
        parse_blueprint(json.dumps(payload), request)


def test_trusted_plans_reject_omitted_constraint_coverage() -> None:
    request = _request()
    slots = _planner_slots(request)
    for slot in slots:
        slot["constraint_ref"] = "tc-001"
        slot["subject_id"] = "geography"
        slot["topic_id"] = "indian_geography"
        slot["category_id"] = "indian_geography"
    payload = {"slots": slots, "requestedTopicEvidence": None}

    with pytest.raises(ValueError, match="CONSTRAINT_COVERAGE_INVALID"):
        parse_blueprint(json.dumps(payload), request)


def test_trusted_planner_payload_carries_references_not_source_spans() -> None:
    request = _request()
    executor = MockModelExecutor(content='{"slots":[],"requestedTopicEvidence":null}')

    RoutedPlannerProvider(LlmOrchestrator(model_executor=executor)).plan(request, tier="strong")

    assert executor.last_messages is not None
    payload = json.loads(executor.last_messages[1].content)
    assert payload["request_constraints_authority"] == "context_only"
    assert payload["trusted_constraints"] == [
        {"constraint_ref": "tc-001", "subject_id": "geography", "topic_id": "indian_geography"},
        {"constraint_ref": "tc-002", "subject_id": "polity", "topic_id": "polity"},
        {"constraint_ref": "tc-003", "subject_id": "history", "topic_id": "history"},
        {"constraint_ref": "tc-004", "subject_id": "science", "topic_id": "science"},
        {"constraint_ref": "tc-005", "subject_id": "economics", "topic_id": "economy"},
    ]
    assert all("sourceText" not in item for item in payload["trusted_constraints"])


def test_deterministic_fallback_uses_trusted_references() -> None:
    class _FailingPlanner:
        def plan(self, *_args: object, **_kwargs: object) -> str:
            raise ValueError("invalid planner output")

    result = BlueprintManager(_FailingPlanner()).build(_request())

    assert result.deterministic_fallback is True
    assert len(result.blueprint.slots) == 50
    assert {slot.constraint_ref for slot in result.blueprint.slots} == {
        "tc-001",
        "tc-002",
        "tc-003",
        "tc-004",
        "tc-005",
    }


def test_constraint_count_infeasibility_fails_before_planning() -> None:
    query = INCIDENT_QUERY.replace("50", "3", 1)

    with pytest.raises(PracticeRequestConstraintCountError) as exc_info:
        _request(query, count=3)

    assert exc_info.value.reason_code == "PRACTICE_REQUEST_CONSTRAINT_COUNT_INFEASIBLE"


def test_deterministic_blueprint_preserves_exact_trusted_identity() -> None:
    request = _request()
    blueprint = deterministic_blueprint(request)

    assert [slot.constraint_ref for slot in blueprint.slots[:5]] == [
        "tc-001",
        "tc-002",
        "tc-003",
        "tc-004",
        "tc-005",
    ]
