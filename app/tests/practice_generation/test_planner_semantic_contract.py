"""Models own subject/topic semantics; deterministic code validates structure only.

Pins the 40Q incident (Request Intelligence judged "Indian Constitution" polity and a
static vocabulary rejected it), the single route-family definition, the compact
planner slot intent (target_skill / concept / pattern_hint) and its delivery to the
generator.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import features.practice_generation.metadata_normalization as metadata_normalization
from features.practice_generation.planning import parse_blueprint, resolve_practice_request
from features.practice_generation.providers import RoutedPlannerProvider, RoutedQuestionGenerator
from features.practice_generation.request_intelligence import token_id, tokenize_query
from features.practice_generation.schemas import (
    GenerationGroup,
    PlannerSlot,
    planner_generation_schema,
)
from schemas.practice_request_intelligence import (
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA,
    PRACTICE_SUBJECT_FAMILIES,
)
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor

INCIDENT = (
    "create mock test of 40 questions. from indian constitution, "
    "math arithmetic, reasoning, general science"
)
# (selected words, normalizedName, family the model assigned)
INCIDENT_TOPICS = (
    ("indian constitution", "Indian Constitution", "polity"),
    ("math arithmetic", "Arithmetic", "math"),
    ("reasoning", "Reasoning", "reasoning"),
    ("general science", "General Science", "science"),
)


def _token_ids(query: str, words: str) -> list[str]:
    tokens = tokenize_query(query)
    wanted = words.split()
    for start in range(len(tokens)):
        window = tokens[start : start + len(wanted)]
        if [token.text.casefold() for token in window] == wanted:
            return [token_id(token.index) for token in window]
    raise AssertionError(words)


class _Interpreter:
    def __init__(self, query: str, topics: tuple[tuple[str, str, str], ...]) -> None:
        self._raw = json.dumps(
            {
                "interpretationStatus": "RESOLVED",
                "requestedCount": 40,
                "topics": [
                    {
                        "tokenIds": _token_ids(query, words),
                        "normalizedName": name,
                        "subjectId": family,
                    }
                    for words, name, family in topics
                ],
                "difficulty": {"mode": "UNSPECIFIED"},
            }
        )

    def interpret(self, **_kwargs) -> str:
        return self._raw


def _incident_request(query: str = INCIDENT):
    return resolve_practice_request(
        request_id="incident", user_id="u", conversation_id="c", turn_id="t",
        query=query, subject="general", topic=None, difficulty="intermediate",
        language="english", exam_id="CAT", exam_stage=None,
        request_interpreter=_Interpreter(query, INCIDENT_TOPICS),
    )


def _planner_output(request, **slot_overrides) -> str:
    references = [f"tc-{index:03d}" for index in range(1, 5)]
    slots = []
    for index in range(1, request.accepted_count + 1):
        constraint = request.trusted_constraints[(index - 1) % 4]
        slots.append(
            {
                "slot_id": f"slot-{index:03d}",
                "constraint_ref": references[(index - 1) % 4],
                "subject_id": constraint.subject_id,
                "topic_id": constraint.topic_id,
                "category_id": constraint.topic_id,
                "difficulty": "intermediate",
                "complexity": "medium",
                "target_skill": f"determine outcome {index}",
                "concept": "successive percentage relation",
                "pattern_hint": f"Given two linked quantities, ask for the third ({index}).",
                "pattern_family_id": None,
                "trap_type": None,
                "not_same_when": [],
                "generation_group_hint": None,
                **slot_overrides,
            }
        )
    return json.dumps({"slots": slots, "requestedTopicEvidence": None})


class TestOneRouteFamilyDefinition:
    def test_every_subject_gate_uses_the_canonical_families(self) -> None:
        planner_subject = planner_generation_schema()["properties"]["slots"]["items"][
            "properties"
        ]["subject_id"]
        ri_subject = PRACTICE_REQUEST_INTELLIGENCE_SCHEMA["properties"]["topics"]["items"][
            "properties"
        ]["subjectId"]

        assert planner_subject["enum"] == list(PRACTICE_SUBJECT_FAMILIES)
        assert ri_subject["enum"] == [*PRACTICE_SUBJECT_FAMILIES, None]
        assert metadata_normalization._SUPPORTED_SUBJECTS == frozenset(PRACTICE_SUBJECT_FAMILIES)


class TestIncidentComposition:
    def test_model_assigned_families_survive_as_trusted_constraints(self) -> None:
        request = _incident_request()

        assert request.accepted_count == 40
        assert [(c.subject_id, c.topic_id) for c in request.trusted_constraints] == [
            ("polity", "indian_constitution"),
            ("math", "arithmetic"),
            ("reasoning", "reasoning"),
            ("science", "general_science"),
        ]

    def test_unknown_token_selection_is_still_rejected(self) -> None:
        interpreter = _Interpreter(INCIDENT, INCIDENT_TOPICS)
        interpreter._raw = interpreter._raw.replace('"T', '"T9', 1)

        request = resolve_practice_request(
            request_id="bad", user_id="u", conversation_id="c", turn_id="t",
            query=INCIDENT, subject="general", topic=None, difficulty="intermediate",
            language="english", exam_id="CAT", exam_stage=None,
            request_interpreter=interpreter,
        )

        assert request.trusted_constraints == ()


class TestPlannerSlotContract:
    def test_planner_output_keeps_intent_and_server_fills_owned_fields(self) -> None:
        request = _incident_request()

        blueprint = parse_blueprint(_planner_output(request), request)

        slot = blueprint.slots[0]
        assert slot.concept == "successive percentage relation"
        assert slot.pattern_hint == "Given two linked quantities, ask for the third (1)."
        assert slot.exam_ids == ["CAT"]
        assert slot.question_type.value == "mcq"
        assert (slot.subject_id, slot.generator_route_hint) == (
            "polity", "polity.generator.intermediate"
        )
        assert {s.subject_id for s in blueprint.slots} == {
            "polity", "math", "reasoning", "science"
        }

    def test_empty_pattern_hint_is_rejected(self) -> None:
        request = _incident_request()

        with pytest.raises((ValueError, ValidationError)):
            parse_blueprint(_planner_output(request, pattern_hint=""), request)

    def test_family_outside_the_canonical_set_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlannerSlot.model_validate(
                {
                    "slot_id": "slot-001", "subject_id": "indian_polity",
                    "topic_id": "t", "category_id": "t", "difficulty": "basic",
                    "complexity": "low", "target_skill": "x",
                    "generator_route_hint": "general.generator.basic",
                }
            )

    def test_blueprint_persisted_before_concept_still_loads(self) -> None:
        slot = PlannerSlot.model_validate(
            {
                "slot_id": "slot-001", "subject_id": "math", "topic_id": "algebra",
                "category_id": "algebra", "difficulty": "basic", "complexity": "low",
                "target_skill": "solve", "variation_hint": "vary_values",
                "reasoning_target": "isolate_variable",
                "generator_route_hint": "math.generator.basic",
            }
        )

        assert (slot.concept, slot.pattern_hint) == (None, None)
        assert slot.variation_hint == "vary_values"


class TestIntentDelivery:
    def test_generator_receives_slot_intent_and_the_rule_to_follow_it(self) -> None:
        request = _incident_request()
        blueprint = parse_blueprint(_planner_output(request), request)
        slot = blueprint.slots[0]
        bucket = next(b for b in blueprint.buckets if b.subject == slot.subject_id)
        executor = MockModelExecutor(content='{"questions":[]}')

        RoutedQuestionGenerator(LlmOrchestrator(model_executor=executor)).generate_slots(
            request=request,
            bucket=bucket,
            group=GenerationGroup(
                group_id="g1", bucket_id=bucket.bucket_id, required_count=1,
                slot_ids=[slot.slot_id],
            ),
            slots=(slot,),
            exclude_normalized_texts=(),
        )

        payload = json.loads(executor.last_messages[1].content)
        sent = payload["slots"][0]
        assert (sent["target_skill"], sent["concept"], sent["pattern_hint"]) == (
            slot.target_skill, slot.concept, slot.pattern_hint
        )
        assert "`pattern_hint` structure, with fresh wording and values" in (
            executor.last_messages[0].content
        )

    def test_reference_question_in_the_request_reaches_the_planner(self) -> None:
        query = (
            "make 40 similar questions: A shopkeeper marks an article 25% above cost "
            "and gives 10% discount. Find the gain percent."
        )
        request = resolve_practice_request(
            request_id="ref", user_id="u", conversation_id="c", turn_id="t",
            query=query, subject="math", topic=None, difficulty="intermediate",
            language="english", exam_id=None, exam_stage=None,
        )
        executor = MockModelExecutor(content='{"slots":[],"requestedTopicEvidence":null}')

        RoutedPlannerProvider(LlmOrchestrator(model_executor=executor)).plan(
            request, tier="strong"
        )

        payload = json.loads(executor.last_messages[1].content)
        assert "marks an article 25% above cost" in payload["request_constraints"]
        assert "any reference question; never copy it" in executor.last_messages[0].content


class TestSimilarQuestionPlanning:
    def test_similar_question_with_trusted_topic_still_reaches_the_planner(self) -> None:
        from features.practice_generation.planning import select_planning_mode
        from features.practice_generation.schemas import PracticeType

        request = _incident_request()
        similar = request.model_copy(
            update={
                "practice_type": PracticeType.SIMILAR_QUESTION,
                "accepted_count": 10,
                "requested_count": 10,
            }
        )
        quick = request.model_copy(update={"accepted_count": 10, "requested_count": 10})

        assert select_planning_mode(similar)[0] == "INTELLIGENCE"
        assert select_planning_mode(quick) == ("DETERMINISTIC", "TRUSTED_TOPIC_CONSTRAINTS")
