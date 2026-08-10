"""Practice prompt organization, composition, and budget tests."""

from __future__ import annotations

import json

import pytest

from features.practice_generation.planning import (
    deterministic_blueprint,
    resolve_practice_request,
)
from features.practice_generation.providers import (
    RoutedPlannerProvider,
    RoutedQuestionGenerator,
    RoutedQuestionVerifier,
)
from features.practice_generation.schemas import GeneratedQuestion, GenerationGroup
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor
from services.llm.orchestration.prompt_resolver import PromptResolver

PROMPT_PATHS = (
    "practice_generation/shared_contract",
    "practice_generation/planners/quant_reasoning",
    "practice_generation/planners/english",
    "practice_generation/planners/factual",
    "practice_generation/question_generator",
    "practice_generation/question_verifier",
    "practice_generation/question_generator_v2",
    "practice_generation/question_verifier_v2",
)


@pytest.mark.parametrize("prompt_path", PROMPT_PATHS)
def test_nested_practice_prompt_loads_within_budget(prompt_path: str) -> None:
    content = (PromptResolver()._prompt_root / f"{prompt_path}.md").read_text(encoding="utf-8")
    assert 100 < len(content) < 1_500


@pytest.mark.parametrize(
    ("subject", "role_text", "prompt_name"),
    [
        ("math", "Quant or Reasoning", "quant_reasoning.md"),
        ("english", "English assessment", "english.md"),
        ("history", "static factual-subject", "factual.md"),
    ],
)
def test_planner_receives_subject_family_prompt_only(
    subject: str,
    role_text: str,
    prompt_name: str,
) -> None:
    executor = MockModelExecutor(
        content=json.dumps({"slots": []})
    )
    request = resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query="Create five algebra questions",
        subject=subject,
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )

    RoutedPlannerProvider(LlmOrchestrator(model_executor=executor)).plan(
        request,
        tier="light",
    )

    assert executor.last_messages is not None
    system = executor.last_messages[0].content
    assert executor.last_route_decision is not None
    assert "# Shared contract" in system
    assert role_text in system
    assert executor.last_route_decision.prompt.endswith(prompt_name)
    assert 'Return exactly `{"slots":[...]}`' in system
    assert "Generate only the assigned questions" not in system
    assert "Independently verify" not in system


def test_composed_prompt_estimate_remains_compact() -> None:
    prompt_root = PromptResolver()._prompt_root
    shared = (prompt_root / "practice_generation/shared_contract.md").read_text(
        encoding="utf-8"
    )
    for role_path in PROMPT_PATHS[1:]:
        role = (prompt_root / f"{role_path}.md").read_text(encoding="utf-8")
        combined = f"{shared}\n\n{role}"
        estimated_tokens = (len(combined) + 3) // 4
        assert len(combined) < 2_000
        assert estimated_tokens < 500


def test_generator_and_verifier_receive_only_their_role_prompt() -> None:
    request = resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query="Create one algebra question",
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )
    bucket = deterministic_blueprint(request).buckets[0]
    generator_executor = MockModelExecutor(content='{"questions":[]}')
    RoutedQuestionGenerator(LlmOrchestrator(model_executor=generator_executor)).generate(
        request=request,
        bucket=bucket,
        group=GenerationGroup(
            group_id="math-algebra-g1",
            bucket_id=bucket.bucket_id,
            required_count=1,
        ),
        exclude_normalized_texts=(),
    )
    assert generator_executor.last_messages is not None
    generator_system = generator_executor.last_messages[0].content
    assert "Generate only the assigned questions" in generator_system
    assert 'Return exactly `{"slots":[...]}`' not in generator_system
    assert "Independently verify" not in generator_system
    assert '"question_type":"mcq"' in generator_system
    assert "mcq|msq|numerical|descriptive" not in generator_system

    verifier_executor = MockModelExecutor(
        content=json.dumps(
            {
                "generation_item_id": "item-1",
                "approved": True,
                "reason_code": "MATCH",
            }
        )
    )
    RoutedQuestionVerifier(LlmOrchestrator(model_executor=verifier_executor)).verify(
        request=request,
        bucket=bucket,
        question=GeneratedQuestion(
            generation_item_id="item-1",
            bucket_id=bucket.bucket_id,
            question="If x plus one equals two, what is x?",
            question_type="mcq",
            options=["0", "1"],
            correct_answer="1",
            solution="Subtract one.",
            subject="math",
            topic="algebra",
            difficulty="intermediate",
        ),
    )
    assert verifier_executor.last_messages is not None
    verifier_system = verifier_executor.last_messages[0].content
    assert "Independently verify" in verifier_system
    assert "Generate only the assigned questions" not in verifier_system
    assert 'Return exactly `{"slots":[...]}`' not in verifier_system


def test_slot_generator_and_verifier_exchange_the_canonical_answer_contract() -> None:
    request = resolve_practice_request(
        request_id="request-v2",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-v2",
        query="Create one algebra question",
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )
    blueprint = deterministic_blueprint(request)
    bucket = blueprint.buckets[0]
    slot = blueprint.slots[0]
    group = GenerationGroup(
        group_id="math-algebra-g1",
        bucket_id=bucket.bucket_id,
        required_count=1,
        slot_ids=[slot.slot_id],
    )
    generator_executor = MockModelExecutor(content='{"questions":[]}')

    RoutedQuestionGenerator(LlmOrchestrator(model_executor=generator_executor)).generate_slots(
        request=request,
        bucket=bucket,
        group=group,
        slots=(slot,),
        exclude_normalized_texts=(),
    )

    assert generator_executor.last_messages is not None
    assert "Generate one independently playable MCQ" in generator_executor.last_messages[0].content
    generator_payload = json.loads(generator_executor.last_messages[1].content)
    assert generator_payload["required_answer_contract"] == {
        "answer_status": "PENDING_VERIFICATION",
        "answer_version": 1,
        "option_ids": ["0", "1", "2", "3"],
    }

    question = GeneratedQuestion(
        schema_version="2",
        generation_item_id="item-1",
        bucket_id=bucket.bucket_id,
        slot_id=slot.slot_id,
        question="If two plus two equals four, which indexed option is correct?",
        question_type="mcq",
        options=[
            {"option_id": "0", "value": "1"},
            {"option_id": "1", "value": "2"},
            {"option_id": "2", "value": "3"},
            {"option_id": "3", "value": "4"},
        ],
        correct_option_id="3",
        correct_answer="4",
        answer_explanation="Two plus two is four.",
        solution="Two plus two is four.",
        subject=slot.subject_id,
        topic=slot.topic_id,
        difficulty=slot.difficulty,
    )
    verifier_executor = MockModelExecutor(
        content=json.dumps(
            {
                "schema_version": "2",
                "generation_item_id": "item-1",
                "slot_id": slot.slot_id,
                "decision": "ACCEPT",
                "independently_solved_option_id": "3",
                "reason_codes": ["INDEPENDENT_SOLUTION_MATCH"],
            }
        )
    )

    RoutedQuestionVerifier(LlmOrchestrator(model_executor=verifier_executor)).verify_slot(
        request=request,
        bucket=bucket,
        slot=slot,
        question=question,
    )

    assert verifier_executor.last_messages is not None
    assert "Independently solve and verify" in verifier_executor.last_messages[0].content
    verifier_payload = json.loads(verifier_executor.last_messages[1].content)
    assert verifier_payload["question"]["submitted_answer"] == {
        "correct_option_id": "3",
        "correct_answer": "4",
    }
