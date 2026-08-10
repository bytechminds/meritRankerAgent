"""Focused unit tests for practice-generation contracts and deterministic policy."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from features.practice_generation.config import (
    PracticeConfigurationError,
    PracticeGenerationConfig,
    get_practice_config,
)
from features.practice_generation.generation import (
    build_generation_groups,
    build_slot_generation_groups,
    parse_partial_generation,
    validate_final_set,
    verification_required,
)
from features.practice_generation.matching import (
    build_reuse_bucket_key,
    match_existing_questions,
    reusable_question_from_item,
)
from features.practice_generation.pattern_context import (
    NoOpPatternContextProvider,
    build_pattern_context_provider,
)
from features.practice_generation.planning import (
    BlueprintManager,
    apply_system_bucket_policy,
    decide_practice_launch,
    deterministic_blueprint,
    parse_blueprint,
    planner_tier,
    resolve_practice_request,
    select_planner_family,
)
from features.practice_generation.question_contract import (
    validate_persisted_playable_question,
    validate_playable_question,
)
from features.practice_generation.schemas import (
    Complexity,
    DemandBucket,
    Difficulty,
    GenerationGroup,
    PlannerFamily,
    PracticeBlueprint,
    PracticeType,
    QuestionType,
    VerificationPolicy,
)
from schemas.llm_routing import RouteRequest
from services.llm.orchestration.route_resolver import resolve_route


def request(
    query: str,
    *,
    count_query: str | None = None,
    subject: str = "math",
    topic: str = "algebra",
):
    return resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query=count_query or query,
        subject=subject,
        topic=topic,
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )


def test_count_above_max_is_clamped_and_retained() -> None:
    resolved = request("Create a full mock", count_query="Create 150 question full mock")
    assert (resolved.requested_count, resolved.accepted_count) == (150, 100)


def test_reuse_bucket_key_matches_backend_versioned_uri_contract() -> None:
    assert (
        build_reuse_bucket_key(
            category="General Knowledge",
            topic="Indian Economy & Policy",
            question_type="MCQ",
            language="EN",
        )
        == "v1#general%20knowledge#indian%20economy%20%26%20policy#mcq#en"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Create five algebra questions",
        "Give me a CAT mock test",
        "Create a 10-minute Reasoning mini mock for CAT.",
        "Create a 20-minute English mini mock for SSC.",
        "Provide a short English quiz",
        "Give me another similar question",
    ],
)
def test_explicit_practice_creation_request_is_eligible(query: str) -> None:
    decision = decide_practice_launch(query, {"intent": "practice"})
    assert decision.eligible is True


@pytest.mark.parametrize(
    "query",
    [
        "How should I practise algebra?",
        "Explain the best strategy for English practice.",
        "Which topics should I practise?",
        "Is this question good for practice?",
    ],
)
def test_practice_advice_request_is_not_eligible(query: str) -> None:
    decision = decide_practice_launch(query, {"intent": "practice"})
    assert decision.eligible is False


@pytest.mark.parametrize(
    ("query", "artifact"),
    [
        ("Generate five percentage questions.", "questions"),
        ("Give me ten English grammar questions.", "questions"),
        ("Create a CAT mock test.", "mock test"),
        ("Create a 10-minute Reasoning mini mock for CAT.", "mini mock"),
        ("Create a 20-minute English mini mocks for SSC.", "mini mocks"),
        ("Make one similar question.", "similar_question"),
        ("Prepare a sectional quiz.", "quiz"),
    ],
)
def test_requested_question_artifact_launches(query: str, artifact: str) -> None:
    decision = decide_practice_launch(query, {"intent": "practice"})
    assert (decision.eligible, decision.requested_artifact) == (True, artifact)


@pytest.mark.parametrize(
    "query",
    [
        "Give me tips for mock tests.",
        "Provide a quiz preparation strategy.",
        "Make a study plan.",
        "Explain how mock tests help.",
        "Explain how mini mocks help.",
        "Give me mini mock strategy.",
        "Which questions should I practise?",
    ],
)
def test_non_question_artifact_does_not_launch(query: str) -> None:
    assert decide_practice_launch(query, {"intent": "practice"}).eligible is False


def test_enabled_runtime_requires_existing_persistence_configuration() -> None:
    config = PracticeGenerationConfig(
        enabled=True,
        pattern_context_enabled=False,
        assessment_table="",
        question_table="",
        question_bank_table="",
        question_bank_category_index="",
        question_test_index="",
        aws_region="ap-south-1",
    )
    with pytest.raises(
        PracticeConfigurationError,
        match="practice/mock-test-quiz/table-name",
    ):
        config.validate_runtime()


def test_disabled_feature_ignores_invalid_tuning_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "false")
    monkeypatch.setenv("PRACTICE_GENERATION_CONCURRENCY", "invalid")
    config = get_practice_config()
    config.validate_runtime()
    assert config.enabled is False


@pytest.mark.parametrize(
    ("subject", "difficulty", "model", "prompt", "max_tokens"),
    [
        (
            "quant_reasoning",
            "basic",
            "general_fast_generator",
            "practice_generation/planners/quant_reasoning.md",
            1200,
        ),
        (
            "quant_reasoning",
            "intermediate",
            "openai_gpt_4_1",
            "practice_generation/planners/quant_reasoning.md",
            1800,
        ),
        (
            "english",
            "advanced",
            "openai_gpt_4_1",
            "practice_generation/planners/english.md",
            2600,
        ),
        (
            "factual",
            "basic",
            "general_fast_generator",
            "practice_generation/planners/factual.md",
            1200,
        ),
    ],
)
def test_planner_family_routes_resolve_existing_models(
    subject: str,
    difficulty: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> None:
    route = resolve_route(
        RouteRequest(
            request_id="practice-route",
            subject=subject,
            task_role="planner",
            difficulty=difficulty,
            intent="practice",
            language="english",
        )
    )
    assert (route.model, route.prompt, route.max_tokens) == (
        model,
        prompt,
        max_tokens,
    )


def test_practice_verifier_route_has_bounded_reasoning_budget() -> None:
    route = resolve_route(
        RouteRequest(
            request_id="practice-verifier-route",
            subject="general",
            task_role="verifier",
            difficulty="default",
            intent="practice",
            language="english",
        )
    )

    assert (route.model, route.max_tokens) == ("openai_o4_mini", 5000)
    assert route.provider_options == {"reasoning_effort": "medium"}


@pytest.mark.parametrize(
    ("subject", "family"),
    [
        ("math", PlannerFamily.QUANT_REASONING),
        ("reasoning", PlannerFamily.QUANT_REASONING),
        ("english", PlannerFamily.ENGLISH),
        ("history", PlannerFamily.FACTUAL),
    ],
)
def test_planner_family_selection_is_subject_aware(
    subject: str,
    family: PlannerFamily,
) -> None:
    resolved = request(
        "Create five practice questions",
        subject=subject,
        topic="topic",
    )
    assert select_planner_family(resolved) is family


def test_exam_year_is_not_misread_as_question_count() -> None:
    resolved = request("Create a CAT 2025 full mock")
    assert (resolved.practice_type, resolved.requested_count) == (
        PracticeType.FULL_MOCK,
        100,
    )


def test_duration_based_mini_mock_uses_quick_practice_default_count() -> None:
    resolved = request("Create a 10-minute Reasoning mini mock for CAT.")
    assert (resolved.practice_type, resolved.requested_count) == (
        PracticeType.QUICK_PRACTICE,
        5,
    )


def test_subject_word_is_not_misread_as_question_count() -> None:
    resolved = request("Create a quiz on one nation one election")
    assert (resolved.practice_type, resolved.requested_count) == (
        PracticeType.QUIZ,
        10,
    )


def test_deterministic_blueprint_for_two_questions_uses_no_planner() -> None:
    class Planner:
        def plan(self, *_args, **_kwargs):
            raise AssertionError("planner must not run")

    result = BlueprintManager(Planner()).build(request("Create two algebra questions"))
    assert (result.planner_calls, result.blueprint.accepted_count) == (0, 2)
    assert result.blueprint.schema_version == "2"
    assert result.blueprint.planner_family is PlannerFamily.QUANT_REASONING
    assert [slot.slot_id for slot in result.blueprint.slots] == ["slot-001", "slot-002"]
    assert len(result.blueprint.buckets) == 1


def test_planner_tiers_follow_complexity_not_subject() -> None:
    assert planner_tier(request("Create five algebra questions")) == "light"
    assert planner_tier(request("Create twenty algebra questions")) == "standard"
    assert planner_tier(request("Create fifty question full mock")) == "strong"


def test_invalid_planner_output_gets_one_repair_then_fallback() -> None:
    class Planner:
        calls = 0

        def plan(self, *_args, **_kwargs):
            self.calls += 1
            return '{"buckets":[]}'

    planner = Planner()
    result = BlueprintManager(planner, repair_limit=1).build(
        request("Create five algebra questions")
    )
    assert (planner.calls, result.repaired, result.deterministic_fallback) == (
        2,
        True,
        True,
    )
    assert result.validation_reason_code == "PLANNER_SLOT_COUNT_MISMATCH"


def test_valid_planner_slot_difficulty_is_not_overridden_by_classifier_difficulty() -> None:
    resolved = request("Create five algebra questions").model_copy(
        update={"difficulty": Difficulty.ADVANCED}
    )
    planner_blueprint = deterministic_blueprint(resolved)
    payload = planner_blueprint.model_dump(mode="json")
    levels = (Difficulty.BASIC, Difficulty.INTERMEDIATE, Difficulty.ADVANCED)
    for index, slot in enumerate(payload["slots"]):
        difficulty = levels[index % len(levels)]
        slot["difficulty"] = difficulty.value
        slot["complexity"] = {
            Difficulty.BASIC: Complexity.LOW,
            Difficulty.INTERMEDIATE: Complexity.MEDIUM,
            Difficulty.ADVANCED: Complexity.HIGH,
        }[difficulty].value
        slot["generator_route_hint"] = f"math.generator.{difficulty.value}"

    class Planner:
        def plan(self, *_args, **_kwargs):
            return json.dumps({"slots": payload["slots"]})

    result = BlueprintManager(Planner()).build(resolved)

    assert result.deterministic_fallback is False
    assert [slot.difficulty for slot in result.blueprint.slots] == [
        Difficulty.BASIC,
        Difficulty.INTERMEDIATE,
        Difficulty.ADVANCED,
        Difficulty.BASIC,
        Difficulty.INTERMEDIATE,
    ]


def test_mixed_deterministic_fallback_uses_slot_specific_routes() -> None:
    resolved = request(
        "provide 10 questions mock test of quant mixed for CAT"
    ).model_copy(update={"difficulty": Difficulty.ADVANCED})

    blueprint = deterministic_blueprint(resolved)
    groups = build_slot_generation_groups(
        blueprint,
        {slot.slot_id for slot in blueprint.slots},
        group_size=5,
        group_max=5,
    )
    slots = {slot.slot_id: slot for slot in blueprint.slots}
    buckets = {bucket.bucket_id: bucket for bucket in blueprint.buckets}

    assert [slot.difficulty for slot in blueprint.slots] == [
        Difficulty.BASIC,
        Difficulty.INTERMEDIATE,
        Difficulty.ADVANCED,
        Difficulty.BASIC,
        Difficulty.INTERMEDIATE,
        Difficulty.ADVANCED,
        Difficulty.BASIC,
        Difficulty.INTERMEDIATE,
        Difficulty.ADVANCED,
        Difficulty.BASIC,
    ]
    for group in groups:
        group_slots = [slots[slot_id] for slot_id in group.slot_ids]
        bucket = buckets[group.bucket_id]
        assert {slot.difficulty for slot in group_slots} == {bucket.difficulty}
        assert {slot.generator_route_hint for slot in group_slots} == {
            f"math.generator.{bucket.difficulty.value}"
        }


def test_mixed_difficulty_survives_safe_async_request_rehydration() -> None:
    resolved = request(
        "provide 10 questions mock test of quant mixed for CAT"
    ).model_copy(update={"difficulty": Difficulty.ADVANCED})
    rehydrated = type(resolved).model_validate(
        {
            **resolved.model_dump(mode="json"),
            "original_query": "Practice generation request",
        }
    )

    blueprint = deterministic_blueprint(rehydrated)
    groups = build_slot_generation_groups(
        blueprint,
        {slot.slot_id for slot in blueprint.slots},
        group_size=5,
        group_max=5,
    )

    assert rehydrated.mixed_difficulty_requested is True
    assert len(blueprint.buckets) == 3
    assert sum(bucket.required_count for bucket in blueprint.buckets) == 10
    assert sum(group.required_count for group in groups) == 10


def test_blueprint_rejects_wrong_total() -> None:
    bucket = deterministic_blueprint(request("Create five algebra questions")).buckets[0]
    with pytest.raises(ValidationError):
        PracticeBlueprint(
            practice_type=PracticeType.QUICK_PRACTICE,
            accepted_count=5,
            buckets=[bucket.model_copy(update={"required_count": 4})],
        )


def test_schema_v1_blueprint_remains_explicitly_compatible() -> None:
    bucket = deterministic_blueprint(request("Create one algebra question")).buckets[0]
    blueprint = PracticeBlueprint(
        schema_version="1",
        practice_type=PracticeType.QUICK_PRACTICE,
        accepted_count=1,
        buckets=[bucket.model_copy(update={"required_count": 1})],
    )
    assert blueprint.schema_version == "1"
    assert blueprint.planner_family is None
    assert blueprint.slots == []


def test_schema_v2_planner_slots_are_canonical_over_supplied_buckets() -> None:
    resolved = request("Create five algebra questions")
    fallback = deterministic_blueprint(resolved)
    raw = json.dumps(
        {
            "slots": [slot.model_dump(mode="json") for slot in fallback.slots],
            "buckets": [
                {
                    "bucket_id": "planner-owned-bucket",
                    "subject": "english",
                    "topic": "grammar",
                    "difficulty": "basic",
                    "question_type": "mcq",
                    "required_count": 5,
                    "question_intent": "Override canonical slot metadata.",
                }
            ],
        }
    )
    blueprint = parse_blueprint(raw, resolved)
    assert blueprint.schema_version == "2"
    assert blueprint.planner_family is PlannerFamily.QUANT_REASONING
    assert all(bucket.subject == "math" for bucket in blueprint.buckets)
    assert all(bucket.bucket_id != "planner-owned-bucket" for bucket in blueprint.buckets)


def test_schema_v2_rejects_duplicate_slot_ids() -> None:
    resolved = request("Create five algebra questions")
    slots = [
        slot.model_dump(mode="json")
        for slot in deterministic_blueprint(resolved).slots
    ]
    slots[1]["slot_id"] = slots[0]["slot_id"]
    with pytest.raises(ValidationError, match="planner slot IDs must be unique"):
        parse_blueprint(json.dumps({"slots": slots}), resolved)


def test_blueprint_rejects_unsupported_subject() -> None:
    with pytest.raises(ValidationError):
        DemandBucket(
            bucket_id="unsupported",
            subject="invented_subject",
            topic="topic",
            difficulty=Difficulty.BASIC,
            question_type=QuestionType.MCQ,
            required_count=1,
            keywords=[],
            question_intent="Create one question.",
            verification_policy=VerificationPolicy.NONE,
        )


def test_reuse_requires_explicit_verified_reusable_platform_metadata() -> None:
    candidate = reusable_question_from_item(
        {
            "qbId": "q1",
            "question": "What is 2 + 2?",
            "answers": json.dumps({"options": ["3", "4", "5", "6"]}),
            "correctAnswer": "4",
            "explanation": "Adding gives 4.",
            "category": "math",
            "difficulty": "MEDIUM",
            "meta": json.dumps(
                {
                    "status": "ACTIVE",
                    "qualityStatus": "VERIFIED",
                    "reusable": True,
                    "visibility": "PLATFORM",
                    "language": "english",
                    "subject": "math",
                    "topic": "algebra",
                    "questionType": "mcq",
                }
            ),
        },
        requested_language="english",
    )
    assert candidate is not None


def test_reuse_rejects_private_or_unverified_candidate() -> None:
    item = {
        "qbId": "q1",
        "question": "What is 2 + 2?",
        "answers": json.dumps({"options": ["3", "4"]}),
        "correctAnswer": "4",
        "explanation": "Four.",
        "category": "math",
        "meta": json.dumps(
            {
                "status": "ACTIVE",
                "qualityStatus": "PENDING",
                "reusable": True,
                "visibility": "PRIVATE",
                "language": "english",
            }
        ),
    }
    assert reusable_question_from_item(item, requested_language="english") is None


def test_solution_requirement_is_applied_by_assessment_policy() -> None:
    item = {
        "qbId": "q-no-solution",
        "question": "Choose the grammatically correct sentence.",
        "answers": json.dumps(
            {"options": ["She runs.", "She run.", "They run.", "They runs."]}
        ),
        "correctAnswer": "She runs.",
        "explanation": None,
        "category": "english",
        "difficulty": "EASY",
        "meta": json.dumps(
            {
                "status": "ACTIVE",
                "qualityStatus": "VERIFIED",
                "reusable": True,
                "visibility": "PLATFORM",
                "language": "english",
                "subject": "english",
                "topic": "grammar",
                "questionType": "mcq",
            }
        ),
    }
    candidate = reusable_question_from_item(item, requested_language="english")
    assert candidate is not None
    optional_solution = DemandBucket(
        bucket_id="english-optional",
        subject="english",
        topic="grammar",
        difficulty=Difficulty.BASIC,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Create one question.",
        verification_policy=VerificationPolicy.NONE,
        solution_required=False,
    )
    required_solution = optional_solution.model_copy(update={"solution_required": True})
    optional_blueprint = PracticeBlueprint(
        practice_type=PracticeType.QUIZ,
        accepted_count=1,
        buckets=[optional_solution],
    )
    required_blueprint = optional_blueprint.model_copy(update={"buckets": [required_solution]})
    assert match_existing_questions(optional_blueprint, [candidate])[0].deficit == 0
    assert match_existing_questions(required_blueprint, [candidate])[0].deficit == 1


def test_matching_is_deterministic_and_prevents_duplicate_selection() -> None:
    blueprint = deterministic_blueprint(request("Create two algebra questions"))
    candidate = reusable_question_from_item(
        {
            "qbId": "q1",
            "question": "Solve x plus one equals two.",
            "answers": json.dumps({"options": ["0", "1", "2", "3"]}),
            "correctAnswer": "1",
            "explanation": "Subtract one.",
            "category": "math",
            "difficulty": "MEDIUM",
            "meta": json.dumps(
                {
                    "status": "ACTIVE",
                    "qualityStatus": "VERIFIED",
                    "reusable": True,
                    "visibility": "PLATFORM",
                    "language": "english",
                    "subject": "math",
                    "topic": "algebra",
                    "questionType": "mcq",
                }
            ),
        },
        requested_language="english",
    )
    assert candidate is not None
    matches = match_existing_questions(blueprint, [candidate, candidate])
    assert (len(matches[0].selected), matches[0].deficit) == (1, 1)


def test_reuse_rejects_wrong_difficulty_instead_of_only_downranking_it() -> None:
    blueprint = deterministic_blueprint(request("Create one algebra question"))
    candidate = reusable_question_from_item(
        {
            "qbId": "q-easy",
            "question": "Solve x plus one equals two.",
            "answers": json.dumps({"options": ["0", "1", "2", "3"]}),
            "correctAnswer": "1",
            "explanation": "Subtract one.",
            "category": "math",
            "difficulty": "EASY",
            "meta": json.dumps(
                {
                    "status": "ACTIVE",
                    "qualityStatus": "VERIFIED",
                    "reusable": True,
                    "visibility": "PLATFORM",
                    "language": "english",
                    "subject": "math",
                    "topic": "algebra",
                    "questionType": "mcq",
                }
            ),
        },
        requested_language="english",
    )
    assert candidate is not None
    assert match_existing_questions(blueprint, [candidate])[0].deficit == 1


def test_intermediate_generation_groups_are_capped_at_four() -> None:
    blueprint = deterministic_blueprint(request("Create twenty algebra questions"))
    groups = build_generation_groups(
        blueprint,
        {blueprint.buckets[0].bucket_id: 20},
        group_size=5,
        group_max=5,
    )
    assert [group.required_count for group in groups] == [4, 4, 4, 4, 4]


def test_advanced_math_generation_groups_are_capped_at_two() -> None:
    blueprint = deterministic_blueprint(request("Create five algebra questions")).model_copy(
        update={
            "buckets": [
                deterministic_blueprint(request("Create five algebra questions"))
                .buckets[0]
                .model_copy(
                    update={
                        "difficulty": Difficulty.ADVANCED,
                        "generation_group_hint": 5,
                    }
                )
            ]
        }
    )
    groups = build_generation_groups(
        blueprint,
        {blueprint.buckets[0].bucket_id: 5},
        group_size=5,
        group_max=5,
    )
    assert [group.required_count for group in groups] == [2, 2, 1]


def test_advanced_reasoning_generation_groups_allow_two_compatible_questions() -> None:
    bucket = DemandBucket(
        bucket_id="reasoning-advanced",
        subject="reasoning",
        topic="arrangements",
        difficulty=Difficulty.ADVANCED,
        question_type=QuestionType.MCQ,
        question_intent="practice",
        required_count=5,
        verification_policy=VerificationPolicy.MANDATORY,
    )
    blueprint = PracticeBlueprint(
        practice_type=PracticeType.QUICK_PRACTICE,
        accepted_count=5,
        buckets=[bucket],
    )

    groups = build_generation_groups(
        blueprint,
        {bucket.bucket_id: 5},
        group_size=3,
        group_max=5,
    )

    assert [group.required_count for group in groups] == [2, 2, 1]


@pytest.mark.parametrize(
    ("subject", "difficulty", "complexity", "expected"),
    [
        ("math", Difficulty.BASIC, Complexity.LOW, [2, 2, 1]),
        ("math", Difficulty.INTERMEDIATE, Complexity.MEDIUM, [4, 1]),
        ("math", Difficulty.ADVANCED, Complexity.LOW, [2, 2, 1]),
        ("reasoning", Difficulty.ADVANCED, Complexity.LOW, [2, 2, 1]),
        ("reasoning", Difficulty.ADVANCED, Complexity.HIGH, [1, 1, 1, 1, 1]),
        ("history", Difficulty.ADVANCED, Complexity.LOW, [5]),
    ],
)
def test_schema_v2_slot_batches_enforce_subject_difficulty_and_complexity_caps(
    subject: str,
    difficulty: Difficulty,
    complexity: Complexity,
    expected: list[int],
) -> None:
    resolved = request(
        "Create five practice questions",
        subject=subject,
        topic="algebra" if subject == "math" else "arrangements",
    )
    original = deterministic_blueprint(resolved)
    payload = original.model_dump(mode="json")
    for slot in payload["slots"]:
        slot.update(
            {
                "subject_id": subject,
                "difficulty": difficulty.value,
                "complexity": complexity.value,
                "generator_route_hint": f"{subject}.generator.{difficulty.value}",
                "generation_group_hint": 5,
            }
        )
    payload.pop("buckets", None)
    blueprint = PracticeBlueprint.model_validate(payload)

    groups = build_slot_generation_groups(
        blueprint,
        {slot.slot_id for slot in blueprint.slots},
        group_size=5,
        group_max=5,
    )

    assert [group.required_count for group in groups] == expected


def test_slot_batches_expand_only_when_the_shared_route_budget_allows() -> None:
    original = deterministic_blueprint(request("Create five algebra questions"))
    payload = original.model_dump(mode="json")
    for slot in payload["slots"]:
        slot.update(
            {
                "difficulty": Difficulty.BASIC.value,
                "complexity": Complexity.LOW.value,
                "generation_group_hint": 5,
            }
        )
    payload.pop("buckets", None)
    blueprint = PracticeBlueprint.model_validate(payload)

    groups = build_slot_generation_groups(
        blueprint,
        {slot.slot_id for slot in blueprint.slots},
        group_size=5,
        group_max=5,
        token_budget_resolver=lambda _subject, _difficulty: 3_200,
    )

    assert [group.required_count for group in groups] == [5]
    assert [group.token_budget for group in groups] == [3_200]


def test_partial_group_retains_valid_sibling_and_reports_deficit() -> None:
    blueprint = deterministic_blueprint(request("Create three algebra questions"))
    bucket = blueprint.buckets[0]
    group = GenerationGroup(
        group_id="g1",
        bucket_id=bucket.bucket_id,
        required_count=3,
    )
    raw = json.dumps(
        {
            "questions": [
                {
                    "generation_item_id": "i1",
                    "bucket_id": bucket.bucket_id,
                    "question": "If x plus 2 equals 5, what is x?",
                    "question_type": "mcq",
                    "options": ["2", "3", "4", "5"],
                    "correct_answer": "3",
                    "solution": "Subtract 2 from both sides.",
                    "subject": "math",
                    "topic": "algebra",
                    "difficulty": "intermediate",
                },
                {"invalid": True},
            ]
        }
    )
    parsed = parse_partial_generation(
        raw,
        group=group,
        bucket=bucket,
        existing_normalized_texts=set(),
    )
    assert (len(parsed.accepted), parsed.rejected_count) == (1, 2)


def test_slot_generation_normalizes_safe_subject_and_topic_labels() -> None:
    resolved = request(
        "Create one time and work question",
        subject="math",
        topic="time_and_work",
    )
    blueprint = deterministic_blueprint(
        resolved.model_copy(update={"difficulty": Difficulty.BASIC})
    )
    slot = blueprint.slots[0]
    bucket = blueprint.buckets[0]
    group = GenerationGroup(
        group_id="g1",
        bucket_id=bucket.bucket_id,
        required_count=1,
        slot_ids=[slot.slot_id],
    )
    raw = json.dumps(
        {
            "questions": [
                {
                    "schema_version": "2",
                    "generation_item_id": "item-slot-001",
                    "bucket_id": bucket.bucket_id,
                    "slot_id": slot.slot_id,
                    "question": (
                        "A can finish a task in six days. "
                        "How many days for two such workers?"
                    ),
                    "question_type": "mcq",
                    "options": [
                        {"option_id": "0", "value": "2"},
                        {"option_id": "1", "value": "3"},
                        {"option_id": "2", "value": "6"},
                        {"option_id": "3", "value": "12"},
                    ],
                    "correct_option_id": "1",
                    "correct_answer": "3",
                    "answer_explanation": "Two equal workers complete twice the work per day.",
                    "solution": "Two equal workers complete twice the work per day.",
                    "subject": "Quant",
                    "topic": "Time and Work",
                    "difficulty": "basic",
                }
            ]
        }
    )

    parsed = parse_partial_generation(
        raw,
        group=group,
        bucket=bucket,
        existing_normalized_texts=set(),
        slots=(slot,),
    )

    assert parsed.rejected_count == 0
    assert [question.slot_id for question in parsed.accepted] == [slot.slot_id]


def test_slot_generation_reports_safe_schema_v2_identity_failure() -> None:
    resolved = request(
        "Create one time and work question",
        subject="math",
        topic="time_and_work",
    )
    blueprint = deterministic_blueprint(
        resolved.model_copy(update={"difficulty": Difficulty.BASIC})
    )
    slot = blueprint.slots[0]
    bucket = blueprint.buckets[0]
    group = GenerationGroup(
        group_id="g1",
        bucket_id=bucket.bucket_id,
        required_count=1,
        slot_ids=[slot.slot_id],
    )
    raw = json.dumps(
        {
            "questions": [
                {
                    "schema_version": "2",
                    "generation_item_id": "item-slot-001",
                    "bucket_id": bucket.bucket_id,
                    "slot_id": slot.slot_id,
                    "question": "A worker completes a task in six days. What is the rate?",
                    "question_type": "mcq",
                    "options": [
                        {"option_id": "0", "value": "One sixth"},
                        {"option_id": "1", "value": "One third"},
                        {"option_id": "2", "value": "One half"},
                        {"option_id": "3", "value": "One"},
                    ],
                    "correct_option_id": "0",
                    "correct_answer": "One sixth",
                    "answer_explanation": "The worker completes one sixth each day.",
                    "solution": "The worker completes one sixth each day.",
                    "subject": slot.subject_id,
                    "topic": slot.topic_id,
                    "difficulty": slot.difficulty.value,
                }
            ]
        }
    ).replace('"generation_item_id": "item-slot-001"', '"generation_item_id": ""')

    parsed = parse_partial_generation(
        raw,
        group=group,
        bucket=bucket,
        existing_normalized_texts=set(),
        slots=(slot,),
    )

    assert parsed.accepted == ()
    assert parsed.rejection_reason_codes == ("SCHEMA_V2_IDENTITY_CONTRACT_INVALID",)


def test_every_item_is_independently_verified_even_for_legacy_selective_policy() -> None:
    bucket = DemandBucket(
        bucket_id="math-algebra",
        subject="math",
        topic="algebra",
        difficulty=Difficulty.INTERMEDIATE,
        question_type=QuestionType.MCQ,
        required_count=3,
        keywords=["algebra"],
        question_intent="Solve an algebra equation.",
        verification_policy=VerificationPolicy.SELECTIVE,
    )
    assert [verification_required(bucket, index) for index in range(3)] == [True, True, True]


@pytest.mark.parametrize(
    ("subject", "difficulty"),
    [("math", Difficulty.INTERMEDIATE), ("reasoning", Difficulty.ADVANCED)],
)
def test_high_risk_policy_is_system_owned_and_mandatory(
    subject: str,
    difficulty: Difficulty,
) -> None:
    original = DemandBucket(
        bucket_id=f"{subject}-risk",
        subject=subject,
        topic="puzzle" if subject == "reasoning" else "algebra",
        difficulty=difficulty,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Create one question.",
        verification_policy=VerificationPolicy.NONE,
    )
    practice_request = request(f"Create one {subject} question").model_copy(
        update={"subject": subject, "difficulty": difficulty}
    )
    blueprint = PracticeBlueprint(
        practice_type=practice_request.practice_type,
        accepted_count=1,
        buckets=[original],
    )
    resolved = apply_system_bucket_policy(blueprint, practice_request)
    assert resolved.buckets[0].verification_policy is VerificationPolicy.MANDATORY


def test_easy_english_avoids_unnecessary_model_verification() -> None:
    bucket = DemandBucket(
        bucket_id="english-grammar",
        subject="english",
        topic="grammar",
        difficulty=Difficulty.BASIC,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Create one grammar question.",
        verification_policy=VerificationPolicy.MANDATORY,
    )
    practice_request = request("Create one English question").model_copy(
        update={"subject": "english", "difficulty": Difficulty.BASIC}
    )
    resolved = apply_system_bucket_policy(
        PracticeBlueprint(
            practice_type=practice_request.practice_type,
            accepted_count=1,
            buckets=[bucket],
        ),
        practice_request,
    )
    assert resolved.buckets[0].verification_policy is VerificationPolicy.NONE


@pytest.mark.parametrize("subject", ["english", "history", "polity", "science"])
def test_every_generated_subject_receives_independent_model_verification(
    subject: str,
) -> None:
    blueprint = deterministic_blueprint(request("Create five algebra questions")).model_copy(
        update={
            "buckets": [
                deterministic_blueprint(request("Create five algebra questions"))
                .buckets[0]
                .model_copy(
                    update={
                        "subject": subject,
                        "difficulty": Difficulty.ADVANCED,
                        "verification_policy": VerificationPolicy.MANDATORY,
                    }
                )
            ]
        }
    )
    resolved = apply_system_bucket_policy(
        blueprint,
        request("Create five algebra questions"),
    )
    assert resolved.buckets[0].verification_policy is VerificationPolicy.NONE
    assert verification_required(resolved.buckets[0], 0) is True


def test_replacement_verification_applies_to_every_subject() -> None:
    bucket = DemandBucket(
        bucket_id="english-grammar",
        subject="english",
        topic="grammar",
        difficulty=Difficulty.BASIC,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Create one grammar question.",
        verification_policy=VerificationPolicy.NONE,
    )
    assert verification_required(bucket, 0, replacement=True) is True
    assert (
        verification_required(
            bucket.model_copy(update={"subject": "math"}),
            0,
            replacement=True,
        )
        is True
    )


def test_ready_gate_rejects_short_and_duplicate_sets() -> None:
    blueprint = deterministic_blueprint(request("Create two algebra questions"))
    item = {
        "questionId": "q1",
        "question": "Question one",
        "answers": "{}",
        "correctAnswer": "A",
        "_practiceMeta": {
            "bucketId": blueprint.buckets[0].bucket_id,
            "verified": True,
        },
    }
    assert validate_final_set(blueprint=blueprint, linked_questions=[item]).ready is False
    duplicate = [item, {**item, "questionId": "q2"}]
    assert validate_final_set(blueprint=blueprint, linked_questions=duplicate).ready is False


@pytest.mark.parametrize(
    ("question_type", "options", "correct_answer", "reason_code"),
    [
        ("mcq", [], "A", "INVALID_OPTION_COUNT"),
        ("mcq", ["A", "", "C", "D"], "A", "EMPTY_OPTION_TEXT"),
        ("mcq", ["A", "a", "C", "D"], "A", "DUPLICATE_OPTIONS"),
        ("mcq", ["A", "B", "C"], "A", "INVALID_OPTION_COUNT"),
        ("mcq", ["A", "B", "C", "D"], "E", "CORRECT_OPTION_NOT_FOUND"),
        ("numerical", [], "4", "UNSUPPORTED_QUESTION_TYPE"),
    ],
)
def test_current_player_question_contract_rejects_unplayable_items(
    question_type: str,
    options: list[str],
    correct_answer: str,
    reason_code: str,
) -> None:
    validation = validate_playable_question(
        question_type=question_type,
        question="What is two plus two?",
        options=options,
        correct_answer=correct_answer,
        solution="Two plus two equals four.",
        solution_required=True,
    )
    assert (validation.valid, validation.reason_code) == (False, reason_code)


def test_current_player_question_contract_accepts_valid_mcq() -> None:
    validation = validate_playable_question(
        question_type="mcq",
        question="What is two plus two?",
        options=["1", "2", "3", "4"],
        correct_answer="4",
        solution="Two plus two equals four.",
        solution_required=True,
    )
    assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")


def test_persisted_schema_v2_answer_contract_matches_index_v1_player_shape() -> None:
    item = {
        "question": "What is two plus two?",
        "options": ["1", "2", "3", "4"],
        "correctAnswer": "4",
        "explanation": "Two plus two equals four.",
        "answers": json.dumps(
            {
                "schemaVersion": "2",
                "optionIdentity": "INDEX_V1",
                "options": [
                    {"optionId": 0, "value": "1"},
                    {"optionId": 1, "value": "2"},
                    {"optionId": 2, "value": "3"},
                    {"optionId": 3, "value": "4"},
                ],
                "correctOptionId": 3,
                "correctAnswer": "4",
                "answerExplanation": "Two plus two equals four.",
                "answerStatus": "VERIFIED",
                "answerVersion": 1,
            }
        ),
        "_practiceMeta": {"questionType": "mcq", "language": "english"},
    }

    validation = validate_persisted_playable_question(
        item,
        expected_question_type="mcq",
        expected_language="english",
        solution_required=True,
    )

    assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")
    dynamodb_item = {
        **item,
        "answers": json.dumps(
            {
                "schemaVersion": "2",
                "optionIdentity": "INDEX_V1",
                "options": [
                    {"optionId": Decimal(index), "value": option}
                    for index, option in enumerate(item["options"])
                ],
                "correctOptionId": Decimal(3),
                "correctAnswer": "4",
                "answerExplanation": "Two plus two equals four.",
                "answerStatus": "VERIFIED",
                "answerVersion": Decimal(1),
            },
            default=str,
        ),
    }
    # DynamoDB deserializes numeric map values as Decimal rather than int.
    dynamodb_answers = json.loads(dynamodb_item["answers"])
    for option in dynamodb_answers["options"]:
        option["optionId"] = Decimal(option["optionId"])
    dynamodb_answers["correctOptionId"] = Decimal(dynamodb_answers["correctOptionId"])
    dynamodb_answers["answerVersion"] = Decimal(dynamodb_answers["answerVersion"])
    dynamodb_item["answers"] = dynamodb_answers
    validation = validate_persisted_playable_question(
        dynamodb_item,
        expected_question_type="mcq",
        expected_language="english",
        solution_required=True,
    )
    assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")
    malformed = {
        **item,
        "answers": item["answers"].replace(
            '"correctOptionId": 3',
            '"correctOptionId": "3"',
        ),
    }
    validation = validate_persisted_playable_question(
        malformed,
        expected_question_type="mcq",
        expected_language="english",
        solution_required=True,
    )
    assert (validation.valid, validation.reason_code) == (
        False,
        "ANSWER_CONTRACT_MISMATCH",
    )


def test_planner_cannot_emit_question_types_the_current_player_cannot_render() -> None:
    blueprint = deterministic_blueprint(request("Create two algebra questions"))
    unsupported = blueprint.model_copy(
        update={
            "buckets": [
                blueprint.buckets[0].model_copy(
                    update={"question_type": QuestionType.NUMERICAL}
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="PLAYER_UNSUPPORTED_QUESTION_TYPE"):
        apply_system_bucket_policy(unsupported, request("Create two algebra questions"))


def test_final_ready_gate_rejects_a_persisted_unplayable_question() -> None:
    blueprint = deterministic_blueprint(request("Create one algebra question"))
    valid = {
        "questionId": "q1",
        "question": "What is two plus two?",
        "options": ["1", "2", "3", "4"],
        "answers": json.dumps({"correctAnswer": "4", "options": ["1", "2", "3", "4"]}),
        "correctAnswer": "4",
        "explanation": "Two plus two equals four.",
        "topic": blueprint.slots[0].topic_id,
        "difficulty": blueprint.slots[0].difficulty.value,
        "_practiceMeta": {
            "bucketId": blueprint.buckets[0].bucket_id,
            "slotId": blueprint.slots[0].slot_id,
            "verified": True,
            "sourceType": "AI_GENERATED",
            "verificationMethod": "INDEPENDENT_MODEL_V2",
            "questionType": "mcq",
            "language": "english",
        },
    }
    assert validate_final_set(blueprint=blueprint, linked_questions=[valid]).ready is True
    invalid = {**valid, "options": []}
    assert validate_final_set(blueprint=blueprint, linked_questions=[invalid]).ready is False


def test_pattern_provider_is_noop_when_disabled_and_rejects_unimplemented_enable() -> None:
    provider = build_pattern_context_provider(enabled=False)
    assert isinstance(provider, NoOpPatternContextProvider)
    assert provider.discover_catalog(request("Create one similar question")) == ()
    with pytest.raises(RuntimeError):
        build_pattern_context_provider(enabled=True)
