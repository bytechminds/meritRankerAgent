"""Focused DynamoDB mapping and conditional-write tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from features.practice_generation.execution_control import (
    ActivePracticeExecutionRegistry,
    bind_practice_execution,
)
from features.practice_generation.matching import (
    ReusableQuestion,
    _candidate_matches_slot,
    build_question_bank_id,
    build_question_bank_version_hash,
    build_reuse_difficulty_prefix,
    build_slot_reuse_bucket_key,
    question_bank_identity_is_intact,
    question_bank_version_hash_from_item,
    reusable_question_from_item,
)
from features.practice_generation.pattern_context import PatternSlotSelection
from features.practice_generation.planning import resolve_practice_request
from features.practice_generation.repositories import (
    AssessmentRepository,
    PracticeRepositoryError,
    QuestionRepository,
    estimate_dynamodb_item_size,
)
from features.practice_generation.resource_validation import IndexProjection
from features.practice_generation.schemas import Difficulty, GeneratedQuestion, PlannerSlot

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()


def _item(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: _SERIALIZER.serialize(entry) for key, entry in value.items()}


def _plain(value: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: _DESERIALIZER.deserialize(entry) for key, entry in value.items()}


class RecordingClient:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.transact_calls: list[dict[str, Any]] = []
        self.query_calls = 0
        self.update_calls: list[dict[str, Any]] = []

    def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        return {}

    def transact_write_items(self, **kwargs):
        self.transact_calls.append(kwargs)
        return {}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        if kwargs.get("ReturnValues"):
            return {
                "Attributes": _item(
                    {
                        "testId": "test-1",
                        "status": "GENERATING",
                        "updatedAt": "now",
                        "meta": {
                            "generationGroups": {
                                "group-1": {
                                    "groupId": "group-1",
                                    "bucketId": "bucket-1",
                                    "state": "RUNNING",
                                }
                            }
                        },
                    }
                )
            }
        return {}

    def query(self, **_kwargs):
        self.query_calls += 1
        count = 1 if self.query_calls == 1 else 2
        return {
            "Items": [
                _item(
                    {
                        "questionId": f"question-{index}",
                        "testId": "test-1",
                        "question": f"Question {index}",
                        "meta": '{"bucketId":"bucket-1","verified":true}',
                    }
                )
                for index in range(count)
            ]
        }


class FencedQuestionClient(RecordingClient):
    def transact_write_items(self, **kwargs):
        self.transact_calls.append(kwargs)
        raise ClientError(
            {
                "Error": {"Code": "TransactionCanceledException", "Message": "cancelled"},
                "CancellationReasons": [
                    {"Code": "ConditionalCheckFailed"},
                    {"Code": "None"},
                ],
            },
            "TransactWriteItems",
        )

    def get_item(self, **_kwargs):
        return {
            "Item": _item(
                {
                    "testId": "test-1",
                    "meta": {"activeExecutionId": "new-execution"},
                }
            )
        }


def _request():
    return resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query="Create five algebra questions",
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )


def test_assessment_creation_uses_existing_model_and_conditional_idempotency() -> None:
    client = RecordingClient()
    repository = AssessmentRepository(client, table_name="MockTestQuiz-table")
    request = _request().model_copy(update={"exam_profile_id": "cat_management_pre"})
    item, duplicate = repository.create_or_get("test-1", request)

    assert duplicate is False
    assert (
        item["testId"],
        item["status"],
        item["visibility"],
        item["origin"],
    ) == ("test-1", "GENERATING", "PRIVATE", "AI_CUSTOM")
    assert client.put_calls[0]["ConditionExpression"] == ("attribute_not_exists(testId)")
    assert "querySummary" not in item["meta"]["practiceRequest"]
    assert "examProfileId" not in item["meta"]
    assert item["meta"]["practiceRequest"]["examProfileId"] == "cat_management_pre"
    assert item["assessmentMode"] == "PRACTICE"
    assert item["examProfileId"] == "cat_management_pre"


def test_generated_question_is_assessment_owned_and_conditionally_linked() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    created = repository.link_generated(
        test_id="test-1",
        question=GeneratedQuestion(
            generation_item_id="item-1",
            bucket_id="bucket-1",
            question="If x plus 1 is 2, what is x?",
            question_type="mcq",
            options=["0", "1", "2", "3"],
            correct_answer="1",
            solution="Subtract one from both sides.",
            subject="math",
            topic="algebra",
            difficulty="intermediate",
        ),
        verified=True,
        group_id="bucket-1-g1",
        generator_route="math.generator.intermediate",
        generator_model="test-generator",
        verification_policy="MANDATORY",
        verification_method="MODEL",
        language="english",
        source_question_bank_id="pattern-v1-linked",
        pattern_selection=PatternSlotSelection(
            patternId="pattern-1",
            patternVersionHash="version-1",
            tier="GUIDANCE_SAFE",
            reason="verified_generation",
        ),
    )

    assert created is True
    put = client.put_calls[0]
    assert put["TableName"] == "Question-table"
    assert put["ConditionExpression"] == ("attribute_not_exists(questionId)")
    stored = _plain(put["Item"])
    assert stored["meta"]["generatorRoute"] == "math.generator.intermediate"
    assert stored["meta"]["generationGroupId"] == "bucket-1-g1"
    assert stored["meta"]["questionType"] == "mcq"
    assert stored["meta"]["language"] == "english"
    assert stored["meta"]["sourceQuestionBankId"] == "pattern-v1-linked"
    assert stored["patternId"] == "pattern-1"
    assert stored["patternVersionHash"] == "version-1"
    assert client.transact_calls == []


def test_generated_question_write_is_transaction_fenced_during_an_execution() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    registry = ActivePracticeExecutionRegistry()
    registry.register("execution-1", "test-1")

    with bind_practice_execution("execution-1", registry):
        created = repository._put_question(
            question_id="question-1",
            test_id="test-1",
            question="Question?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            solution="Solution",
            subject="math",
            topic="algebra",
            difficulty="basic",
            meta={"verified": True},
        )

    assert created is True
    assert client.put_calls == []
    transaction = client.transact_calls[0]["TransactItems"]
    assert transaction[0]["ConditionCheck"]["ExpressionAttributeValues"] == _item(
        {":execution": "execution-1"}
    )


def test_old_executor_cannot_persist_after_a_new_execution_claims() -> None:
    client = FencedQuestionClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    registry = ActivePracticeExecutionRegistry()
    registry.register("old-execution", "test-1")

    with bind_practice_execution("old-execution", registry):
        with pytest.raises(PracticeRepositoryError, match="PRACTICE_EXECUTION_FENCED"):
            repository._put_question(
                question_id="question-1",
                test_id="test-1",
                question="Question?",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                solution="Solution",
                subject="math",
                topic="algebra",
                difficulty="basic",
                meta={"verified": True},
            )
    assert client.update_calls == []


def test_final_question_updates_retain_the_execution_fence() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    registry = ActivePracticeExecutionRegistry()
    registry.register("execution-1", "test-1")

    with bind_practice_execution("execution-1", registry):
        repository.assign_positions(
            [
                {
                    "questionId": "question-1",
                    "testId": "test-1",
                    "_practiceMeta": {"bucketId": "bucket-1"},
                }
            ]
        )

    assert client.update_calls == []
    transaction = client.transact_calls[0]["TransactItems"]
    assert transaction[0]["ConditionCheck"]["ExpressionAttributeValues"] == _item(
        {":execution": "execution-1"}
    )
    assert transaction[1]["Update"]["Key"] == _item({"questionId": "question-1"})


def test_verified_pattern_question_persists_authoritative_link_and_returns_qb_id() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    question = GeneratedQuestion(
        generation_item_id="item-1",
        bucket_id="bucket-1",
        question="If x plus 1 is 2, what is x?",
        question_type="mcq",
        options=["0", "1", "2", "3"],
        correct_answer="1",
        solution="Subtract one from both sides.",
        subject="math",
        topic="algebra",
        difficulty="intermediate",
    )
    slot = PlannerSlot(
        slot_id="slot-001",
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty="intermediate",
        complexity="medium",
        exam_ids=["CAT"],
        question_type="mcq",
        target_skill="solve_linear_equation",
        variation_hint="vary_coefficients",
        generator_route_hint="math.generator.intermediate",
    )

    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=question,
        slot=slot,
        language="english",
        pattern_id="pattern-1",
        pattern_version_hash="version-1",
    )

    assert qb_id is not None
    stored = _plain(client.put_calls[0]["Item"])
    assert stored["qbId"] == qb_id
    assert stored["patternId"] == "pattern-1"
    assert stored["patternVersionHash"] == "version-1"
    assert stored["patternLinkEvidence"] == "VERIFIED_GENERATION"
    assert stored["meta"]["qualityStatus"] == "VERIFIED"


def test_schema_v2_generated_question_persists_private_versioned_answer_contract() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )

    repository.link_generated(
        test_id="test-v2",
        question=GeneratedQuestion.model_validate(
            {
                "schema_version": "2",
                "generation_item_id": "item-slot-001",
                "bucket_id": "bucket-1",
                "slot_id": "slot-001",
                "question": "If x plus one is two, what is x?",
                "question_type": "mcq",
                "options": [
                    {"option_id": "0", "value": "0"},
                    {"option_id": "1", "value": "1"},
                    {"option_id": "2", "value": "2"},
                    {"option_id": "3", "value": "3"},
                ],
                "correct_option_id": "1",
                "correct_answer": "1",
                "answer_explanation": "Subtract one from both sides.",
                "solution": "A non-authoritative alternate explanation.",
                "subject": "math",
                "topic": "algebra",
                "difficulty": "intermediate",
            }
        ),
        verified=True,
        group_id="group-1",
        generator_route="math.generator.intermediate",
        generator_model="test-generator",
        verification_policy="MANDATORY",
        verification_method="INDEPENDENT_MODEL_V2",
        language="english",
    )

    stored = _plain(client.put_calls[0]["Item"])
    answer = json.loads(stored["answers"])
    assert answer == {
        "schemaVersion": "2",
        "optionIdentity": "INDEX_V1",
        "options": [
            {"optionId": 0, "value": "0"},
            {"optionId": 1, "value": "1"},
            {"optionId": 2, "value": "2"},
            {"optionId": 3, "value": "3"},
        ],
        "correctOptionId": 1,
        "correctAnswer": "1",
        "answerExplanation": "Subtract one from both sides.",
        "answerStatus": "VERIFIED",
        "answerVersion": 1,
    }
    assert stored["explanation"] == "Subtract one from both sides."
    assert stored["meta"]["slotId"] == "slot-001"


def test_answer_rebalancing_preserves_schema_v2_answer_authority() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )
    questions = [
        {
            "questionId": f"question-{index}",
            "options": ["correct", "wrong-1", "wrong-2", "wrong-3"],
            "correctAnswer": "correct",
            "explanation": "Verified explanation.",
            "answers": json.dumps({"answerVersion": 1}),
            "_practiceMeta": {"bucketId": "bucket-1", "schemaVersion": "2"},
        }
        for index in range(4)
    ]

    changed, positions = repository.rebalance_answer_positions("test-v2", questions)

    assert changed is True
    assert sorted(positions) == [0, 1, 2, 3]
    for call in client.update_calls:
        values = _plain(call["ExpressionAttributeValues"])
        answer = json.loads(values[":answers"])
        assert answer["schemaVersion"] == "2"
        assert answer["answerStatus"] == "VERIFIED"
        assert answer["answerVersion"] == 2
        assert type(answer["correctOptionId"]) is int
        assert answer["options"][answer["correctOptionId"]]["value"] == "correct"
        assert [option["optionId"] for option in answer["options"]] == [0, 1, 2, 3]


def test_reused_question_retains_question_bank_provenance_without_promoting_pattern_identity(
) -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )

    repository.link_reused(
        test_id="test-1",
        bucket_id="bucket-1",
        question=ReusableQuestion(
            question_id="bank-1",
            question="What is one plus one?",
            options=("1", "2", "3", "4"),
            correct_answer="2",
            solution="Add the values.",
            subject="math",
            topic="arithmetic",
            difficulty="medium",
            question_type="mcq",
            language="english",
            source="QuestionBank",
            source_updated_at="2026-07-01T00:00:00Z",
            pattern_id="forged-question-bank-pattern",
            pattern_version_hash="forged-version",
            pattern_link_evidence="VERIFIED_GENERATION",
        ),
    )

    stored = _plain(client.put_calls[0]["Item"])
    assert stored["meta"]["sourceType"] == "QUESTION_BANK"
    assert stored["meta"]["sourceQuestionBankId"] == "bank-1"
    assert stored["meta"]["sourceVersion"] == "2026-07-01T00:00:00Z"
    assert stored["meta"]["questionType"] == "mcq"
    assert stored["meta"]["language"] == "english"
    assert "patternId" not in stored
    assert "patternVersionHash" not in stored


def test_runtime_reused_question_persists_only_the_trusted_selection_identity() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )

    repository.link_reused(
        test_id="test-1",
        bucket_id="bucket-1",
        question=ReusableQuestion(
            question_id="bank-1",
            question="What is one plus one?",
            options=("1", "2", "3", "4"),
            correct_answer="2",
            solution="Add the values.",
            subject="math",
            topic="arithmetic",
            difficulty="medium",
            question_type="mcq",
            language="english",
            source="QuestionBank",
            source_updated_at="2026-07-01T00:00:00Z",
            pattern_id="forged-question-bank-pattern",
            pattern_version_hash="forged-version",
            pattern_link_evidence="VERIFIED_GENERATION",
        ),
        trusted_pattern_selection=PatternSlotSelection(
            patternId="runtime-selected-pattern",
            patternVersionHash="runtime-selected-version",
            tier="REUSE_SAFE",
            reason="authoritative_playable_question_compatible",
        ),
    )

    stored = _plain(client.put_calls[0]["Item"])
    assert stored["patternId"] == "runtime-selected-pattern"
    assert stored["patternVersionHash"] == "runtime-selected-version"


def test_reused_question_rejects_non_reuse_safe_selection_identity() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )

    repository.link_reused(
        test_id="test-1",
        bucket_id="bucket-1",
        question=ReusableQuestion(
            question_id="bank-1",
            question="What is one plus one?",
            options=("1", "2", "3", "4"),
            correct_answer="2",
            solution="Add the values.",
            subject="math",
            topic="arithmetic",
            difficulty="medium",
            question_type="mcq",
            language="english",
            source="QuestionBank",
            source_updated_at="2026-07-01T00:00:00Z",
        ),
        trusted_pattern_selection=PatternSlotSelection(
            patternId="guidance-only-pattern",
            patternVersionHash="guidance-only-version",
            tier="GUIDANCE_SAFE",
            reason="guidance",
        ),
    )

    stored = _plain(client.put_calls[0]["Item"])
    assert "patternId" not in stored
    assert "patternVersionHash" not in stored


def test_linked_question_query_does_not_treat_gsi_as_authoritative_counter() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )

    linked = repository.list_linked("test-1")

    assert (len(linked), client.query_calls) == (1, 1)


class PaginatedQueryClient:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []
        self.batch_calls: list[dict[str, Any]] = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        page = len(self.queries)
        return {
            "Items": [
                _item(
                    {
                        "qbId": f"bank-{page}",
                        "category": "math",
                        "difficulty": "MEDIUM",
                        "tags": ["algebra"],
                        "meta": {
                            "status": "ACTIVE",
                            "qualityStatus": "VERIFIED",
                            "reusable": True,
                            "subject": "math",
                            "topic": "algebra",
                            "questionType": "mcq",
                        },
                    }
                )
            ],
            "LastEvaluatedKey": _item({"qbId": f"bank-{page}"}),
            "ConsumedCapacity": {"CapacityUnits": 0.5},
        }

    def batch_get_item(self, **kwargs):
        self.batch_calls.append(kwargs)
        return {"Responses": {"QuestionBank-table": []}}

    def scan(self, **_kwargs):
        raise AssertionError("Scan must never be called")


def test_question_bank_retrieval_is_query_only_bounded_and_projected() -> None:
    client = PaginatedQueryClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
        query_page_size=1,
        query_max_pages=2,
    )

    result = repository.query_reuse_candidates(category="math", limit=10)

    assert (result.page_count, result.has_more_pages, len(result.items)) == (2, True, 2)
    assert all(call["IndexName"] == "getByCategory" for call in client.queries)
    assert all("ProjectionExpression" in call for call in client.queries)
    assert client.batch_calls == []


class HydratingQueryClient(PaginatedQueryClient):
    def batch_get_item(self, **kwargs):
        self.batch_calls.append(kwargs)
        requested_id = _plain(kwargs["RequestItems"]["QuestionBank-table"]["Keys"][0])["qbId"]
        return {
            "Responses": {
                "QuestionBank-table": [
                    _item(
                        {
                            "qbId": requested_id,
                            "category": "math",
                            "difficulty": "MEDIUM",
                            "tags": ["algebra"],
                            "meta": {"status": "ACTIVE"},
                        }
                    )
                ]
            }
        }


def test_keys_only_reuse_projection_hydrates_candidates_by_primary_key() -> None:
    client = HydratingQueryClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
        question_bank_reuse_index="getByReuseBucket",
        question_bank_reuse_projection=IndexProjection(
            projection_type="KEYS_ONLY",
            projected_attributes=frozenset({"qbId", "reuseBucketKey", "reuseSortKey"}),
            query_mode="IDS_AND_BATCH_GET",
        ),
        query_page_size=1,
        query_max_pages=1,
    )

    result = repository.query_topic_reuse_candidates(
        reuse_bucket_key="math#algebra#mcq#english",
        limit=1,
        difficulty_prefix="v1#medium#",
    )

    assert len(result.items) == 1
    assert client.queries[0]["IndexName"] == "getByReuseBucket"
    assert client.queries[0]["ProjectionExpression"] == "qbId"
    assert (
        _plain(client.queries[0]["ExpressionAttributeValues"])[":lookup"]
        == "math#algebra#mcq#english"
    )
    assert client.queries[0]["KeyConditionExpression"] == (
        "#lookup = :lookup AND begins_with(#sort_key, :sort_key_prefix)"
    )
    assert client.queries[0]["ExpressionAttributeNames"]["#sort_key"] == "reuseSortKey"
    assert (
        _plain(client.queries[0]["ExpressionAttributeValues"])[":sort_key_prefix"]
        == "v1#medium#"
    )
    assert len(client.batch_calls) == 1


def test_legacy_reuse_query_logs_safe_aws_diagnostics_and_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeniedClient:
        def query(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "sensitive provider detail",
                    }
                },
                "Query",
            )

    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "features.practice_generation.repositories.emit_practice_event",
        lambda name, **kwargs: events.append((name, kwargs)),
    )
    repository = QuestionRepository(
        DeniedClient(),
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
        question_bank_reuse_index="getByReuseBucket",
        pattern_context_enabled=False,
        pattern_reuse_enabled=False,
    )

    with pytest.raises(PracticeRepositoryError, match="QUESTION_BANK_QUERY_FAILED"):
        repository.query_topic_reuse_candidates(
            reuse_bucket_key="math#algebra#mcq#english",
            limit=5,
        )

    name, event = events[0]
    assert name == "PRACTICE_REPOSITORY_FAILURE"
    assert event["details"] == {
        "reasonCode": "QUESTION_BANK_QUERY_FAILED",
        "operation": "Query",
        "logicalTable": "QuestionBank",
        "logicalIndex": "QuestionBank.reuse",
        "awsExceptionType": "ClientError",
        "awsErrorCode": "AccessDeniedException",
        "patternContextEnabled": False,
        "patternReuseEnabled": False,
        "fallbackDecision": "fatal_legacy_repository_failure",
    }
    assert "sensitive provider detail" not in str(event)


def test_selected_question_bank_records_are_fetched_by_primary_keys_only() -> None:
    client = PaginatedQueryClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )

    repository.get_reuse_candidates(["bank-1", "bank-1", "bank-2"])

    request = client.batch_calls[0]["RequestItems"]["QuestionBank-table"]
    assert len(request["Keys"]) == 2
    assert request["ConsistentRead"] is True


def test_dynamodb_item_estimator_counts_attribute_names_and_nested_values() -> None:
    small = estimate_dynamodb_item_size({"testId": "t", "meta": {"phase": "READY"}})
    larger = estimate_dynamodb_item_size(
        {"testId": "long-test-id", "meta": {"phase": "READY", "groups": [1, 2, 3]}}
    )
    assert larger > small > len("tREADY")


class PlayableClient(RecordingClient):
    def __init__(self, *, ready: bool) -> None:
        super().__init__()
        self.ready = ready
        self.last_query: dict[str, Any] | None = None

    def get_item(self, **_kwargs):
        return {
            "Item": _item(
                {
                    "testId": "test-1",
                    "userId": "user-1",
                    "status": "READY" if self.ready else "GENERATING",
                    "live": self.ready,
                    "meta": {
                        "playable": self.ready,
                        "acceptedCount": 1,
                        "readyCount": 1,
                        "readyQuestionIds": ["question-1"],
                        "practiceRequest": {"language": "english", "includeSolutions": True},
                    },
                }
            )
        }

    def query(self, **kwargs):
        self.query_calls += 1
        self.last_query = kwargs
        return {
            "Items": (
                [
                    _item(
                        {
                            "questionId": "question-1",
                            "testId": "test-1",
                            "position": 1,
                            "question": "What is two plus two?",
                            "options": ["1", "2", "3", "4"],
                            "answers": json.dumps(
                                {"correctAnswer": "4", "options": ["1", "2", "3", "4"]}
                            ),
                            "correctAnswer": "4",
                            "explanation": "Two plus two equals four.",
                            "meta": {"questionType": "mcq", "language": "english"},
                        }
                    )
                ]
                if self.ready
                else []
            )
        }


class LaggingPlayableClient(PlayableClient):
    def query(self, **kwargs):
        self.query_calls += 1
        self.last_query = kwargs
        return {"Items": []}

    def batch_get_item(self, **kwargs):
        requested = kwargs["RequestItems"]["Question-table"]["Keys"]
        return {
            "Responses": {
                "Question-table": [
                    _item(
                        {
                            "questionId": _plain(key)["questionId"],
                            "testId": "test-1",
                            "position": 1,
                            "question": "What is two plus two?",
                            "options": ["1", "2", "3", "4"],
                            "answers": json.dumps(
                                {"correctAnswer": "4", "options": ["1", "2", "3", "4"]}
                            ),
                            "correctAnswer": "4",
                            "explanation": "Two plus two equals four.",
                            "meta": {"questionType": "mcq", "language": "english"},
                        }
                    )
                    for key in requested
                ]
            }
        }


def test_student_question_fetch_requires_ready_and_uses_test_id_gsi() -> None:
    blocked = PlayableClient(ready=False)
    blocked_repository = QuestionRepository(
        blocked,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )
    assert blocked_repository.list_playable("test-1", user_id="user-1") == []
    assert blocked.query_calls == 0

    ready = PlayableClient(ready=True)
    ready_repository = QuestionRepository(
        ready,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )
    assert ready_repository.list_playable("test-1", user_id="other-user") == []
    assert ready.query_calls == 0
    assert len(ready_repository.list_playable("test-1", user_id="user-1")) == 1
    assert ready.last_query is not None
    assert ready.last_query["IndexName"] == "getQuestionsByTestId"
    assert ready.last_query["KeyConditionExpression"] == "#test = :test"


def test_student_question_fetch_falls_back_to_strong_manifest_reads_on_gsi_lag() -> None:
    client = LaggingPlayableClient(ready=True)
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )

    questions = repository.list_playable("test-1", user_id="user-1")

    assert [question["questionId"] for question in questions] == ["question-1"]


class RetryingBatchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def batch_get_item(self, **kwargs):
        self.calls.append(kwargs)
        request = kwargs["RequestItems"]["Question-table"]
        if len(self.calls) == 1:
            return {
                "Responses": {"Question-table": []},
                "UnprocessedKeys": {"Question-table": request},
            }
        return {
            "Responses": {
                "Question-table": [
                    _item(
                        {
                            "questionId": _plain(key)["questionId"],
                            "testId": "test-1",
                        }
                    )
                    for key in request["Keys"]
                ]
            }
        }


def test_manifest_batch_get_retries_unprocessed_keys_once_for_100_questions() -> None:
    client = RetryingBatchClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )

    questions = repository.get_questions_by_ids([f"question-{index}" for index in range(100)])

    assert len(questions) == 100
    assert len(client.calls) == 2
    assert len(client.calls[0]["RequestItems"]["Question-table"]["Keys"]) == 100


def test_meta_updates_use_document_paths_without_full_meta_replacement() -> None:
    client = RecordingClient()
    repository = AssessmentRepository(client, table_name="MockTestQuiz-table")

    repository.update(
        "test-1",
        meta_updates={"phase": "GENERATING", "progressPercent": 10},
    )

    expression = client.update_calls[0]["UpdateExpression"]
    assert "#meta.#field0" in expression
    assert "#meta = " not in expression


def test_bucket_recovery_is_conditioned_on_test_slot_and_verification() -> None:
    client = RecordingClient()
    repository = QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="getQuestionsByTestId",
        question_bank_category_index="getByCategory",
    )

    repaired = repository.repair_bucket_assignments(
        "test-1",
        {"question-1": ("slot-002", "slot-bucket-002")},
    )

    assert repaired == 1
    call = client.update_calls[0]
    assert call["UpdateExpression"] == (
        "SET #meta.#bucket = :bucket, #updated = :updated"
    )
    assert call["ConditionExpression"] == (
        "#test = :test AND #meta.#slot = :slot AND #meta.#verified = :verified"
    )
    values = _plain(call["ExpressionAttributeValues"])
    assert {
        key: values[key]
        for key in (":bucket", ":slot", ":verified", ":test")
    } == {
        ":bucket": "slot-bucket-002",
        ":slot": "slot-002",
        ":verified": True,
        ":test": "test-1",
    }


def test_four_group_updates_target_independent_atomic_paths() -> None:
    client = RecordingClient()
    repository = AssessmentRepository(client, table_name="MockTestQuiz-table")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda group_id: repository.update_group(
                    "test-1",
                    group_id,
                    state="COMPLETED",
                ),
                [f"group-{index}" for index in range(4)],
            )
        )

    group_aliases = {call["ExpressionAttributeNames"]["#group"] for call in client.update_calls}
    assert group_aliases == {"group-0", "group-1", "group-2", "group-3"}
    assert all("#meta = " not in call["UpdateExpression"] for call in client.update_calls)


def test_meta_size_guard_rejects_oversized_update_before_write() -> None:
    client = RecordingClient()
    repository = AssessmentRepository(
        client,
        table_name="MockTestQuiz-table",
        meta_safe_size_bytes=64_000,
    )

    with pytest.raises(
        PracticeRepositoryError,
        match="ASSESSMENT_META_SIZE_LIMIT_EXCEEDED",
    ):
        repository.update(
            "test-1",
            meta_updates={"blueprint": "x" * 70_000},
        )
    assert client.update_calls == []


class ClaimClient(RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.claimed = False

    def update_item(self, **kwargs):
        if "claimCount" not in str(kwargs.get("ExpressionAttributeNames")):
            return super().update_item(**kwargs)
        if self.claimed:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConditionalCheckFailedException",
                        "Message": "claimed",
                    }
                },
                "UpdateItem",
            )
        self.claimed = True
        self.update_calls.append(kwargs)
        return {
            "Attributes": _item(
                {
                    "testId": "test-1",
                    "status": "GENERATING",
                    "meta": {
                        "generationGroups": {
                            "group-1": {
                                "groupId": "group-1",
                                "bucketId": "bucket-1",
                                "state": "RUNNING",
                            }
                        }
                    },
                }
            )
        }


def test_duplicate_group_claim_is_rejected() -> None:
    client = ClaimClient()
    repository = AssessmentRepository(client, table_name="MockTestQuiz-table")

    assert repository.claim_group("test-1", "group-1") is not None
    assert repository.claim_group("test-1", "group-1") is None
    condition = client.update_calls[0]["ConditionExpression"]
    assert "#state = :pending" in condition


def test_group_transition_updates_existing_group() -> None:
    client = RecordingClient()
    repository = AssessmentRepository(client, table_name="MockTestQuiz-table")

    repository.update_group(
        "test-1",
        "group-1",
        state="COMPLETED",
    )

    condition = client.update_calls[0]["ConditionExpression"]
    assert condition == "attribute_exists(#meta.#groups.#group)"


def _promotion_question(**overrides: Any) -> GeneratedQuestion:
    payload: dict[str, Any] = {
        "generation_item_id": "item-1",
        "bucket_id": "bucket-1",
        "question": "If x plus 1 is 2, what is x?",
        "question_type": "mcq",
        "options": ["0", "1", "2", "3"],
        "correct_answer": "1",
        "solution": "Subtract one from both sides.",
        "subject": "math",
        "topic": "algebra",
        "difficulty": "intermediate",
    }
    payload.update(overrides)
    return GeneratedQuestion.model_validate(payload)


def _promotion_slot() -> PlannerSlot:
    return PlannerSlot(
        slot_id="slot-001",
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty="intermediate",
        complexity="medium",
        exam_ids=["CAT"],
        question_type="mcq",
        target_skill="solve_linear_equation",
        variation_hint="vary_coefficients",
        generator_route_hint="math.generator.intermediate",
    )


def _promotion_repository(client: Any) -> QuestionRepository:
    return QuestionRepository(
        client,
        assessment_table="MockTestQuiz-table",
        question_table="Question-table",
        question_bank_table="QuestionBank-table",
        question_test_index="questionsByTestIdAndCreatedAt",
        question_bank_category_index="questionBanksByCategory",
    )


def test_verified_question_promotes_without_any_pattern_linkage() -> None:
    client = RecordingClient()
    repository = _promotion_repository(client)

    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )

    assert qb_id is not None
    assert qb_id.startswith("qb-v2-")
    stored = _plain(client.put_calls[0]["Item"])
    assert "patternId" not in stored
    assert "patternVersionHash" not in stored
    assert "patternLinkEvidence" not in stored
    assert stored["source"] == "SYSTEM_VERIFIED_PRACTICE"
    # Reuse eligibility is meta-based, so a Pattern-less row must still be readable
    # by the unchanged exact-reuse path.
    assert stored["meta"]["status"] == "ACTIVE"
    assert stored["meta"]["qualityStatus"] == "VERIFIED"
    assert stored["meta"]["reusable"] is True
    assert stored["meta"]["visibility"] == "PLATFORM"
    assert stored["reuseBucketKey"]
    assert stored["reuseSortKey"].startswith("v1#medium#")


def test_pattern_less_promoted_row_is_accepted_by_existing_reuse_gates() -> None:
    client = RecordingClient()
    repository = _promotion_repository(client)
    repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    stored = _plain(client.put_calls[0]["Item"])

    candidate = reusable_question_from_item(stored, requested_language="english")

    assert candidate is not None
    assert candidate.question_id == stored["qbId"]
    assert candidate.pattern_family_id is None


def test_question_identity_is_stable_across_retry_and_pattern_attachment() -> None:
    first_client = RecordingClient()
    retry_client = RecordingClient()
    pattern_client = RecordingClient()

    unlinked = _promotion_repository(first_client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    retried = _promotion_repository(retry_client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    linked = _promotion_repository(pattern_client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
        pattern_id="pattern-1",
        pattern_version_hash="version-1",
    )

    assert unlinked == retried == linked
    assert _plain(pattern_client.put_calls[0]["Item"])["patternId"] == "pattern-1"


def test_question_identity_separates_materially_different_playable_content() -> None:
    slot = _promotion_slot()
    repository = _promotion_repository(RecordingClient())

    baseline = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(), slot=slot, language="english"
    )
    other_options = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(options=["0", "1", "5", "9"]),
        slot=slot,
        language="english",
    )
    other_answer = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(correct_answer="2"),
        slot=slot,
        language="english",
    )
    other_values = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(question="If x plus 7 is 2, what is x?"),
        slot=slot,
        language="english",
    )

    assert len({baseline, other_options, other_answer, other_values}) == 4


def test_question_identity_ignores_explanation_and_classification_only_changes() -> None:
    repository = _promotion_repository(RecordingClient())
    baseline = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(), slot=_promotion_slot(), language="english"
    )
    reworded_solution = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(solution="Take one away from each side."),
        slot=_promotion_slot(),
        language="english",
    )

    assert baseline == reworded_solution


class _ConditionalPutClient(RecordingClient):
    """Second writer of the same logical Question loses the conditional put."""

    def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")


def test_duplicate_promotion_returns_one_identity_without_overwrite() -> None:
    client = _ConditionalPutClient()
    repository = _promotion_repository(client)

    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )

    assert qb_id is not None
    assert client.put_calls[0]["ConditionExpression"] == "attribute_not_exists(qbId)"


class _ConditionalDynamoClient(RecordingClient):
    """Enforces attribute_not_exists(qbId) the way DynamoDB does."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[str, dict[str, Any]] = {}

    def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        key = kwargs["Item"]["qbId"]["S"]
        if kwargs.get("ConditionExpression") == "attribute_not_exists(qbId)" and key in self.rows:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.rows[key] = kwargs["Item"]
        return {}


def test_phase_a_round_trip_preserves_the_full_playable_question() -> None:
    client = _ConditionalDynamoClient()
    slot = _promotion_slot()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(), slot=slot, language="english"
    )
    stored = _plain(client.rows[qb_id])

    candidate = reusable_question_from_item(stored, requested_language="english")

    assert candidate is not None
    assert candidate.question_id == qb_id
    assert candidate.question == "If x plus 1 is 2, what is x?"
    assert candidate.options == ("0", "1", "2", "3")
    assert candidate.correct_answer == "1"
    assert candidate.solution == "Subtract one from both sides."
    assert candidate.language == "english"
    assert candidate.difficulty == "medium"
    assert candidate.exam_ids == ("CAT",)
    assert candidate.question_type == "mcq"
    # Reuse keys must let getByReuseBucket find it again for the same demand.
    assert stored["reuseBucketKey"] == build_slot_reuse_bucket_key(slot, language="english")
    assert stored["reuseSortKey"].startswith(
        build_reuse_difficulty_prefix(slot.difficulty.value)
    )


def test_second_promotion_never_attaches_pattern_linkage_to_an_existing_row() -> None:
    """Characterises the create-only contract: the conditional put is a no-op."""
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    first = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(), slot=_promotion_slot(), language="english"
    )
    second = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
        pattern_id="pattern-1",
        pattern_version_hash="version-1",
    )

    assert first == second
    stored = _plain(client.rows[first])
    assert "patternId" not in stored
    assert stored["source"] == "SYSTEM_VERIFIED_PRACTICE"


def test_second_promotion_never_rewrites_classification_or_solution() -> None:
    """Create-only rows keep the first classification and first explanation."""
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    first = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(solution="First explanation."),
        slot=_promotion_slot(),
        language="english",
    )
    repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(solution="Corrected explanation."),
        slot=_promotion_slot(),
        language="english",
    )
    stored = _plain(client.rows[first])

    assert stored["explanation"] == "First explanation."
    assert stored["difficulty"] == "MEDIUM"


def test_option_order_is_part_of_playable_identity() -> None:
    baseline = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(options=["0", "1", "2", "3"]),
        slot=_promotion_slot(),
        language="english",
    )
    reordered = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(options=["3", "2", "1", "0"]),
        slot=_promotion_slot(),
        language="english",
    )

    assert baseline != reordered


def test_identity_segments_resist_delimiter_injection() -> None:
    shifted = build_question_bank_id(
        subject="math",
        difficulty="intermediate",
        exam_ids=["CAT"],
        question_type="mcq",
        question="What is 12|34 plus one?",
        options=("1", "2", "3", "4"),
        correct_answer="1",
        language="english",
    )
    split = build_question_bank_id(
        subject="math",
        difficulty="intermediate",
        exam_ids=["CAT"],
        question_type="mcq",
        question="What is 12",
        options=("34 plus one?", "2", "3", "4"),
        correct_answer="1",
        language="english",
    )

    assert shifted != split


def test_identity_uses_canonical_language_normalization() -> None:
    ids = {
        build_question_bank_id(
            subject="math",
            difficulty="intermediate",
            exam_ids=["CAT"],
            question_type="mcq",
            question="What is 12 plus one?",
            options=("1", "2", "3", "4"),
            correct_answer="1",
            language=language,
        )
        for language in ("english", "English", "EN", "en")
    }

    assert len(ids) == 1


def _identity(**overrides: Any) -> str | None:
    payload: dict[str, Any] = {
        "subject": "math",
        "difficulty": "basic",
        "exam_ids": ["CAT"],
        "language": "english",
        "question_type": "mcq",
        "question": "Two angles of a triangle are 55 and 75 degrees. Find the third.",
        "options": ("40", "50", "60", "70"),
        "correct_answer": "50",
    }
    payload.update(overrides)
    return build_question_bank_id(**payload)


def test_strict_compatibility_dimensions_separate_identity() -> None:
    baseline = _identity()

    assert baseline is not None
    assert _identity(subject="physics") != baseline
    assert _identity(difficulty="advanced") != baseline
    assert _identity(exam_ids=["GMAT"]) != baseline
    assert _identity(language="hindi") != baseline
    assert _identity(question_type="msq") != baseline


def test_exact_playable_content_separates_identity() -> None:
    baseline = _identity()

    assert _identity(options=("40", "50", "60", "99")) != baseline
    assert _identity(correct_answer="60") != baseline
    assert (
        _identity(question="Two angles of a triangle are 35 and 75 degrees. Find the third.")
        != baseline
    )


def test_exam_scope_identity_is_order_and_duplicate_insensitive() -> None:
    assert _identity(exam_ids=["CAT", "XAT"]) == _identity(exam_ids=["XAT", "CAT"])
    assert _identity(exam_ids=["CAT", "CAT"]) == _identity(exam_ids=["CAT"])
    assert _identity(exam_ids=["CAT", "XAT"]) != _identity(exam_ids=["CAT"])


def test_difficulty_aliases_resolve_to_one_identity() -> None:
    assert _identity(difficulty="basic") == _identity(difficulty="easy")
    assert _identity(difficulty="intermediate") == _identity(difficulty="medium")
    assert _identity(difficulty="advanced") == _identity(difficulty="hard")


def test_identity_is_none_when_a_strict_dimension_cannot_canonicalize() -> None:
    assert _identity(subject="not a real subject") is None
    assert _identity(difficulty="somewhat tricky") is None
    assert _identity(language="klingon") is None
    assert _identity(exam_ids=["not an exam id"]) is None


def test_classification_scopes_produce_two_distinct_rows() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    basic = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    harder = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot().model_copy(
            update={"difficulty": Difficulty.ADVANCED, "exam_ids": ["GMAT"]}
        ),
        language="english",
    )

    assert basic != harder
    assert len(client.rows) == 2
    first = _plain(client.rows[basic])
    second = _plain(client.rows[harder])
    assert first["difficulty"] == "MEDIUM"
    assert second["difficulty"] == "HARD"
    assert first["meta"]["examIds"] == ["CAT"]
    assert second["meta"]["examIds"] == ["GMAT"]


def test_pattern_link_attaches_then_stays_idempotent_and_never_overwrites() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    assert "patternId" not in _plain(client.rows[qb_id])

    assert repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id=qb_id, pattern_id="P1", pattern_version_hash="V1"
    )
    update = client.update_calls[-1]
    assert update["ConditionExpression"] == (
        "attribute_exists(qbId) AND "
        "(attribute_not_exists(patternId) OR patternId = :pattern)"
    )
    # Only Pattern-linkage fields and updatedAt may appear in the update expression.
    for protected in (
        "question",
        "answers",
        "correctAnswer",
        "explanation",
        "difficulty",
        "category",
        "meta",
        "source",
        "reuseBucketKey",
        "reuseSortKey",
    ):
        assert protected not in update["UpdateExpression"]


def test_pattern_link_attach_refuses_missing_identifiers() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)

    assert not repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id="", pattern_id="P1", pattern_version_hash="V1"
    )
    assert not repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id="qb-v2-x", pattern_id="", pattern_version_hash="V1"
    )
    assert client.update_calls == []


def test_pattern_link_conflict_is_reported_without_failing_practice() -> None:
    class _ConflictingClient(_ConditionalDynamoClient):
        def update_item(self, **kwargs):
            self.update_calls.append(kwargs)
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem"
            )

    client = _ConflictingClient()
    repository = _promotion_repository(client)

    assert not repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id="qb-v2-x", pattern_id="P2", pattern_version_hash="V2"
    )


def test_duplicate_promotion_with_pattern_enriches_the_existing_row() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    first = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    again = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
        pattern_id="P1",
        pattern_version_hash="V1",
    )

    assert first == again
    assert len(client.update_calls) == 1


def test_duplicate_promotion_without_pattern_performs_no_update() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    for _ in range(2):
        repository.persist_verified_question(
            test_id="test-1",
            question=_promotion_question(),
            slot=_promotion_slot(),
            language="english",
        )

    assert client.update_calls == []


def _promoted_row(client: _ConditionalDynamoClient | None = None) -> dict[str, Any]:
    client = client or _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    return _plain(client.rows[qb_id])


def test_versioned_row_passes_identity_integrity() -> None:
    assert question_bank_identity_is_intact(_promoted_row())
    assert reusable_question_from_item(_promoted_row(), requested_language="english") is not None


def test_in_place_mutation_of_identity_fields_is_rejected_from_reuse() -> None:
    for field, value in (
        ("question", "A completely different question stem entirely?"),
        ("correctAnswer", "2"),
        ("answers", json.dumps({"options": ["0", "1", "2", "9"]})),
        ("difficulty", "HARD"),
    ):
        row = _promoted_row()
        row[field] = value
        assert not question_bank_identity_is_intact(row), field
        assert reusable_question_from_item(row, requested_language="english") is None, field


def test_in_place_mutation_of_identity_metadata_is_rejected_from_reuse() -> None:
    for key, value in (("subject", "physics"), ("examIds", ["GMAT"]), ("language", "hindi")):
        row = _promoted_row()
        row["meta"] = {**row["meta"], key: value}
        assert not question_bank_identity_is_intact(row), key


def test_solution_only_correction_keeps_identity_valid() -> None:
    row = _promoted_row()
    row["explanation"] = "A corrected and much clearer explanation."

    assert question_bank_identity_is_intact(row)
    candidate = reusable_question_from_item(row, requested_language="english")
    assert candidate is not None
    assert candidate.solution == "A corrected and much clearer explanation."


def test_legacy_pattern_v1_rows_keep_existing_eligibility() -> None:
    row = _promoted_row()
    row["qbId"] = "pattern-v1-" + "a" * 32

    # Legacy ids predate this formula and must not be recomputed or rejected.
    assert question_bank_identity_is_intact(row)
    candidate = reusable_question_from_item(row, requested_language="english")
    assert candidate is not None
    assert candidate.question_id == row["qbId"]


def test_path_a_cannot_cross_classification_scopes() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    basic_slot = _promotion_slot()
    hard_slot = _promotion_slot().model_copy(
        update={"difficulty": Difficulty.ADVANCED, "exam_ids": ["GMAT"]}
    )
    basic_id = repository.persist_verified_question(
        test_id="test-1", question=_promotion_question(), slot=basic_slot, language="english"
    )
    hard_id = repository.persist_verified_question(
        test_id="test-1", question=_promotion_question(), slot=hard_slot, language="english"
    )
    basic_row = _plain(client.rows[basic_id])
    hard_row = _plain(client.rows[hard_id])

    basic_candidate = reusable_question_from_item(basic_row, requested_language="english")
    hard_candidate = reusable_question_from_item(hard_row, requested_language="english")
    assert basic_candidate is not None and hard_candidate is not None
    # Each row is only compatible with its own strict scope.
    assert _candidate_matches_slot(basic_candidate, basic_slot, requested_language="english")
    assert not _candidate_matches_slot(hard_candidate, basic_slot, requested_language="english")
    assert _candidate_matches_slot(hard_candidate, hard_slot, requested_language="english")
    assert not _candidate_matches_slot(basic_candidate, hard_slot, requested_language="english")


def test_round_trip_with_pattern_present_from_first_write() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
        pattern_id="P1",
        pattern_version_hash="V1",
    )
    row = _plain(client.rows[qb_id])

    assert row["patternId"] == "P1"
    assert row["source"] == "PATTERN_VERIFIED_PRACTICE"
    assert question_bank_identity_is_intact(row)
    candidate = reusable_question_from_item(row, requested_language="english")
    assert candidate is not None
    assert candidate.options == ("0", "1", "2", "3")


def test_round_trip_survives_later_pattern_enrichment() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    before = reusable_question_from_item(_plain(client.rows[qb_id]), requested_language="english")
    assert before is not None

    repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id=qb_id, pattern_id="P1", pattern_version_hash="V1"
    )
    # Simulate the persisted effect of the scoped update expression.
    row = _plain(client.rows[qb_id])
    row.update(
        {
            "patternId": "P1",
            "patternVersionHash": "V1",
            "patternLinkEvidence": "VERIFIED_GENERATION",
        }
    )

    after = reusable_question_from_item(row, requested_language="english")
    assert after is not None
    # Enrichment must not disturb identity or any playable field.
    assert question_bank_identity_is_intact(row)
    assert after.question_id == before.question_id
    assert after.question == before.question
    assert after.options == before.options
    assert after.correct_answer == before.correct_answer
    assert after.solution == before.solution


def _version(**overrides: Any) -> str | None:
    payload: dict[str, Any] = {
        "subject": "math",
        "difficulty": "basic",
        "exam_ids": ["CAT"],
        "language": "english",
        "question_type": "mcq",
        "topic": "geometry",
        "category": "geometry",
        "question": "Two angles of a triangle are 55 and 75 degrees. Find the third.",
        "options": ("40", "50", "60", "70"),
        "correct_answer": "50",
        "solution": "Interior angles sum to 180 degrees.",
    }
    payload.update(overrides)
    return build_question_bank_version_hash(**payload)


def test_identical_question_produces_one_version_hash() -> None:
    assert _version() is not None
    assert _version() == _version()


def test_authoritative_content_changes_change_the_version_hash() -> None:
    baseline = _version()

    assert _version(question="Two angles are 35 and 75 degrees. Find the third.") != baseline
    assert _version(options=("40", "50", "60", "99")) != baseline
    assert _version(options=("70", "60", "50", "40")) != baseline
    assert _version(correct_answer="60") != baseline
    assert _version(solution="A different explanation entirely.") != baseline


def test_strict_compatibility_changes_change_the_version_hash() -> None:
    baseline = _version()

    assert _version(subject="physics") != baseline
    assert _version(difficulty="advanced") != baseline
    assert _version(exam_ids=["GMAT"]) != baseline
    assert _version(language="hindi") != baseline
    assert _version(question_type="msq") != baseline


def test_semantic_representation_changes_change_the_version_hash() -> None:
    baseline = _version()

    # topic and category stay out of qbId but are embedded/strict, so a change
    # must invalidate an indexed vector.
    assert _version(topic="algebra") != baseline
    assert _version(category="algebra") != baseline


def test_exam_ordering_alone_keeps_the_version_hash_stable() -> None:
    assert _version(exam_ids=["CAT", "XAT"]) == _version(exam_ids=["XAT", "CAT"])
    assert _version(exam_ids=["CAT", "CAT"]) == _version(exam_ids=["CAT"])


def test_version_hash_fails_closed_on_uncanonicalizable_metadata() -> None:
    assert _version(subject="not a subject") is None
    assert _version(difficulty="sort of hard") is None
    assert _version(language="klingon") is None
    assert _version(exam_ids=["not an exam"]) is None


def test_promoted_row_stores_a_recomputable_version_hash() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])

    assert row["versionHash"]
    assert question_bank_version_hash_from_item(row) == row["versionHash"]


def test_transient_and_pattern_fields_never_move_the_version_hash() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])
    baseline = question_bank_version_hash_from_item(row)

    for mutation in (
        {"updatedAt": "2030-01-01T00:00:00Z"},
        {"createdAt": "2030-01-01T00:00:00Z"},
        {"patternId": "P1", "patternVersionHash": "V1"},
        {"patternVersionHash": "V2"},
        {"patternLinkEvidence": "VERIFIED_GENERATION"},
        {"qbId": "qb-v2-" + "b" * 32},
        {"source": "PATTERN_VERIFIED_PRACTICE"},
    ):
        assert question_bank_version_hash_from_item({**row, **mutation}) == baseline, mutation


def test_pattern_family_change_does_not_move_the_version_hash() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])
    enriched = {**row, "meta": {**row["meta"], "patternFamilyId": "some_family"}}

    # Not embedded, not vector metadata, not student-facing: re-checked live instead.
    assert question_bank_version_hash_from_item(enriched) == (
        question_bank_version_hash_from_item(row)
    )


def test_pattern_attachment_keeps_identity_and_version_hash_stable() -> None:
    client = _ConditionalDynamoClient()
    repository = _promotion_repository(client)
    qb_id = repository.persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])
    before = question_bank_version_hash_from_item(row)

    repository.attach_verified_pattern_link_if_absent(
        test_id="test-1", qb_id=qb_id, pattern_id="P1", pattern_version_hash="V1"
    )
    enriched = {
        **row,
        "patternId": "P1",
        "patternVersionHash": "V1",
        "patternLinkEvidence": "VERIFIED_GENERATION",
        "updatedAt": "2030-01-01T00:00:00Z",
    }

    assert question_bank_identity_is_intact(enriched)
    assert question_bank_version_hash_from_item(enriched) == before
    assert enriched["qbId"] == qb_id


def test_solution_correction_keeps_identity_but_moves_the_version_hash() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])
    corrected = {**row, "explanation": "A corrected and clearer explanation."}

    # Same durable Question, newer authoritative content: an old vector must
    # fail parity until it is reindexed.
    assert question_bank_identity_is_intact(corrected)
    assert corrected["qbId"] == qb_id
    assert question_bank_version_hash_from_item(corrected) != row["versionHash"]


def test_stored_version_hash_is_advisory_not_authoritative() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    row = _plain(client.rows[qb_id])
    # An administrative edit that forgets to refresh the stored hash.
    tampered = {**row, "explanation": "Silently rewritten explanation."}

    assert tampered["versionHash"] == row["versionHash"]
    assert question_bank_version_hash_from_item(tampered) != tampered["versionHash"]


def test_legacy_rows_have_no_version_hash_and_stay_reusable() -> None:
    client = _ConditionalDynamoClient()
    qb_id = _promotion_repository(client).persist_verified_question(
        test_id="test-1",
        question=_promotion_question(),
        slot=_promotion_slot(),
        language="english",
    )
    legacy = _plain(client.rows[qb_id])
    legacy["qbId"] = "pattern-v1-" + "a" * 32
    legacy.pop("versionHash")

    assert question_bank_identity_is_intact(legacy)
    assert reusable_question_from_item(legacy, requested_language="english") is not None
    # Recomputation still works, so Phase C/D can index legacy rows deliberately.
    assert question_bank_version_hash_from_item(legacy) is not None


def _bucket(topic: str, *, category: str | None = None) -> str | None:
    # Constructed, not model_copy: PlannerSlot's field validator canonicalizes the
    # taxonomy identifiers, and that is what the real planner path exercises.
    slot = PlannerSlot(
        slot_id="slot-001",
        subject_id="math",
        topic_id=topic,
        category_id=category or topic,
        difficulty="basic",
        complexity="low",
        exam_ids=[],
        question_type="mcq",
        target_skill="selling_price_from_discount",
        variation_hint="vary_values",
        generator_route_hint="math.generator.basic",
    )
    return build_slot_reuse_bucket_key(slot, language="english")


def test_reuse_bucket_partitions_on_the_planner_topic_label() -> None:
    """Characterises the incident: the same concept under two planner labels lands
    in two different GSI partitions, so the second request queries an empty bucket."""
    first_run = _bucket("profit_and_loss")
    second_run = _bucket("percentage_discount")

    assert first_run != second_run
    assert first_run == "v1#profit_and_loss#profit_and_loss#mcq#english"
    assert second_run == "v1#percentage_discount#percentage_discount#mcq#english"


def test_identical_topic_labels_share_one_reuse_bucket() -> None:
    # Reuse works whenever the planner reproduces the same label, independent of
    # the numbers in the question.
    assert _bucket("percentage_discount") == _bucket("percentage_discount")


def test_reuse_bucket_is_stable_across_label_casing_and_spacing() -> None:
    # The canonical normalizer already absorbs presentation differences; only a
    # genuinely different label produces a different bucket.
    assert _bucket("Percentage Discount") == _bucket("percentage_discount")
    assert _bucket("percentage-discount") == _bucket("percentage_discount")


def test_distinct_concepts_never_share_a_reuse_bucket() -> None:
    buckets = {
        _bucket(topic)
        for topic in (
            "linear_equations",
            "quadratic_equations",
            "triangle_angle_sum",
            "pythagorean_theorem",
            "simple_interest",
            "compound_interest",
            "probability",
            "geometry",
        )
    }

    # False-positive protection: distinct educational concepts stay separated.
    assert len(buckets) == 8
