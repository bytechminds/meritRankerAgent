"""Focused DynamoDB mapping and conditional-write tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from features.practice_generation.matching import ReusableQuestion
from features.practice_generation.planning import resolve_practice_request
from features.practice_generation.repositories import (
    AssessmentRepository,
    PracticeRepositoryError,
    QuestionRepository,
    estimate_dynamodb_item_size,
)
from features.practice_generation.resource_validation import IndexProjection
from features.practice_generation.schemas import GeneratedQuestion

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
    item, duplicate = repository.create_or_get("test-1", _request())

    assert duplicate is False
    assert (
        item["testId"],
        item["status"],
        item["visibility"],
        item["origin"],
    ) == ("test-1", "GENERATING", "PRIVATE", "AI_CUSTOM")
    assert client.put_calls[0]["ConditionExpression"] == ("attribute_not_exists(testId)")
    assert "querySummary" not in item["meta"]["practiceRequest"]


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
    assert client.transact_calls == []
    assert client.update_calls == []


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


def test_reused_question_retains_question_bank_provenance() -> None:
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
    )

    stored = _plain(client.put_calls[0]["Item"])
    assert stored["meta"]["sourceType"] == "QUESTION_BANK"
    assert stored["meta"]["sourceQuestionBankId"] == "bank-1"
    assert stored["meta"]["sourceVersion"] == "2026-07-01T00:00:00Z"
    assert stored["meta"]["questionType"] == "mcq"
    assert stored["meta"]["language"] == "english"


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
