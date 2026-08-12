"""Focused startup validation for existing practice tables and indexes."""

from __future__ import annotations

from copy import deepcopy

import pytest

from features.practice_generation.config import (
    PracticeConfigurationError,
    PracticeGenerationConfig,
)
from features.practice_generation.resource_validation import (
    validate_practice_resources,
)


def _config(**updates) -> PracticeGenerationConfig:
    base = PracticeGenerationConfig(
        enabled=True,
        pattern_context_enabled=False,
        pattern_reuse_enabled=False,
        assessment_table="assessment",
        question_table="question",
        question_bank_table="bank",
        question_bank_category_index="getByCategory",
        question_test_index="getQuestionsByTestId",
        aws_region="ap-south-1",
        appsync_graphql_endpoint="https://example.appsync-api.ap-south-1.amazonaws.com/graphql",
        assessment_table_arn="arn:assessment",
        question_table_arn="arn:question",
        question_bank_table_arn="arn:bank",
        question_bank_reuse_index="getByReuseBucket",
        resource_schema_version="1",
        reuse_key_contract_version="1",
    )
    return PracticeGenerationConfig(**(base.__dict__ | updates))


class Dynamo:
    def __init__(self) -> None:
        self.calls = 0
        self.tables = {
            "assessment": {
                "TableStatus": "ACTIVE",
                "TableArn": "arn:assessment",
                "KeySchema": [{"AttributeName": "testId", "KeyType": "HASH"}],
            },
            "question": {
                "TableStatus": "ACTIVE",
                "TableArn": "arn:question",
                "KeySchema": [{"AttributeName": "questionId", "KeyType": "HASH"}],
                "GlobalSecondaryIndexes": [
                    {
                        "IndexName": "getQuestionsByTestId",
                        "IndexStatus": "ACTIVE",
                        "KeySchema": [
                            {"AttributeName": "testId", "KeyType": "HASH"},
                            {"AttributeName": "createdAt", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    }
                ],
            },
            "bank": {
                "TableStatus": "ACTIVE",
                "TableArn": "arn:bank",
                "KeySchema": [{"AttributeName": "qbId", "KeyType": "HASH"}],
                "GlobalSecondaryIndexes": [
                    {
                        "IndexName": "getByCategory",
                        "IndexStatus": "ACTIVE",
                        "KeySchema": [
                            {"AttributeName": "category", "KeyType": "HASH"},
                            {"AttributeName": "createdAt", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                    {
                        "IndexName": "getByReuseBucket",
                        "IndexStatus": "ACTIVE",
                        "KeySchema": [
                            {"AttributeName": "reuseBucketKey", "KeyType": "HASH"},
                            {"AttributeName": "reuseSortKey", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "KEYS_ONLY"},
                    },
                ],
            },
        }

    def describe_table(self, *, TableName):
        self.calls += 1
        return {"Table": deepcopy(self.tables[TableName])}


def test_active_required_tables_and_indexes_validate() -> None:
    capabilities = validate_practice_resources(
        _config(),
        dynamodb=Dynamo(),
    )
    assert capabilities.question_bank_reuse.query_mode == "IDS_AND_BATCH_GET"


def test_missing_gsi_blocks_feature_enablement() -> None:
    dynamo = Dynamo()
    dynamo.tables["bank"]["GlobalSecondaryIndexes"] = []
    with pytest.raises(PracticeConfigurationError, match="index is missing"):
        validate_practice_resources(_config(), dynamodb=dynamo)


def test_inactive_gsi_blocks_feature_enablement() -> None:
    dynamo = Dynamo()
    dynamo.tables["question"]["GlobalSecondaryIndexes"][0]["IndexStatus"] = "CREATING"
    with pytest.raises(PracticeConfigurationError, match="not ACTIVE"):
        validate_practice_resources(_config(), dynamodb=dynamo)


def test_feature_disabled_performs_no_resource_validation() -> None:
    dynamo = Dynamo()
    validate_practice_resources(
        _config(enabled=False),
        dynamodb=dynamo,
    )
    assert dynamo.calls == 0
