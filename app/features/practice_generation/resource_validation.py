"""Fail-fast validation for configured practice DynamoDB tables and indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from features.practice_generation.config import (
    PracticeConfigurationError,
    PracticeGenerationConfig,
)
from features.practice_generation.events import emit_practice_event

_REUSE_RANKING_ATTRIBUTES = frozenset(
    {"qbId", "category", "difficulty", "tags", "meta", "updatedAt", "source"}
)


@dataclass(frozen=True)
class IndexProjection:
    projection_type: str
    projected_attributes: frozenset[str]
    query_mode: str


@dataclass(frozen=True)
class PracticeResourceCapabilities:
    question_bank_reuse: IndexProjection
    question_bank_category: IndexProjection
    question_test: IndexProjection


def _key_schema(value: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(entry.get("AttributeName") or ""), str(entry.get("KeyType") or "")) for entry in value
    )


def _describe_table(client: Any, table_name: str) -> dict[str, Any]:
    try:
        return dict(client.describe_table(TableName=table_name).get("Table") or {})
    except ClientError as exc:
        raise PracticeConfigurationError(
            "Configured practice DynamoDB resource could not be described."
        ) from exc


def _require_table_key(
    table: dict[str, Any],
    *,
    expected: tuple[tuple[str, str], ...],
    expected_arn: str,
) -> None:
    if str(table.get("TableStatus") or "") != "ACTIVE":
        raise PracticeConfigurationError("Configured practice DynamoDB table is not ACTIVE.")
    if _key_schema(list(table.get("KeySchema") or [])) != expected:
        raise PracticeConfigurationError(
            "Configured practice DynamoDB table key schema does not match."
        )
    if expected_arn and str(table.get("TableArn") or "") != expected_arn:
        raise PracticeConfigurationError("Configured practice DynamoDB table ARN does not match.")


def _require_index(
    table: dict[str, Any],
    *,
    index_name: str,
    expected: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    indexes = {
        str(index.get("IndexName") or ""): index
        for index in list(table.get("GlobalSecondaryIndexes") or [])
    }
    index = indexes.get(index_name)
    if index is None:
        raise PracticeConfigurationError("Configured practice DynamoDB index is missing.")
    if str(index.get("IndexStatus") or "") != "ACTIVE":
        raise PracticeConfigurationError("Configured practice DynamoDB index is not ACTIVE.")
    if _key_schema(list(index.get("KeySchema") or [])) != expected:
        raise PracticeConfigurationError(
            "Configured practice DynamoDB index key schema does not match."
        )
    return index


def _projection(
    table: dict[str, Any],
    index: dict[str, Any],
    *,
    ranking_required: bool,
) -> IndexProjection:
    projection = dict(index.get("Projection") or {})
    projection_type = str(projection.get("ProjectionType") or "").upper()
    if projection_type not in {"ALL", "INCLUDE", "KEYS_ONLY"}:
        raise PracticeConfigurationError(
            "Configured practice DynamoDB index projection is unknown."
        )
    projected = {
        name
        for name, _key_type in (
            _key_schema(list(table.get("KeySchema") or []))
            + _key_schema(list(index.get("KeySchema") or []))
        )
        if name
    }
    projected.update(
        str(name) for name in list(projection.get("NonKeyAttributes") or []) if str(name)
    )
    if projection_type == "ALL":
        projected.add("*")
    query_mode = (
        "PROJECTED_METADATA"
        if projection_type == "ALL" or (ranking_required and _REUSE_RANKING_ATTRIBUTES <= projected)
        else "IDS_AND_BATCH_GET"
    )
    return IndexProjection(
        projection_type=projection_type,
        projected_attributes=frozenset(projected),
        query_mode=query_mode,
    )


def validate_practice_resources(
    config: PracticeGenerationConfig,
    *,
    dynamodb: Any,
) -> PracticeResourceCapabilities | None:
    """Validate existing physical resources without exposing their names in logs."""
    config.validate_runtime()
    if not config.enabled:
        return None

    assessment = _describe_table(dynamodb, config.assessment_table)
    question = _describe_table(dynamodb, config.question_table)
    question_bank = _describe_table(dynamodb, config.question_bank_table)
    _require_table_key(
        assessment,
        expected=(("testId", "HASH"),),
        expected_arn=config.assessment_table_arn,
    )
    _require_table_key(
        question,
        expected=(("questionId", "HASH"),),
        expected_arn=config.question_table_arn,
    )
    _require_table_key(
        question_bank,
        expected=(("qbId", "HASH"),),
        expected_arn=config.question_bank_table_arn,
    )
    category_index = _require_index(
        question_bank,
        index_name=config.question_bank_category_index,
        expected=(("category", "HASH"), ("createdAt", "RANGE")),
    )
    reuse_index = _require_index(
        question_bank,
        index_name=config.question_bank_reuse_index,
        expected=(("reuseBucketKey", "HASH"), ("reuseSortKey", "RANGE")),
    )
    question_index = _require_index(
        question,
        index_name=config.question_test_index,
        expected=(("testId", "HASH"), ("createdAt", "RANGE")),
    )
    category_projection = _projection(
        question_bank,
        category_index,
        ranking_required=True,
    )
    reuse_projection = _projection(
        question_bank,
        reuse_index,
        ranking_required=True,
    )
    question_projection = _projection(
        question,
        question_index,
        ranking_required=False,
    )
    capabilities = PracticeResourceCapabilities(
        question_bank_reuse=reuse_projection,
        question_bank_category=category_projection,
        question_test=question_projection,
    )
    emit_practice_event(
        "PRACTICE_RUNTIME_VALIDATED",
        test_id="runtime",
        status="completed",
        details={
            "logicalTable": "MockTestQuiz/Question/QuestionBank",
            "logicalIndex": ("QuestionBank.reuse/QuestionBank.category/Question.testId"),
            "reuseProjectionType": reuse_projection.projection_type,
            "reuseProjectedAttributes": ",".join(sorted(reuse_projection.projected_attributes)),
            "reuseQueryMode": reuse_projection.query_mode,
            "categoryProjectionType": category_projection.projection_type,
            "categoryProjectedAttributes": ",".join(
                sorted(category_projection.projected_attributes)
            ),
            "categoryQueryMode": category_projection.query_mode,
            "questionProjectionType": question_projection.projection_type,
            "questionProjectedAttributes": ",".join(
                sorted(question_projection.projected_attributes)
            ),
            "questionQueryMode": question_projection.query_mode,
        },
    )
    return capabilities
