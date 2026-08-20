"""Centralized environment configuration for practice generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from practice_limits import MAX_PRACTICE_QUESTIONS

if TYPE_CHECKING:
    from features.practice_generation.resource_contract import PracticeResourceContract


PRACTICE_RUNTIME_REVISION = "practice-v2-structured-output-capacity-20260811.6"


class PracticeConfigurationError(RuntimeError):
    """Raised when an enabled practice runtime lacks required configuration."""


@dataclass(frozen=True)
class PracticeGenerationConfig:
    enabled: bool
    pattern_context_enabled: bool
    pattern_reuse_enabled: bool
    assessment_table: str
    question_table: str
    question_bank_table: str
    question_bank_category_index: str
    question_test_index: str
    aws_region: str
    appsync_graphql_endpoint: str = ""
    assessment_table_arn: str = ""
    question_table_arn: str = ""
    question_bank_table_arn: str = ""
    question_bank_reuse_index: str = ""
    resource_schema_version: str = ""
    reuse_key_contract_version: str = ""
    # Promotion of verifier-approved system Questions into the reusable QuestionBank.
    # Independent of Pattern reuse by contract; it stays off until the runtime role
    # actually carries dynamodb:PutItem on the QuestionBank table.
    question_bank_promotion_enabled: bool = False
    # Question semantic reuse (Phase D). "off" keeps behaviour bit-for-bit unchanged,
    # "shadow" runs full retrieval + authoritative validation for calibration without
    # serving anything, "on" allows validated candidates to fill deficits.
    question_semantic_reuse_mode: str = "off"
    question_semantic_top_k: int = 12
    # UNCALIBRATED. Directional only, from a 19-vector Dev probe. Serving stays off
    # until a labelled calibration set sets this from evidence.
    question_semantic_threshold: float = 0.30
    max_questions: int = MAX_PRACTICE_QUESTIONS
    generation_group_size: int = 3
    generation_group_max: int = 5
    # Bounded fan-out for the existing per-question verifier calls within one
    # generation group. Transport concurrency only: the verifier route, model,
    # prompt, and one-question-per-call isolation are unchanged.
    verification_max_concurrency: int = 3
    item_retry_limit: int = 1
    planner_repair_limit: int = 1
    question_bank_query_limit: int = 50
    meta_safe_size_bytes: int = 280_000
    question_bank_query_page_size: int = 25
    question_bank_max_pages: int = 2
    question_bank_max_candidates_per_bucket: int = 75
    question_bank_max_total_candidates: int = 500
    query_latency_warning_ms: int = 1_000
    finalization_max_attempts: int = 3
    recovery_stale_seconds: int = 120
    max_wall_time_seconds: int = 900

    def validate_runtime(self) -> None:
        if not self.enabled:
            return
        missing = [
            name
            for name, value in {
                "practice/mock-test-quiz/table-name": self.assessment_table,
                "practice/mock-test-quiz/table-arn": self.assessment_table_arn,
                "practice/question/table-name": self.question_table,
                "practice/question/table-arn": self.question_table_arn,
                "practice/question-bank/table-name": self.question_bank_table,
                "practice/question-bank/table-arn": self.question_bank_table_arn,
                "practice/question-bank/category-index-name": (self.question_bank_category_index),
                "practice/question-bank/reuse-index-name": (self.question_bank_reuse_index),
                "practice/question/test-id-index-name": self.question_test_index,
                "practice/resource-contract-version": self.resource_schema_version,
                "practice/reuse-key-contract-version": (self.reuse_key_contract_version),
                "APPSYNC_GRAPHQL_ENDPOINT": self.appsync_graphql_endpoint,
            }.items()
            if not value
        ]
        if missing:
            raise PracticeConfigurationError(
                "Missing practice configuration: " + ", ".join(missing)
            )
        if self.resource_schema_version != "1":
            raise PracticeConfigurationError("Unsupported practice resource schema version.")
        if self.reuse_key_contract_version != "1":
            raise PracticeConfigurationError("Unsupported practice reuse-key contract version.")


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise PracticeConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _semantic_reuse_mode() -> str:
    mode = os.getenv("PRACTICE_QUESTION_SEMANTIC_REUSE_MODE", "off").strip().lower()
    if mode not in {"off", "shadow", "on"}:
        raise PracticeConfigurationError(
            "PRACTICE_QUESTION_SEMANTIC_REUSE_MODE must be off, shadow, or on."
        )
    return mode


def _semantic_threshold() -> float:
    value = float(os.getenv("PRACTICE_QUESTION_SEMANTIC_THRESHOLD", "0.30"))
    if not 0.0 < value <= 1.0:
        raise PracticeConfigurationError(
            "PRACTICE_QUESTION_SEMANTIC_THRESHOLD must be within (0.0, 1.0]."
        )
    return value


def _contract_or_none(
    enabled: bool,
    resource_contract: PracticeResourceContract | None,
) -> PracticeResourceContract | None:
    if not enabled:
        return None
    if resource_contract is not None:
        return resource_contract
    from features.practice_generation.resource_contract import (  # noqa: PLC0415
        load_practice_resource_contract,
    )

    return load_practice_resource_contract()


def _bounded_int_alias(
    name: str,
    legacy_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        raw = os.getenv(legacy_name, str(default))
    value = int(raw)
    if not minimum <= value <= maximum:
        raise PracticeConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def get_practice_config(
    *,
    resource_contract: PracticeResourceContract | None = None,
) -> PracticeGenerationConfig:
    enabled = os.getenv("PRACTICE_GENERATION_ENABLED", "false").lower() == "true"
    contract = _contract_or_none(enabled, resource_contract)
    config = PracticeGenerationConfig(
        enabled=enabled,
        pattern_context_enabled=(
            os.getenv("PRACTICE_PATTERN_CONTEXT_ENABLED", "false").lower() == "true"
        ),
        pattern_reuse_enabled=(
            os.getenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "false").lower() == "true"
        ),
        question_bank_promotion_enabled=(
            os.getenv("PRACTICE_QUESTION_BANK_PROMOTION_ENABLED", "false").lower() == "true"
        ),
        question_semantic_reuse_mode=_semantic_reuse_mode(),
        question_semantic_top_k=(
            _bounded_int("PRACTICE_QUESTION_SEMANTIC_TOP_K", 12, 1, 50) if enabled else 12
        ),
        question_semantic_threshold=_semantic_threshold(),
        assessment_table=contract.assessment_table if contract else "",
        question_table=contract.question_table if contract else "",
        question_bank_table=contract.question_bank_table if contract else "",
        question_bank_category_index=(contract.question_bank_category_index if contract else ""),
        question_test_index=contract.question_test_index if contract else "",
        aws_region=(
            contract.region_name
            if contract
            else os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "")).strip()
        ),
        appsync_graphql_endpoint=os.getenv("APPSYNC_GRAPHQL_ENDPOINT", "").strip(),
        assessment_table_arn=contract.assessment_table_arn if contract else "",
        question_table_arn=contract.question_table_arn if contract else "",
        question_bank_table_arn=(contract.question_bank_table_arn if contract else ""),
        question_bank_reuse_index=(contract.question_bank_reuse_index if contract else ""),
        resource_schema_version=contract.schema_version if contract else "",
        reuse_key_contract_version=(contract.reuse_key_contract_version if contract else ""),
        generation_group_size=(
            _bounded_int("PRACTICE_GENERATION_GROUP_SIZE", 3, 1, 5) if enabled else 3
        ),
        generation_group_max=(
            _bounded_int("PRACTICE_GENERATION_GROUP_MAX", 5, 1, 5) if enabled else 5
        ),
        verification_max_concurrency=(
            _bounded_int("PRACTICE_VERIFICATION_MAX_CONCURRENCY", 3, 1, 5) if enabled else 3
        ),
        item_retry_limit=(_bounded_int("PRACTICE_ITEM_RETRY_LIMIT", 1, 0, 1) if enabled else 1),
        planner_repair_limit=(
            _bounded_int("PRACTICE_PLANNER_REPAIR_LIMIT", 1, 0, 1) if enabled else 1
        ),
        question_bank_query_limit=(
            _bounded_int("PRACTICE_QUESTION_BANK_QUERY_LIMIT", 50, 1, 50) if enabled else 50
        ),
        meta_safe_size_bytes=(
            _bounded_int("PRACTICE_META_SAFE_SIZE_BYTES", 280_000, 64_000, 290_000)
            if enabled
            else 280_000
        ),
        question_bank_query_page_size=(
            _bounded_int_alias(
                "PRACTICE_QB_QUERY_PAGE_SIZE",
                "PRACTICE_QUESTION_BANK_QUERY_PAGE_SIZE",
                25,
                1,
                50,
            )
            if enabled
            else 25
        ),
        question_bank_max_pages=(
            _bounded_int_alias(
                "PRACTICE_QB_MAX_PAGES_PER_BUCKET",
                "PRACTICE_QUESTION_BANK_MAX_PAGES",
                2,
                1,
                5,
            )
            if enabled
            else 2
        ),
        question_bank_max_candidates_per_bucket=(
            _bounded_int(
                "PRACTICE_QB_MAX_CANDIDATES_PER_BUCKET",
                75,
                1,
                100,
            )
            if enabled
            else 75
        ),
        question_bank_max_total_candidates=(
            _bounded_int(
                "PRACTICE_QB_MAX_TOTAL_CANDIDATES_PER_ASSESSMENT",
                500,
                1,
                2_000,
            )
            if enabled
            else 500
        ),
        query_latency_warning_ms=(
            _bounded_int("PRACTICE_QUERY_LATENCY_WARNING_MS", 1_000, 100, 10_000)
            if enabled
            else 1_000
        ),
        finalization_max_attempts=(
            _bounded_int("PRACTICE_FINALIZATION_MAX_ATTEMPTS", 3, 1, 5) if enabled else 3
        ),
        recovery_stale_seconds=(
            _bounded_int("PRACTICE_RECOVERY_STALE_SECONDS", 120, 30, 3_600)
            if enabled
            else 120
        ),
        max_wall_time_seconds=(
            _bounded_int("PRACTICE_MAX_WALL_TIME_SECONDS", 900, 60, 3_600)
            if enabled
            else 900
        ),
    )
    if config.generation_group_size > config.generation_group_max:
        raise PracticeConfigurationError(
            "PRACTICE_GENERATION_GROUP_SIZE cannot exceed PRACTICE_GENERATION_GROUP_MAX."
        )
    return config
