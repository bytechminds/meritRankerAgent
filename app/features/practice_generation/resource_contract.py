"""Versioned SSM contract for Amplify-owned practice resources."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from features.practice_generation.config import PracticeConfigurationError
from features.practice_generation.events import emit_practice_event
from services.aws_client_factory import get_ssm_client

PRACTICE_RESOURCE_PARAMETER_ROOT = "/meritranker/agent-runtime/v1/practice"
PRACTICE_RESOURCE_SCHEMA_VERSION = "1"
PRACTICE_REUSE_KEY_CONTRACT_VERSION = "1"

_PARAMETER_SUFFIXES = {
    "schema_version": "resource-contract-version",
    "reuse_key_contract_version": "reuse-key-contract-version",
    "assessment_table": "mock-test-quiz/table-name",
    "assessment_table_arn": "mock-test-quiz/table-arn",
    "question_table": "question/table-name",
    "question_table_arn": "question/table-arn",
    "question_bank_table": "question-bank/table-name",
    "question_bank_table_arn": "question-bank/table-arn",
    "question_bank_category_index": "question-bank/category-index-name",
    "question_bank_reuse_index": "question-bank/reuse-index-name",
    "question_test_index": "question/test-id-index-name",
}

_lock = threading.Lock()
_cached: PracticeResourceContract | None = None


@dataclass(frozen=True)
class PracticeResourceContract:
    schema_version: str
    reuse_key_contract_version: str
    assessment_table: str
    assessment_table_arn: str
    question_table: str
    question_table_arn: str
    question_bank_table: str
    question_bank_table_arn: str
    question_bank_category_index: str
    question_bank_reuse_index: str
    question_test_index: str
    region_name: str


def _parameter_paths(root: str) -> dict[str, str]:
    return {field: f"{root.rstrip('/')}/{suffix}" for field, suffix in _PARAMETER_SUFFIXES.items()}


def load_practice_resource_contract(
    *,
    ssm_client: Any | None = None,
    region_name: str | None = None,
    parameter_root: str | None = None,
) -> PracticeResourceContract:
    """Load the complete practice contract once; failed reads are never cached."""
    global _cached  # noqa: PLW0603
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        resolved_region = (
            region_name
            or os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
        )
        if not resolved_region:
            raise PracticeConfigurationError("Missing practice resource configuration: AWS_REGION")
        root = (
            parameter_root
            or os.getenv("PRACTICE_RESOURCE_PARAMETER_ROOT", "").strip()
            or PRACTICE_RESOURCE_PARAMETER_ROOT
        )
        paths = _parameter_paths(root)
        client = ssm_client or get_ssm_client(resolved_region)
        try:
            parameters: list[dict[str, Any]] = []
            parameter_names = list(paths.values())
            for offset in range(0, len(parameter_names), 10):
                response = client.get_parameters(
                    Names=parameter_names[offset : offset + 10],
                    WithDecryption=False,
                )
                parameters.extend(list(response.get("Parameters") or []))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "unknown")
            emit_practice_event(
                "PRACTICE_RESOURCE_VALIDATION_FAILED",
                test_id="runtime",
                status="failed",
                details={"reasonCode": "PRACTICE_RESOURCE_CONTRACT_READ_FAILED"},
            )
            raise PracticeConfigurationError(
                f"Practice resource configuration read failed: {code}"
            ) from exc
        values = {
            str(item.get("Name") or ""): str(item.get("Value") or "").strip() for item in parameters
        }
        missing = [path for path in paths.values() if not values.get(path)]
        if missing:
            emit_practice_event(
                "PRACTICE_RESOURCE_VALIDATION_FAILED",
                test_id="runtime",
                status="failed",
                details={"reasonCode": "PRACTICE_RESOURCE_CONTRACT_INCOMPLETE"},
            )
            raise PracticeConfigurationError(
                "Missing practice resource configuration: " + ", ".join(missing)
            )
        resolved = {field: values[path] for field, path in paths.items()}
        if resolved["schema_version"] != PRACTICE_RESOURCE_SCHEMA_VERSION:
            raise PracticeConfigurationError("Unsupported practice resource schema version.")
        if resolved["reuse_key_contract_version"] != PRACTICE_REUSE_KEY_CONTRACT_VERSION:
            raise PracticeConfigurationError("Unsupported practice reuse-key contract version.")
        _cached = PracticeResourceContract(
            **resolved,
            region_name=resolved_region,
        )
        emit_practice_event(
            "PRACTICE_RESOURCE_CONTRACT_LOADED",
            test_id="runtime",
            status="completed",
            details={
                "schemaVersion": _cached.schema_version,
                "reuseKeyContractVersion": _cached.reuse_key_contract_version,
            },
        )
        return _cached


def reset_practice_resource_contract_for_tests() -> None:
    global _cached  # noqa: PLW0603
    _cached = None
