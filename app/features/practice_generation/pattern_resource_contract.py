"""Conditionally-required SSM contract for Amplify-owned Pattern resources.

Feature behaviour stays in environment flags; Pattern infrastructure identity is
authoritative in SSM.  Deployed runtimes already receive these values as CDK-injected
environment variables, so an explicit environment value always wins and SSM is read
only for what is still missing.  Nothing here is resolved unless Pattern Intelligence
is explicitly enabled.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from features.practice_generation.config import PracticeConfigurationError
from features.practice_generation.events import emit_practice_event
from services.aws_client_factory import get_ssm_client

PATTERN_RESOURCE_PARAMETER_ROOT = "/meritranker/agent-runtime/v1/pattern-intelligence"
PRACTICE_QUESTION_BANK_PARAMETER_ROOT = "/meritranker/agent-runtime/v1/practice/question-bank"
PRACTICE_ATTEMPT_PARAMETER_ROOT = "/meritranker/agent-runtime/v1/practice/attempt"

# Field -> (parameter path, environment variable already honoured by config.Settings).
_PARAMETER_SOURCES = {
    "pattern_vector_index_arn": (
        f"{PATTERN_RESOURCE_PARAMETER_ROOT}/vector/index-arn",
        "S3_VECTOR_PATTERN_INDEX_ARN",
    ),
    "pattern_table_name": (
        f"{PATTERN_RESOURCE_PARAMETER_ROOT}/pattern/table-name",
        "DYNAMODB_PATTERN_TABLE",
    ),
    "question_bank_table_name": (
        f"{PRACTICE_QUESTION_BANK_PARAMETER_ROOT}/table-name",
        "DYNAMODB_QUESTION_BANK_TABLE",
    ),
    "question_bank_pattern_index_name": (
        f"{PRACTICE_QUESTION_BANK_PARAMETER_ROOT}/pattern-index-name",
        "DYNAMODB_QUESTION_BANK_PATTERN_INDEX",
    ),
    "practice_attempt_table_name": (
        f"{PRACTICE_ATTEMPT_PARAMETER_ROOT}/table-name",
        "DYNAMODB_PRACTICE_ATTEMPT_TABLE",
    ),
    "practice_attempt_user_index_name": (
        f"{PRACTICE_ATTEMPT_PARAMETER_ROOT}/user-activity-index-name",
        "DYNAMODB_PRACTICE_ATTEMPT_USER_INDEX",
    ),
}
# Student attempt history is only consulted when direct reuse is enabled, so these
# identifiers are required exactly when PATTERN_INTELLIGENCE_REUSE_ENABLED is true.
_REUSE_ONLY_FIELDS = frozenset(
    {"practice_attempt_table_name", "practice_attempt_user_index_name"}
)
# S3 Vectors index ARNs end with `.../index/<index-name>`.
_INDEX_ARN_SUFFIX = re.compile(r"/index/(?P<index_name>[^/]+)/?$")

_lock = threading.Lock()
_cached: PatternResourceContract | None = None


@dataclass(frozen=True)
class PatternResourceContract:
    """Immutable process-memory Pattern resource identity."""

    pattern_vector_index_arn: str
    pattern_table_name: str
    question_bank_table_name: str
    question_bank_pattern_index_name: str
    practice_attempt_table_name: str
    practice_attempt_user_index_name: str
    pattern_vector_index_name: str
    source: str


def index_name_from_arn(index_arn: str) -> str:
    """Derive the index name the vector client selects on from its authoritative ARN."""
    match = _INDEX_ARN_SUFFIX.search(index_arn.strip())
    return match.group("index_name") if match else ""


def load_pattern_resource_contract(
    *,
    ssm_client: Any | None = None,
    region_name: str | None = None,
    reuse_enabled: bool | None = None,
) -> PatternResourceContract:
    """Resolve Pattern resource identity once per process; failures are never cached."""
    global _cached  # noqa: PLW0603
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        require_reuse_resources = (
            os.getenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "false").strip().lower() == "true"
            if reuse_enabled is None
            else reuse_enabled
        )
        resolved = {
            field: os.getenv(variable, "").strip()
            for field, (_, variable) in _PARAMETER_SOURCES.items()
        }
        missing_fields = [field for field, value in resolved.items() if not value]
        source = "ENV"
        if missing_fields:
            source = "ENV+SSM" if len(missing_fields) < len(resolved) else "SSM"
            resolved.update(
                _read_parameters(
                    fields=missing_fields,
                    ssm_client=ssm_client,
                    region_name=region_name,
                )
            )
        required_fields = set(_PARAMETER_SOURCES) - (
            set() if require_reuse_resources else _REUSE_ONLY_FIELDS
        )
        still_missing = sorted(
            _PARAMETER_SOURCES[field][0]
            for field, value in resolved.items()
            if not value and field in required_fields
        )
        if still_missing:
            _fail(
                "PATTERN_RESOURCE_CONTRACT_INCOMPLETE",
                "Missing Pattern resource configuration: " + ", ".join(still_missing),
            )
        index_name = index_name_from_arn(resolved["pattern_vector_index_arn"])
        if not index_name:
            _fail(
                "PATTERN_RESOURCE_VECTOR_ARN_INVALID",
                "Pattern vector index ARN does not identify an index.",
            )
        _cached = PatternResourceContract(
            **resolved,
            pattern_vector_index_name=index_name,
            source=source,
        )
        emit_practice_event(
            "PATTERN_RESOURCE_CONTRACT_LOADED",
            test_id="runtime",
            status="completed",
            details={"source": source},
        )
        return _cached


def _read_parameters(
    *,
    fields: list[str],
    ssm_client: Any | None,
    region_name: str | None,
) -> dict[str, str]:
    """Read every still-missing identifier in one bounded GetParameters call."""
    resolved_region = (
        region_name
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )
    if not resolved_region:
        _fail(
            "PATTERN_RESOURCE_REGION_MISSING",
            "Missing Pattern resource configuration: AWS_REGION",
        )
    paths = {_PARAMETER_SOURCES[field][0]: field for field in fields}
    client = ssm_client or get_ssm_client(resolved_region)
    try:
        response = client.get_parameters(Names=list(paths), WithDecryption=False)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "unknown")
        _fail(
            "PATTERN_RESOURCE_CONTRACT_READ_FAILED",
            f"Pattern resource configuration read failed: {code}",
        )
    values: dict[str, str] = {}
    for item in list(response.get("Parameters") or []):
        field = paths.get(str(item.get("Name") or ""))
        if field:
            values[field] = str(item.get("Value") or "").strip()
    return values


def _fail(reason_code: str, message: str) -> None:
    emit_practice_event(
        "PRACTICE_RESOURCE_VALIDATION_FAILED",
        test_id="runtime",
        status="failed",
        details={"reasonCode": reason_code},
    )
    raise PracticeConfigurationError(message)


def reset_pattern_resource_contract_for_tests() -> None:
    global _cached  # noqa: PLW0603
    _cached = None
