"""Wallet and ledger table identity from the published agent-runtime contract.

The deployed runtime receives these table names as environment variables the CDK
injects from SSM, and that path is unchanged. This resolver is the fallback used
only when those variables are absent — chiefly local development — so credits
resolve their resource identity the same way Practice and conversation
persistence already do, from one published source of truth.

Mirrors ``features/practice_generation/resource_contract.py``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from services.aws_client_factory import get_ssm_client

STUDENT_CREDIT_PARAMETER_ROOT = "/meritranker/agent-runtime/v1/credits"

# Only the table names are needed to construct the repository. The published
# ARNs are consumed by CDK for IAM and are deliberately not read here.
_PARAMETER_SUFFIXES = {
    "user_credits_table": "user-credits/table-name",
    "credit_ledger_table": "credit-ledger/table-name",
}

_lock = threading.Lock()
_cached: StudentCreditResourceContract | None = None


class StudentCreditResourceError(RuntimeError):
    """Raised when the published credit table identity cannot be resolved."""


@dataclass(frozen=True)
class StudentCreditResourceContract:
    user_credits_table: str
    credit_ledger_table: str


def _parameter_paths(root: str) -> dict[str, str]:
    return {field: f"{root}/{suffix}" for field, suffix in _PARAMETER_SUFFIXES.items()}


def load_student_credit_resource_contract(
    *,
    ssm_client: Any | None = None,
    region_name: str | None = None,
    parameter_root: str | None = None,
) -> StudentCreditResourceContract:
    """Resolve both table names once. A failed read is never cached."""
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
            raise StudentCreditResourceError(
                "Missing student credit resource configuration: AWS_REGION"
            )
        root = (
            parameter_root
            or os.getenv("STUDENT_CREDIT_PARAMETER_ROOT", "").strip()
            or STUDENT_CREDIT_PARAMETER_ROOT
        )
        paths = _parameter_paths(root)
        client = ssm_client or get_ssm_client(resolved_region)
        try:
            response = client.get_parameters(
                Names=list(paths.values()), WithDecryption=False
            )
        except (BotoCoreError, ClientError) as exc:
            raise StudentCreditResourceError(
                "Student credit resource configuration read failed."
            ) from exc
        values = {
            str(item.get("Name") or ""): str(item.get("Value") or "").strip()
            for item in (response.get("Parameters") or [])
        }
        missing = [path for path in paths.values() if not values.get(path)]
        if missing:
            raise StudentCreditResourceError(
                "Missing student credit resource configuration: " + ", ".join(missing)
            )
        _cached = StudentCreditResourceContract(
            **{field: values[path] for field, path in paths.items()}
        )
        return _cached


def reset_student_credit_resource_contract_for_tests() -> None:
    global _cached  # noqa: PLW0603
    with _lock:
        _cached = None
