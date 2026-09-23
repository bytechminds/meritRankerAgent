"""Strict loader for the bundled, versioned student credit policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from schemas.student_credits import StudentCreditPolicy, StudentCreditPolicyDocument

DEFAULT_STUDENT_CREDIT_POLICY_PATH = (
    Path(__file__).resolve().parent / "config" / "student_credit_policy.yaml"
)


class StudentCreditPolicyConfigurationError(ValueError):
    """Raised when the bundled student business policy is unavailable or invalid."""


def load_student_credit_policy(
    *,
    enforcement_enabled: bool,
    dry_run: bool,
    path: Path | None = None,
) -> StudentCreditPolicy:
    """Load the sole student pricing/authorization source into the runtime policy."""
    policy_path = path or DEFAULT_STUDENT_CREDIT_POLICY_PATH
    try:
        raw: Any = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StudentCreditPolicyConfigurationError(
            "Unable to load student credit policy configuration."
        ) from exc
    if not isinstance(raw, dict):
        raise StudentCreditPolicyConfigurationError(
            "Student credit policy configuration must be a YAML mapping."
        )
    try:
        document = StudentCreditPolicyDocument.model_validate(raw)
    except ValidationError as exc:
        raise StudentCreditPolicyConfigurationError(
            "Invalid student credit policy configuration."
        ) from exc
    return document.build_runtime_policy(
        enforcement_enabled=enforcement_enabled,
        dry_run=dry_run,
    )
