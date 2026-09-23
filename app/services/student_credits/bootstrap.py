"""Process-scoped construction for student credit enforcement."""

from __future__ import annotations

from config import get_settings
from services.aws_client_factory import get_dynamodb_client
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime


def build_student_credit_runtime() -> StudentCreditRuntime | None:
    """Return the credit runtime, or None when enforcement is disabled.

    A None runtime is the plug-out guarantee: no wallet read, no settlement, and
    no behaviour change for Doubt Solver.
    """
    settings = get_settings()
    if not settings.student_credit_enforcement_enabled:
        return None
    policy = settings.student_credit_policy
    if policy is None:
        raise RuntimeError("Student credit policy was not loaded while enforcement is enabled.")
    repository = StudentCreditRepository(
        user_credits_table=settings.dynamodb_user_credits_table,
        credit_ledger_table=settings.dynamodb_credit_ledger_table,
        client=get_dynamodb_client(settings.dynamodb_region or None),
    )
    return StudentCreditRuntime(policy=policy, repository=repository)
