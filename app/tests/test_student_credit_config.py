"""Startup validation and the plug-out guarantee for student credit config."""

from __future__ import annotations

from decimal import Decimal

import pytest

import config as config_module
from config import ConfigurationError, get_settings
from services.student_credits.bootstrap import build_student_credit_runtime


def _enable(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    monkeypatch.setenv("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "true")
    monkeypatch.setenv("DYNAMODB_USER_CREDITS_TABLE", "UserCredits-dev")
    monkeypatch.setenv("DYNAMODB_CREDIT_LEDGER_TABLE", "CreditLedger-dev")
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    config_module._settings = None


def test_defaults_disable_enforcement_and_build_no_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented default, independent of whatever the ambient env sets."""
    monkeypatch.delenv("STUDENT_CREDIT_ENFORCEMENT_ENABLED", raising=False)
    config_module._settings = None
    settings = get_settings()

    assert settings.student_credit_enforcement_enabled is False
    assert build_student_credit_runtime() is None


def test_disabled_enforcement_ignores_invalid_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured deployment must keep starting exactly as before."""
    monkeypatch.setenv("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "false")
    monkeypatch.setenv("CREDITS_PER_USD", "0")
    monkeypatch.setenv("TARGET_GROSS_MARGIN", "5")
    config_module._settings = None

    assert get_settings().student_credit_enforcement_enabled is False
    assert build_student_credit_runtime() is None


def test_enabled_enforcement_parses_the_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    settings = get_settings()

    assert settings.student_credit_credits_per_usd == Decimal("50")
    assert settings.student_credit_target_gross_margin == Decimal("0.40")
    assert settings.student_credit_rounding_mode == "CEIL"
    assert settings.student_credit_dry_run is True

    runtime = build_student_credit_runtime()
    assert runtime is not None
    assert runtime.policy.dry_run is True
    assert runtime.policy.margin_divisor == Decimal("0.60")


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("CREDITS_PER_USD", "0", "CREDITS_PER_USD must be greater than zero."),
        ("CREDITS_PER_USD", "-1", "CREDITS_PER_USD must be greater than zero."),
        ("CREDITS_PER_USD", "abc", "CREDITS_PER_USD must be a decimal number."),
        ("TARGET_GROSS_MARGIN", "1", "TARGET_GROSS_MARGIN must be at least 0"),
        ("TARGET_GROSS_MARGIN", "1.5", "TARGET_GROSS_MARGIN must be at least 0"),
        ("TARGET_GROSS_MARGIN", "-0.1", "TARGET_GROSS_MARGIN must be at least 0"),
        ("TARGET_GROSS_MARGIN", "x", "TARGET_GROSS_MARGIN must be a decimal number."),
        ("STUDENT_CREDIT_ROUNDING_MODE", "FLOOR", "must be 'CEIL'"),
    ],
)
def test_invalid_policy_fails_fast(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, message: str
) -> None:
    _enable(monkeypatch, **{variable: value})

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


@pytest.mark.parametrize(
    "variable",
    ["DYNAMODB_USER_CREDITS_TABLE", "DYNAMODB_CREDIT_LEDGER_TABLE"],
)
def test_missing_table_identity_fails_fast(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    """Unset env AND an unresolvable published contract still fails fast."""
    def _unavailable() -> tuple[str, str]:
        raise RuntimeError("ssm unavailable")

    monkeypatch.setattr(config_module, "_credit_tables_from_ssm", _unavailable)
    _enable(monkeypatch, **{variable: ""})

    with pytest.raises(ConfigurationError, match=f"{variable} is required"):
        get_settings()


def test_environment_table_names_win_and_skip_the_ssm_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects deployed behaviour: CDK-injected values never trigger SSM."""
    calls: list[str] = []

    def _should_not_run() -> tuple[str, str]:
        calls.append("ssm")
        return ("ssm-user", "ssm-ledger")

    monkeypatch.setattr(config_module, "_credit_tables_from_ssm", _should_not_run)
    _enable(
        monkeypatch,
        DYNAMODB_USER_CREDITS_TABLE="env-user",
        DYNAMODB_CREDIT_LEDGER_TABLE="env-ledger",
    )
    settings = get_settings()

    assert settings.dynamodb_user_credits_table == "env-user"
    assert settings.dynamodb_credit_ledger_table == "env-ledger"
    assert calls == []


def test_absent_environment_falls_back_to_the_published_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_credit_tables_from_ssm",
        lambda: ("ssm-user", "ssm-ledger"),
    )
    _enable(
        monkeypatch,
        DYNAMODB_USER_CREDITS_TABLE="",
        DYNAMODB_CREDIT_LEDGER_TABLE="",
    )
    settings = get_settings()

    assert settings.dynamodb_user_credits_table == "ssm-user"
    assert settings.dynamodb_credit_ledger_table == "ssm-ledger"


def test_enforcement_off_never_resolves_credit_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plug-out: no credit table identity is required or looked up."""
    calls: list[str] = []
    monkeypatch.setattr(
        config_module,
        "_credit_tables_from_ssm",
        lambda: calls.append("ssm") or ("u", "l"),
    )
    monkeypatch.setenv("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "false")
    monkeypatch.delenv("DYNAMODB_USER_CREDITS_TABLE", raising=False)
    monkeypatch.delenv("DYNAMODB_CREDIT_LEDGER_TABLE", raising=False)
    config_module._settings = None

    settings = get_settings()

    assert settings.student_credit_enforcement_enabled is False
    assert calls == []
    assert build_student_credit_runtime() is None


def test_dry_run_still_requires_resolvable_table_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration integrity is separate from runtime DynamoDB availability."""
    def _unavailable() -> tuple[str, str]:
        raise RuntimeError("ssm unavailable")

    monkeypatch.setattr(config_module, "_credit_tables_from_ssm", _unavailable)
    _enable(
        monkeypatch,
        STUDENT_CREDIT_DRY_RUN="true",
        DYNAMODB_USER_CREDITS_TABLE="",
        DYNAMODB_CREDIT_LEDGER_TABLE="",
    )

    with pytest.raises(ConfigurationError, match="DYNAMODB_USER_CREDITS_TABLE is required"):
        get_settings()


def test_margin_boundary_zero_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, TARGET_GROSS_MARGIN="0")

    assert get_settings().student_credit_target_gross_margin == Decimal("0")


def test_real_debit_requires_dry_run_to_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch, STUDENT_CREDIT_DRY_RUN="false")
    runtime = build_student_credit_runtime()

    assert runtime is not None
    assert runtime.policy.dry_run is False
