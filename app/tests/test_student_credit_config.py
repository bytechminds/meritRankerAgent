"""Startup validation and the plug-out guarantee for student credit config."""

from __future__ import annotations

from decimal import Decimal

import pytest
import yaml

import config as config_module
import student_credit_policy as policy_module
from config import ConfigurationError, get_settings
from services.student_credits.bootstrap import build_student_credit_runtime
from student_credit_policy import (
    DEFAULT_STUDENT_CREDIT_POLICY_PATH,
    StudentCreditPolicyConfigurationError,
    load_student_credit_policy,
)


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


def test_disabled_enforcement_does_not_load_credit_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """An unconfigured deployment must keep starting exactly as before."""
    monkeypatch.setenv("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "false")
    monkeypatch.setattr(
        policy_module,
        "DEFAULT_STUDENT_CREDIT_POLICY_PATH",
        tmp_path / "missing-student-credit-policy.yaml",
    )
    config_module._settings = None

    settings = get_settings()
    assert settings.student_credit_enforcement_enabled is False
    assert settings.student_credit_policy is None
    assert build_student_credit_runtime() is None


def test_enabled_enforcement_loads_the_versioned_business_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asserts the documented dry-run default, so an ambient .env value must not leak in.
    monkeypatch.delenv("STUDENT_CREDIT_DRY_RUN", raising=False)
    _enable(monkeypatch)
    settings = get_settings()
    policy = settings.student_credit_policy

    assert policy is not None
    assert policy.credits_per_usd == Decimal("100")
    assert policy.target_gross_margin == Decimal("0.60")
    assert policy.rounding_mode == "CEIL"
    assert settings.student_credit_dry_run is True
    assert policy.doubt_authorization_credits == 5
    assert policy.practice_min_authorization_credits == 5
    assert policy.practice_authorization_credits_per_question == 5

    runtime = build_student_credit_runtime()
    assert runtime is not None
    assert runtime.policy == policy
    assert runtime.policy.dry_run is True
    assert runtime.policy.margin_divisor == Decimal("0.40")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("pricing", "target_gross_margin", "1"),
        ("currency", "usd_to_inr_business_rate", "0"),
        ("currency", "credit_value_inr", "0"),
        ("pricing", "rounding_mode", "FLOOR"),
        ("authorization", "doubt_min_credits", 0),
        ("authorization", "practice_min_credits", 0),
        ("authorization", "practice_credits_per_question", 0),
    ],
)
def test_invalid_versioned_policy_fails_strict_validation(
    tmp_path,
    section: str,
    key: str,
    value: str | int,
) -> None:
    raw = yaml.safe_load(DEFAULT_STUDENT_CREDIT_POLICY_PATH.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "student_credit_policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(StudentCreditPolicyConfigurationError, match="Invalid"):
        load_student_credit_policy(
            enforcement_enabled=True,
            dry_run=True,
            path=path,
        )


def test_versioned_policy_rejects_an_unknown_version_or_field(tmp_path) -> None:
    raw = yaml.safe_load(DEFAULT_STUDENT_CREDIT_POLICY_PATH.read_text(encoding="utf-8"))
    raw["policy_version"] = "student-credit-v2"
    raw["unexpected"] = True
    path = tmp_path / "student_credit_policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(StudentCreditPolicyConfigurationError, match="Invalid"):
        load_student_credit_policy(
            enforcement_enabled=True,
            dry_run=True,
            path=path,
        )


def test_enabled_enforcement_fails_safely_when_policy_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        policy_module,
        "DEFAULT_STUDENT_CREDIT_POLICY_PATH",
        tmp_path / "missing-student-credit-policy.yaml",
    )
    _enable(monkeypatch)

    with pytest.raises(ConfigurationError, match="Unable to load student credit policy"):
        get_settings()


def test_business_policy_environment_values_do_not_override_the_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(
        monkeypatch,
        CREDITS_PER_USD="1",
        TARGET_GROSS_MARGIN="0",
        STUDENT_CREDIT_ROUNDING_MODE="FLOOR",
        STUDENT_CREDIT_DOUBT_AUTHORIZATION_CREDITS="1",
        STUDENT_CREDIT_PRACTICE_MIN_AUTHORIZATION_CREDITS="1",
        STUDENT_CREDIT_PRACTICE_AUTHORIZATION_CREDITS_PER_QUESTION="99",
    )

    policy = get_settings().student_credit_policy

    assert policy is not None
    assert policy.credits_per_usd == Decimal("100")
    assert policy.target_gross_margin == Decimal("0.60")
    assert policy.rounding_mode == "CEIL"
    assert policy.doubt_authorization_credits == 5
    assert policy.practice_min_authorization_credits == 5
    assert policy.practice_authorization_credits_per_question == 5


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


def test_real_debit_requires_dry_run_to_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch, STUDENT_CREDIT_DRY_RUN="false")
    runtime = build_student_credit_runtime()

    assert runtime is not None
    assert runtime.policy.dry_run is False
