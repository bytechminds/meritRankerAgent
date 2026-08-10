"""Versioned SSM practice resource contract tests."""

from __future__ import annotations

import pytest

from features.practice_generation.config import (
    PracticeConfigurationError,
    get_practice_config,
)
from features.practice_generation.resource_contract import (
    PRACTICE_RESOURCE_PARAMETER_ROOT,
    load_practice_resource_contract,
    reset_practice_resource_contract_for_tests,
)

_VALUES = {
    "resource-contract-version": "1",
    "reuse-key-contract-version": "1",
    "mock-test-quiz/table-name": "assessment",
    "mock-test-quiz/table-arn": "arn:assessment",
    "question/table-name": "question",
    "question/table-arn": "arn:question",
    "question-bank/table-name": "bank",
    "question-bank/table-arn": "arn:bank",
    "question-bank/category-index-name": "category-index",
    "question-bank/reuse-index-name": "reuse-index",
    "question/test-id-index-name": "question-index",
}


class Ssm:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(_VALUES if values is None else values)
        self.calls = 0

    def get_parameters(self, *, Names, WithDecryption):
        self.calls += 1
        assert WithDecryption is False
        return {
            "Parameters": [
                {
                    "Name": name,
                    "Value": self.values[name.removeprefix(f"{PRACTICE_RESOURCE_PARAMETER_ROOT}/")],
                }
                for name in Names
                if name.removeprefix(f"{PRACTICE_RESOURCE_PARAMETER_ROOT}/") in self.values
            ]
        }


@pytest.fixture(autouse=True)
def _reset_contract() -> None:
    reset_practice_resource_contract_for_tests()


def test_complete_versioned_contract_resolves_and_is_cached() -> None:
    ssm = Ssm()
    first = load_practice_resource_contract(
        ssm_client=ssm,
        region_name="ap-south-1",
    )
    second = load_practice_resource_contract(
        ssm_client=ssm,
        region_name="ap-south-1",
    )

    assert first is second
    assert first.question_bank_reuse_index == "reuse-index"
    assert first.assessment_table_arn == "arn:assessment"
    assert ssm.calls == 2


def test_missing_parameter_blocks_enablement() -> None:
    values = dict(_VALUES)
    del values["question/table-arn"]
    with pytest.raises(
        PracticeConfigurationError,
        match="Missing practice resource configuration",
    ):
        load_practice_resource_contract(
            ssm_client=Ssm(values),
            region_name="ap-south-1",
        )


def test_schema_version_fails_closed() -> None:
    values = dict(_VALUES)
    values["resource-contract-version"] = "unsupported"
    with pytest.raises(PracticeConfigurationError, match="Unsupported practice"):
        load_practice_resource_contract(
            ssm_client=Ssm(values),
            region_name="ap-south-1",
        )


def test_reuse_key_contract_version_fails_closed() -> None:
    values = dict(_VALUES)
    values["reuse-key-contract-version"] = "unsupported"
    with pytest.raises(PracticeConfigurationError, match="reuse-key contract"):
        load_practice_resource_contract(
            ssm_client=Ssm(values),
            region_name="ap-south-1",
        )


def test_disabled_feature_does_not_load_ssm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "false")
    config = get_practice_config()
    assert config.enabled is False
    assert config.assessment_table == ""


def test_enabled_config_uses_contract_without_physical_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    contract = load_practice_resource_contract(
        ssm_client=Ssm(),
        region_name="ap-south-1",
    )
    config = get_practice_config(resource_contract=contract)

    assert config.assessment_table == "assessment"
    assert config.question_bank_reuse_index == "reuse-index"
    assert config.question_test_index == "question-index"
