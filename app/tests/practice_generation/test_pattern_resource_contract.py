"""Startup resolution of Pattern resource identity from env-then-SSM."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from features.practice_generation.config import PracticeConfigurationError
from features.practice_generation.pattern_resource_contract import (
    index_name_from_arn,
    load_pattern_resource_contract,
    reset_pattern_resource_contract_for_tests,
)

_ARN = "arn:aws:s3vectors:ap-south-1:111122223333:bucket/mr-vectors/index/patterns-v1"
_PATHS = {
    "/meritranker/agent-runtime/v1/pattern-intelligence/vector/index-arn": _ARN,
    "/meritranker/agent-runtime/v1/pattern-intelligence/pattern/table-name": "PatternGraph-dev",
    "/meritranker/agent-runtime/v1/practice/question-bank/table-name": "QuestionBank-dev",
    "/meritranker/agent-runtime/v1/practice/question-bank/pattern-index-name": "byPattern",
    "/meritranker/agent-runtime/v1/practice/attempt/table-name": "PracticeAttempt-dev",
    "/meritranker/agent-runtime/v1/practice/attempt/user-activity-index-name": "byUser",
}
_PATTERN_ENV = (
    "S3_VECTOR_PATTERN_INDEX_ARN",
    "DYNAMODB_PATTERN_TABLE",
    "DYNAMODB_QUESTION_BANK_TABLE",
    "DYNAMODB_QUESTION_BANK_PATTERN_INDEX",
    "DYNAMODB_PRACTICE_ATTEMPT_TABLE",
    "DYNAMODB_PRACTICE_ATTEMPT_USER_INDEX",
    "PATTERN_INTELLIGENCE_REUSE_ENABLED",
)


class _Ssm:
    def __init__(self, values: dict[str, str] | None = None, error: str | None = None) -> None:
        self._values = values if values is not None else dict(_PATHS)
        self._error = error
        self.calls: list[tuple[str, ...]] = []

    def get_parameters(self, *, Names: list[str], WithDecryption: bool) -> dict[str, Any]:  # noqa: N803
        del WithDecryption
        self.calls.append(tuple(Names))
        if self._error:
            raise ClientError(
                {"Error": {"Code": self._error, "Message": "denied"}}, "GetParameters"
            )
        return {
            "Parameters": [
                {"Name": name, "Value": self._values[name]}
                for name in Names
                if name in self._values
            ]
        }


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    reset_pattern_resource_contract_for_tests()
    for name in _PATTERN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    yield
    reset_pattern_resource_contract_for_tests()


def test_missing_env_resolves_every_identifier_from_one_batched_call() -> None:
    ssm = _Ssm()

    contract = load_pattern_resource_contract(ssm_client=ssm)

    assert len(ssm.calls) == 1
    assert set(ssm.calls[0]) == set(_PATHS)
    assert contract.pattern_table_name == "PatternGraph-dev"
    assert contract.question_bank_table_name == "QuestionBank-dev"
    assert contract.question_bank_pattern_index_name == "byPattern"
    assert contract.source == "SSM"


def test_explicit_env_values_win_and_skip_ssm_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S3_VECTOR_PATTERN_INDEX_ARN", _ARN)
    monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "env-pattern-table")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_TABLE", "env-qb-table")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_PATTERN_INDEX", "env-index")
    monkeypatch.setenv("DYNAMODB_PRACTICE_ATTEMPT_TABLE", "env-attempt")
    monkeypatch.setenv("DYNAMODB_PRACTICE_ATTEMPT_USER_INDEX", "env-attempt-index")
    ssm = _Ssm()

    contract = load_pattern_resource_contract(ssm_client=ssm)

    assert ssm.calls == []
    assert contract.pattern_table_name == "env-pattern-table"
    assert contract.source == "ENV"


def test_partial_env_reads_only_the_remaining_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "env-pattern-table")
    ssm = _Ssm()

    contract = load_pattern_resource_contract(ssm_client=ssm)

    assert len(ssm.calls) == 1
    assert (
        "/meritranker/agent-runtime/v1/pattern-intelligence/pattern/table-name" not in ssm.calls[0]
    )
    assert contract.pattern_table_name == "env-pattern-table"
    assert contract.pattern_vector_index_arn == _ARN
    assert contract.source == "ENV+SSM"


def test_contract_is_resolved_once_per_process() -> None:
    ssm = _Ssm()

    first = load_pattern_resource_contract(ssm_client=ssm)
    second = load_pattern_resource_contract(ssm_client=ssm)

    assert first is second
    assert len(ssm.calls) == 1


def test_missing_parameter_fails_with_an_explicit_configuration_error() -> None:
    partial = dict(_PATHS)
    del partial["/meritranker/agent-runtime/v1/practice/question-bank/pattern-index-name"]

    with pytest.raises(PracticeConfigurationError) as exc:
        load_pattern_resource_contract(ssm_client=_Ssm(partial))

    assert "pattern-index-name" in str(exc.value)


def test_access_denied_fails_clearly_without_leaking_detail() -> None:
    with pytest.raises(PracticeConfigurationError) as exc:
        load_pattern_resource_contract(ssm_client=_Ssm(error="AccessDeniedException"))

    message = str(exc.value)
    assert "AccessDeniedException" in message
    assert "denied" not in message


def test_failures_are_never_cached() -> None:
    with pytest.raises(PracticeConfigurationError):
        load_pattern_resource_contract(ssm_client=_Ssm(error="AccessDeniedException"))

    contract = load_pattern_resource_contract(ssm_client=_Ssm())
    assert contract.pattern_table_name == "PatternGraph-dev"


@pytest.mark.parametrize(
    ("arn", "expected"),
    [
        (_ARN, "patterns-v1"),
        (f"{_ARN}/", "patterns-v1"),
        ("arn:aws:s3vectors:r:1:bucket/b/index/other-index", "other-index"),
        ("not-an-arn", ""),
    ],
)
def test_index_name_is_derived_from_the_authoritative_arn(arn: str, expected: str) -> None:
    assert index_name_from_arn(arn) == expected


def test_unparseable_vector_arn_fails_instead_of_using_a_stale_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S3_VECTOR_PATTERN_INDEX_ARN", "arn:aws:s3vectors:r:1:bucket/b")
    monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "t")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_TABLE", "q")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_PATTERN_INDEX", "i")

    with pytest.raises(PracticeConfigurationError) as exc:
        load_pattern_resource_contract(ssm_client=_Ssm())

    assert "does not identify an index" in str(exc.value)


class _RecordingSsm(_Ssm):
    """Fails loudly if any SSM call happens when none is expected."""


def _build_provider(monkeypatch: pytest.MonkeyPatch, ssm: _Ssm):
    from features.practice_generation import pattern_resource_contract as prc
    from features.practice_generation.pattern_context import build_pattern_context_provider

    monkeypatch.setattr(prc, "get_ssm_client", lambda region: ssm)
    return build_pattern_context_provider(enabled=True)


def test_pattern_disabled_performs_zero_ssm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from features.practice_generation.pattern_context import NoOpPatternContextProvider

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "false")
    ssm = _Ssm()

    provider = _build_provider(monkeypatch, ssm)

    assert isinstance(provider, NoOpPatternContextProvider)
    assert ssm.calls == []


def test_pattern_enabled_builds_the_runtime_provider_from_ssm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from features.practice_generation.pattern_context import RuntimePatternContextProvider

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "false")
    ssm = _Ssm()

    provider = _build_provider(monkeypatch, ssm)

    assert isinstance(provider, RuntimePatternContextProvider)
    assert len(ssm.calls) == 1


def test_enabled_pattern_with_unresolvable_resources_fails_instead_of_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")

    with pytest.raises(PracticeConfigurationError):
        _build_provider(monkeypatch, _Ssm({}))


def test_enabled_pattern_surfaces_access_denied_instead_of_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")

    with pytest.raises(PracticeConfigurationError):
        _build_provider(monkeypatch, _Ssm(error="AccessDeniedException"))


def test_deployed_env_injection_still_builds_without_any_ssm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CDK-injected environment variables must keep working unchanged."""
    from features.practice_generation.pattern_context import RuntimePatternContextProvider

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("S3_VECTOR_PATTERN_INDEX_ARN", _ARN)
    monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "PatternGraph-prod")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_TABLE", "QuestionBank-prod")
    monkeypatch.setenv("DYNAMODB_QUESTION_BANK_PATTERN_INDEX", "byPattern")
    monkeypatch.setenv("DYNAMODB_PRACTICE_ATTEMPT_TABLE", "PracticeAttempt-prod")
    monkeypatch.setenv("DYNAMODB_PRACTICE_ATTEMPT_USER_INDEX", "byUser")
    ssm = _Ssm()

    provider = _build_provider(monkeypatch, ssm)

    assert isinstance(provider, RuntimePatternContextProvider)
    assert ssm.calls == []


def test_no_ssm_call_occurs_during_slot_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    ssm = _Ssm()
    provider = _build_provider(monkeypatch, ssm)
    calls_after_startup = len(ssm.calls)

    provider.resolve_slots(request=_practice_request(), slots=())

    assert len(ssm.calls) == calls_after_startup == 1


def _practice_request():
    from features.practice_generation.schemas import PracticeGenerationRequest

    return PracticeGenerationRequest(
        request_id="r",
        user_id="u",
        conversation_id="c",
        turn_id="t",
        original_query="Create two algebra questions",
        practice_type="QUIZ",
        requested_count=2,
        accepted_count=2,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra practice",
    )


def test_attempt_history_resources_are_required_only_when_reuse_is_enabled() -> None:
    """Reuse-off must not demand attempt-history identifiers."""
    without_attempt = {
        path: value for path, value in _PATHS.items() if "/practice/attempt/" not in path
    }

    contract = load_pattern_resource_contract(ssm_client=_Ssm(without_attempt), reuse_enabled=False)

    assert contract.pattern_table_name == "PatternGraph-dev"
    assert contract.practice_attempt_table_name == ""


def test_reuse_enabled_fails_fast_when_attempt_history_is_unresolvable() -> None:
    """Reuse-on with no attempt-history identifiers must not start silently."""
    without_attempt = {
        path: value for path, value in _PATHS.items() if "/practice/attempt/" not in path
    }

    with pytest.raises(PracticeConfigurationError) as exc:
        load_pattern_resource_contract(ssm_client=_Ssm(without_attempt), reuse_enabled=True)

    assert "/practice/attempt/" in str(exc.value)


def test_reuse_enabled_resolves_attempt_history_from_ssm() -> None:
    contract = load_pattern_resource_contract(ssm_client=_Ssm(), reuse_enabled=True)

    assert contract.practice_attempt_table_name == "PracticeAttempt-dev"
    assert contract.practice_attempt_user_index_name == "byUser"


def test_reuse_gate_passes_using_ssm_resolved_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the launcher gate read raw env only, so SSM-backed reuse
    identifiers were invisible and startup failed with
    'Pattern reuse requires QuestionBank Pattern and practice-history indexes.'
    """
    from features.practice_generation import agentcore_async
    from features.practice_generation import pattern_resource_contract as prc

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "true")
    monkeypatch.setattr(prc, "get_ssm_client", lambda region: _Ssm())

    import config as cfg_module

    cfg_module._settings = None
    overlaid = agentcore_async._with_pattern_resources(cfg_module.get_settings())

    assert overlaid.dynamodb_question_bank_pattern_index == "byPattern"
    assert overlaid.dynamodb_practice_attempt_table == "PracticeAttempt-dev"
    assert overlaid.dynamodb_practice_attempt_user_index == "byUser"
    # The exact condition the launcher gate evaluates.
    assert all(
        (
            overlaid.pattern_intelligence_enabled,
            overlaid.dynamodb_question_bank_pattern_index,
            overlaid.dynamodb_practice_attempt_table,
            overlaid.dynamodb_practice_attempt_user_index,
        )
    )
