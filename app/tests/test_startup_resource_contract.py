"""Regression guard for the startup resource-resolution invariant.

Invariant under test: stable SSM-backed resource metadata is resolved during
application startup, and request paths consume the cached values without
performing any SSM resource discovery.

These tests deliberately assert a BOUND on startup activity plus a ZERO
post-startup delta, rather than pinning an exact startup call count, so adding
or batching a legitimate startup parameter does not produce a false failure.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from botocore.exceptions import ClientError

from features.practice_generation.config import PracticeConfigurationError
from features.practice_generation.pattern_resource_contract import (
    load_pattern_resource_contract,
    reset_pattern_resource_contract_for_tests,
)
from features.practice_generation.resource_contract import (
    load_practice_resource_contract,
    reset_practice_resource_contract_for_tests,
)
from services.conversation.runtime_config import (
    load_conversation_runtime_config,
    reset_conversation_runtime_config_for_tests,
)

# Upper bound on distinct startup GetParameters calls. One batched call per
# resource contract, plus headroom for the practice contract's 10-name chunking.
_MAX_STARTUP_SSM_CALLS = 8
# Every startup call must actually batch; a regression to one-call-per-parameter
# would show up as many single-name calls.
_MIN_BATCHED_NAMES = 2

_PRACTICE_ROOT = "/meritranker/agent-runtime/v1/practice"
_PATTERN_ROOT = "/meritranker/agent-runtime/v1/pattern-intelligence"
_HISTORY_ROOT = "/meritranker/agent-runtime/v1/conversation-history"
_SESSION_ROOT = "/meritranker/agent-runtime/v1/conversation-session"

_DDB_ARN = "arn:aws:dynamodb:ap-south-1:1:table"

_PARAMETERS: dict[str, str] = {
    f"{_PRACTICE_ROOT}/resource-contract-version": "1",
    f"{_PRACTICE_ROOT}/reuse-key-contract-version": "1",
    f"{_PRACTICE_ROOT}/mock-test-quiz/table-name": "MockTestQuiz-test",
    f"{_PRACTICE_ROOT}/mock-test-quiz/table-arn": f"{_DDB_ARN}/MockTestQuiz-test",
    f"{_PRACTICE_ROOT}/question/table-name": "Question-test",
    f"{_PRACTICE_ROOT}/question/table-arn": f"{_DDB_ARN}/Question-test",
    f"{_PRACTICE_ROOT}/question/test-id-index-name": "byTestId",
    f"{_PRACTICE_ROOT}/question-bank/table-name": "QuestionBank-test",
    f"{_PRACTICE_ROOT}/question-bank/table-arn": f"{_DDB_ARN}/QuestionBank-test",
    f"{_PRACTICE_ROOT}/question-bank/category-index-name": "byCategory",
    f"{_PRACTICE_ROOT}/question-bank/reuse-index-name": "byReuse",
    f"{_PRACTICE_ROOT}/question-bank/pattern-index-name": "byPattern",
    f"{_PRACTICE_ROOT}/attempt/table-name": "PracticeAttempt-test",
    f"{_PRACTICE_ROOT}/attempt/user-activity-index-name": "byUserActivity",
    f"{_PATTERN_ROOT}/vector/index-arn": (
        "arn:aws:s3vectors:ap-south-1:1:bucket/b-test/index/patterns-v1"
    ),
    f"{_PATTERN_ROOT}/pattern/table-name": "Pattern-test",
    f"{_HISTORY_ROOT}/table-name": "ConversationHistory-test",
    f"{_HISTORY_ROOT}/table-arn": f"{_DDB_ARN}/ConversationHistory-test",
    f"{_SESSION_ROOT}/table-name": "ConversationSession-test",
    f"{_SESSION_ROOT}/table-arn": f"{_DDB_ARN}/ConversationSession-test",
}


class _CountingSsm:
    """Records every GetParameters call so phase deltas can be asserted."""

    def __init__(self, *, values: dict[str, str] | None = None, error: str | None = None) -> None:
        self._values = _PARAMETERS if values is None else values
        self._error = error
        self.calls: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    def get_parameters(self, *, Names: list[str], WithDecryption: bool = False) -> dict[str, Any]:  # noqa: N803
        del WithDecryption
        with self._lock:
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

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)


@pytest.fixture
def ssm(monkeypatch: pytest.MonkeyPatch) -> _CountingSsm:
    """Route every resource contract's SSM client to one counting fake."""
    client = _CountingSsm()
    _install(monkeypatch, lambda region=None: client)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    _reset_all_contracts()
    yield client
    _reset_all_contracts()


def _install(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    from features.practice_generation import pattern_resource_contract, resource_contract
    from services.conversation import runtime_config

    for module in (resource_contract, pattern_resource_contract, runtime_config):
        monkeypatch.setattr(module, "get_ssm_client", factory)


def _reset_all_contracts() -> None:
    reset_practice_resource_contract_for_tests()
    reset_pattern_resource_contract_for_tests()
    reset_conversation_runtime_config_for_tests()


def _startup_bootstrap() -> None:
    """Resolve exactly what main.py resolves at import time, in the same order."""
    load_conversation_runtime_config()
    load_practice_resource_contract()
    load_pattern_resource_contract()


def _request_path_operations() -> None:
    """Every resource-consuming entry point a request or async operation reaches."""
    load_conversation_runtime_config()
    load_practice_resource_contract()
    load_pattern_resource_contract()


def test_startup_resolves_all_resource_metadata_in_bounded_batched_calls(
    ssm: _CountingSsm,
) -> None:
    _startup_bootstrap()

    assert ssm.call_count > 0, "startup must actually resolve resource metadata"
    assert ssm.call_count <= _MAX_STARTUP_SSM_CALLS
    multi_name_calls = [names for names in ssm.calls if len(names) >= _MIN_BATCHED_NAMES]
    assert multi_name_calls, "startup reads must be batched, not one call per parameter"


def test_first_request_performs_zero_ssm_resource_discovery(ssm: _CountingSsm) -> None:
    _startup_bootstrap()
    after_startup = ssm.call_count

    _request_path_operations()

    assert ssm.call_count - after_startup == 0


def test_second_request_performs_zero_ssm_resource_discovery(ssm: _CountingSsm) -> None:
    _startup_bootstrap()
    after_startup = ssm.call_count

    _request_path_operations()
    delta_first = ssm.call_count - after_startup
    _request_path_operations()
    delta_second = ssm.call_count - after_startup

    assert (delta_first, delta_second) == (0, 0)


def test_practice_configuration_after_startup_performs_zero_ssm(
    ssm: _CountingSsm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Practice enabled, so get_practice_config actually consumes the contract."""
    from features.practice_generation.config import get_practice_config

    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    _startup_bootstrap()
    after_startup = ssm.call_count

    first = get_practice_config()
    second = get_practice_config()

    assert ssm.call_count - after_startup == 0
    assert first.assessment_table == second.assessment_table == "MockTestQuiz-test"
    assert first.question_bank_table == "QuestionBank-test"


def test_practice_disabled_never_consults_the_resource_contract(
    ssm: _CountingSsm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled optional feature must not require its resources."""
    from features.practice_generation.config import get_practice_config

    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "false")
    _reset_all_contracts()

    config = get_practice_config()

    assert config.enabled is False
    assert config.assessment_table == ""
    assert ssm.call_count == 0


def test_pattern_retrieval_provider_construction_performs_zero_ssm(
    ssm: _CountingSsm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from features.practice_generation.pattern_context import (
        RuntimePatternContextProvider,
        build_pattern_context_provider,
    )

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "false")
    _startup_bootstrap()
    after_startup = ssm.call_count

    provider = build_pattern_context_provider(enabled=True)

    assert isinstance(provider, RuntimePatternContextProvider)
    assert ssm.call_count - after_startup == 0


def test_doubt_solver_exam_profile_resolution_performs_zero_ssm(
    ssm: _CountingSsm,
) -> None:
    """Doubt Solver's only SSM-backed dependency is the exam-profile snapshot."""
    from services.doubt_solver.exam_profile_cache import ExamProfileRuntime

    class _Source:
        def __init__(self) -> None:
            self.loads = 0

        def load_active_profiles(self):
            self.loads += 1
            return ()

    source = _Source()
    runtime = ExamProfileRuntime(source=source)
    runtime.load()
    after_startup = ssm.call_count

    for _ in range(3):
        runtime.resolve(exam_profile_id=None, exam_id="CAT", exam_stage=None)

    assert source.loads == 1
    assert ssm.call_count - after_startup == 0


def test_conversation_session_and_memory_resources_resolve_once(ssm: _CountingSsm) -> None:
    first = load_conversation_runtime_config()
    after_startup = ssm.call_count

    second = load_conversation_runtime_config()

    assert first is second
    assert first.table_name == "ConversationHistory-test"
    assert first.conversation_session_table_name == "ConversationSession-test"
    assert ssm.call_count - after_startup == 0


def test_concurrent_requests_cannot_trigger_additional_resolution(
    ssm: _CountingSsm,
) -> None:
    _startup_bootstrap()
    after_startup = ssm.call_count
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            _request_path_operations()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert ssm.call_count - after_startup == 0


def test_concurrent_cold_bootstrap_resolves_each_contract_exactly_once(
    ssm: _CountingSsm,
) -> None:
    """A cold start with concurrent arrivals must not duplicate resolution."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            _startup_bootstrap()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert ssm.call_count <= _MAX_STARTUP_SSM_CALLS, (
        f"cold bootstrap under concurrency duplicated resolution: {ssm.calls}"
    )
    # All eight threads must observe the identical immutable snapshots.
    assert load_practice_resource_contract() is load_practice_resource_contract()
    assert load_pattern_resource_contract() is load_pattern_resource_contract()


def test_startup_resource_resolution_duration_is_measured(
    ssm: _CountingSsm,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Record aggregate startup resource-resolution cost without pinning a threshold."""
    with caplog.at_level(logging.INFO, logger=__name__):
        started = time.perf_counter()
        _startup_bootstrap()
        duration_ms = (time.perf_counter() - started) * 1000
        parameter_count = sum(len(names) for names in ssm.calls)
        logging.getLogger(__name__).info(
            "startup_resource_resolution ssm_call_count=%d ssm_parameter_count=%d duration_ms=%.2f",
            ssm.call_count,
            parameter_count,
            duration_ms,
        )

    assert "startup_resource_resolution" in caplog.text
    assert parameter_count > 0
    assert duration_ms >= 0


def test_missing_required_parameter_blocks_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    incomplete = {k: v for k, v in _PARAMETERS.items() if "mock-test-quiz" not in k}
    client = _CountingSsm(values=incomplete)
    _install(monkeypatch, lambda region=None: client)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    _reset_all_contracts()
    try:
        with pytest.raises(PracticeConfigurationError):
            load_practice_resource_contract()
    finally:
        _reset_all_contracts()


def test_access_denied_blocks_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _CountingSsm(error="AccessDeniedException")
    _install(monkeypatch, lambda region=None: client)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    _reset_all_contracts()
    try:
        with pytest.raises(PracticeConfigurationError):
            load_practice_resource_contract()
    finally:
        _reset_all_contracts()


# --------------------------------------------------------------------------------
# Readiness semantics: no event may claim status=ready while mandatory startup
# initialization can still fail.
# --------------------------------------------------------------------------------

_MAIN_SOURCE = (__import__("pathlib").Path(__file__).resolve().parents[1] / "main.py").read_text(
    encoding="utf-8"
)


def _source_index(needle: str) -> int:
    index = _MAIN_SOURCE.find(needle)
    assert index >= 0, f"startup marker not found in main.py: {needle!r}"
    return index


def test_authoritative_ready_event_follows_every_mandatory_startup_step() -> None:
    """runtime_ready must appear after resource/runtime/graph construction."""
    ready = _source_index('"runtime_ready"')
    for marker in (
        "build_conversation_persistence_service()",
        "build_demo_graph()",
        "build_doubt_solver_graph()",
        "build_practice_async_launcher(",
        "Agent initialised",
    ):
        assert _source_index(marker) < ready, (
            f"{marker} must run before the authoritative ready event"
        )


def test_early_runtime_started_event_does_not_claim_readiness() -> None:
    """The pre-initialization event keeps identity diagnostics but not status=ready."""
    started = _source_index('"runtime_started"')
    ready = _source_index('"runtime_ready"')
    early_block = _MAIN_SOURCE[started:ready]
    assert 'status="started"' in early_block
    assert 'status="ready"' not in early_block


def test_practice_bootstrap_failure_precedes_the_ready_event() -> None:
    """A Practice/Pattern resource failure aborts before runtime_ready is reachable."""
    launcher = _source_index("build_practice_async_launcher(")
    ready = _source_index('"runtime_ready"')
    assert launcher < ready
    # No try/except swallows the launcher failure, so it propagates out of import.
    guarded = _MAIN_SOURCE[launcher:ready]
    assert "except PracticeConfigurationError" not in guarded
    assert "except Exception" not in guarded


def test_exactly_one_authoritative_ready_event_exists_in_startup() -> None:
    assert _MAIN_SOURCE.count('"runtime_ready"') == 1
    assert _MAIN_SOURCE.count('"runtime_started"') == 1


def test_both_startup_lifecycle_events_are_registered() -> None:
    from observability.events import EVENT_NAMES

    assert {"runtime_started", "runtime_ready"} <= EVENT_NAMES


def test_ready_event_details_survive_the_sanitizer_and_leak_no_resources() -> None:
    """Aggregate feature flags only — never table names, ARNs, or index names."""
    from observability.events import sanitize_details

    details = {
        "practiceEnabled": True,
        "patternContextEnabled": True,
        "patternReuseEnabled": False,
        "orchestratedDoubtSolverEnabled": True,
    }
    safe = sanitize_details(details)

    assert len(safe) == len(details), f"a readiness field was silently dropped: {safe}"
    rendered = " ".join(str(value) for value in safe.values())
    for forbidden in ("arn:", "table", "index", "MockTestQuiz", "QuestionBank", "Pattern-"):
        assert forbidden not in rendered


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("true", True), ("false", False), ("", False)],
)
def test_pattern_reuse_resolves_exactly_from_configuration(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    """No precedence bug: the resolved value mirrors the configured value."""
    import config as config_module

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", env_value)
    config_module._settings = None

    settings = config_module.get_settings()

    assert settings.pattern_intelligence_reuse_enabled is expected
    assert settings.pattern_intelligence_enabled is True


def test_pattern_reuse_without_master_flag_is_rejected_at_config_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing safety invariant must not regress."""
    import config as config_module

    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "false")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_REUSE_ENABLED", "true")
    config_module._settings = None

    with pytest.raises(config_module.ConfigurationError):
        config_module.get_settings()
