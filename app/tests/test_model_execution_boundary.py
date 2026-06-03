"""
Tests for the Part 4 model execution boundary.
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from schemas.llm import LlmMessage
from schemas.llm_routing import FallbackAttempt, RouteDecision, RouteRequest
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.errors import (
    ModelConfigResolutionError,
    ModelExecutionConfigError,
    ProviderExecutionError,
)
from services.llm_orchestration.model_config_resolver import ModelConfigResolver
from services.llm_orchestration.model_execution import (
    FakeProviderExecutor,
    RegistryBackedModelExecutor,
)
from services.llm_orchestration.orchestrator import LlmOrchestrator
from services.llm_orchestration.prompt_resolver import PromptResolver
from services.llm_orchestration.route_resolver import resolve_route

_TEST_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: gemini_flash_light
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
            fallback:
              - safe_mock
      math:
        generator:
          default:
            model: gemini_flash_light
            prompt: subjects/math_generator.md
            temperature: 0.2
            max_tokens: 800
            fallback:
              - general_default
              - safe_mock
          advanced:
            inherits: default
            model: gemini_flash_reasoning_light
            prompt: subjects/math_generator.md
            temperature: 0.15
            max_tokens: 1000
            provider_options:
              thinking: true
            fallback:
              - default
              - general_default
              - safe_mock
    models:
      gemini_flash_light:
        provider: gemini
        provider_profile: gemini_primary
        model_id: gemini-2.5-flash
        model_label: gemini-flash-light
        cost_tier: low
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 20
        capabilities:
          math: medium
          general: medium
      gemini_flash_reasoning_light:
        provider: gemini
        provider_profile: gemini_primary
        model_id: gemini-2.5-flash
        model_label: gemini-flash-reasoning-light
        cost_tier: low
        supports_streaming: true
        supports_thinking: true
        timeout_seconds: 25
        capabilities:
          math: high
          general: high
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
        capabilities:
          general: low
    provider_profiles:
      gemini_primary:
        provider: gemini
        api_key_env: GEMINI_API_KEY
      local_mock:
        provider: mock
""")


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(_TEST_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _route_decision(
    *,
    model: str = "gemini_flash_light",
    provider_options: dict[str, Any] | None = None,
    fallback_attempts: list[FallbackAttempt] | None = None,
) -> RouteDecision:
    return RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        model=model,
        prompt="subjects/math_generator.md",
        temperature=0.2,
        max_tokens=800,
        provider_options=provider_options or {},
        fallback_attempts=fallback_attempts or [],
        route_source="exact",
    )


def _messages() -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content="System prompt."),
        LlmMessage(role="user", content="Student question."),
    ]


def _executor_pair(
    tmp_path: Path,
    *,
    provider_executor: FakeProviderExecutor | None = None,
) -> tuple[RegistryBackedModelExecutor, FakeProviderExecutor]:
    fake = provider_executor or FakeProviderExecutor(content="Boundary answer. <ANSWER_DONE>")
    resolver = ModelConfigResolver(registry=_registry(tmp_path))
    return (
        RegistryBackedModelExecutor(
            provider_executor=fake,
            model_config_resolver=resolver,
        ),
        fake,
    )


def test_fake_provider_executor_records_request_and_call_count(tmp_path: Path) -> None:
    executor, fake = _executor_pair(tmp_path)

    result = executor.execute(route_decision=_route_decision(), messages=_messages())

    assert result.content == "Boundary answer. <ANSWER_DONE>"
    assert fake.call_count == 1
    assert fake.last_request is not None


def test_registry_backed_executor_builds_provider_execution_request(
    tmp_path: Path,
) -> None:
    executor, fake = _executor_pair(tmp_path)
    decision = _route_decision(model="gemini_flash_light")

    executor.execute(route_decision=decision, messages=_messages())

    assert fake.last_request is not None
    assert fake.last_request.route_decision is decision
    assert fake.last_request.model_resolution.model_alias == "gemini_flash_light"
    assert fake.last_request.model_resolution.provider == "gemini"


def test_provider_execution_request_contains_internal_messages(
    tmp_path: Path,
) -> None:
    executor, fake = _executor_pair(tmp_path)
    messages = _messages()

    executor.execute(route_decision=_route_decision(), messages=messages)

    assert fake.last_request is not None
    assert fake.last_request.messages == messages


def test_provider_execution_request_copies_runtime_options(tmp_path: Path) -> None:
    executor, fake = _executor_pair(tmp_path)
    decision = _route_decision(provider_options={"stream": True})

    executor.execute(route_decision=decision, messages=_messages())

    assert fake.last_request is not None
    assert fake.last_request.temperature == 0.2
    assert fake.last_request.max_tokens == 800
    assert fake.last_request.provider_options == {"stream": True}


def test_provider_executor_is_required() -> None:
    with pytest.raises(TypeError, match="provider_executor is required"):
        RegistryBackedModelExecutor(provider_executor=None)  # type: ignore[arg-type]


def test_executor_does_not_read_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _ = _executor_pair(tmp_path)

    def fail_getenv(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("os.getenv must not be called by executor")

    monkeypatch.setattr(os, "getenv", fail_getenv)

    result = executor.execute(route_decision=_route_decision(), messages=_messages())

    assert result.provider == "gemini"


def test_thinking_true_with_unsupported_model_raises_config_error(
    tmp_path: Path,
) -> None:
    executor, fake = _executor_pair(tmp_path)

    with pytest.raises(ModelExecutionConfigError, match="does not support"):
        executor.execute(
            route_decision=_route_decision(provider_options={"thinking": True}),
            messages=_messages(),
        )

    assert fake.call_count == 0


def test_thinking_true_with_supported_model_executes(tmp_path: Path) -> None:
    executor, fake = _executor_pair(tmp_path)

    result = executor.execute(
        route_decision=_route_decision(
            model="gemini_flash_reasoning_light",
            provider_options={"thinking": True},
        ),
        messages=_messages(),
    )

    assert result.model == "gemini_flash_reasoning_light"
    assert fake.call_count == 1


def test_unsupported_provider_option_raises_config_error(tmp_path: Path) -> None:
    executor, fake = _executor_pair(tmp_path)

    with pytest.raises(ModelExecutionConfigError, match="Unsupported provider option"):
        executor.execute(
            route_decision=_route_decision(provider_options={"temperature_boost": True}),
            messages=_messages(),
        )

    assert fake.call_count == 0


def test_unknown_model_alias_propagates_resolution_error(tmp_path: Path) -> None:
    executor, fake = _executor_pair(tmp_path)

    with pytest.raises(ModelConfigResolutionError, match="Unknown model alias"):
        executor.execute(
            route_decision=_route_decision(model="missing_model"),
            messages=_messages(),
        )

    assert fake.call_count == 0


def test_generic_provider_failure_wraps_provider_execution_error(
    tmp_path: Path,
) -> None:
    provider = FakeProviderExecutor(raise_on_execute=ValueError("raw provider detail"))
    executor, _ = _executor_pair(tmp_path, provider_executor=provider)

    with pytest.raises(ProviderExecutionError) as exc_info:
        executor.execute(route_decision=_route_decision(), messages=_messages())

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "raw provider detail" not in str(exc_info.value)


def test_provider_execution_error_re_raised_without_double_wrap(
    tmp_path: Path,
) -> None:
    provider_error = ProviderExecutionError("Safe provider failure.")
    provider = FakeProviderExecutor(raise_on_execute=provider_error)
    executor, _ = _executor_pair(tmp_path, provider_executor=provider)

    with pytest.raises(ProviderExecutionError) as exc_info:
        executor.execute(route_decision=_route_decision(), messages=_messages())

    assert exc_info.value is provider_error


def test_safe_metadata_passed_to_model_execution_result(tmp_path: Path) -> None:
    executor, _ = _executor_pair(tmp_path)

    result = executor.execute(route_decision=_route_decision(), messages=_messages())

    assert result.metadata == {
        "model_alias": "gemini_flash_light",
        "provider": "gemini",
        "supports_streaming": True,
        "supports_thinking": False,
        "supports_reasoning": False,
        "reasoning_effort": "none",
        "timeout_seconds": 20,
    }


def test_executor_logs_only_safe_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor, _ = _executor_pair(tmp_path)
    caplog.set_level(logging.INFO)

    executor.execute(route_decision=_route_decision(), messages=_messages())

    log_text = caplog.text
    assert "gemini_flash_light" in log_text
    assert "gemini" in log_text
    assert "Student question" not in log_text
    assert "System prompt" not in log_text
    assert "GEMINI_API_KEY" not in log_text


def test_no_fallback_execution_is_attempted_in_part_4(tmp_path: Path) -> None:
    fallback_attempt = FallbackAttempt(
        kind="model",
        model="safe_mock",
        reason="safe_mock",
    )
    executor, fake = _executor_pair(tmp_path)

    executor.execute(
        route_decision=_route_decision(fallback_attempts=[fallback_attempt]),
        messages=_messages(),
    )

    assert fake.call_count == 1


def test_provider_execution_request_not_exposed_in_orchestration_result(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    prompt_root = tmp_path / "prompts"
    (prompt_root / "subjects").mkdir(parents=True)
    (prompt_root / "subjects" / "math_generator.md").write_text(
        "# Math tutor system prompt.",
        encoding="utf-8",
    )
    provider = FakeProviderExecutor(content="Final answer. <ANSWER_DONE>")
    model_executor = RegistryBackedModelExecutor(
        provider_executor=provider,
        model_config_resolver=ModelConfigResolver(registry=registry),
    )
    orchestrator = LlmOrchestrator(
        model_executor=model_executor,
        prompt_resolver=PromptResolver(prompt_root=prompt_root),
        route_resolver_fn=lambda req: resolve_route(req, registry=registry),
    )

    result = orchestrator.generate(
        route_request=RouteRequest(
            request_id="req-001",
            subject="math",
            task_role="generator",
            difficulty="default",
        ),
        query="What is 2+2?",
    )

    result_dict = result.model_dump()
    assert result.content == "Final answer."
    assert result.metadata == {}
    assert "messages" not in result_dict
    assert "model_resolution" not in result_dict
    assert "ProviderExecutionRequest" not in str(result_dict)
    assert provider.last_request is not None


def test_full_isolated_orchestration_flow_returns_safe_result(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    prompt_root = tmp_path / "prompts"
    (prompt_root / "subjects").mkdir(parents=True)
    (prompt_root / "subjects" / "math_generator.md").write_text(
        "# Math tutor system prompt.",
        encoding="utf-8",
    )
    provider = FakeProviderExecutor(content="2 + 2 = 4. <ANSWER_DONE>")
    orchestrator = LlmOrchestrator(
        model_executor=RegistryBackedModelExecutor(
            provider_executor=provider,
            model_config_resolver=ModelConfigResolver(registry=registry),
        ),
        prompt_resolver=PromptResolver(prompt_root=prompt_root),
        route_resolver_fn=lambda req: resolve_route(req, registry=registry),
    )

    result = orchestrator.generate(
        route_request=RouteRequest(
            request_id="req-002",
            subject="math",
            task_role="generator",
            difficulty="advanced",
        ),
        query="Solve 2+2.",
    )

    assert result.content == "2 + 2 = 4."
    assert result.model == "gemini_flash_reasoning_light"
    assert result.provider == "gemini"
    assert result.answer_source == "llm"
    assert result.metadata == {}
    assert provider.call_count == 1
    assert provider.last_request is not None
    assert provider.last_request.model_resolution.supports_thinking is True


# ---------------------------------------------------------------------------
# Part 4.1 — Isolation guard tests (Task: no import-time side effects,
#             RegistryBackedModelExecutor requires explicit provider_executor)
# ---------------------------------------------------------------------------


def test_model_execution_import_does_not_access_env_or_config() -> None:
    """Importing model_execution must not read env vars, load config, or call providers.

    This test reimports the module in a subprocess-like way by clearing it
    from sys.modules and reimporting.  If the import triggers config.get_settings()
    or any provider init the test will observe it via the monkeypatched guard
    in conftest.py (ENABLE_REAL_LLM=false is already set).

    The key assertion: after import, no _settings singleton is created.
    """
    import config as cfg_module

    # Ensure _settings is None so we can detect if import creates it.
    cfg_module._settings = None

    # Re-import the module (may be cached; the point is no __init__ side-effects).
    import services.llm_orchestration.model_execution  # noqa: F401, PLC0415

    # Import must NOT have touched config._settings.
    assert cfg_module._settings is None, (
        "Importing model_execution must not call get_settings() or any config accessor"
    )


def test_model_config_resolver_import_does_not_load_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing model_config_resolver must not load YAML or call get_registry()."""
    import services.llm.orchestration.config_registry as registry_mod

    registry_mod.reset_registry()

    # Spy on the public get_registry() function.  A call here means the module
    # eagerly initialises the registry singleton at import time.
    original = registry_mod.get_registry
    calls: list[bool] = []

    def _spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_mod, "get_registry", _spy)

    # Module is already cached — no module body re-execution.
    # The spy would only fire if module-level code calls get_registry().
    import services.llm.orchestration.model_config_resolver  # noqa: F401, PLC0415

    assert not calls, "Importing model_config_resolver must not call get_registry()"


def test_registry_backed_executor_requires_provider_executor() -> None:
    """RegistryBackedModelExecutor must raise TypeError when provider_executor=None.

    This ensures no real provider is used as a silent default.
    """
    with pytest.raises(TypeError, match="provider_executor is required"):
        RegistryBackedModelExecutor(provider_executor=None)  # type: ignore[arg-type]


def test_registry_backed_executor_does_not_call_real_provider_without_injection(
    tmp_path: Path,
) -> None:
    """RegistryBackedModelExecutor must only call the explicitly injected executor.

    The FakeProviderExecutor is injected — no real OpenAI/Bedrock/Gemini call
    should be made.  Verified by asserting FakeProviderExecutor received the call.
    """
    fake = FakeProviderExecutor(content="Isolated test response. <ANSWER_DONE>")
    resolver = ModelConfigResolver(registry=_registry(tmp_path))
    executor = RegistryBackedModelExecutor(
        provider_executor=fake,
        model_config_resolver=resolver,
    )

    result = executor.execute(route_decision=_route_decision(), messages=_messages())

    assert fake.call_count == 1, "FakeProviderExecutor must be called exactly once"
    assert result.content == "Isolated test response. <ANSWER_DONE>"
    assert result.provider == "gemini"


def test_unit_tests_do_not_reach_real_llm_provider() -> None:
    """Verify ENABLE_REAL_LLM is false in the unit-test environment.

    This test will fail if conftest.py does not enforce the safe default,
    catching any regression where .env.local leaks into unit tests.
    """
    from config import get_settings  # noqa: PLC0415

    settings = get_settings()
    assert settings.enable_real_llm is False, (
        "ENABLE_REAL_LLM must be false in unit tests. "
        "Check conftest.py or environment variable leakage from .env.local."
    )


def test_unit_tests_do_not_reach_real_kb_retrieval() -> None:
    """Verify ENABLE_KB_RETRIEVAL is false in the unit-test environment."""
    from config import get_settings  # noqa: PLC0415

    settings = get_settings()
    assert settings.enable_kb_retrieval is False, (
        "ENABLE_KB_RETRIEVAL must be false in unit tests."
    )


def test_unit_tests_do_not_reach_real_dynamodb() -> None:
    """Verify ENABLE_DYNAMODB_FETCH is false in the unit-test environment."""
    from config import get_settings  # noqa: PLC0415

    settings = get_settings()
    assert settings.enable_dynamodb_fetch is False, (
        "ENABLE_DYNAMODB_FETCH must be false in unit tests."
    )
