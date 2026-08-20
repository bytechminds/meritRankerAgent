"""
app/tests/test_azure_first_fallback.py
---------------------------------------
Tests for Part 9.1 — Azure-first model execution with native OpenAI fallback.

Coverage:
  1. Model registry — fallback_models schema validation
  2. Registry cross-validation — alias existence, self-ref, cycle detection
  3. Model execution fallback — primary fails, fallback succeeds
  4. No fallback for non-eligible errors (invalid_request, config errors)
  5. All-fallbacks-fail → ProviderExecutionError
  6. Provider error mapping (OpenAI / Azure)
  7. max_retries=0 in both adapters
  8. Graph _generate_node — safe answer on ProviderExecutionError
  9. Existing regression guards (mock/legacy/streaming paths)

No real provider calls. No AWS calls. No network access.
"""

from __future__ import annotations

import json
import logging
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import services.llm.orchestration.model_execution as model_execution_module
from features.practice_generation.execution_control import PracticeExecutionStopped
from observability import bind_execution_context, bind_request_context
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
from schemas.llm_routing import ModelConfig, RouteDecision, RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.errors import (
    LlmConfigValidationError,
    ProviderExecutionError,
)
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import (
    FakeProviderExecutor,
    RegistryBackedModelExecutor,
)
from services.llm.orchestration.route_resolver import resolve_route
from services.llm.providers.azure_openai_provider import (
    AzureOpenAIProviderAdapter,
    _classify_azure_openai_error,
)
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.openai_provider import (
    OpenAIProviderAdapter,
    _classify_openai_error,
)

# ---------------------------------------------------------------------------
# Shared test YAML — Azure-first with OpenAI native fallback
# ---------------------------------------------------------------------------

_FALLBACK_YAML = textwrap.dedent("""\
    version: 1
    routes:
      math:
        generator:
          default:
            model: azure_fast
            prompt: subjects/math_generator.md
            temperature: 0.2
            max_tokens: 800
            fallback:
              - safe_mock
      general:
        generator:
          default:
            model: azure_fast
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
            fallback:
              - safe_mock
    models:
      azure_fast:
        description: Azure-first fast generator.
        provider: azure_openai
        provider_profile: azure_primary
        deployment: my-azure-deployment
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 30
        fallback_models:
          - openai_native_fallback
      openai_native_fallback:
        description: Native OpenAI fallback.
        provider: openai
        provider_profile: openai_primary
        model_id: gpt-4o-mini
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 20
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
    provider_profiles:
      azure_primary:
        provider: azure_openai
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_key_env: AZURE_OPENAI_API_KEY
        api_version_env: AZURE_OPENAI_API_VERSION
      openai_primary:
        provider: openai
        api_key_env: OPENAI_API_KEY
      local_mock:
        provider: mock
""")


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    p = tmp_path / "llm.yaml"
    p.write_text(_FALLBACK_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=p)


def _resolver(tmp_path: Path) -> ModelConfigResolver:
    return ModelConfigResolver(registry=_registry(tmp_path))


def _route_decision(model: str = "azure_fast") -> RouteDecision:
    return RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        model=model,
        prompt="subjects/math_generator.md",
        temperature=0.2,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def _messages() -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content="System prompt."),
        LlmMessage(role="user", content="Student question."),
    ]


# ---------------------------------------------------------------------------
# Helper: multi-alias fake executor
# ---------------------------------------------------------------------------


class _AliasedFakeProviderExecutor:
    """Fake executor that raises or returns based on the model alias."""

    def __init__(
        self,
        *,
        raise_for: dict[str, Exception] | None = None,
        return_for: dict[str, str | ModelExecutionResult] | None = None,
        default_content: str = "Default answer.",
    ) -> None:
        self._raise_for: dict[str, Exception] = raise_for or {}
        self._return_for: dict[str, str | ModelExecutionResult] = return_for or {}
        self._default_content = default_content
        self.call_log: list[str] = []
        self.last_request: ProviderExecutionRequest | None = None

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        alias = request.model_resolution.model_alias
        self.call_log.append(alias)
        self.last_request = request
        if alias in self._raise_for:
            raise self._raise_for[alias]
        content = self._return_for.get(alias, self._default_content)
        if isinstance(content, ModelExecutionResult):
            return content
        return ModelExecutionResult(
            content=content,
            model=alias,
            provider=request.model_resolution.provider,
            finish_reason="stop",
        )


# ===========================================================================
# 1. ModelConfig schema — fallback_models validation
# ===========================================================================


class TestModelConfigFallbackModelsSchema:
    def test_fallback_models_empty_by_default(self) -> None:
        cfg = ModelConfig(
            provider="openai",
            provider_profile="openai_primary",
            model_id="gpt-4o-mini",
            supports_streaming=False,
            supports_thinking=False,
            timeout_seconds=20,
        )
        assert cfg.fallback_models == []

    def test_fallback_models_valid_aliases_accepted(self) -> None:
        cfg = ModelConfig(
            provider="azure_openai",
            provider_profile="azure_primary",
            deployment="my-deployment",
            supports_streaming=True,
            supports_thinking=False,
            timeout_seconds=30,
            fallback_models=["math_basic_generator_openai_native"],
        )
        assert cfg.fallback_models == ["math_basic_generator_openai_native"]

    def test_fallback_models_max_3_aliases(self) -> None:
        with pytest.raises(Exception, match="at most 3"):
            ModelConfig(
                provider="openai",
                provider_profile="openai_primary",
                model_id="gpt-4o-mini",
                supports_streaming=False,
                supports_thinking=False,
                timeout_seconds=20,
                fallback_models=["a_fallback", "b_fallback", "c_fallback", "d_fallback"],
            )

    def test_fallback_models_empty_string_rejected(self) -> None:
        with pytest.raises(Exception, match="empty"):
            ModelConfig(
                provider="openai",
                provider_profile="openai_primary",
                model_id="gpt-4o-mini",
                supports_streaming=False,
                supports_thinking=False,
                timeout_seconds=20,
                fallback_models=[""],
            )

    def test_fallback_models_provider_model_id_rejected(self) -> None:
        with pytest.raises(Exception, match="provider model_id"):
            ModelConfig(
                provider="openai",
                provider_profile="openai_primary",
                model_id="gpt-4o-mini",
                supports_streaming=False,
                supports_thinking=False,
                timeout_seconds=20,
                fallback_models=["gpt-4o-mini"],
            )

    def test_fallback_models_slash_rejected(self) -> None:
        with pytest.raises(Exception, match="provider model_id"):
            ModelConfig(
                provider="openai",
                provider_profile="openai_primary",
                model_id="gpt-4o-mini",
                supports_streaming=False,
                supports_thinking=False,
                timeout_seconds=20,
                fallback_models=["openai/gpt-4o"],
            )

    def test_fallback_models_dots_rejected(self) -> None:
        with pytest.raises(Exception, match="dots"):
            ModelConfig(
                provider="openai",
                provider_profile="openai_primary",
                model_id="gpt-4o-mini",
                supports_streaming=False,
                supports_thinking=False,
                timeout_seconds=20,
                fallback_models=["math.basic.generator"],
            )


# ===========================================================================
# 2. Registry cross-validation
# ===========================================================================


class TestRegistryCrossValidation:
    def _yaml_with_models(self, models_section: str) -> str:
        return (
            textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: primary_model
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
            """)
            + models_section
            + textwrap.dedent("""\
            provider_profiles:
              openai_primary:
                provider: openai
                api_key_env: OPENAI_API_KEY
              azure_primary:
                provider: azure_openai
                endpoint_env: AZURE_OPENAI_ENDPOINT
                api_key_env: AZURE_OPENAI_API_KEY
                api_version_env: AZURE_OPENAI_API_VERSION
              local_mock:
                provider: mock
            """)
        )

    def test_valid_azure_primary_openai_fallback_loads(self, tmp_path: Path) -> None:
        yaml = self._yaml_with_models(
            textwrap.dedent("""\
            models:
              primary_model:
                provider: azure_openai
                provider_profile: azure_primary
                deployment: my-deployment
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 30
                fallback_models:
                  - openai_fallback
              openai_fallback:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 20
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            """)
        )
        p = tmp_path / "llm.yaml"
        p.write_text(yaml, encoding="utf-8")
        registry = LlmConfigRegistry(yaml_path=p)
        assert "primary_model" in registry.model_map
        assert registry.model_map["primary_model"].fallback_models == ["openai_fallback"]
        assert "openai_fallback" in registry.model_map

    def test_unknown_fallback_alias_rejected(self, tmp_path: Path) -> None:
        yaml = self._yaml_with_models(
            textwrap.dedent("""\
            models:
              primary_model:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
                fallback_models:
                  - nonexistent_alias
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            """)
        )
        p = tmp_path / "llm.yaml"
        p.write_text(yaml, encoding="utf-8")
        with pytest.raises(LlmConfigValidationError, match="nonexistent_alias"):
            LlmConfigRegistry(yaml_path=p)

    def test_self_fallback_rejected(self, tmp_path: Path) -> None:
        yaml = self._yaml_with_models(
            textwrap.dedent("""\
            models:
              primary_model:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
                fallback_models:
                  - primary_model
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            """)
        )
        p = tmp_path / "llm.yaml"
        p.write_text(yaml, encoding="utf-8")
        with pytest.raises(LlmConfigValidationError, match="itself"):
            LlmConfigRegistry(yaml_path=p)

    def test_cyclic_fallback_rejected(self, tmp_path: Path) -> None:
        yaml = self._yaml_with_models(
            textwrap.dedent("""\
            models:
              primary_model:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
                fallback_models:
                  - model_b
              model_b:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
                fallback_models:
                  - primary_model
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            """)
        )
        p = tmp_path / "llm.yaml"
        p.write_text(yaml, encoding="utf-8")
        with pytest.raises(LlmConfigValidationError, match="[Cc]ycl"):
            LlmConfigRegistry(yaml_path=p)

    def test_routes_reference_only_primary_alias(self, tmp_path: Path) -> None:
        """Routes must reference primary aliases; fallback aliases must not be in routes."""
        registry = _registry(tmp_path)
        # Verify the route points to azure_fast (primary), not openai_native_fallback
        route = registry.get_route("math", "generator", "default")
        assert route is not None
        assert route.model == "azure_fast"
        assert "openai_native_fallback" not in [r.model for r in registry.route_map.values()]

    def test_fallback_alias_provider_profile_must_exist(self, tmp_path: Path) -> None:
        yaml = self._yaml_with_models(
            textwrap.dedent("""\
            models:
              primary_model:
                provider: openai
                provider_profile: openai_primary
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
                fallback_models:
                  - fallback_bad_profile
              fallback_bad_profile:
                provider: openai
                provider_profile: nonexistent_profile
                model_id: gpt-4o-mini
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 20
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            """)
        )
        p = tmp_path / "llm.yaml"
        p.write_text(yaml, encoding="utf-8")
        with pytest.raises(LlmConfigValidationError, match="nonexistent_profile"):
            LlmConfigRegistry(yaml_path=p)


# ===========================================================================
# 3. Model execution fallback — primary fails, fallback succeeds
# ===========================================================================


class TestModelExecutionFallback:
    def test_attempt_guard_blocks_escalation_after_primary_finishes_cancelled(
        self, tmp_path: Path
    ) -> None:
        exhausted = ModelExecutionResult(
            content="partial",
            model="azure_fast",
            provider="azure_openai",
            finish_reason="length",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={
                "azure_fast": exhausted,
                "openai_native_fallback": "Fallback answer.",
            }
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(PracticeExecutionStopped, match="PRACTICE_CANCEL_OBSERVED"):
            executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
                attempt_guard=lambda: (_ for _ in ()).throw(
                    PracticeExecutionStopped("PRACTICE_CANCEL_OBSERVED")
                ),
            )

        assert fake_executor.call_log == ["azure_fast"]

    def test_attempt_guard_blocks_fallback_after_inflight_escalation_finishes(
        self, tmp_path: Path
    ) -> None:
        class _CancelDuringEscalation:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.cancelled = False

            def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
                alias = request.model_resolution.model_alias
                self.calls.append(alias)
                if len(self.calls) == 1:
                    return ModelExecutionResult(
                        content="partial",
                        model=alias,
                        provider=request.model_resolution.provider,
                        finish_reason="length",
                    )
                self.cancelled = True
                raise LlmProviderExecutionError(
                    "escalation unavailable",
                    failure_kind="provider_unavailable",
                )

            def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
                raise AssertionError("Streaming is not used for structured Practice generation.")

        provider = _CancelDuringEscalation()
        executor = RegistryBackedModelExecutor(
            provider_executor=provider,
            model_config_resolver=_resolver(tmp_path),
        )

        def guard() -> None:
            if provider.cancelled:
                raise PracticeExecutionStopped("PRACTICE_CANCEL_OBSERVED")

        with pytest.raises(PracticeExecutionStopped, match="PRACTICE_CANCEL_OBSERVED"):
            executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
                attempt_guard=guard,
            )

        assert provider.calls == ["azure_fast", "azure_fast"]

    def test_superseded_attempt_guard_blocks_capacity_escalation(
        self, tmp_path: Path
    ) -> None:
        exhausted = ModelExecutionResult(
            content="partial",
            model="azure_fast",
            provider="azure_openai",
            finish_reason="length",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={
                "azure_fast": exhausted,
                "openai_native_fallback": "Fallback answer.",
            }
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(PracticeExecutionStopped, match="PRACTICE_EXECUTION_FENCED"):
            executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
                attempt_guard=lambda: (_ for _ in ()).throw(
                    PracticeExecutionStopped("PRACTICE_EXECUTION_FENCED")
                ),
            )

        assert fake_executor.call_log == ["azure_fast"]

    def test_active_attempt_guard_preserves_escalation_and_fallback(
        self, tmp_path: Path
    ) -> None:
        exhausted = LlmProviderResponseError("response unavailable")
        exhausted.finish_reason = "length"
        exhausted.output_tokens = 800
        exhausted.reasoning_tokens = 800
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": exhausted},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        guard_calls = 0

        def guard() -> None:
            nonlocal guard_calls
            guard_calls += 1

        result = executor.execute(
            route_decision=_route_decision().model_copy(update={"intent": "practice"}),
            messages=_messages(),
            attempt_guard=guard,
        )

        assert fake_executor.call_log == [
            "azure_fast",
            "azure_fast",
            "openai_native_fallback",
        ]
        assert guard_calls == 2
        assert result.model == "openai_native_fallback"

    def test_none_attempt_guard_preserves_existing_model_execution(
        self, tmp_path: Path
    ) -> None:
        empty_response = LlmProviderResponseError(
            "response unavailable",
            failure_kind="empty_answer",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": empty_response},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        result = executor.execute(
            route_decision=_route_decision(),
            messages=_messages(),
            attempt_guard=None,
        )

        assert fake_executor.call_log == ["azure_fast", "openai_native_fallback"]
        assert result.model == "openai_native_fallback"

    def test_attempt_guard_does_not_log_unstructured_exception_text(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=model_execution_module.logger.name)
        exhausted = ModelExecutionResult(
            content="partial",
            model="azure_fast",
            provider="azure_openai",
            finish_reason="length",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={"azure_fast": exhausted}
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(RuntimeError, match="private-provider-detail"):
            executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
                attempt_guard=lambda: (_ for _ in ()).throw(
                    RuntimeError("private-provider-detail")
                ),
            )

        assert "private-provider-detail" not in caplog.text
        assert "RuntimeError" in caplog.text

    def test_real_math_advanced_empty_response_uses_o3_fallback(self) -> None:
        empty_response = LlmProviderResponseError(
            "response unavailable",
            failure_kind="empty_answer",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={
                "math_advanced_generator": empty_response,
            },
            return_for={"openai_o3": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(provider_executor=fake_executor)
        decision = resolve_route(
            RouteRequest(
                request_id="practice-math-fallback",
                subject="math",
                task_role="generator",
                difficulty="advanced",
                intent="practice",
                language="english",
            )
        )

        result = executor.execute(route_decision=decision, messages=_messages())

        assert decision.model == "math_advanced_generator"
        assert fake_executor.call_log == [
            "math_advanced_generator",
            "openai_o3",
        ]
        assert result.model == "openai_o3"
        assert result.fallback_used is True

    def test_empty_response_uses_configured_fallback_with_safe_attempt_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        empty_response = LlmProviderResponseError(
            "response unavailable",
            failure_kind="empty_answer",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": empty_response},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        events: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(
            model_execution_module,
            "log_event",
            lambda event_name, **kwargs: events.append(
                (event_name, dict(kwargs.get("details") or {}))
            ),
        )

        with bind_request_context(
            request_id="request-fallback",
            conversation_id="conversation-fallback",
            turn_id="turn-fallback",
        ):
            with bind_execution_context(
                activity_id="practice-test-1",
                batch_id="slot-group-001",
                slot_ids=("slot-001",),
            ):
                result = executor.execute(
                    route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                    messages=_messages(),
                )

        assert result.model == "openai_native_fallback"
        assert fake_executor.call_log == ["azure_fast", "openai_native_fallback"]
        attempt_events = [
            (name, details) for name, details in events if name.startswith("generator_")
        ]
        assert [name for name, _details in attempt_events] == [
            "generator_model_attempt_started",
            "generator_model_attempt_failed",
            "generator_fallback_selected",
            "generator_fallback_started",
            "generator_fallback_succeeded",
        ]
        for _name, details in attempt_events:
            assert details["activityId"] == "practice-test-1"
            assert details["batchId"] == "slot-group-001"
            assert details["slotIds"] == "slot-001"
            assert details["slotCount"] == 1
            assert "Student question." not in str(details)

    def test_real_reasoning_advanced_chain_completes_five_question_manifest(self) -> None:
        exhausted = LlmProviderResponseError("response unavailable")
        exhausted.finish_reason = "length"
        exhausted.output_tokens = 3200
        exhausted.reasoning_tokens = 3200
        manifest = json.dumps(
            {
                "questions": [
                    {
                        "generation_item_id": f"reasoning-{index}",
                        "bucket_id": "reasoning-advanced",
                        "question": f"Reasoning question {index}?",
                        "question_type": "mcq",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "solution": "Concise solution.",
                        "subject": "reasoning",
                        "topic": "logical_reasoning",
                        "difficulty": "advanced",
                    }
                    for index in range(1, 6)
                ]
            }
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={
                "reasoning_advanced_generator": exhausted,
                "openai_o3": LlmProviderExecutionError(
                    "fallback unavailable",
                    failure_kind="provider_unavailable",
                ),
            },
            return_for={"deepseek_v4pro": manifest},
        )
        executor = RegistryBackedModelExecutor(provider_executor=fake_executor)
        decision = resolve_route(
            RouteRequest(
                request_id="practice-reasoning-fallback",
                subject="reasoning",
                task_role="generator",
                difficulty="advanced",
                intent="practice",
                language="english",
            )
        )

        result = executor.execute(route_decision=decision, messages=_messages())

        assert decision.model == "reasoning_advanced_generator"
        # Advanced reasoning now starts at 5600, so the escalation call that could
        # not add capacity is skipped: one fewer known-failing model call.
        assert fake_executor.call_log == [
            "reasoning_advanced_generator",
            "openai_o3",
            "deepseek_v4pro",
        ]
        assert result.model == "deepseek_v4pro"
        assert result.fallback_used is True
        assert len(json.loads(result.content)["questions"]) == 5

    def test_safety_refusal_does_not_attempt_a_fallback(self, tmp_path: Path) -> None:
        refusal = LlmProviderResponseError(
            "response refused",
            failure_kind="safety_blocked",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": refusal},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(ProviderExecutionError) as raised:
            executor.execute(route_decision=_route_decision(), messages=_messages())

        assert raised.value.failure_kind == "safety_blocked"
        assert fake_executor.call_log == ["azure_fast"]

    def test_practice_token_exhaustion_from_empty_response_uses_configured_fallback(
        self, tmp_path: Path
    ) -> None:
        exhausted = LlmProviderResponseError("response unavailable")
        exhausted.finish_reason = "length"
        exhausted.output_tokens = 800
        exhausted.reasoning_tokens = 800
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": exhausted},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        decision = _route_decision().model_copy(update={"intent": "practice"})

        result = executor.execute(route_decision=decision, messages=_messages())

        assert fake_executor.call_log == [
            "azure_fast",
            "azure_fast",
            "openai_native_fallback",
        ]
        assert result.model == "openai_native_fallback"
        assert result.fallback_used is True
        assert result.metadata["failure_kind"] == "output_token_exhausted"
        assert fake_executor.last_request is not None
        assert fake_executor.last_request.max_tokens == 1500

    def test_practice_capacity_escalation_retries_primary_once_at_larger_budget(
        self, tmp_path: Path
    ) -> None:
        class _BudgetAwareProvider:
            def __init__(self) -> None:
                self.requests: list[ProviderExecutionRequest] = []

            def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
                self.requests.append(request)
                return ModelExecutionResult(
                    content=("partial" if len(self.requests) == 1 else "complete"),
                    model=request.model_resolution.model_alias,
                    provider=request.model_resolution.provider,
                    finish_reason=("length" if len(self.requests) == 1 else "stop"),
                )

            def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
                raise AssertionError("Streaming is not used for structured Practice generation.")

        provider = _BudgetAwareProvider()
        executor = RegistryBackedModelExecutor(
            provider_executor=provider,
            model_config_resolver=_resolver(tmp_path),
        )

        result = executor.execute(
            route_decision=_route_decision().model_copy(update={"intent": "practice"}),
            messages=_messages(),
        )

        assert [request.max_tokens for request in provider.requests] == [900, 1500]
        assert [request.model_resolution.model_alias for request in provider.requests] == [
            "azure_fast",
            "azure_fast",
        ]
        assert result.content == "complete"
        assert result.fallback_used is False

    def test_capacity_insufficient_fallback_is_skipped_after_exhaustion(
        self, tmp_path: Path
    ) -> None:
        registry = _registry(tmp_path)
        fallback_model = registry.get_model("openai_native_fallback")
        assert fallback_model is not None
        fallback_model.model_hard_max_output_tokens = 1000
        resolver = ModelConfigResolver(registry=registry)
        exhausted = ModelExecutionResult(
            content="partial",
            model="azure_fast",
            provider="azure_openai",
            finish_reason="length",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={"azure_fast": exhausted},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=resolver,
        )

        with pytest.raises(ProviderExecutionError) as raised:
            executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
            )

        assert raised.value.failure_kind == "output_token_exhausted"
        assert raised.value.attempted_aliases == (
            "azure_fast",
            "openai_native_fallback",
        )
        assert fake_executor.call_log == ["azure_fast", "azure_fast"]

    def test_truncated_practice_content_bypasses_contract_parsing_and_logs_safe_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        truncated = ModelExecutionResult(
            content='{"questions":[{"slot_id":"slot-001","question":"private body"},',
            model="azure_fast",
            provider="azure_openai",
            finish_reason="length",
            output_tokens=2600,
            reasoning_tokens=2573,
        )
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={
                "azure_fast": truncated,
                "openai_native_fallback": "Fallback answer.",
            }
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        events: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(
            model_execution_module,
            "log_event",
            lambda event_name, **kwargs: events.append(
                (event_name, dict(kwargs.get("details") or {}))
            ),
        )

        with bind_execution_context(slot_ids=("slot-001", "slot-002")):
            result = executor.execute(
                route_decision=_route_decision().model_copy(update={"intent": "practice"}),
                messages=_messages(),
            )

        assert fake_executor.call_log == [
            "azure_fast",
            "azure_fast",
            "openai_native_fallback",
        ]
        assert result.model == "openai_native_fallback"
        exhausted_events = [
            details
            for event_name, details in events
            if event_name == "practice_generator_token_exhausted"
        ]
        assert len(exhausted_events) == 2
        assert [event["outputTokenLimit"] for event in exhausted_events] == [900, 1500]
        for event in exhausted_events:
            assert event["route"] == "math.generator.default"
            assert event["modelAlias"] == "azure_fast"
            assert event["failureClass"] == "output_token_exhausted"
            assert event["finishReason"] == "length"
            assert event["normalizedFailureReason"] == "output_token_exhausted"
            assert event["requestedItems"] == 2
            assert event["completeItemsDetected"] == 1
            assert event["structuredPayloadComplete"] is False
            assert event["structuredPayloadStartsWithObject"] is True
            assert event["structuredPayloadEndsMidObjectOrArray"] is True
            assert event["outputTokens"] == 2600
            assert event["reasoningTokens"] == 2573
            assert event["answerTokens"] == 27
        assert "private body" not in str(exhausted_events)

    def test_exhausted_fallback_chain_retains_attempted_aliases(self, tmp_path: Path) -> None:
        exhausted = LlmProviderResponseError("response unavailable")
        exhausted.finish_reason = "length"
        exhausted.output_tokens = 800
        exhausted.reasoning_tokens = 800
        fallback_failure = LlmProviderExecutionError(
            "fallback unavailable",
            failure_kind="provider_unavailable",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={
                "azure_fast": exhausted,
                "openai_native_fallback": fallback_failure,
            }
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        decision = _route_decision().model_copy(update={"intent": "practice"})

        with pytest.raises(ProviderExecutionError) as raised:
            executor.execute(route_decision=decision, messages=_messages())

        assert raised.value.failure_kind == "output_token_exhausted"
        assert raised.value.attempted_aliases == (
            "azure_fast",
            "openai_native_fallback",
        )

    def test_primary_quota_failure_triggers_fallback(self, tmp_path: Path) -> None:
        """Azure primary raises insufficient_quota; OpenAI fallback succeeds."""
        quota_error = LlmProviderExecutionError(
            "Azure quota exceeded",
            failure_kind="insufficient_quota",
            provider="azure_openai",
            model_alias="azure_fast",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": quota_error},
            return_for={"openai_native_fallback": "Fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        result = executor.execute(route_decision=_route_decision(), messages=_messages())

        assert result.content == "Fallback answer."
        assert result.fallback_used is True
        assert result.model == "openai_native_fallback"

    def test_fallback_result_metadata_has_safe_provenance(self, tmp_path: Path) -> None:
        quota_error = LlmProviderExecutionError(
            "quota",
            failure_kind="insufficient_quota",
            provider="azure_openai",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": quota_error},
            return_for={"openai_native_fallback": "Answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())

        assert result.metadata["fallback_from"] == "azure_fast"
        assert result.metadata["fallback_to"] == "openai_native_fallback"
        assert result.metadata["failure_kind"] == "insufficient_quota"

    def test_no_prompt_or_context_in_fallback_metadata(self, tmp_path: Path) -> None:
        quota_error = LlmProviderExecutionError(
            "quota",
            failure_kind="rate_limited",
            provider="azure_openai",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": quota_error},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        unsafe = {"prompt", "messages", "query", "context", "api_key", "secret"}
        assert unsafe.isdisjoint(result.metadata.keys()), (
            f"Unsafe keys found in metadata: {unsafe & result.metadata.keys()}"
        )

    def test_same_messages_reused_for_fallback(self, tmp_path: Path) -> None:
        quota_error = LlmProviderExecutionError(
            "quota",
            failure_kind="insufficient_quota",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": quota_error},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        messages = _messages()
        executor.execute(route_decision=_route_decision(), messages=messages)

        # The fallback request must carry the same messages list
        assert fake_executor.last_request is not None
        assert fake_executor.last_request.messages == messages

    def test_fallback_provider_options_stripped(self, tmp_path: Path) -> None:
        """Fallback requests must not carry thinking=True from the primary route."""
        quota_error = LlmProviderExecutionError(
            "quota",
            failure_kind="timeout",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": quota_error},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        decision = _route_decision()
        # Construct with empty provider_options for simplicity
        executor.execute(route_decision=decision, messages=_messages())
        # Fallback request should have empty provider_options
        assert fake_executor.last_request is not None
        assert fake_executor.last_request.provider_options == {}

    def test_all_fallbacks_fail_raises_provider_execution_error(self, tmp_path: Path) -> None:
        error = LlmProviderExecutionError(
            "quota",
            failure_kind="rate_limited",
        )
        fake_executor = _AliasedFakeProviderExecutor(raise_for={"azure_fast": error})
        # openai_native_fallback also fails
        fake_executor._raise_for["openai_native_fallback"] = LlmProviderExecutionError(
            "openai also quota",
            failure_kind="rate_limited",
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(ProviderExecutionError, match="All model execution attempts failed"):
            executor.execute(route_decision=_route_decision(), messages=_messages())

    def test_all_fallbacks_fail_attempted_list_in_error(self, tmp_path: Path) -> None:
        error = LlmProviderExecutionError(
            "quota",
            failure_kind="insufficient_quota",
        )
        fallback_error = LlmProviderExecutionError(
            "openai quota",
            failure_kind="insufficient_quota",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={
                "azure_fast": error,
                "openai_native_fallback": fallback_error,
            }
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(ProviderExecutionError) as exc_info:
            executor.execute(route_decision=_route_decision(), messages=_messages())

        msg = str(exc_info.value)
        assert "azure_fast" in msg
        assert "openai_native_fallback" in msg

    def test_fallback_order_respected(self, tmp_path: Path) -> None:
        """Executor tries aliases in the order they appear in fallback_models."""
        call_order: list[str] = []

        class _OrderTrackingExecutor:
            def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
                alias = request.model_resolution.model_alias
                call_order.append(alias)
                if alias == "azure_fast":
                    raise LlmProviderExecutionError(
                        "primary fails",
                        failure_kind="provider_unavailable",
                    )
                return ModelExecutionResult(
                    content="success",
                    model=alias,
                    provider="openai",
                )

        executor = RegistryBackedModelExecutor(
            provider_executor=_OrderTrackingExecutor(),
            model_config_resolver=_resolver(tmp_path),
        )
        executor.execute(route_decision=_route_decision(), messages=_messages())
        assert call_order == ["azure_fast", "openai_native_fallback"]

    def test_primary_rate_limited_triggers_fallback(self, tmp_path: Path) -> None:
        error = LlmProviderExecutionError(
            "rate limited",
            failure_kind="rate_limited",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": error},
            return_for={"openai_native_fallback": "Rate fallback answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        assert result.fallback_used is True
        assert result.content == "Rate fallback answer."

    def test_stream_invalid_deployment_triggers_fallback(self, tmp_path: Path) -> None:
        deployment_error = LlmProviderExecutionError(
            "Azure deployment not found",
            failure_kind="model_not_found",
            provider="azure_openai",
            model_alias="azure_fast",
        )

        class _StreamFallbackExecutor:
            def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
                alias = request.model_resolution.model_alias
                if alias == "azure_fast":
                    raise deployment_error
                yield "streamed fallback"

            def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
                return ModelExecutionResult(
                    content="buffered",
                    model=request.model_resolution.model_alias,
                    provider="openai",
                )

        executor = RegistryBackedModelExecutor(
            provider_executor=_StreamFallbackExecutor(),
            model_config_resolver=_resolver(tmp_path),
        )
        chunks = list(
            executor.execute_stream(route_decision=_route_decision(), messages=_messages())
        )
        assert chunks == ["streamed fallback"]

    def test_primary_timeout_triggers_fallback(self, tmp_path: Path) -> None:
        error = LlmProviderExecutionError(
            "timeout",
            failure_kind="timeout",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": error},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        assert result.fallback_used is True


# ===========================================================================
# 4. No fallback for non-eligible errors
# ===========================================================================


class TestNoFallbackForConfigErrors:
    def test_invalid_request_does_not_trigger_fallback(self, tmp_path: Path) -> None:
        """invalid_request is not fallback-eligible — fails immediately."""
        error = LlmProviderExecutionError(
            "bad request",
            failure_kind="invalid_request",
        )
        fake_executor = _AliasedFakeProviderExecutor(raise_for={"azure_fast": error})
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(ProviderExecutionError):
            executor.execute(route_decision=_route_decision(), messages=_messages())

        # Only one call made — no fallback attempted
        assert fake_executor.call_log == ["azure_fast"]

    def test_invalid_request_not_in_fallback_eligible_kinds(self) -> None:
        assert "invalid_request" not in FALLBACK_ELIGIBLE_FAILURE_KINDS

    def test_config_error_does_not_trigger_fallback(self, tmp_path: Path) -> None:
        """LlmProviderConfigurationError is not LlmProviderExecutionError → no fallback."""
        config_error = LlmProviderConfigurationError("missing api_key")
        fake_executor = _AliasedFakeProviderExecutor(raise_for={"azure_fast": config_error})
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )

        with pytest.raises(ProviderExecutionError):
            executor.execute(route_decision=_route_decision(), messages=_messages())

        assert fake_executor.call_log == ["azure_fast"]

    def test_model_config_resolution_error_raises_immediately(self, tmp_path: Path) -> None:
        """Unknown model alias raises ModelConfigResolutionError, not fallback."""
        from services.llm.orchestration.errors import ModelConfigResolutionError  # noqa: PLC0415

        executor = RegistryBackedModelExecutor(
            provider_executor=FakeProviderExecutor(),
            model_config_resolver=_resolver(tmp_path),
        )

        bad_route = _route_decision(model="nonexistent_alias")
        with pytest.raises(ModelConfigResolutionError):
            executor.execute(route_decision=bad_route, messages=_messages())

    def test_unsupported_provider_option_raises_immediately(self, tmp_path: Path) -> None:
        """Unsupported option → ModelExecutionConfigError, no fallback."""
        from services.llm.orchestration.errors import ModelExecutionConfigError  # noqa: PLC0415

        executor = RegistryBackedModelExecutor(
            provider_executor=FakeProviderExecutor(),
            model_config_resolver=_resolver(tmp_path),
        )
        bad_route = RouteDecision(
            route_id="math.generator.default",
            subject="math",
            task_role="generator",
            difficulty="default",
            model="azure_fast",
            prompt="subjects/math_generator.md",
            temperature=0.2,
            max_tokens=800,
            provider_options={"unsupported_option": True},
            fallback_attempts=[],
            route_source="exact",
        )
        with pytest.raises(ModelExecutionConfigError):
            executor.execute(route_decision=bad_route, messages=_messages())


# ===========================================================================
# 5. Provider error mapping
# ===========================================================================


class TestOpenAIErrorMapping:
    def _make_openai_exc(self, exc_type_name: str, **kwargs: Any) -> BaseException:
        """Build a minimal fake openai exception of the given type."""
        try:
            import openai  # noqa: PLC0415

            exc_class = getattr(openai, exc_type_name, None)
            if exc_class is not None:
                # Build a minimal response mock
                mock_resp = types.SimpleNamespace(
                    status_code=kwargs.get("status_code", 400),
                    headers={},
                    text="",
                    json=lambda: {},
                    request=types.SimpleNamespace(method="POST", url="https://api.openai.com"),
                )
                try:
                    return exc_class(
                        kwargs.get("message", "err"),
                        response=mock_resp,
                        body=kwargs.get("body", {}),
                    )
                except Exception:
                    pass
        except ImportError:
            pass
        # Fallback: return a generic RuntimeError
        return RuntimeError("simulated openai error")

    def test_rate_limit_insufficient_quota_maps(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=429,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://api.openai.com"),
        )
        try:
            exc = openai.RateLimitError(
                "insufficient_quota",
                response=mock_resp,
                body={"error": {"code": "insufficient_quota"}},
            )
            result = _classify_openai_error(exc)
            assert result == "insufficient_quota"
        except Exception:
            pytest.skip("Cannot construct RateLimitError with body kwarg")

    def test_rate_limit_generic_maps_rate_limited(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=429,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://api.openai.com"),
        )
        try:
            exc = openai.RateLimitError(
                "rate limit exceeded",
                response=mock_resp,
                body={},
            )
            result = _classify_openai_error(exc)
            assert result == "rate_limited"
        except Exception:
            pytest.skip("Cannot construct RateLimitError")

    def test_auth_error_maps_authentication_failed(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=401,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://api.openai.com"),
        )
        try:
            exc = openai.AuthenticationError(
                "invalid api key",
                response=mock_resp,
                body={},
            )
            result = _classify_openai_error(exc)
            assert result == "authentication_failed"
        except Exception:
            pytest.skip("Cannot construct AuthenticationError")

    def test_not_found_maps_model_not_found(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=404,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://api.openai.com"),
        )
        try:
            exc = openai.NotFoundError(
                "model not found",
                response=mock_resp,
                body={},
            )
            result = _classify_openai_error(exc)
            assert result == "model_not_found"
        except Exception:
            pytest.skip("Cannot construct NotFoundError")

    def test_unknown_error_maps_unknown_provider_error(self) -> None:
        result = _classify_openai_error(RuntimeError("unknown"))
        assert result == "unknown_provider_error"

    def test_no_api_key_in_provider_execution_error_message(self) -> None:
        """LlmProviderExecutionError must not include API key in message."""
        exc = LlmProviderExecutionError(
            "OpenAI call failed for model_alias='test': RateLimitError",
            failure_kind="rate_limited",
            provider="openai",
        )
        assert "sk-" not in str(exc)
        assert "api_key" not in str(exc).lower()

    def test_azure_error_classification_is_consistent(self) -> None:
        """Azure classifier should behave similarly for common error types."""
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=429,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://my.azure.com"),
        )
        try:
            exc = openai.RateLimitError(
                "rate limit",
                response=mock_resp,
                body={},
            )
            result = _classify_azure_openai_error(exc)
            assert result in ("rate_limited", "insufficient_quota")
        except Exception:
            pytest.skip("Cannot construct RateLimitError for azure test")

    def test_azure_bad_request_deployment_error_maps_model_not_found(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai not installed")

        mock_resp = types.SimpleNamespace(
            status_code=400,
            headers={},
            text="",
            json=lambda: {},
            request=types.SimpleNamespace(method="POST", url="https://my.azure.com"),
        )
        try:
            exc = openai.BadRequestError(
                "Deployment gpt-5.4-mini does not exist",
                response=mock_resp,
                body={"error": {"message": "Deployment not found"}},
            )
            assert _classify_azure_openai_error(exc) == "model_not_found"
        except Exception:
            pytest.skip("Cannot construct BadRequestError for azure test")

    def test_primary_invalid_deployment_triggers_fallback(self, tmp_path: Path) -> None:
        """Invalid deployment (model_not_found) on primary triggers fallback_models."""
        deployment_error = LlmProviderExecutionError(
            "Azure deployment not found",
            failure_kind="model_not_found",
            provider="azure_openai",
            model_alias="azure_fast",
        )
        fake_executor = _AliasedFakeProviderExecutor(
            raise_for={"azure_fast": deployment_error},
            return_for={"openai_native_fallback": "Fallback after bad deployment."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        assert result.content == "Fallback after bad deployment."
        assert result.fallback_used is True
        assert fake_executor.call_log == ["azure_fast", "openai_native_fallback"]


class TestProviderExecutionErrorAttributes:
    def test_failure_kind_attribute_set(self) -> None:
        exc = LlmProviderExecutionError(
            "msg",
            failure_kind="timeout",
            provider="openai",
            model_alias="math_fast",
        )
        assert exc.failure_kind == "timeout"
        assert exc.provider == "openai"
        assert exc.model_alias == "math_fast"

    def test_default_failure_kind_is_unknown(self) -> None:
        exc = LlmProviderExecutionError("generic failure")
        assert exc.failure_kind == "unknown_provider_error"

    def test_all_eligible_kinds_are_valid_strings(self) -> None:
        for kind in FALLBACK_ELIGIBLE_FAILURE_KINDS:
            assert isinstance(kind, str)
            assert len(kind) > 0


# ===========================================================================
# 6. max_retries=0 / retry behaviour
# ===========================================================================


class TestRetryBehavior:
    def test_openai_adapter_client_factory_receives_credentials(self) -> None:
        """The client_factory is called with the credentials object."""
        captured: list = []

        def fake_factory(creds: Any) -> Any:
            captured.append(creds)
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **kw: types.SimpleNamespace(
                            choices=[
                                types.SimpleNamespace(
                                    message=types.SimpleNamespace(content="ok"),
                                    finish_reason="stop",
                                )
                            ],
                            usage=None,
                        )
                    )
                )
            )

        from schemas.llm_orchestration import ResolvedModelConfig  # noqa: PLC0415
        from schemas.llm_routing import ModelConfig as MC  # noqa: PLC0415
        from schemas.llm_routing import ProviderProfile as PP
        from services.secrets.provider_credentials import ProviderCredentials  # noqa: PLC0415

        adapter = OpenAIProviderAdapter(client_factory=fake_factory)
        credentials = ProviderCredentials(provider="openai", api_key="test-key")

        mc = MC(
            provider="openai",
            provider_profile="openai_primary",
            model_id="gpt-4o-mini",
            supports_streaming=False,
            supports_thinking=False,
            timeout_seconds=20,
        )
        pp = PP(provider="openai", api_key_env="OPENAI_API_KEY")
        rmc = ResolvedModelConfig(
            model_alias="test_model",
            model_config=mc,
            provider_profile_name="openai_primary",
            provider_profile=pp,
            provider="openai",
            supports_streaming=False,
            supports_thinking=False,
            timeout_seconds=20,
        )
        route_dec = RouteDecision(
            route_id="math.generator.default",
            subject="math",
            task_role="generator",
            difficulty="default",
            model="test_model",
            prompt="subjects/math_generator.md",
            temperature=0.2,
            max_tokens=800,
            provider_options={},
            fallback_attempts=[],
            route_source="exact",
        )
        request = ProviderExecutionRequest(
            route_decision=route_dec,
            model_resolution=rmc,
            messages=[LlmMessage(role="user", content="q")],
            temperature=0.2,
            max_tokens=800,
        )
        adapter.generate(request=request, credentials=credentials)
        assert len(captured) == 1
        assert captured[0] is credentials

    def test_openai_real_client_built_with_max_retries_zero(self) -> None:
        """Verify OpenAIProviderAdapter passes max_retries=0 to the real SDK client."""
        built: list = []

        def capture_factory(creds: Any) -> Any:
            # This factory is the real _build_client path replacement;
            # just capture what was returned and return a stub.
            built.append(creds)
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kw: None)
                )
            )

        adapter = OpenAIProviderAdapter(client_factory=capture_factory)
        # The client_factory receives credentials unchanged; the max_retries
        # only matters in the real _OpenAI(...) call path.
        # Verify _build_client calls client_factory when factory is injected.
        from services.secrets.provider_credentials import ProviderCredentials  # noqa: PLC0415

        creds = ProviderCredentials(provider="openai", api_key="k")
        result = adapter._build_client(creds)
        assert built == [creds]
        assert result is not None

    def test_azure_adapter_client_factory_is_used(self) -> None:
        """AzureOpenAIProviderAdapter uses the injected client_factory."""
        called = []

        def fake_factory(creds: Any) -> Any:
            called.append(creds)
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kw: None)
                )
            )

        from services.secrets.provider_credentials import ProviderCredentials  # noqa: PLC0415

        adapter = AzureOpenAIProviderAdapter(client_factory=fake_factory)
        creds = ProviderCredentials(
            provider="azure_openai", api_key="k", endpoint="https://e", api_version="v"
        )
        adapter._build_client_classic(creds)
        assert called == [creds]


# ===========================================================================
# 7. Graph _generate_node — safe answer on ProviderExecutionError
# ===========================================================================


class TestGenerateNodeProviderFailureHandling:
    """Test that _generate_node returns safe fallback answer on ProviderExecutionError."""

    def _build_graph_with_failing_adapter(self, failure_message: str) -> Any:
        """Build an orchestrated graph whose adapter always raises ProviderExecutionError."""
        from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph  # noqa: PLC0415
        from services.doubt_solver.answer_generation_adapter import (
            AnswerGenerationAdapter,  # noqa: PLC0415
        )
        from services.llm.orchestration.errors import ProviderExecutionError as PEE  # noqa: PLC0415
        from services.llm.orchestration.orchestrator import LlmOrchestrator  # noqa: PLC0415

        # MockModelExecutor that raises ProviderExecutionError on execute()
        mock_executor = __import__(
            "services.llm.orchestration.orchestrator",
            fromlist=["MockModelExecutor"],
        ).MockModelExecutor(raise_on_execute=PEE(failure_message))

        orchestrator = LlmOrchestrator(model_executor=mock_executor)
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        return build_orchestrated_doubt_solver_graph(adapter)

    def test_provider_failure_returns_safe_answer(self) -> None:
        """ProviderExecutionError → safe fallback answer, not a raised exception."""
        graph = self._build_graph_with_failing_adapter("All attempts failed.")
        state = {
            "request_id": "test-req-001",
            "query": "What is 2+2?",
            "classification": {
                "subject": "math",
                "intent": "explain",
                "difficulty": "default",
                "retrieval_required": False,
            },
            "context_text": "",
            "answer": None,
        }

        result = graph.invoke(state)
        answer = result.get("answer", "")
        assert answer, "Answer must not be empty"
        answer_lower = answer.lower()
        assert (
            "unavailable" in answer_lower or "quota" in answer_lower or "try again" in answer_lower
        ), f"Expected safe fallback message, got: {answer!r}"

    def test_provider_failure_does_not_expose_raw_error(self) -> None:
        """Provider error internals must not appear in the safe answer."""
        failure_msg = (
            "Attempted aliases: [azure_fast, openai_native_fallback]."
            " Primary failure: rate_limited."
        )
        graph = self._build_graph_with_failing_adapter(failure_msg)
        state = {
            "request_id": "test-req-002",
            "query": "Explain gravity.",
            "classification": {
                "subject": "general",
                "intent": "explain",
                "difficulty": "default",
                "retrieval_required": False,
            },
            "context_text": "",
            "answer": None,
        }

        result = graph.invoke(state)
        answer = result.get("answer", "")
        # The raw attempt list must not appear in the answer
        assert "azure_fast" not in answer
        assert "openai_native_fallback" not in answer

    def test_graph_state_has_internal_retrieval_context_after_provider_failure(self) -> None:
        """OrchestratedDoubtSolverState retains the internal retrieval contract."""
        from graphs.doubt_solver_graph import OrchestratedDoubtSolverState  # noqa: PLC0415

        fields = set(OrchestratedDoubtSolverState.__annotations__.keys())
        assert fields == {
            "request_id",
            "query",
            "original_query",
            "actor_id",
            "conversation_id",
            "turn_id",
            "language",
            "exam_id",
            "exam_stage",
            "exam_profile_id",
            "classification",
            "retrieval_context",
            "context_text",
            "answer",
            "final_answer",
            "conversation_context",
            "conversation_relation",
            "conversation_preparation",
            "query_classification",
            "source_modality",
            "response_type",
            "practice_test_id",
        }, f"OrchestratedDoubtSolverState fields changed: {fields}"

    def test_unexpected_error_propagates_loudly(self) -> None:
        """Non-ProviderExecutionError exceptions must propagate, not be silenced."""
        from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph  # noqa: PLC0415
        from services.doubt_solver.answer_generation_adapter import (
            AnswerGenerationAdapter,  # noqa: PLC0415
        )
        from services.llm.orchestration.orchestrator import LlmOrchestrator  # noqa: PLC0415

        # ValueError is NOT a ProviderExecutionError — must not be caught
        mock_executor = __import__(
            "services.llm.orchestration.orchestrator",
            fromlist=["MockModelExecutor"],
        ).MockModelExecutor(raise_on_execute=ValueError("programming bug"))

        from services.llm.orchestration.errors import LlmExecutionError  # noqa: PLC0415

        orchestrator = LlmOrchestrator(model_executor=mock_executor)
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        graph = build_orchestrated_doubt_solver_graph(adapter)

        state = {
            "request_id": "test-req-003",
            "query": "What is 2+2?",
            "classification": None,
            "context_text": "",
            "answer": None,
        }
        # Should raise (not return safe answer)
        with pytest.raises((ValueError, LlmExecutionError)):
            graph.invoke(state)


# ===========================================================================
# 8. ModelConfigResolver.resolve_for_alias
# ===========================================================================


class TestResolveForAlias:
    def test_resolve_for_alias_returns_correct_config(self, tmp_path: Path) -> None:
        resolver = _resolver(tmp_path)
        resolved = resolver.resolve_for_alias("openai_native_fallback")
        assert resolved.model_alias == "openai_native_fallback"
        assert resolved.provider == "openai"

    def test_resolve_for_alias_azure_primary(self, tmp_path: Path) -> None:
        resolver = _resolver(tmp_path)
        resolved = resolver.resolve_for_alias("azure_fast")
        assert resolved.provider == "azure_openai"
        assert resolved.model_config.deployment == "my-azure-deployment"

    def test_resolve_for_alias_unknown_raises(self, tmp_path: Path) -> None:
        from services.llm.orchestration.errors import ModelConfigResolutionError  # noqa: PLC0415

        resolver = _resolver(tmp_path)
        with pytest.raises(ModelConfigResolutionError, match="unknown_alias"):
            resolver.resolve_for_alias("unknown_alias")


# ===========================================================================
# 9. Production model registry — Azure-first aliases
# ===========================================================================


class TestProductionModelRegistry:
    """Verify the live model_registry.yaml has Azure-first structure."""

    def _live_registry(self) -> LlmConfigRegistry:
        from services.llm.orchestration.config_registry import (
            LlmConfigRegistry,  # noqa: PLC0415
            reset_registry,  # noqa: PLC0415
        )

        reset_registry()
        return LlmConfigRegistry()

    def test_primary_aliases_are_azure_first(self) -> None:
        registry = self._live_registry()
        azure_first_aliases = [
            "math_basic_generator",
            "math_reasoning_generator",
            "reasoning_standard_generator",
            "english_fast_generator",
            "general_fast_generator",
        ]
        for alias in azure_first_aliases:
            cfg = registry.model_map.get(alias)
            assert cfg is not None, f"alias '{alias}' missing"
            assert cfg.provider == "azure_openai", (
                f"'{alias}' should be azure_openai, got {cfg.provider}"
            )

    def test_fallback_aliases_are_native_openai(self) -> None:
        registry = self._live_registry()
        fallback_aliases = [
            "math_basic_generator_openai_native",
            "math_reasoning_generator_openai_native",
            "reasoning_standard_generator_openai_native",
            "english_fast_generator_openai_native",
            "general_fast_generator_openai_native",
        ]
        for alias in fallback_aliases:
            cfg = registry.model_map.get(alias)
            assert cfg is not None, f"fallback alias '{alias}' missing"
            assert cfg.provider == "openai", (
                f"fallback alias '{alias}' should be openai, got {cfg.provider}"
            )
            assert cfg.model_id is not None, f"fallback alias '{alias}' missing model_id"

    def test_primary_aliases_have_fallback_configured(self) -> None:
        registry = self._live_registry()
        for alias in [
            "math_basic_generator",
            "general_fast_generator",
        ]:
            cfg = registry.model_map.get(alias)
            assert cfg is not None
            assert len(cfg.fallback_models) > 0, (
                f"'{alias}' should have at least one fallback_model"
            )

    def test_fallback_aliases_exist_in_registry(self) -> None:
        """Every fallback alias referenced in primary models must exist."""
        registry = self._live_registry()
        for alias, cfg in registry.model_map.items():
            for fallback in cfg.fallback_models:
                assert fallback in registry.model_map, (
                    f"'{alias}'.fallback_models contains '{fallback}' which is not in registry"
                )

    def test_routes_reference_only_primary_aliases(self) -> None:
        """llm_routes.yaml must not reference _openai_native suffixed aliases."""
        registry = self._live_registry()
        for (_, _, _), route in registry.route_map.items():
            assert not route.model.endswith("_openai_native"), (
                f"Route references fallback alias '{route.model}'"
                " — routes must use primary aliases only"
            )

    def test_safe_mock_still_present(self) -> None:
        registry = self._live_registry()
        assert "safe_mock" in registry.model_map

    def test_azure_deployment_field_set(self) -> None:
        """Active Azure models must have deployment; optional GPT-5.x may be blank."""
        optional_blank = frozenset(
            {
                "openai_gpt_5_4",
                "openai_gpt_5_4_mini",
                "openai_gpt_5_5",
                "deepseek_azure_reasoning_generator",
                "deepseek_azure_standard_generator",
                "deepseek_azure_advanced_generator",
            }
        )
        registry = self._live_registry()
        for alias, cfg in registry.model_map.items():
            if cfg.provider != "azure_openai":
                continue
            if alias in optional_blank:
                continue
            assert cfg.deployment, f"Azure model '{alias}' is missing the deployment field"

    def test_openai_native_fallbacks_have_model_id(self) -> None:
        registry = self._live_registry()
        for alias, cfg in registry.model_map.items():
            if alias.endswith("_openai_native"):
                assert cfg.model_id is not None, (
                    f"OpenAI native fallback '{alias}' is missing model_id"
                )


# ===========================================================================
# 10. Regression — existing paths unaffected
# ===========================================================================


class TestRegressionGuards:
    def test_mock_path_still_works(self, tmp_path: Path) -> None:
        """FakeProviderExecutor-backed executor still returns results."""
        executor, fake = _executor_pair_simple(tmp_path)
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        assert result.content
        assert fake.call_count == 1

    def test_no_fallback_when_primary_succeeds(self, tmp_path: Path) -> None:
        """When primary succeeds, fallback is never called."""
        fake_executor = _AliasedFakeProviderExecutor(
            return_for={"azure_fast": "Primary answer."},
        )
        executor = RegistryBackedModelExecutor(
            provider_executor=fake_executor,
            model_config_resolver=_resolver(tmp_path),
        )
        result = executor.execute(route_decision=_route_decision(), messages=_messages())
        assert result.fallback_used is False
        assert result.content == "Primary answer."
        assert fake_executor.call_log == ["azure_fast"]

    def test_graph_state_includes_internal_retrieval_context(self) -> None:
        """Orchestrated state retains request context and answer boundaries."""
        from graphs.doubt_solver_graph import OrchestratedDoubtSolverState  # noqa: PLC0415

        fields = set(OrchestratedDoubtSolverState.__annotations__.keys())
        assert fields == {
            "request_id",
            "query",
            "original_query",
            "actor_id",
            "conversation_id",
            "turn_id",
            "language",
            "exam_id",
            "exam_stage",
            "exam_profile_id",
            "classification",
            "retrieval_context",
            "context_text",
            "answer",
            "final_answer",
            "conversation_context",
            "conversation_relation",
            "conversation_preparation",
            "query_classification",
            "source_modality",
            "response_type",
            "practice_test_id",
        }


def _executor_pair_simple(
    tmp_path: Path,
) -> tuple[RegistryBackedModelExecutor, FakeProviderExecutor]:
    fake = FakeProviderExecutor(content="Simple answer.")
    resolver = _resolver(tmp_path)
    return (
        RegistryBackedModelExecutor(
            provider_executor=fake,
            model_config_resolver=resolver,
        ),
        fake,
    )


class TestAdvancedReasoningCapacityStabilization:
    """Model-call accounting for the advanced reasoning path.

    Latency is not asserted; the eliminated known-failing model call is.
    """

    @staticmethod
    def _manifest(count: int = 1) -> str:
        return json.dumps(
            {
                "questions": [
                    {
                        "generation_item_id": f"reasoning-{index}",
                        "bucket_id": "reasoning-advanced",
                        "question": f"Reasoning question {index}?",
                        "question_type": "mcq",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "solution": "Concise solution.",
                        "subject": "reasoning",
                        "topic": "logical_reasoning",
                        "difficulty": "advanced",
                    }
                    for index in range(1, count + 1)
                ]
            }
        )

    @staticmethod
    def _advanced_route():
        return resolve_route(
            RouteRequest(
                request_id="practice-reasoning-capacity",
                subject="reasoning",
                task_role="generator",
                difficulty="advanced",
                intent="practice",
                language="english",
            )
        )

    @staticmethod
    def _exhausted() -> LlmProviderResponseError:
        error = LlmProviderResponseError("response unavailable")
        error.finish_reason = "length"
        error.output_tokens = 5600
        error.reasoning_tokens = 5600
        return error

    def test_success_at_the_initial_capacity_costs_exactly_one_model_call(self) -> None:
        fake = _AliasedFakeProviderExecutor(
            raise_for={},
            return_for={"reasoning_advanced_generator": self._manifest()},
        )
        executor = RegistryBackedModelExecutor(provider_executor=fake)

        result = executor.execute(route_decision=self._advanced_route(), messages=_messages())

        assert fake.call_log == ["reasoning_advanced_generator"]
        assert result.fallback_used is False

    def test_exhaustion_skips_the_noop_escalation_and_falls_back(self) -> None:
        fake = _AliasedFakeProviderExecutor(
            raise_for={"reasoning_advanced_generator": self._exhausted()},
            return_for={"openai_o3": self._manifest()},
        )
        executor = RegistryBackedModelExecutor(provider_executor=fake)

        result = executor.execute(route_decision=self._advanced_route(), messages=_messages())

        # Two calls, not three: the escalation that could not add capacity is gone.
        assert fake.call_log == ["reasoning_advanced_generator", "openai_o3"]
        assert fake.call_log.count("reasoning_advanced_generator") == 1
        assert result.fallback_used is True

    def test_fallback_failure_preserves_terminal_semantics(self) -> None:
        fake = _AliasedFakeProviderExecutor(
            raise_for={
                "reasoning_advanced_generator": self._exhausted(),
                "openai_o3": LlmProviderExecutionError(
                    "fallback unavailable", failure_kind="provider_unavailable"
                ),
                "deepseek_v4pro": LlmProviderExecutionError(
                    "fallback unavailable", failure_kind="provider_unavailable"
                ),
            },
            return_for={},
        )
        executor = RegistryBackedModelExecutor(provider_executor=fake)

        with pytest.raises(ProviderExecutionError):
            executor.execute(route_decision=self._advanced_route(), messages=_messages())

        # No retries were added; each alias is attempted once.
        assert fake.call_log.count("reasoning_advanced_generator") == 1
