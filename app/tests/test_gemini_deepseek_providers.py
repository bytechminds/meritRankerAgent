"""
app/tests/test_gemini_deepseek_providers.py
--------------------------------------------
Unit tests for Gemini and DeepSeek provider wiring (no real API calls).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import get_settings
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import RouteDecision
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import ProviderAdapterExecutor
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.gemini_provider import (
    GeminiProviderAdapter,
    _active_native_schema_builder,
)
from services.llm.providers.openai_compatible_adapter import (
    DeepSeekProviderAdapter,
    classify_gemini_error,
)
from services.llm.providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.provider_credentials import ProviderCredentialResolver, ProviderCredentials


def _fake_completion(content: str = "Answer text.", finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


def _route_decision(
    *,
    model: str,
    route_id: str = "general.generator.gemini_test",
) -> RouteDecision:
    return RouteDecision(
        route_id=route_id,
        subject="general",
        task_role="generator",
        difficulty="gemini_test",
        model=model,
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=900,
        provider_options={},
        fallback=["default", "safe_mock"],
        fallback_attempts=[],
        route_source="exact",
    )


def _make_request(
    tmp_path: Path,
    yaml_text: str,
    route: RouteDecision,
) -> ProviderExecutionRequest:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    registry = LlmConfigRegistry(yaml_path=yaml_path)
    resolver = ModelConfigResolver(registry=registry)
    resolution = resolver.resolve(route)
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=resolution,
        messages=[LlmMessage(role="user", content="Explain photosynthesis.")],
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        provider_options={},
    )


_GEMINI_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: general_fast_generator
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 900
          gemini_test:
            model: gemini_flash_text
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 900
            fallback:
              - default
              - safe_mock
    models:
      gemini_flash_text:
        provider: gemini
        provider_profile: gemini_primary
        model_id: gemini-2.5-flash
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 30
        fallback_models:
          - general_fast_generator
      general_fast_generator:
        provider: azure_openai
        provider_profile: azure_foundry_v1
        deployment: gpt-4.1-mini
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 30
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
    provider_profiles:
      gemini_primary:
        provider: gemini
        api_key_env: GEMINI_API_KEY
        optional_api_key: true
      azure_foundry_v1:
        provider: azure_openai
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_key_env: AZURE_OPENAI_API_KEY
      local_mock:
        provider: mock
""")


class TestProviderFactoryGeminiDeepSeek:
    def test_gemini_resolves(self) -> None:
        factory = ProviderAdapterFactory()
        assert isinstance(factory.get_provider("gemini"), GeminiProviderAdapter)

    def test_deepseek_resolves(self) -> None:
        factory = ProviderAdapterFactory()
        assert isinstance(factory.get_provider("deepseek"), DeepSeekProviderAdapter)


class TestGeminiDeepSeekConfigDefaults:
    def test_gemini_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as config_module

        config_module._settings = None
        settings = get_settings()
        assert settings.gemini_default_model == "gemini-2.5-flash-lite"
        assert settings.gemini_text_model == "gemini-2.5-flash"
        assert settings.gemini_timeout_seconds == 30

    def test_deepseek_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as config_module

        config_module._settings = None
        settings = get_settings()
        assert settings.deepseek_default_model == "deepseek-chat"
        assert settings.deepseek_reasoner_model == "deepseek-reasoner"

    def test_missing_api_keys_do_not_break_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as config_module

        config_module._settings = None
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        settings = get_settings()
        assert settings.gemini_api_key == ""
        assert settings.deepseek_api_key == ""


class TestGeminiAdapterExecution:
    def test_request_intelligence_uses_the_canonical_native_schema(self, tmp_path: Path) -> None:
        from schemas.practice_request_intelligence import PRACTICE_REQUEST_INTELLIGENCE_SCHEMA

        route = _route_decision(model="gemini_flash_text").model_copy(
            update={
                "task_role": "request_intelligence",
                "prompt": "practice_generation/request_intelligence.md",
            }
        )
        request = _make_request(tmp_path, _GEMINI_YAML, route)
        schema_builder = _active_native_schema_builder(request)
        assert schema_builder is not None
        assert schema_builder() == PRACTICE_REQUEST_INTELLIGENCE_SCHEMA

        other_prompt = request.model_copy(
            update={"route_decision": route.model_copy(update={"prompt": "other.md"})}
        )
        assert _active_native_schema_builder(other_prompt) is None

    def test_classifier_uses_native_structured_output_schema(self) -> None:
        captured: dict[str, object] = {}

        class Models:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    text='{"intent":"solve_question","subject":"math","confidence":0.96}',
                    parsed=None,
                    candidates=[SimpleNamespace(finish_reason="STOP")],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        candidates_token_count=25,
                        total_token_count=125,
                    ),
                )

        registry = LlmConfigRegistry()
        route = RouteDecision(
            route_id="general.classifier.default",
            subject="general",
            task_role="classifier",
            difficulty="default",
            model="doubt_solver_classifier_gemini",
            prompt="classification_semantics.md",
            temperature=0.1,
            max_tokens=650,
            provider_options={},
            fallback=[],
            fallback_attempts=[],
            route_source="exact",
        )
        request = ProviderExecutionRequest(
            route_decision=route,
            model_resolution=ModelConfigResolver(registry=registry).resolve(route),
            messages=[
                LlmMessage(role="system", content="Classify without solving."),
                LlmMessage(role="user", content="What is 20% of 500?"),
            ],
            temperature=0.1,
            max_tokens=650,
            provider_options={},
        )
        result = GeminiProviderAdapter(
            client_factory=lambda _credentials, _timeout: SimpleNamespace(
                models=Models()
            )
        ).generate(
            request=request,
            credentials=ProviderCredentials(provider="gemini", api_key="test-key"),
        )

        config = captured["config"]
        schema = config.response_json_schema
        assert captured["model"] == "gemini-3.1-flash-lite"
        assert config.response_mime_type == "application/json"
        assert "classification_source" not in schema["properties"]
        assert "requires_recent_conversation" not in schema["properties"]
        assert set(schema["required"]) == set(schema["properties"])
        assert result.provider == "gemini"
        assert result.total_tokens == 125
        assert result.metadata["native_structured_output"] is True

    def test_generate_returns_normalized_response(self, tmp_path: Path) -> None:
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = SimpleNamespace(
            text="Gemini answer.",
            parsed=None,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
                total_token_count=30,
            ),
        )

        adapter = GeminiProviderAdapter(
            client_factory=lambda _creds, _timeout: fake_client,
        )
        request = _make_request(
            tmp_path,
            _GEMINI_YAML,
            _route_decision(model="gemini_flash_text"),
        )
        creds = ProviderCredentials(provider="gemini", api_key="test-key")

        result = adapter.generate(request=request, credentials=creds)

        assert result.content == "Gemini answer."
        assert result.provider == "gemini"
        assert result.finish_reason == "stop"
        assert result.normalized_finish_reason == "completed"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        call_kwargs = fake_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["config"].max_output_tokens == 900

    def test_missing_api_key_raises_provider_not_configured(self, tmp_path: Path) -> None:
        adapter = GeminiProviderAdapter(client_factory=lambda _c, _t: MagicMock())
        request = _make_request(
            tmp_path,
            _GEMINI_YAML,
            _route_decision(model="gemini_flash_text"),
        )
        creds = ProviderCredentials(provider="gemini", api_key=None)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=request, credentials=creds)

        assert exc_info.value.failure_kind == "provider_not_configured"


class TestDeepSeekAdapterExecution:
    def test_length_response_without_final_content_preserves_safe_usage(
        self,
    ) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="private reasoning must never be surfaced",
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=935,
                completion_tokens=2600,
                total_tokens=3535,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2573),
            ),
        )
        route = RouteDecision(
            route_id="math.generator.advanced",
            subject="math",
            task_role="generator",
            difficulty="advanced",
            model="math_advanced_generator",
            prompt="subjects/math_generator.md",
            temperature=0.15,
            max_tokens=3600,
            provider_options={},
            fallback=[],
            fallback_attempts=[],
            route_source="exact",
            intent="practice",
        )
        request = ProviderExecutionRequest(
            route_decision=route,
            model_resolution=ModelConfigResolver(
                registry=LlmConfigRegistry()
            ).resolve(route),
            messages=[LlmMessage(role="user", content="Generate one question.")],
            temperature=route.temperature,
            max_tokens=route.max_tokens,
            provider_options={},
        )
        adapter = DeepSeekProviderAdapter(client_factory=lambda _c, _t: fake_client)

        with pytest.raises(LlmProviderResponseError) as raised:
            adapter.generate(
                request=request,
                credentials=ProviderCredentials(provider="deepseek", api_key="test-key"),
            )

        error = raised.value
        assert error.finish_reason == "length"
        assert error.normalized_finish_reason == "output_token_exhausted"
        assert error.output_tokens == 2600
        assert error.reasoning_tokens == 2573
        assert "private reasoning" not in str(error)
        assert fake_client.chat.completions.create.call_args.kwargs["max_tokens"] == 3600

    def test_missing_api_key_raises_provider_not_configured(self, tmp_path: Path) -> None:
        adapter = DeepSeekProviderAdapter(client_factory=lambda _c, _t: MagicMock())
        yaml = textwrap.dedent("""\
            version: 1
            routes:
              math:
                generator:
                  default:
                    model: math_basic_generator
                    prompt: subjects/math_generator.md
                    temperature: 0.2
                    max_tokens: 900
                  deepseek_test:
                    model: deepseek_reasoning_generator
                    prompt: subjects/math_generator.md
                    temperature: 0.15
                    max_tokens: 2600
            models:
              deepseek_reasoning_generator:
                provider: deepseek
                provider_profile: deepseek_primary
                model_id: deepseek-reasoner
                supports_streaming: true
                supports_thinking: true
                timeout_seconds: 90
              math_basic_generator:
                provider: azure_openai
                provider_profile: azure_foundry_v1
                deployment: gpt-4.1-mini
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 30
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
            provider_profiles:
              deepseek_primary:
                provider: deepseek
                api_key_env: DEEPSEEK_API_KEY
                optional_api_key: true
              azure_foundry_v1:
                provider: azure_openai
                endpoint_env: AZURE_OPENAI_ENDPOINT
                api_key_env: AZURE_OPENAI_API_KEY
              local_mock:
                provider: mock
        """)
        request = _make_request(
            tmp_path,
            yaml,
            _route_decision(
                model="deepseek_reasoning_generator",
                route_id="math.generator.deepseek_test",
            ),
        )
        creds = ProviderCredentials(provider="deepseek", api_key=None)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=request, credentials=creds)

        assert exc_info.value.failure_kind == "provider_not_configured"


class TestGeminiDeepSeekRegistry:
    def test_production_registry_loads_new_aliases(self) -> None:
        reg = LlmConfigRegistry()
        for alias in (
            "gemini_flash_lite_text",
            "gemini_flash_text",
            "gemini_image_extractor",
            "deepseek_standard_generator",
            "deepseek_reasoning_generator",
            "deepseek_advanced_generator",
        ):
            assert alias in reg.model_map, f"missing alias {alias}"

    def test_active_routes_still_point_to_azure_aliases(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "default")
        assert route is not None
        assert route.model == "math_basic_generator"
        assert reg.model_map[route.model].provider == "azure_openai"


class TestGeminiDeepSeekFallback:
    def test_provider_not_configured_is_fallback_eligible(self) -> None:
        assert "provider_not_configured" in FALLBACK_ELIGIBLE_FAILURE_KINDS

    def test_safety_blocked_is_not_fallback_eligible(self) -> None:
        assert "safety_blocked" not in FALLBACK_ELIGIBLE_FAILURE_KINDS


class TestGeminiErrorMapping:
    def test_safety_blocked_classification(self) -> None:
        exc = Exception("Request blocked due to safety settings")
        assert classify_gemini_error(exc) == "safety_blocked"


class TestProviderAdapterExecutorMissingKey:
    def test_gemini_missing_env_reaches_adapter_not_secret_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        request = _make_request(
            tmp_path,
            _GEMINI_YAML,
            _route_decision(model="gemini_flash_text"),
        )
        factory = ProviderAdapterFactory()
        cred_resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            executor.execute(request)

        assert exc_info.value.failure_kind == "provider_not_configured"
