"""
app/tests/test_llm_payload_shaping.py
---------------------------------------
Unit tests for provider/model capability payload shaping (Azure reasoning fix).
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import ModelConfig, RouteDecision
from services.llm.providers.azure_openai_provider import (
    AzureOpenAIProviderAdapter,
    _azure_error_diagnostics,
    _is_unsupported_parameter_error,
    _log_azure_call_failure,
)
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderResponseError,
)
from services.llm.providers.payload_shaping import (
    build_azure_openai_chat_completion_kwargs,
    effective_supports_temperature,
    effective_token_budget_param,
)
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.model_config_resolver import ModelConfigResolver
from services.secrets.provider_credentials import ProviderCredentials

_REASONING_YAML = textwrap.dedent("""\
    version: 1
    routes:
      reasoning:
        generator:
          advanced:
            model: reasoning_advanced_generator
            prompt: subjects/reasoning_generator.md
            temperature: 0.3
            max_tokens: 3200
            provider_options:
              thinking: true
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 1
      reasoning_advanced_generator:
        provider: azure_openai
        provider_profile: azure_primary
        deployment: o4-mini
        supports_streaming: true
        supports_thinking: false
        supports_reasoning: true
        reasoning_effort: medium
        timeout_seconds: 45
      openai_gpt_4_1_mini:
        provider: azure_openai
        provider_profile: azure_primary
        deployment: gpt-4.1-mini
        supports_streaming: true
        supports_thinking: false
        supports_reasoning: false
        timeout_seconds: 25
    provider_profiles:
      local_mock:
        provider: mock
      azure_primary:
        provider: azure_openai
        api_key_env: AZURE_OPENAI_API_KEY
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_version_env: AZURE_OPENAI_API_VERSION
""")


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(_REASONING_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _route_decision(model: str = "reasoning_advanced_generator") -> RouteDecision:
    return RouteDecision(
        route_id="reasoning.generator.advanced",
        subject="reasoning",
        task_role="generator",
        difficulty="advanced",
        model=model,
        prompt="subjects/reasoning_generator.md",
        temperature=0.3,
        max_tokens=3200,
        provider_options={"thinking": True},
        fallback_attempts=[],
        route_source="exact",
    )


def _make_request(
    tmp_path: Path,
    *,
    model: str = "reasoning_advanced_generator",
    provider_options: dict | None = None,
) -> ProviderExecutionRequest:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))
    route = _route_decision(model=model)
    opts = (
        {"thinking": True}
        if provider_options is None and model == "reasoning_advanced_generator"
        else (provider_options or {})
    )
    route = route.model_copy(update={"provider_options": opts})
    model_resolution = resolver.resolve(route)
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=model_resolution,
        messages=[LlmMessage(role="user", content="Solve this puzzle.")],
        temperature=0.3,
        max_tokens=3200,
        provider_options=dict(opts),
    )


class TestModelConfigReasoningCapabilities:
    def test_azure_reasoning_model_gets_completion_token_budget(self) -> None:
        cfg = ModelConfig(
            provider="azure_openai",
            provider_profile="azure_foundry_v1",
            deployment="o4-mini",
            supports_streaming=True,
            supports_reasoning=True,
            timeout_seconds=45,
        )
        assert effective_token_budget_param(cfg) == "max_completion_tokens"
        assert effective_supports_temperature(cfg) is False

    def test_azure_standard_model_keeps_max_tokens_and_temperature(self) -> None:
        cfg = ModelConfig(
            provider="azure_openai",
            provider_profile="azure_foundry_v1",
            deployment="gpt-4.1-mini",
            supports_streaming=True,
            supports_reasoning=False,
            timeout_seconds=25,
        )
        assert effective_token_budget_param(cfg) == "max_tokens"
        assert effective_supports_temperature(cfg) is True

    def test_production_registry_o4_mini_capabilities(self) -> None:
        registry = LlmConfigRegistry()
        cfg = registry.get_model("openai_o4_mini")
        assert cfg is not None
        assert effective_token_budget_param(cfg) == "max_completion_tokens"
        assert effective_supports_temperature(cfg) is False
        assert cfg.supports_reasoning is True

    def test_production_registry_gpt_4_1_mini_unchanged(self) -> None:
        registry = LlmConfigRegistry()
        cfg = registry.get_model("openai_gpt_4_1_mini")
        assert cfg is not None
        assert effective_token_budget_param(cfg) == "max_tokens"
        assert effective_supports_temperature(cfg) is True


class TestAzureReasoningPayloadShaping:
    def test_o4_mini_uses_max_completion_tokens_not_max_tokens(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path)
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="o4-mini",
            stream=False,
        )
        assert kwargs["max_completion_tokens"] == 3200
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert meta.token_budget_param_used == "max_completion_tokens"
        assert "max_tokens" in meta.dropped_params
        assert "temperature" in meta.dropped_params

    def test_o4_mini_drops_thinking_and_reasoning_effort_by_default(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path)
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="o4-mini",
        )
        assert "thinking" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert meta.reasoning_param_sent is False
        assert "reasoning_effort" in meta.dropped_params

    def test_o4_mini_sends_route_reasoning_effort(self, tmp_path: Path) -> None:
        request = _make_request(
            tmp_path,
            provider_options={"reasoning_effort": "low"},
        )
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="o4-mini",
        )
        assert kwargs["reasoning_effort"] == "low"
        assert meta.reasoning_param_sent is True
        assert "reasoning_effort" not in meta.dropped_params

    def test_gpt_4_1_mini_preserves_max_tokens_and_temperature(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, model="openai_gpt_4_1_mini")
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="gpt-4.1-mini",
        )
        assert kwargs["max_tokens"] == 3200
        assert kwargs["temperature"] == 0.3
        assert "max_completion_tokens" not in kwargs
        assert meta.token_budget_param_used == "max_tokens"


class FakeAzureOpenAIClient:
    def __init__(
        self,
        *,
        content: object = "Answer.",
        refusal: str | None = None,
        usage: object | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.received_kwargs: dict = {}

        def _create(**kwargs):  # noqa: ANN202
            self.received_kwargs = kwargs
            message = types.SimpleNamespace(content=content, refusal=refusal)
            choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
            resolved_usage = (
                usage
                if usage is not None
                else types.SimpleNamespace(prompt_tokens=1, completion_tokens=2)
            )
            return types.SimpleNamespace(choices=[choice], usage=resolved_usage)

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )


class TestAzureAdapterPayloadIntegration:
    def test_adapter_sends_reasoning_safe_payload_for_o4_mini(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient()
        adapter = AzureOpenAIProviderAdapter(
            client_factory=lambda _creds: fake,
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        adapter.generate(request=_make_request(tmp_path), credentials=creds)
        assert fake.received_kwargs["max_completion_tokens"] == 3200
        assert "max_tokens" not in fake.received_kwargs
        assert "temperature" not in fake.received_kwargs

    def test_adapter_preserves_standard_gpt_payload(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient()
        adapter = AzureOpenAIProviderAdapter(
            client_factory=lambda _creds: fake,
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        adapter.generate(
            request=_make_request(tmp_path, model="openai_gpt_4_1_mini"),
            credentials=creds,
        )
        assert fake.received_kwargs["max_tokens"] == 3200
        assert fake.received_kwargs["temperature"] == 0.3
        assert "max_completion_tokens" not in fake.received_kwargs

    def test_adapter_normalizes_content_parts(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient(
            content=[
                {"type": "text", "text": '{"status":"MATCH",'},
                types.SimpleNamespace(text='"reason":"ok"}'),
            ]
        )
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )

        result = adapter.generate(request=_make_request(tmp_path), credentials=creds)

        assert result.content == '{"status":"MATCH","reason":"ok"}'

    def test_adapter_allows_missing_usage_metadata(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient(content="Answer.", usage=types.SimpleNamespace())
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )

        result = adapter.generate(request=_make_request(tmp_path), credentials=creds)

        assert result.content == "Answer."
        assert result.usage_source == "unavailable"

    @pytest.mark.parametrize(
        ("content", "refusal", "failure_kind"),
        [
            (None, None, "empty_answer"),
            ("", None, "empty_answer"),
            (None, "safety refusal", "safety_blocked"),
        ],
    )
    def test_adapter_normalizes_empty_or_refused_response(
        self,
        tmp_path: Path,
        content: object,
        refusal: str | None,
        failure_kind: str,
    ) -> None:
        fake = FakeAzureOpenAIClient(content=content, refusal=refusal)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )

        with pytest.raises(LlmProviderResponseError) as error:
            adapter.generate(request=_make_request(tmp_path), credentials=creds)

        assert error.value.provider_usage is not None
        assert error.value.failure_kind == failure_kind

    def test_adapter_retains_safe_usage_for_exhausted_reasoning_response(
        self, tmp_path: Path
    ) -> None:
        fake = FakeAzureOpenAIClient(
            content=None,
            finish_reason="length",
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=3200,
                completion_tokens_details=types.SimpleNamespace(reasoning_tokens=3200),
            ),
        )
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )

        with pytest.raises(LlmProviderResponseError) as raised:
            adapter.generate(request=_make_request(tmp_path), credentials=creds)

        assert (
            raised.value.finish_reason,
            raised.value.output_tokens,
            raised.value.reasoning_tokens,
        ) == ("length", 3200, 3200)


class TestAzureUnsupportedParameterErrors:
    def test_unsupported_parameter_classified_as_fallback_eligible(self) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: max_tokens",
                    "param": "max_tokens",
                    "type": "invalid_request_error",
                }
            },
        )
        assert _is_unsupported_parameter_error(exc) is True
        assert "unsupported_parameter" in FALLBACK_ELIGIBLE_FAILURE_KINDS

    def test_error_diagnostics_includes_safe_param_field(self) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: temperature",
                    "param": "temperature",
                    "type": "invalid_request_error",
                }
            },
        )
        diag = _azure_error_diagnostics(exc)
        assert diag["status_code"] == 400
        assert diag["provider_error_code"] == "unsupported_parameter"
        assert diag["provider_error_param"] == "temperature"
        assert diag["provider_error_type"] == "invalid_request_error"

    def test_failure_log_does_not_include_secrets(self, caplog: pytest.LogCaptureFixture) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: max_tokens",
                    "param": "max_tokens",
                }
            },
        )
        with caplog.at_level("WARNING"):
            _log_azure_call_failure(
                exc=exc,
                operation="generate",
                model_alias="reasoning_advanced_generator",
                deployment="o4-mini",
                route_id="reasoning.generator.advanced",
                azure_api_mode="azure_openai_v1",
                failure_kind="unsupported_parameter",
            )
        combined = " ".join(caplog.messages)
        assert "sk-" not in combined
        assert "Solve this puzzle" not in combined


class FakeStreamAzureClient:
    """Fake Azure client for stream chunk normalization tests."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks
        self.received_kwargs: dict = {}

        def _create(**kwargs):  # noqa: ANN202
            self.received_kwargs = kwargs
            return iter(self._chunks)

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )


def _make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    has_choices: bool = True,
    delta_none: bool = False,
) -> object:
    if not has_choices:
        return types.SimpleNamespace(choices=[])
    delta = None if delta_none else types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(finish_reason=finish_reason, delta=delta)
    return types.SimpleNamespace(choices=[choice])


class TestAzureStreamNormalization:
    """Azure generate_stream must handle unusual o4-mini chunk shapes."""

    def test_normal_text_chunks_emitted(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk("Hello "),
            _make_chunk("world"),
            _make_chunk(finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        result = list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert result == ["Hello ", "world"]

    def test_empty_choices_chunks_skipped(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk(has_choices=False),
            _make_chunk("answer"),
            _make_chunk(finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        result = list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert result == ["answer"]

    def test_none_delta_chunks_skipped(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk(delta_none=True),
            _make_chunk("content"),
            _make_chunk(finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        result = list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert result == ["content"]

    def test_none_content_chunks_skipped(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk(content=None),
            _make_chunk("text"),
            _make_chunk(content=None, finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        result = list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert result == ["text"]

    def test_all_empty_chunks_produces_no_output(self, tmp_path: Path) -> None:
        """All-empty stream — e.g. o4-mini returning only internal reasoning."""
        chunks = [
            _make_chunk(content=None),
            _make_chunk(content=None),
            _make_chunk(content=None, finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        result = list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert result == []
        assert adapter.last_stream_finish_reason == "stop"

    def test_empty_stream_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_make_chunk(content=None, finish_reason="stop")]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        with caplog.at_level("WARNING"):
            list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert any("empty_stream" in m for m in caplog.messages)

    def test_finish_reason_captured_on_empty_content_chunk(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk("hi"),
            _make_chunk(content=None, finish_reason="stop"),
        ]
        fake = FakeStreamAzureClient(chunks)
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake)
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://x.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        list(adapter.generate_stream(request=_make_request(tmp_path), credentials=creds))
        assert adapter.last_stream_finish_reason == "stop"


class TestPlannerNativeStructuredOutput:
    """Practice's planner/generator/verifier roles get a native strict json_schema
    response_format; Doubt Solver's calls through the same shared task_roles do not.

    Gated on the exact prompt path (route_decision.prompt), not on intent: the
    classifier maps a practice_question intent to the literal string "practice"
    (academic_classifier.ACADEMIC_INTENT_MAP) even for requests that fall through to
    Doubt Solver's ordinary free-text "generate" node (Practice launch ineligible or
    disabled), so intent=="practice" cannot tell the two apart. The legacy schema-v1
    Practice generator/verifier prompts also use a different wire shape than schema-v2
    and must not get this schema either — prompt-path gating excludes those too.
    """

    def _with(self, request, **route_updates):
        route = request.route_decision.model_copy(update=route_updates)
        return request.model_copy(update={"route_decision": route})

    def test_planner_role_gets_native_structured_output(self, tmp_path: Path) -> None:
        request = self._with(
            _make_request(tmp_path),
            task_role="planner",
            prompt="practice_generation/planners/factual.md",
        )
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" in kwargs
        response_format = kwargs["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"slots", "requestedTopicEvidence"}
        assert schema["additionalProperties"] is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "practice_generation/question_generator_v2.md",
            "practice_generation/question_generator_factual.md",
            "practice_generation/question_repair.md",
            "practice_generation/question_regenerator.md",
        ],
    )
    def test_generator_v2_prompts_get_native_structured_output(
        self, tmp_path: Path, prompt: str
    ) -> None:
        """Every schema-v2 generator prompt — initial, factual, repair, regenerate —
        shares the one wire shape and must all get the native schema alike."""
        request = self._with(
            _make_request(tmp_path), task_role="generator", prompt=prompt
        )
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" in kwargs
        schema = kwargs["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["questions"]["items"]["required"] == [
            "schema_version", "bucket_id", "slot_id", "question", "question_type",
            "options", "correct_option_id", "subject", "topic", "difficulty",
        ]

    def test_verifier_v2_prompt_gets_native_structured_output(
        self, tmp_path: Path
    ) -> None:
        request = self._with(
            _make_request(tmp_path),
            task_role="verifier",
            prompt="practice_generation/question_verifier_v2.md",
        )
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" in kwargs
        schema = kwargs["response_format"]["json_schema"]["schema"]
        assert "evidence_urls" in schema["required"]

    def test_other_roles_get_no_response_format(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path)  # task_role="generator", default test prompt
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" not in kwargs

    @pytest.mark.parametrize(
        ("task_role", "prompt"),
        [
            ("generator", "subjects/math_generator.md"),
            ("generator", "subjects/general_generator.md"),
            ("verifier", "answer_correctness_verifier.md"),
        ],
    )
    def test_doubt_solver_prompts_get_no_response_format_even_with_intent_practice(
        self, tmp_path: Path, task_role: str, prompt: str
    ) -> None:
        """Regression for the found defect: the classifier maps practice_question to
        intent="practice" even on requests that fall through to Doubt Solver's ordinary
        free-text generate node. A naive intent=="practice" gate would wrongly force a
        free-text Doubt Solver answer into the strict Practice MCQ JSON schema — the
        prompt-path gate must reject it regardless of intent."""
        request = self._with(
            _make_request(tmp_path), task_role=task_role, prompt=prompt, intent="practice"
        )
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" not in kwargs

    @pytest.mark.parametrize(
        ("task_role", "prompt"),
        [
            ("generator", "practice_generation/question_generator.md"),
            ("verifier", "practice_generation/question_verifier.md"),
        ],
    )
    def test_legacy_v1_practice_prompts_get_no_response_format(
        self, tmp_path: Path, task_role: str, prompt: str
    ) -> None:
        """Regression for the found defect: the legacy schema-v1 Practice
        generator/verifier prompts use a different wire shape (plain-string options,
        correct_answer/solution, no schema_version) than schema-v2 — forcing them into
        the v2 strict schema would reject every legitimate v1 response."""
        request = self._with(
            _make_request(tmp_path), task_role=task_role, prompt=prompt, intent="practice"
        )
        kwargs, _ = build_azure_openai_chat_completion_kwargs(request=request, deployment="dep")
        assert "response_format" not in kwargs

    @pytest.mark.parametrize(
        "schema_name",
        ["planner_generation_schema", "practice_generator_generation_schema",
         "practice_verifier_generation_schema"],
    )
    def test_generation_schema_never_encodes_length_or_count_constraints(
        self, schema_name: str
    ) -> None:
        """maxLength/item-count bounds stay local (Pydantic); the provider only gets shape."""
        import features.practice_generation.schemas as schemas_module

        forbidden_keys = {
            "maxLength", "minLength", "minItems", "maxItems", "pattern", "uniqueItems",
        }

        def check(obj: object) -> None:
            if isinstance(obj, dict):
                found = forbidden_keys & obj.keys()
                assert not found, found
                for value in obj.values():
                    check(value)
            elif isinstance(obj, list):
                for item in obj:
                    check(item)

        check(getattr(schemas_module, schema_name)())

    @pytest.mark.parametrize(
        "schema_name",
        ["planner_generation_schema", "practice_generator_generation_schema",
         "practice_verifier_generation_schema"],
    )
    def test_generation_schema_is_strict_mode_self_consistent(
        self, schema_name: str
    ) -> None:
        """Every property must be required and every object closed, or the provider rejects it."""
        import features.practice_generation.schemas as schemas_module

        def check(obj: dict) -> None:
            if obj.get("type") == "object":
                assert set(obj["properties"]) == set(obj["required"])
                assert obj["additionalProperties"] is False
                for value in obj["properties"].values():
                    check(value)
            if obj.get("type") == "array":
                check(obj["items"])

        check(getattr(schemas_module, schema_name)())

    def test_planner_generation_schema_round_trips_through_real_pydantic_validation(
        self,
    ) -> None:
        """The compiled schema's shape must actually match what PracticeBlueprint accepts."""
        from features.practice_generation.schemas import (
            PracticeBlueprint,
            planner_generation_schema,
        )

        schema = planner_generation_schema()
        slot_props = set(schema["properties"]["slots"]["items"]["properties"])
        evidence_props = set(schema["properties"]["requestedTopicEvidence"]["items"]["properties"])
        slot_model = PracticeBlueprint.model_fields["slots"].annotation.__args__[0]
        assert slot_props == set(slot_model.model_fields)
        assert evidence_props == {"source_text", "topic_id"} or evidence_props == {
            "sourceText",
            "topicId",
        }

    def test_generator_schema_round_trips_through_real_pydantic_validation(self) -> None:
        """A schema-valid instance must be accepted by GeneratedQuestion unchanged."""
        from features.practice_generation.schemas import (
            GeneratedQuestion,
            practice_generator_generation_schema,
        )

        schema = practice_generator_generation_schema()
        question_props = set(schema["properties"]["questions"]["items"]["properties"])
        assert question_props == {
            "schema_version", "bucket_id", "slot_id", "question", "question_type",
            "options", "correct_option_id", "subject", "topic", "difficulty",
        }
        sample = {
            "schema_version": "2", "bucket_id": "b1", "slot_id": "slot-001",
            "question": "What is 2+2?", "question_type": "mcq",
            "options": [
                {"option_id": "0", "value": "3"}, {"option_id": "1", "value": "4"},
                {"option_id": "2", "value": "5"}, {"option_id": "3", "value": "6"},
            ],
            "correct_option_id": "1", "subject": "math", "topic": "addition",
            "difficulty": "basic",
        }
        question = GeneratedQuestion.model_validate(sample)
        assert question.correct_answer == "4"

    def test_verifier_schema_round_trips_through_real_pydantic_validation(self) -> None:
        """A schema-valid instance must be accepted by VerificationResult unchanged,
        including evidence_urls — missing from the prior Gemini-only hand-written copy."""
        from features.practice_generation.schemas import (
            VerificationResult,
            practice_verifier_generation_schema,
        )

        schema = practice_verifier_generation_schema()
        assert set(schema["properties"]) == {
            "schema_version", "generation_item_id", "slot_id", "decision",
            "valid_option_ids", "answer_explanation", "reason_codes", "evidence_urls",
        }
        sample = {
            "schema_version": "2", "generation_item_id": "item-slot-001",
            "slot_id": "slot-001", "decision": "ACCEPT", "valid_option_ids": ["1"],
            "answer_explanation": "Option 1 follows from the calculation.",
            "reason_codes": ["SINGLE_VALID_OPTION"], "evidence_urls": [],
        }
        result = VerificationResult.model_validate(sample)
        assert result.is_approved is True
