"""Azure provider clients honor the resolved model ``timeout_seconds``.

Without the timeout the OpenAI SDK default (600 s read) applied, so a stalled Azure
call ran for 152 s / 195.6 s before the configured fallback could start. The registry
stays the only timeout source: nothing here is keyed on a model name.

The bounded-fallback tests use the real SDK against local sockets — one that accepts
and never answers, one that returns a valid completion — so the timeout that fires is
the SDK's own, classified by the existing error mapping and handled by the existing
fallback executor.
"""

from __future__ import annotations

import json
import socket
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openai
import pytest

from schemas.doubt_solver import DoubtSolverStreamEvent
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import RouteDecision, RouteRequest
from services.doubt_solver.stream_transport import StreamCancellation, stream_events_as_sse
from services.llm.orchestration.config_registry import LlmConfigRegistry, get_registry
from services.llm.orchestration.errors import ProviderExecutionError
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import (
    ProviderAdapterExecutor,
    RegistryBackedModelExecutor,
)
from services.llm.orchestration.route_resolver import resolve_route
from services.llm.providers.azure_openai_provider import AzureOpenAIProviderAdapter
from services.llm.providers.errors import LlmProviderExecutionError
from services.llm.providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.provider_credentials import ProviderCredentialResolver, ProviderCredentials

_V1_CREDENTIALS = ProviderCredentials(
    provider="azure_openai",
    api_key="test-key",
    endpoint="https://example.openai.azure.com/openai/v1",
    azure_api_mode="azure_openai_v1",
)
_CLASSIC_CREDENTIALS = ProviderCredentials(
    provider="azure_openai",
    api_key="test-key",
    endpoint="https://example.openai.azure.com",
    api_version="2024-02-01",
)


# ---------------------------------------------------------------------------
# A1. Client construction: the configured value reaches the SDK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeout_seconds", [30, 60, 90])
def test_v1_client_uses_the_configured_timeout(timeout_seconds: int) -> None:
    client = AzureOpenAIProviderAdapter()._build_client_v1(_V1_CREDENTIALS, timeout_seconds)

    assert client.timeout == float(timeout_seconds)
    effective = client._client.timeout  # noqa: SLF001 - the httpx timeout the SDK sends with
    assert effective.read == float(timeout_seconds)
    assert effective.connect == float(timeout_seconds)
    assert client.max_retries == 0


@pytest.mark.parametrize("timeout_seconds", [30, 60, 90])
def test_classic_client_uses_the_configured_timeout(timeout_seconds: int) -> None:
    client = AzureOpenAIProviderAdapter()._build_client_classic(
        _CLASSIC_CREDENTIALS, timeout_seconds
    )

    assert client.timeout == float(timeout_seconds)
    effective = client._client.timeout  # noqa: SLF001
    assert effective.read == float(timeout_seconds)
    assert client.max_retries == 0


# ---------------------------------------------------------------------------
# A1/A2. Invocation paths take the timeout from the resolved registry config
# ---------------------------------------------------------------------------


class _ConstructorSpy:
    """Stands in for the SDK class; records constructor kwargs, then stops the call."""

    def __init__(self) -> None:
        self.constructed: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.constructed.append(kwargs)

        def _create(**_kw: object) -> object:
            raise RuntimeError("stop after construction")

        return type("_Client", (), {"chat": type("_Chat", (), {
            "completions": type("_Completions", (), {"create": staticmethod(_create)})()
        })()})()


def _production_request(alias: str) -> ProviderExecutionRequest:
    """A request for ``alias`` resolved through the production registry."""
    base = resolve_route(
        RouteRequest(
            request_id="timeout-propagation",
            subject="general",
            task_role="verifier",
            difficulty="default",
            intent="solve",
            language="english",
        )
    )
    # Provider options are route-specific; the timeout under test is per model alias.
    decision = base.model_copy(update={"model": alias, "provider_options": {}})
    return ProviderExecutionRequest(
        route_decision=decision,
        model_resolution=ModelConfigResolver(registry=get_registry()).resolve(decision),
        messages=[LlmMessage(role="user", content="Q.")],
        temperature=0.0,
        max_tokens=50,
    )


_REPRESENTATIVE_AZURE_ALIASES = [
    "math_intermediate_generator",
    "openai_gpt_4_1",
    "openai_gpt_5_6_terra",
    "azure_deepseek_v4_pro",
]


@pytest.mark.parametrize("alias", _REPRESENTATIVE_AZURE_ALIASES)
@pytest.mark.parametrize("streaming", [False, True])
def test_invocation_sends_the_registry_timeout_for_representative_azure_aliases(
    alias: str, streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ConstructorSpy()
    monkeypatch.setattr(openai, "OpenAI", spy)
    request = _production_request(alias)
    configured = get_registry().model_map[alias].timeout_seconds
    adapter = AzureOpenAIProviderAdapter()

    with pytest.raises(LlmProviderExecutionError):
        if streaming:
            list(adapter.generate_stream(request=request, credentials=_V1_CREDENTIALS))
        else:
            adapter.generate(request=request, credentials=_V1_CREDENTIALS)

    assert request.model_resolution.timeout_seconds == configured
    assert [kwargs["timeout"] for kwargs in spy.constructed] == [float(configured)]


@pytest.mark.parametrize("streaming", [False, True])
def test_classic_invocation_sends_the_registry_timeout(
    streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ConstructorSpy()
    monkeypatch.setattr(openai, "AzureOpenAI", spy)
    request = _production_request("openai_gpt_5_6_terra")
    adapter = AzureOpenAIProviderAdapter()

    with pytest.raises(LlmProviderExecutionError):
        if streaming:
            list(adapter.generate_stream(request=request, credentials=_CLASSIC_CREDENTIALS))
        else:
            adapter.generate(request=request, credentials=_CLASSIC_CREDENTIALS)

    assert [kwargs["timeout"] for kwargs in spy.constructed] == [
        float(request.model_resolution.timeout_seconds)
    ]


# ---------------------------------------------------------------------------
# A3/A4. Real SDK timeout -> existing classification -> existing bounded fallback
# ---------------------------------------------------------------------------

_VERDICT = (
    '{"status":"MATCH","independent_answer":"2.5",'
    '"single_defensible_answer":true,"reason":"ok"}'
)


class _HangingServer:
    """Accepts connections and reads requests but never responds."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self.connections = 0
        self._held: list[socket.socket] = []
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self.connections += 1
            self._held.append(conn)

    def close(self) -> None:
        self._stop.set()
        for conn in self._held:
            conn.close()
        self._sock.close()


class _HealthyServer:
    """Returns one valid chat completion per POST."""

    def __init__(self) -> None:
        owner = self
        self.requests = 0

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                owner.requests += 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": "fallback-deployment",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": _VERDICT}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _fallback_registry(tmp_path: Path, *, fallback_profile: str) -> LlmConfigRegistry:
    path = tmp_path / "llm.yaml"
    path.write_text(textwrap.dedent(f"""\
        version: 1
        routes:
          general:
            verifier:
              default:
                model: azure_primary_model
                prompt: answer_correctness_verifier.md
                temperature: 0.0
                max_tokens: 200
        models:
          azure_primary_model:
            provider: azure_openai
            provider_profile: azure_hanging
            deployment: primary-deployment
            supports_streaming: true
            supports_thinking: false
            timeout_seconds: 1
            fallback_models:
              - azure_fallback_model
          azure_fallback_model:
            provider: azure_openai
            provider_profile: {fallback_profile}
            deployment: fallback-deployment
            supports_streaming: true
            supports_thinking: false
            timeout_seconds: 2
          safe_mock:
            provider: mock
            provider_profile: local_mock
            model_id: local-mock
            supports_streaming: true
            supports_thinking: false
            timeout_seconds: 1
        provider_profiles:
          azure_hanging:
            provider: azure_openai
            azure_api_mode: azure_openai_v1
            endpoint_env: TEST_AZURE_HANGING_ENDPOINT
            api_key_env: TEST_AZURE_KEY
          azure_healthy:
            provider: azure_openai
            azure_api_mode: azure_openai_v1
            endpoint_env: TEST_AZURE_HEALTHY_ENDPOINT
            api_key_env: TEST_AZURE_KEY
          local_mock:
            provider: mock
    """), encoding="utf-8")
    return LlmConfigRegistry(yaml_path=path)


class _RecordingAzureAdapter(AzureOpenAIProviderAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: list[tuple[str, str, float]] = []

    def generate(self, *, request, credentials):  # noqa: ANN001, ANN201
        started = time.monotonic()
        alias = request.model_resolution.model_alias
        try:
            result = super().generate(request=request, credentials=credentials)
        except LlmProviderExecutionError as exc:
            self.attempts.append((alias, str(exc.failure_kind), time.monotonic() - started))
            raise
        self.attempts.append((alias, "ok", time.monotonic() - started))
        return result


def _stack(
    tmp_path: Path, *, fallback_profile: str
) -> tuple[RegistryBackedModelExecutor, _RecordingAzureAdapter]:
    registry = _fallback_registry(tmp_path, fallback_profile=fallback_profile)
    adapter = _RecordingAzureAdapter()
    executor = RegistryBackedModelExecutor(
        provider_executor=ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(secret_resolver=EnvSecretResolver()),
            provider_factory=ProviderAdapterFactory(adapter_map={"azure_openai": adapter}),
        ),
        model_config_resolver=ModelConfigResolver(registry=registry),
    )
    return executor, adapter


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_HangingServer, _HealthyServer]]:
    hanging, healthy = _HangingServer(), _HealthyServer()
    monkeypatch.setenv("TEST_AZURE_KEY", "test-key")
    monkeypatch.setenv("TEST_AZURE_HANGING_ENDPOINT", f"http://127.0.0.1:{hanging.port}/openai/v1")
    monkeypatch.setenv("TEST_AZURE_HEALTHY_ENDPOINT", f"http://127.0.0.1:{healthy.port}/openai/v1")
    yield hanging, healthy
    hanging.close()
    healthy.close()


def _verifier_decision() -> RouteDecision:
    return RouteDecision(
        route_id="general.verifier.default",
        subject="general",
        task_role="verifier",
        difficulty="default",
        model="azure_primary_model",
        prompt="answer_correctness_verifier.md",
        temperature=0.0,
        max_tokens=200,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def test_stalled_primary_times_out_at_its_bound_and_falls_back_once(
    tmp_path: Path, servers: tuple[_HangingServer, _HealthyServer]
) -> None:
    hanging, healthy = servers
    executor, adapter = _stack(tmp_path, fallback_profile="azure_healthy")

    started = time.monotonic()
    result = executor.execute(
        route_decision=_verifier_decision(),
        messages=[LlmMessage(role="user", content="Q.")],
    )
    elapsed = time.monotonic() - started

    assert [(alias, kind) for alias, kind, _ in adapter.attempts] == [
        ("azure_primary_model", "timeout"),
        ("azure_fallback_model", "ok"),
    ]
    primary_elapsed = adapter.attempts[0][2]
    assert 0.9 <= primary_elapsed < 3.0
    assert elapsed < 5.0
    assert hanging.connections == 1
    assert healthy.requests == 1
    assert result.model == "azure_fallback_model"
    assert result.fallback_used is True
    assert result.content == _VERDICT


def test_primary_and_fallback_both_stalled_fail_bounded_without_looping(
    tmp_path: Path, servers: tuple[_HangingServer, _HealthyServer]
) -> None:
    hanging, healthy = servers
    executor, adapter = _stack(tmp_path, fallback_profile="azure_hanging")

    started = time.monotonic()
    with pytest.raises(ProviderExecutionError) as exc_info:
        executor.execute(
            route_decision=_verifier_decision(),
            messages=[LlmMessage(role="user", content="Q.")],
        )
    elapsed = time.monotonic() - started

    assert [(alias, kind) for alias, kind, _ in adapter.attempts] == [
        ("azure_primary_model", "timeout"),
        ("azure_fallback_model", "timeout"),
    ]
    assert exc_info.value.failure_kind == "timeout"
    assert exc_info.value.attempted_aliases == ("azure_primary_model", "azure_fallback_model")
    assert elapsed < 1 + 2 + 2.5
    assert hanging.connections == 2
    assert healthy.requests == 0


def test_both_stalled_verifier_ends_the_stream_with_one_terminal_and_no_later_heartbeat(
    tmp_path: Path, servers: tuple[_HangingServer, _HealthyServer]
) -> None:
    """The bounded failure reaches the unchanged transport as one terminal frame."""
    import asyncio  # noqa: PLC0415

    del servers
    executor, adapter = _stack(tmp_path, fallback_profile="azure_hanging")
    request_id = "timeout-stream"

    def events() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(type="status", request_id=request_id, stage="verifying",
                                     label="Verifying...")
        with pytest.raises(ProviderExecutionError):
            executor.execute(
                route_decision=_verifier_decision(),
                messages=[LlmMessage(role="user", content="Q.")],
            )
        # A verifier provider failure becomes the unavailable terminal the streaming
        # service emits (AnswerCorrectnessVerifier fails closed on provider errors).
        yield DoubtSolverStreamEvent(
            type="error", request_id=request_id, stage="failed", label="Unable to complete",
            metadata={"retryable": False, "code": "ANSWER_VERIFICATION_UNAVAILABLE"},
        )

    async def collect() -> list[bytes]:
        return [
            frame
            async for frame in stream_events_as_sse(
                events(),
                request_id=request_id,
                cancellation=StreamCancellation(),
                heartbeat_interval_seconds=0.5,
            )
        ]

    started = time.monotonic()
    frames = asyncio.run(collect())
    elapsed = time.monotonic() - started

    kinds = [
        "heartbeat" if frame.startswith(b":") else json.loads(frame[5:])["type"]
        for frame in frames
    ]
    assert kinds.count("error") == 1
    assert kinds[-1] == "error"
    assert "heartbeat" in kinds
    assert [kind for _, kind, _ in adapter.attempts] == ["timeout", "timeout"]
    assert elapsed < 1 + 2 + 3
