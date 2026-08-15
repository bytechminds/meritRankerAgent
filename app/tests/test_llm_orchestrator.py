"""
app/tests/test_llm_orchestrator.py
------------------------------------
Unit tests for LlmOrchestrator (Part 3 — LLM Orchestration Foundation).

All tests use injected fakes:
- MockModelExecutor for model execution.
- PromptResolver(tmp_path) for prompt isolation.
- A route_resolver_fn lambda returning a fixed RouteDecision (no YAML needed).

No real LLM calls.  No provider SDK.  No boto3.  No AWS calls.  No graph.

Test coverage:
1.  Orchestrator resolves route and returns result with correct route_decision.
2.  model_executor receives the RouteDecision from route_resolver_fn.
3.  model_executor receives exactly two LlmMessage objects.
4.  result.content comes from ModelExecutionResult.content.
5.  result contains route_id / model / fallback_used fields.
6.  route_resolver_fn injection is used (custom lambda called).
7.  prompt_resolver injection is used (custom PromptResolver called).
8.  MockModelExecutor success path end-to-end.
9.  MockModelExecutor with raise_on_execute wraps as LlmExecutionError.
10. route resolution failure (LlmRouteNotFoundError) is not swallowed.
11. prompt resolver failure (PromptNotFoundError) is not swallowed.
12. empty query raises LlmOrchestratorError.
13. whitespace-only query raises LlmOrchestratorError.
14. query over MAX_QUERY_CHARS raises LlmOrchestratorError.
15. no provider/AWS/network call happens (socket-level assertion).
16. classification reaches PromptResolver / user message.
17. context appears in user message only (not system message).
18. OrchestrationResult does not expose messages/query/context/prompt.
19. ModelExecutionResult metadata rejects unsafe keys.
20. OrchestrationResult metadata rejects unsafe keys.
21. MockModelExecutor records last_route_decision / last_messages / call_count.
22. repeated generate() calls reuse injected resolver and call_count increments.
23. answer_source = "mock" when provider="mock".
24. answer_source = "fallback" when fallback_used=True.
25. answer_source = "llm" when provider is non-mock and fallback_used=False.
26. Part 1/2 tests still pass (verified via combined pytest command in docs).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from features.practice_generation.execution_control import PracticeExecutionStopped
from schemas.llm_orchestration import ModelExecutionResult, OrchestrationResult
from schemas.llm_routing import RouteDecision, RouteRequest
from services.llm_orchestration.errors import (
    LlmExecutionError,
    LlmOrchestratorError,
    LlmRouteNotFoundError,
    PromptNotFoundError,
    ProviderExecutionError,
)
from services.llm_orchestration.orchestrator import (
    MAX_QUERY_CHARS,
    LlmOrchestrator,
    MockModelExecutor,
    create_mock_orchestrator_for_tests,
)
from services.llm_orchestration.prompt_resolver import PromptResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_route_decision(
    model: str = "gemini_flash_light",
    prompt: str = "main.md",
    overlays: list[str] | None = None,
    subject: str = "math",
    fallback_used: bool = False,
) -> RouteDecision:
    """Construct a minimal RouteDecision for testing."""
    return RouteDecision(
        route_id=f"{subject}.generator.default",
        subject=subject,
        task_role="generator",
        difficulty="default",
        model=model,
        prompt=prompt,
        overlays=overlays or [],
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )


def _make_route_request(
    subject: str = "math",
    request_id: str = "req-001",
) -> RouteRequest:
    """Construct a minimal RouteRequest for testing."""
    return RouteRequest(
        request_id=request_id,
        subject=subject,
        task_role="generator",
        difficulty="default",
    )


def _write(tmp_path: Path, rel_path: str, content: str) -> None:
    """Write a file under tmp_path, creating parent directories as needed."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


def _make_prompt_resolver(tmp_path: Path, prompt: str = "main.md") -> PromptResolver:
    """Write a minimal prompt file and return a PromptResolver rooted at tmp_path."""
    _write(tmp_path, prompt, "# Test System Prompt\nYou are a helpful tutor.")
    return PromptResolver(prompt_root=tmp_path)


def _fixed_route_resolver(
    decision: RouteDecision,
) -> Any:
    """Return a lambda that always returns the given RouteDecision."""
    return lambda _req: decision


# ---------------------------------------------------------------------------
# Test 1 — Orchestrator resolves route and returns result with route_decision
# ---------------------------------------------------------------------------


def test_result_contains_route_decision(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        content="Answer here.",
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(
        route_request=_make_route_request(),
        query="What is 2+2?",
    )
    assert result.route_decision.route_id == "math.generator.default"
    assert result.route_decision.subject == "math"


def test_generate_structured_returns_json_without_answer_finalization(tmp_path: Path) -> None:
    _write(tmp_path, "structured.md", "Return JSON only.")
    _write(tmp_path, "shared.md", "Shared structured contract.")
    decision = _make_route_decision(prompt="unused.md")
    executor = MockModelExecutor(content='{"questions":[]}')
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        prompt_resolver=PromptResolver(prompt_root=tmp_path),
        route_resolver_fn=_fixed_route_resolver(decision),
    )

    result = orchestrator.generate_structured(
        route_request=_make_route_request(),
        user_content='{"count":1}',
        prompt="structured.md",
        overlays=["shared.md"],
    )

    assert result.content == '{"questions":[]}'
    assert result.final_answer is None
    assert executor.call_count == 1
    assert executor.last_messages is not None
    assert "Return JSON only." in executor.last_messages[0].content
    assert executor.last_messages[1].content == '{"count":1}'


def test_generate_structured_propagates_expensive_attempt_stop_signal(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "structured.md", "Return JSON only.")
    decision = _make_route_decision(prompt="unused.md")
    executor = MockModelExecutor(raise_on_execute=RuntimeError("retry blocked"))
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        prompt_resolver=PromptResolver(prompt_root=tmp_path),
        route_resolver_fn=_fixed_route_resolver(decision),
    )

    def guard() -> None:
        raise PracticeExecutionStopped("PRACTICE_CANCEL_OBSERVED")

    with pytest.raises(PracticeExecutionStopped, match="PRACTICE_CANCEL_OBSERVED"):
        orchestrator.generate_structured(
            route_request=_make_route_request(),
            user_content='{"count":1}',
            prompt="structured.md",
            attempt_guard=guard,
        )


# ---------------------------------------------------------------------------
# Test 2 — model_executor receives the RouteDecision
# ---------------------------------------------------------------------------


def test_executor_receives_route_decision(tmp_path: Path) -> None:
    decision = _make_route_decision(model="test_model_alias")
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    orchestrator.generate(route_request=_make_route_request(), query="Test question?")
    assert executor.last_route_decision is not None
    assert executor.last_route_decision.model == "test_model_alias"
    assert executor.last_route_decision.route_id == "math.generator.default"


# ---------------------------------------------------------------------------
# Test 3 — model_executor receives exactly two LlmMessage objects
# ---------------------------------------------------------------------------


def test_executor_receives_exactly_two_messages(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    orchestrator.generate(route_request=_make_route_request(), query="Explain gravity.")
    assert executor.last_messages is not None
    assert len(executor.last_messages) == 2
    assert executor.last_messages[0].role == "system"
    assert executor.last_messages[1].role == "user"


# ---------------------------------------------------------------------------
# Test 4 — result.content comes from ModelExecutionResult.content
# ---------------------------------------------------------------------------


def test_result_content_from_executor(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        content="The answer is 42.",
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="What is life?")
    assert result.content == "The answer is 42."


# ---------------------------------------------------------------------------
# Test 5 — result contains route_id / model / fallback_used fields
# ---------------------------------------------------------------------------


def test_result_contains_route_id_model_fallback(tmp_path: Path) -> None:
    decision = _make_route_decision(model="gemini_flash_light")
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="Question?")
    assert result.route_decision.route_id == "math.generator.default"
    assert result.model == "gemini_flash_light"
    assert result.fallback_used is False


# ---------------------------------------------------------------------------
# Test 6 — route_resolver_fn injection is used
# ---------------------------------------------------------------------------


def test_route_resolver_fn_injection(tmp_path: Path) -> None:
    call_log: list[RouteRequest] = []
    decision = _make_route_decision(subject="reasoning")

    def tracking_resolver(req: RouteRequest) -> RouteDecision:
        call_log.append(req)
        return decision

    _write(tmp_path, "main.md", "# System prompt.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=tracking_resolver,
    )
    orchestrator.generate(route_request=_make_route_request(subject="reasoning"), query="Q?")
    assert len(call_log) == 1
    assert call_log[0].subject == "reasoning"


# ---------------------------------------------------------------------------
# Test 7 — prompt_resolver injection is used
# ---------------------------------------------------------------------------


def test_prompt_resolver_injection(tmp_path: Path) -> None:
    decision = _make_route_decision()
    _write(tmp_path, "main.md", "# Injected prompt content UNIQUE_MARKER.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    orchestrator.generate(route_request=_make_route_request(), query="Query?")
    assert executor.last_messages is not None
    assert "UNIQUE_MARKER" in executor.last_messages[0].content


# ---------------------------------------------------------------------------
# Test 8 — MockModelExecutor success path end-to-end
# ---------------------------------------------------------------------------


def test_mock_executor_success_path(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        content="Mock content.",
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="Test?")
    assert result.content == "Mock content."
    assert executor.call_count == 1
    assert result.answer_source == "mock"


# ---------------------------------------------------------------------------
# Test 9 — MockModelExecutor with raise_on_execute wraps as LlmExecutionError
# ---------------------------------------------------------------------------


def test_executor_exception_wrapped_as_llm_execution_error(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
        raise_on_execute=RuntimeError("Provider down"),
    )
    with pytest.raises(LlmExecutionError) as exc_info:
        orchestrator.generate(route_request=_make_route_request(), query="Test?")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    # Safe message — must not expose query or prompt content
    assert "Provider down" not in str(exc_info.value)
    assert "math.generator.default" in str(exc_info.value)


def test_provider_execution_error_propagates_without_double_wrap(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    provider_error = ProviderExecutionError("Provider executor failed safely.")
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
        raise_on_execute=provider_error,
    )

    with pytest.raises(ProviderExecutionError) as exc_info:
        orchestrator.generate(route_request=_make_route_request(), query="Test?")

    assert exc_info.value is provider_error


def test_generic_executor_failure_wraps_as_llm_execution_error(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
        raise_on_execute=ValueError("Raw provider detail"),
    )

    with pytest.raises(LlmExecutionError) as exc_info:
        orchestrator.generate(route_request=_make_route_request(), query="Test?")

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "Raw provider detail" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 10 — route resolution failure (LlmRouteNotFoundError) is not swallowed
# ---------------------------------------------------------------------------


def test_route_not_found_error_propagates(tmp_path: Path) -> None:
    def failing_resolver(_req: RouteRequest) -> RouteDecision:
        raise LlmRouteNotFoundError("No route for role=generator subject=unknown")

    _write(tmp_path, "main.md", "# System.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=failing_resolver,
    )
    with pytest.raises(LlmRouteNotFoundError):
        orchestrator.generate(route_request=_make_route_request(), query="Q?")


# ---------------------------------------------------------------------------
# Test 11 — prompt resolver failure (PromptNotFoundError) is not swallowed
# ---------------------------------------------------------------------------


def test_prompt_not_found_propagates(tmp_path: Path) -> None:
    # Create a route that points to a file that does not exist
    decision = _make_route_decision(prompt="missing_prompt.md")
    # Do NOT write missing_prompt.md — it must trigger PromptNotFoundError
    empty_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=empty_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    with pytest.raises(PromptNotFoundError):
        orchestrator.generate(route_request=_make_route_request(), query="Q?")


# ---------------------------------------------------------------------------
# Test 12 — empty query raises LlmOrchestratorError
# ---------------------------------------------------------------------------


def test_empty_query_raises(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    with pytest.raises(LlmOrchestratorError, match="empty"):
        orchestrator.generate(route_request=_make_route_request(), query="")


# ---------------------------------------------------------------------------
# Test 13 — whitespace-only query raises LlmOrchestratorError
# ---------------------------------------------------------------------------


def test_whitespace_query_raises(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    with pytest.raises(LlmOrchestratorError, match="empty"):
        orchestrator.generate(route_request=_make_route_request(), query="   \n  ")


# ---------------------------------------------------------------------------
# Test 14 — query over MAX_QUERY_CHARS raises LlmOrchestratorError
# ---------------------------------------------------------------------------


def test_over_limit_query_raises(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    long_query = "x" * (MAX_QUERY_CHARS + 1)
    with pytest.raises(LlmOrchestratorError, match="maximum allowed length"):
        orchestrator.generate(route_request=_make_route_request(), query=long_query)


# ---------------------------------------------------------------------------
# Test 15 — no provider/AWS/network call happens (socket assertion)
# ---------------------------------------------------------------------------


def test_no_network_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure generate() never opens any network socket."""
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        content="No network needed.",
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )

    def assert_no_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"Network connection attempted in test: connect({args!r})"
        )

    monkeypatch.setattr(socket.socket, "connect", assert_no_connect)
    result = orchestrator.generate(route_request=_make_route_request(), query="Hello?")
    assert result.content == "No network needed."


# ---------------------------------------------------------------------------
# Test 16 — classification reaches PromptResolver / user message
# ---------------------------------------------------------------------------


def test_classification_in_user_message(tmp_path: Path) -> None:
    decision = _make_route_decision()
    _write(tmp_path, "main.md", "# System prompt.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    classification = {"subject": "math", "intent": "solve", "difficulty": "basic"}
    orchestrator.generate(
        route_request=_make_route_request(),
        query="What is 2+2?",
        classification=classification,
    )
    assert executor.last_messages is not None
    user_msg_content = executor.last_messages[1].content
    # Classification fields should appear in the user message
    assert "math" in user_msg_content or "solve" in user_msg_content


# ---------------------------------------------------------------------------
# Test 17 — context appears in user message only (not system message)
# ---------------------------------------------------------------------------


def test_context_in_user_message_only(tmp_path: Path) -> None:
    decision = _make_route_decision()
    _write(tmp_path, "main.md", "# System prompt without any context.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    context = "UNIQUE_CONTEXT_STRING_XYZ"
    orchestrator.generate(
        route_request=_make_route_request(),
        query="Question?",
        context=context,
    )
    assert executor.last_messages is not None
    system_content = executor.last_messages[0].content
    user_content = executor.last_messages[1].content
    assert "UNIQUE_CONTEXT_STRING_XYZ" not in system_content
    assert "UNIQUE_CONTEXT_STRING_XYZ" in user_content


# ---------------------------------------------------------------------------
# Test 18 — OrchestrationResult does not expose messages/query/context/prompt
# ---------------------------------------------------------------------------


def test_orchestration_result_has_no_sensitive_fields(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(
        route_request=_make_route_request(),
        query="What is the capital of France?",
        context="Retrieved context here.",
    )
    # OrchestrationResult must not have messages, query, context, or prompt fields
    result_dict = result.model_dump()
    assert "messages" not in result_dict
    assert "query" not in result_dict
    assert "context" not in result_dict
    assert "prompt" not in result_dict
    assert "system_prompt" not in result_dict
    assert "user_prompt" not in result_dict


# ---------------------------------------------------------------------------
# Test 19 — ModelExecutionResult metadata rejects unsafe keys
# ---------------------------------------------------------------------------


def test_model_execution_result_metadata_rejects_unsafe_keys() -> None:
    for bad_key in ("prompt", "system_prompt", "user_prompt", "messages",
                    "query", "context", "api_key", "secret", "credential"):
        with pytest.raises(ValidationError, match="sensitive keys"):
            ModelExecutionResult(
                content="Some content.",
                model="test_model",
                metadata={bad_key: "value"},
            )


# ---------------------------------------------------------------------------
# Test 20 — OrchestrationResult metadata rejects unsafe keys
# ---------------------------------------------------------------------------


def test_orchestration_result_metadata_rejects_unsafe_keys(tmp_path: Path) -> None:
    decision = _make_route_decision()
    for bad_key in ("prompt", "query", "context", "api_key", "secret", "credential"):
        with pytest.raises(ValidationError, match="sensitive keys"):
            OrchestrationResult(
                content="Some content.",
                route_decision=decision,
                model="test_model",
                answer_source="mock",
                metadata={bad_key: "value"},
            )


# ---------------------------------------------------------------------------
# Test 21 — MockModelExecutor records last_route_decision/last_messages/call_count
# ---------------------------------------------------------------------------


def test_mock_executor_records_call_details(tmp_path: Path) -> None:
    decision = _make_route_decision(model="record_model")
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    assert executor.call_count == 0
    assert executor.last_route_decision is None
    assert executor.last_messages is None

    orchestrator.generate(route_request=_make_route_request(), query="First call.")

    assert executor.call_count == 1
    assert executor.last_route_decision is not None
    assert executor.last_route_decision.model == "record_model"
    assert executor.last_messages is not None
    assert len(executor.last_messages) == 2


# ---------------------------------------------------------------------------
# Test 22 — repeated generate() calls reuse injected resolver, call_count increments
# ---------------------------------------------------------------------------


def test_repeated_calls_increment_call_count(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    resolve_call_count = [0]

    def counting_resolver(req: RouteRequest) -> RouteDecision:
        resolve_call_count[0] += 1
        return decision

    orchestrator, executor = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=counting_resolver,
    )
    orchestrator.generate(route_request=_make_route_request(), query="Call one.")
    orchestrator.generate(route_request=_make_route_request(), query="Call two.")
    orchestrator.generate(route_request=_make_route_request(), query="Call three.")

    assert executor.call_count == 3
    assert resolve_call_count[0] == 3


# ---------------------------------------------------------------------------
# Test 23 — answer_source = "mock" when provider="mock"
# ---------------------------------------------------------------------------


def test_answer_source_mock_when_provider_mock(tmp_path: Path) -> None:
    decision = _make_route_decision()
    prompt_resolver = _make_prompt_resolver(tmp_path)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="Q?")
    # MockModelExecutor returns provider="mock" by default
    assert result.answer_source == "mock"
    assert result.provider == "mock"


# ---------------------------------------------------------------------------
# Test 24 — answer_source = "fallback" when fallback_used=True
# ---------------------------------------------------------------------------


def test_answer_source_fallback_when_fallback_used(tmp_path: Path) -> None:
    decision = _make_route_decision()
    _write(tmp_path, "main.md", "# System.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    # Use a custom executor that returns fallback_used=True
    executor = MockModelExecutor(content="Fallback answer.", fallback_used=True)
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="Q?")
    assert result.answer_source == "fallback"
    assert result.fallback_used is True


# ---------------------------------------------------------------------------
# Test 25 — answer_source = "llm" when provider is non-mock and fallback_used=False
# ---------------------------------------------------------------------------


def test_answer_source_llm_when_real_provider(tmp_path: Path) -> None:
    decision = _make_route_decision()
    _write(tmp_path, "main.md", "# System.")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    # Custom executor that returns provider="gemini" and fallback_used=False
    executor = MockModelExecutor(
        content="Real LLM answer.",
        provider="gemini",
        fallback_used=False,
    )
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_fixed_route_resolver(decision),
    )
    result = orchestrator.generate(route_request=_make_route_request(), query="Q?")
    assert result.answer_source == "llm"
    assert result.provider == "gemini"
    assert result.fallback_used is False


# ---------------------------------------------------------------------------
# Test 26 — ModelExecutionResult field validations
# ---------------------------------------------------------------------------


def test_model_execution_result_field_validation() -> None:
    # content must be non-empty
    with pytest.raises(ValidationError):
        ModelExecutionResult(content="", model="test_model")

    # model must be non-empty
    with pytest.raises(ValidationError):
        ModelExecutionResult(content="ok", model="")

    # input_tokens must be >= 0
    with pytest.raises(ValidationError):
        ModelExecutionResult(content="ok", model="m", input_tokens=-1)

    # output_tokens must be >= 0
    with pytest.raises(ValidationError):
        ModelExecutionResult(content="ok", model="m", output_tokens=-1)

    # latency_ms must be >= 0
    with pytest.raises(ValidationError):
        ModelExecutionResult(content="ok", model="m", latency_ms=-1)

    # valid minimal construction works
    result = ModelExecutionResult(content="ok", model="m")
    assert result.content == "ok"
    assert result.fallback_used is False
    assert result.metadata == {}
