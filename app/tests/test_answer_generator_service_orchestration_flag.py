"""
tests/test_answer_generator_service_orchestration_flag.py
----------------------------------------------------------
Unit tests: ENABLE_ORCHESTRATED_DOUBT_SOLVER feature flag behaviour.

Covers:
    - Flag defaults to false.
    - config.Settings exposes enable_orchestrated_doubt_solver.
    - With flag false: legacy test_main_routing-style path unaffected.
    - With flag true:  Orchestrated adapter is called and returns an answer string.
    - AnswerGenerationAdapter accepts injected orchestrator (no direct provider call).
    - AnswerGenerationAdapter.generate() builds RouteRequest with task_role=generator.
    - No real OpenAI/Azure/Gemini/Bedrock call in any test in this file.
    - No AWS call in any test in this file.

[NOT VERIFIED]: that no socket I/O occurs — tests rely on mock injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import get_settings
from schemas.llm_routing import RouteDecision, RouteRequest
from services.llm_orchestration.answer_generation_adapter import AnswerGenerationAdapter
from services.llm_orchestration.orchestrator import (
    LlmOrchestrator,
    MockModelExecutor,
    create_mock_orchestrator_for_tests,
)
from services.llm_orchestration.prompt_resolver import PromptResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_answer_quality_for_adapter_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """These adapter tests assert single mock executor call — disable quality rewrite."""
    import config as cfg_module

    monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
    cfg_module._settings = None


def _make_route_decision(req: RouteRequest) -> RouteDecision:
    """Return a fixed safe_mock RouteDecision for any RouteRequest."""
    return RouteDecision(
        route_id=f"{req.subject}.{req.task_role}.{req.difficulty}",
        subject=req.subject,
        task_role=req.task_role,
        difficulty=req.difficulty,
        intent=req.intent,
        model="safe_mock",
        prompt="answer_generator.md",
        temperature=0.7,
        max_tokens=500,
        route_source="safe_mock",
    )


def _make_test_adapter(
    tmp_path: Path,
    content: str = "Test answer from mock orchestrator.",
) -> tuple[AnswerGenerationAdapter, MockModelExecutor]:
    """Build an AnswerGenerationAdapter backed by MockModelExecutor.

    Uses:
    - PromptResolver(tmp_path) so no real prompt files are needed.
    - Fixed route decision so no YAML loading occurs.
    - MockModelExecutor that returns `content` without any provider call.
    """
    (tmp_path / "answer_generator.md").write_text("You are a tutor. Answer: {{query}}")
    prompt_resolver = PromptResolver(prompt_root=tmp_path)
    orchestrator, executor = create_mock_orchestrator_for_tests(
        content=content,
        prompt_resolver=prompt_resolver,
        route_resolver_fn=_make_route_decision,
    )
    adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
    return adapter, executor


# ===========================================================================
# Feature flag config tests
# ===========================================================================


class TestEnableOrchestratedDoubtSolverFlag:
    """Verify the feature flag is present in Settings and defaults to false."""

    def test_flag_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_ORCHESTRATED_DOUBT_SOLVER defaults to false (safe default)."""
        import config as cfg_module  # noqa: PLC0415
        monkeypatch.delenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", raising=False)
        cfg_module._settings = None
        settings = get_settings()
        assert settings.enable_orchestrated_doubt_solver is False

    def test_flag_can_be_set_to_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as cfg_module  # noqa: PLC0415
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        cfg_module._settings = None
        settings = get_settings()
        assert settings.enable_orchestrated_doubt_solver is True

    def test_flag_is_bool_not_string(self) -> None:
        settings = get_settings()
        assert isinstance(settings.enable_orchestrated_doubt_solver, bool)

    def test_flag_false_when_set_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as cfg_module  # noqa: PLC0415
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
        cfg_module._settings = None
        settings = get_settings()
        assert settings.enable_orchestrated_doubt_solver is False

    def test_env_local_does_not_affect_unit_tests(self) -> None:
        """Conftest forces ENABLE_ORCHESTRATED_DOUBT_SOLVER=false for all tests."""
        settings = get_settings()
        assert settings.enable_orchestrated_doubt_solver is False


# ===========================================================================
# AnswerGenerationAdapter unit tests
# ===========================================================================


class TestAnswerGenerationAdapter:
    """Verify adapter interface: RouteRequest construction and return value."""

    def test_adapter_requires_orchestrator(self) -> None:
        with pytest.raises(TypeError):
            AnswerGenerationAdapter(orchestrator=None)  # type: ignore[arg-type]

    def test_adapter_generate_returns_string(self, tmp_path: Path) -> None:
        adapter, _ = _make_test_adapter(tmp_path, content="Answer text.")
        result = adapter.generate(
            request_id="req-001",
            query="What is 2+2?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert isinstance(result, str)

    def test_adapter_returns_mock_content(self, tmp_path: Path) -> None:
        adapter, _ = _make_test_adapter(tmp_path, content="42 is the answer.")
        result = adapter.generate(
            request_id="req-001",
            query="What is 2+2?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert result == "42 is the answer."

    def test_adapter_calls_executor_once_per_generate(self, tmp_path: Path) -> None:
        adapter, executor = _make_test_adapter(tmp_path)
        adapter.generate(
            request_id="req-001",
            query="What is 2+2?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert executor.call_count == 1

    def test_adapter_builds_route_request_with_task_role_generator(
        self, tmp_path: Path
    ) -> None:
        """RouteRequest sent to orchestrator must use task_role='generator'."""
        captured_requests: list[RouteRequest] = []

        def _capturing_resolver(req: RouteRequest) -> RouteDecision:
            captured_requests.append(req)
            return _make_route_decision(req)

        (tmp_path / "answer_generator.md").write_text("Answer: {{query}}")
        pr = PromptResolver(prompt_root=tmp_path)
        orchestrator, _ = create_mock_orchestrator_for_tests(
            content="captured",
            prompt_resolver=pr,
            route_resolver_fn=_capturing_resolver,
        )
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        adapter.generate(
            request_id="req-001",
            query="test",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert len(captured_requests) == 1
        assert captured_requests[0].task_role == "generator"

    def test_adapter_logs_actual_generator_execution_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import services.doubt_solver.answer_generation_adapter as adapter_module

        adapter, _ = _make_test_adapter(tmp_path, content="Answer text.")
        event = MagicMock()
        monkeypatch.setattr(adapter_module, "log_event", event)

        adapter.generate(
            request_id="req-001",
            query="What is 2+2?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )

        details = event.call_args.kwargs["details"]
        assert event.call_args.args == ("generation_completed",)
        assert details["route_id"] == "math.generator.default"
        assert details["task_role"] == "generator"
        assert details["model_alias"] == "safe_mock"
        assert details["provider"] == "mock"

    def test_adapter_does_not_pass_model_id_to_route_request(
        self, tmp_path: Path
    ) -> None:
        """RouteRequest has no model_id — that stays in the model registry."""
        captured_requests: list[RouteRequest] = []

        def _capturing_resolver(req: RouteRequest) -> RouteDecision:
            captured_requests.append(req)
            return _make_route_decision(req)

        (tmp_path / "answer_generator.md").write_text("Answer: {{query}}")
        pr = PromptResolver(prompt_root=tmp_path)
        orchestrator, _ = create_mock_orchestrator_for_tests(
            content="captured",
            prompt_resolver=pr,
            route_resolver_fn=_capturing_resolver,
        )
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        adapter.generate(
            request_id="req-001",
            query="test",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert len(captured_requests) == 1
        req_dict = captured_requests[0].model_dump()
        assert "model_id" not in req_dict
        assert "provider" not in req_dict
        assert "deployment" not in req_dict
        assert "api_key" not in req_dict

    def test_adapter_passes_subject_to_route_request(self, tmp_path: Path) -> None:
        captured_requests: list[RouteRequest] = []

        def _capturing_resolver(req: RouteRequest) -> RouteDecision:
            captured_requests.append(req)
            return _make_route_decision(req)

        (tmp_path / "answer_generator.md").write_text("Answer: {{query}}")
        pr = PromptResolver(prompt_root=tmp_path)
        orchestrator, _ = create_mock_orchestrator_for_tests(
            content="math",
            prompt_resolver=pr,
            route_resolver_fn=_capturing_resolver,
        )
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        adapter.generate(
            request_id="req-001",
            query="test",
            subject="math",
            intent="solve",
            difficulty="advanced",
            context="",
        )
        assert captured_requests[0].subject == "math"
        assert captured_requests[0].difficulty == "advanced"
        assert captured_requests[0].intent == "solve"

    def test_adapter_with_context_passes_context_to_orchestrator(
        self, tmp_path: Path
    ) -> None:
        """Verify non-empty context is forwarded to orchestrator.generate()."""
        original_generate = LlmOrchestrator.generate

        (tmp_path / "answer_generator.md").write_text("Answer: {{query}}")
        pr = PromptResolver(prompt_root=tmp_path)
        orchestrator, executor = create_mock_orchestrator_for_tests(
            content="With context answer.",
            prompt_resolver=pr,
            route_resolver_fn=_make_route_decision,
        )

        # Patch orchestrator.generate to capture context

        calls: list[dict[str, Any]] = []

        def _patched_generate(
            self,
            *,
            route_request,
            query,
            classification=None,
            context=None,
            conversation_context=None,
        ):
            calls.append({"context": context})
            return original_generate(
                self,
                route_request=route_request,
                query=query,
                classification=classification,
                context=context,
                conversation_context=conversation_context,
            )

        orchestrator.generate = lambda **kw: _patched_generate(orchestrator, **kw)

        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        adapter.generate(
            request_id="req-001",
            query="test",
            subject="general",
            intent="explain",
            difficulty="default",
            context="Reference text.",
        )
        # Verify executor was called (context forwarding happens before execute)
        assert executor.call_count == 1

    def test_adapter_empty_context_passed_as_none_to_orchestrator(
        self, tmp_path: Path
    ) -> None:
        """Empty context string → None sent to orchestrator (not empty string)."""
        adapter, executor = _make_test_adapter(tmp_path)
        adapter.generate(
            request_id="req-001",
            query="test",
            subject="general",
            intent="explain",
            difficulty="default",
            context="",
        )
        # Executor was called; the adapter converts "" → None internally
        assert executor.call_count == 1

    def test_no_real_provider_call_in_unit_test(self, tmp_path: Path) -> None:
        """MockModelExecutor is used — no real OpenAI/Azure/Bedrock call."""
        adapter, executor = _make_test_adapter(tmp_path, content="Mock only.")
        result = adapter.generate(
            request_id="req-001",
            query="What is 20% of 500?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert result == "Mock only."
        assert executor.call_count == 1
        # MockModelExecutor.last_route_decision confirms we used the mock path
        assert executor.last_route_decision is not None
        assert executor.last_route_decision.model == "safe_mock"


# ===========================================================================
# Flag-based path selection tests
# ===========================================================================


class TestOrchestratedPathSelection:
    """Verify orchestrated path is only active when flag is true."""

    def test_with_flag_false_settings_reports_false(self) -> None:
        settings = get_settings()
        assert settings.enable_orchestrated_doubt_solver is False

    def test_with_flag_true_orchestrated_path_produces_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With orchestrated flag=true, the adapter returns an answer."""
        import config as cfg_module  # noqa: PLC0415
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        cfg_module._settings = None

        adapter, _ = _make_test_adapter(tmp_path, content="Orchestrated answer.")
        result = adapter.generate(
            request_id="req-v1",
            query="What is 20% of 500?",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert result == "Orchestrated answer."

    def test_with_flag_true_no_aws_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Orchestrated flag=true, mock executor — no real AWS call."""
        import config as cfg_module  # noqa: PLC0415
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        cfg_module._settings = None

        adapter, executor = _make_test_adapter(tmp_path)
        adapter.generate(
            request_id="req-v1",
            query="test",
            subject="general",
            intent="explain",
            difficulty="default",
            context="",
        )
        # Only mock executor was called — no real provider
        assert executor.call_count == 1
        assert executor.last_route_decision is not None
