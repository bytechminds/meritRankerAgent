"""Tests for answer completion marker and bounded continuation."""

from __future__ import annotations

import logging

import pytest

import config as cfg_module
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult
from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_completion import (
    AnswerCompletionPolicy,
    StreamingMarkerFilter,
    build_continuation_messages,
    continuation_max_tokens,
    has_completion_marker,
    is_answer_generation_route,
    needs_continuation,
    resolve_generator_route_subject,
    should_run_continuation,
    strip_completion_marker,
)
from services.doubt_solver.answer_quality import detect_final_answer
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor


def _reset_settings() -> None:
    cfg_module._settings = None


class TestAnswerCompletionPolicy:
    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANSWER_COMPLETION_MARKER", "<ANSWER_DONE>")
        monkeypatch.setenv("ANSWER_CONTINUATION_ENABLED", "true")
        monkeypatch.setenv("ANSWER_CONTINUATION_MAX_ATTEMPTS", "1")
        _reset_settings()
        policy = AnswerCompletionPolicy.from_settings()
        assert policy.marker == "<ANSWER_DONE>"
        assert policy.continuation_enabled is True
        assert policy.continuation_max_attempts == 1

    def test_marker_found_no_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert (
            needs_continuation("Final Answer: 42 <ANSWER_DONE>", "stop", policy) is False
        )

    def test_finish_reason_length_triggers_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert needs_continuation("partial answer", "length", policy) is True

    def test_missing_marker_triggers_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert needs_continuation("partial answer without marker", "stop", policy) is True

    def test_empty_content_blocks_continuation(self) -> None:
        """Empty partial_content must not trigger continuation (would raise ValidationError)."""
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert needs_continuation("", "stop", policy) is False
        assert needs_continuation("", "length", policy) is False
        assert needs_continuation("", None, policy) is False

    def test_whitespace_only_content_blocks_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert needs_continuation("   ", "stop", policy) is False
        assert needs_continuation("\n\t\n", "length", policy) is False

    def test_should_run_continuation_blocks_on_empty(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert should_run_continuation(
            "",
            "stop",
            policy,
            provider="azure_openai",
            task_role="generator",
        ) is False

    def test_should_run_continuation_allows_non_empty(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert should_run_continuation(
            "partial answer without marker",
            "stop",
            policy,
            provider="azure_openai",
            task_role="generator",
        ) is True

    def test_stop_with_final_answer_no_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        text = "**Final Answer:**\n\\(15\\) km/h"
        assert needs_continuation(text, "stop", policy) is False
        assert detect_final_answer(text) is True

    def test_continuation_disabled(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=False,
            continuation_max_attempts=1,
        )
        assert needs_continuation("partial", "length", policy) is False


class TestContinuationHelpers:
    def test_strip_marker(self) -> None:
        text = "Answer is 5.\n<ANSWER_DONE>"
        assert strip_completion_marker(text, "<ANSWER_DONE>") == "Answer is 5."

    def test_streaming_marker_filter(self) -> None:
        filt = StreamingMarkerFilter("<ANSWER_DONE>")
        parts = []
        for piece in ("Done ", "<ANS", "WER_DONE>"):
            out = filt.feed(piece)
            if out:
                parts.append(out)
        tail = filt.flush()
        if tail:
            parts.append(tail)
        assert "".join(parts) == "Done "
        assert "<ANSWER_DONE>" not in "".join(parts)

    def test_continuation_messages_do_not_restart(self) -> None:
        base = [LlmMessage(role="system", content="sys"), LlmMessage(role="user", content="q")]
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        cont = build_continuation_messages(
            base, partial_content="Step 3: ...", policy=policy
        )
        assert cont[-1].role == "user"
        assert "Do not restart" in cont[-1].content
        assert cont[-2].role == "assistant"
        assert cont[-2].content == "Step 3: ..."

    def test_continuation_max_tokens_bounded(self) -> None:
        assert continuation_max_tokens(difficulty="basic", route_subject="general") == 500
        assert continuation_max_tokens(difficulty="intermediate", route_subject="math") == 700
        assert continuation_max_tokens(difficulty="advanced", route_subject="reasoning") == 1000
        assert continuation_max_tokens(difficulty="default", route_subject="practice") == 1000

    def test_route_subject_mapping(self) -> None:
        assert resolve_generator_route_subject(subject="general", intent="practice") == "practice"
        assert (
            resolve_generator_route_subject(
                subject="general",
                intent="explain",
                web_search_reason="current_affairs",
            )
            == "current_affairs"
        )
        assert resolve_generator_route_subject(subject="math", intent="solve") == "math"


class TestGeneratorRouteBudgets:
    def test_math_intermediate_budget(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        route = LlmConfigRegistry().get_route("math", "generator", "intermediate")
        assert route is not None
        assert route.max_tokens == 1500

    def test_practice_route_budget(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        route = LlmConfigRegistry().get_route("practice", "generator", "default")
        assert route is not None
        assert route.max_tokens == 2600

    def test_current_affairs_route_budget(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        route = LlmConfigRegistry().get_route("current_affairs", "generator", "default")
        assert route is not None
        assert route.max_tokens == 1800


class TestOrchestratorContinuation:
    def test_continuation_runs_once_on_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        cfg_module._settings = None
        primary = MockModelExecutor(content="Partial solution", finish_reason="length")
        continuation = MockModelExecutor(
            content=" final part <ANSWER_DONE>",
            finish_reason="stop",
        )
        call = {"n": 0}

        class _SwitchingExecutor:
            last_stream_finish_reason = "stop"

            def execute(self, *, route_decision, messages):
                call["n"] += 1
                if call["n"] == 1:
                    return primary.execute(route_decision=route_decision, messages=messages)
                return continuation.execute(route_decision=route_decision, messages=messages)

        orchestrator = LlmOrchestrator(model_executor=_SwitchingExecutor())
        result = orchestrator.generate(
            route_request=RouteRequest(
                request_id="c1",
                subject="math",
                task_role="generator",
                difficulty="intermediate",
                intent="solve",
            ),
            query="Solve speed problem",
        )
        assert call["n"] == 2
        assert has_completion_marker(result.content + "<ANSWER_DONE>", "<ANSWER_DONE>") or (
            "final part" in result.content
        )
        assert "<ANSWER_DONE>" not in result.content

    def test_marker_present_skips_second_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        cfg_module._settings = None
        executor = MockModelExecutor(content="Done <ANSWER_DONE>", finish_reason="stop")
        orchestrator = LlmOrchestrator(model_executor=executor)
        result = orchestrator.generate(
            route_request=RouteRequest(
                request_id="c2",
                subject="math",
                task_role="generator",
                difficulty="basic",
                intent="solve",
            ),
            query="2+2",
        )
        assert executor.call_count == 1
        assert "<ANSWER_DONE>" not in result.content

    def test_stop_with_final_answer_skips_continuation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        cfg_module._settings = None
        executor = MockModelExecutor(
            content="**Final Answer:**\n\\(15\\) km/h",
            finish_reason="stop",
        )
        orchestrator = LlmOrchestrator(model_executor=executor)
        result = orchestrator.generate(
            route_request=RouteRequest(
                request_id="c3",
                subject="math",
                task_role="generator",
                difficulty="intermediate",
                intent="solve",
            ),
            query="speed problem",
        )
        assert executor.call_count == 1
        assert "15" in result.content
        assert "<ANSWER_DONE>" not in result.content

    def test_malformed_math_triggers_rewrite_not_continuation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
        monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
        cfg_module._settings = None
        primary = MockModelExecutor(
            content="Actually check setup. Final Answer: $15$ km/h",
            finish_reason="stop",
        )
        rewrite = MockModelExecutor(
            content=(
                "**Final Answer:**\n\\(15\\) km/h\n<ANSWER_DONE>"
            ),
            finish_reason="stop",
        )
        call = {"n": 0}

        class _SwitchingExecutor:
            last_stream_finish_reason = "stop"

            def execute(self, *, route_decision, messages):
                call["n"] += 1
                if call["n"] == 1:
                    return primary.execute(route_decision=route_decision, messages=messages)
                return rewrite.execute(route_decision=route_decision, messages=messages)

        orchestrator = LlmOrchestrator(model_executor=_SwitchingExecutor())
        result = orchestrator.generate(
            route_request=RouteRequest(
                request_id="rw1",
                subject="math",
                task_role="generator",
                difficulty="intermediate",
                intent="solve",
            ),
            query="speed",
        )
        assert call["n"] == 2
        assert "$" not in result.content
        assert "15" in result.content

    def test_complete_rewrite_without_marker_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
        monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
        cfg_module._settings = None
        outputs = iter(
            [
                "Actually check setup. **Final Answer:** $15$ km/h <ANSWER_DONE>",
                r"**Final Answer:** \(15\) km/h",
            ]
        )

        class _Executor:
            last_stream_finish_reason = "stop"

            def execute(self, *, route_decision, messages):
                return ModelExecutionResult(
                    content=next(outputs),
                    model=route_decision.model,
                    finish_reason="stop",
                )

        result = LlmOrchestrator(model_executor=_Executor()).generate(
            route_request=RouteRequest(
                request_id="rw-missing-marker",
                subject="math",
                task_role="generator",
                difficulty="intermediate",
                intent="solve",
            ),
            query="speed",
        )

        assert result.content == r"**Final Answer:** \(15\) km/h"
        assert result.final_answer is not None
        assert result.final_answer.quality_status == "passed_quality_gate"
        assert any(
            getattr(record, "observability_event", {}).get("details", {}).get("outcome")
            == "marker_missing_but_complete"
            for record in caplog.records
        )

    def test_empty_rewrite_fails_safely(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
        monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
        cfg_module._settings = None
        outputs = iter(["Actually. **Final Answer:** $15$ <ANSWER_DONE>", " "])

        class _Executor:
            last_stream_finish_reason = "stop"

            def execute(self, *, route_decision, messages):
                return ModelExecutionResult(
                    content=next(outputs),
                    model=route_decision.model,
                    finish_reason="stop",
                )

        result = LlmOrchestrator(model_executor=_Executor()).generate(
            route_request=RouteRequest(
                request_id="rw-empty",
                subject="math",
                task_role="generator",
                difficulty="intermediate",
                intent="solve",
            ),
            query="speed",
        )

        assert result.final_answer is not None
        assert result.final_answer.quality_status == "failed_quality_gate"
        assert any(
            getattr(record, "observability_event", {}).get("details", {}).get("outcome")
            == "provider_empty"
            for record in caplog.records
        )


class TestCompletionRouteGating:
    def test_classifier_route_skips_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert (
            should_run_continuation(
                "partial json without marker",
                "stop",
                policy,
                provider="azure_openai",
                task_role="classifier",
                route_id="general.classifier.default",
            )
            is False
        )

    def test_generator_route_allows_continuation(self) -> None:
        policy = AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        )
        assert (
            should_run_continuation(
                "partial answer",
                "length",
                policy,
                provider="azure_openai",
                task_role="generator",
                route_id="math.generator.default",
            )
            is True
        )

    def test_is_answer_generation_route(self) -> None:
        assert is_answer_generation_route(task_role="generator") is True
        assert is_answer_generation_route(route_id="math.generator.default") is True
        assert is_answer_generation_route(task_role="classifier") is False

    def test_classifier_orchestrator_single_call(self) -> None:
        executor = MockModelExecutor(content='{"subject":"math"}', finish_reason="stop")
        orchestrator = LlmOrchestrator(model_executor=executor)
        result = orchestrator.generate(
            route_request=RouteRequest(
                request_id="cls-1",
                subject="general",
                task_role="classifier",
                difficulty="default",
                intent="classify",
            ),
            query="40 km/hr speed problem",
        )
        assert executor.call_count == 1
        assert result.content == '{"subject":"math"}'


class TestOrchestratorEmptyStream:
    class _EmptyStreamExecutor:
        last_stream_finish_reason = "length"

        def execute_stream(self, *, route_decision, messages, on_before_fallback=None):  # noqa: ANN001
            return
            yield  # pragma: no cover

        def execute(self, *, route_decision, messages):  # noqa: ANN001
            raise AssertionError("not used")

    def test_empty_stream_raises_controlled_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from services.llm.orchestration.errors import LlmExecutionError

        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        cfg_module._settings = None
        orchestrator = LlmOrchestrator(model_executor=self._EmptyStreamExecutor())
        with pytest.raises(LlmExecutionError, match="no visible output"):
            list(
                orchestrator.generate_stream(
                    route_request=RouteRequest(
                        request_id="empty-stream-1",
                        subject="reasoning",
                        task_role="generator",
                        difficulty="advanced",
                        intent="solve",
                    ),
                    query="A reasoning puzzle",
                )
            )
