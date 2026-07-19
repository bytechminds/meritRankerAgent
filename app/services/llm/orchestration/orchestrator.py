"""
app/services/llm_orchestration/orchestrator.py
------------------------------------------------
LlmOrchestrator — service-level coordinator for the orchestration layer (Part 3).

Responsibilities:
- Accept a RouteRequest + query + optional classification + optional context.
- Resolve a RouteDecision via an injected route resolver function.
- Build composed LlmMessages via an injected PromptResolver.
- Call the injected ModelExecutor boundary.
- Return a safe, normalised OrchestrationResult.

Design invariants:
- No LLM provider SDK calls.
- No AWS calls.
- No YAML parsing.
- No prompt file reading directly; delegates to PromptResolver.
- No secrets fetched, no secrets logged.
- Query and context are NEVER logged.
- model_executor is always required — no implicit mock in production path.

Non-goals (deferred):
- Real model_router / provider adapter wiring (Part 5+).
- SecretResolver.
- AgentCore config bundle prompt source.
- Langfuse prompt management.
- Provider-level fallback.
- Graph node integration.
- Memory, cache, verifier, visual formatter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

from config import get_settings
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult, OrchestrationResult
from schemas.llm_routing import RouteDecision, RouteRequest
from services.doubt_solver.answer_completion import (
    AnswerCompletionPolicy,
    StreamingMarkerFilter,
    build_continuation_messages,
    continuation_max_tokens,
    has_completion_marker,
    is_answer_generation_route,
    log_answer_completion,
    log_answer_generation_budget,
    marker_missing_but_answer_complete,
    should_run_continuation,
    strip_completion_marker,
)
from services.doubt_solver.answer_quality import (
    AnswerQualityPolicy,
    apply_safe_sanitizer,
    build_rewrite_messages,
    detect_final_answer,
    generation_failure_message,
    log_answer_quality_rewrite,
    log_answer_quality_validation,
    plain_text_fallback,
    rewrite_max_tokens,
    strip_duplicate_final_answer_section,
    validate_answer_quality,
)
from services.llm.orchestration.errors import (
    LlmExecutionError,
    LlmOrchestrationError,
    LlmOrchestratorError,
    ProviderExecutionError,
)
from services.llm.orchestration.prompt_resolver import PromptResolver, get_prompt_resolver
from services.llm.orchestration.route_resolver import resolve_route

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS: int = 4_000

# ---------------------------------------------------------------------------
# ModelExecutor Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelExecutor(Protocol):
    """Minimal boundary for any model backend.

    Implementations must be stateless between calls except for optional
    recording of the last call (for testing purposes).
    """

    def execute(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
    ) -> ModelExecutionResult:
        """Execute a model call and return the normalised result.

        Args:
            route_decision: The resolved route (model alias, temperature, etc.).
            messages:       The composed system + user messages from PromptResolver.

        Returns:
            ModelExecutionResult with content and safe metadata.

        Raises:
            Any exception — LlmOrchestrator wraps it as LlmExecutionError.
        """
        ...

    def execute_stream(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
    ) -> Iterator[str]:
        """Execute a model call and yield answer text chunks."""
        ...


# ---------------------------------------------------------------------------
# MockModelExecutor
# ---------------------------------------------------------------------------


class MockModelExecutor:
    """Test-only ModelExecutor that returns a configurable canned response.

    Usage:
        orchestrator, executor = create_mock_orchestrator_for_tests(content="2+2=4")
        result = orchestrator.generate(route_request=..., query="What is 2+2?")
        assert result.content == "2+2=4"
        assert executor.call_count == 1

    Attributes:
        last_route_decision: The RouteDecision from the most recent execute() call.
        last_messages:       The messages list from the most recent execute() call.
        call_count:          Total number of execute() invocations since construction.
    """

    def __init__(
        self,
        content: str = "Mock response. <ANSWER_DONE>",
        raise_on_execute: Exception | None = None,
        provider: str | None = "mock",
        fallback_used: bool = False,
        notify_fallback_on_stream: bool = False,
        finish_reason: str | None = "stop",
    ) -> None:
        self._content = content
        self._raise_on_execute = raise_on_execute
        self._provider = provider
        self._fallback_used = fallback_used
        self._notify_fallback_on_stream = notify_fallback_on_stream
        self._finish_reason = finish_reason
        self.last_route_decision: RouteDecision | None = None
        self.last_messages: list[LlmMessage] | None = None
        self.last_stream_finish_reason: str | None = None
        self.call_count: int = 0

    def execute(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
    ) -> ModelExecutionResult:
        """Record the call, optionally raise, or return canned content."""
        self.last_route_decision = route_decision
        self.last_messages = messages
        self.call_count += 1

        if self._raise_on_execute is not None:
            raise self._raise_on_execute

        return ModelExecutionResult(
            content=self._content,
            model=route_decision.model,
            provider=self._provider,
            finish_reason=self._finish_reason,
            fallback_used=self._fallback_used,
        )

    def execute_stream(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
        on_before_fallback: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        """Record the call, optionally raise, or yield canned content in chunks."""
        self.last_route_decision = route_decision
        self.last_messages = messages
        self.call_count += 1
        self.last_stream_finish_reason = self._finish_reason

        if self._raise_on_execute is not None:
            raise self._raise_on_execute

        if self._notify_fallback_on_stream and on_before_fallback is not None:
            on_before_fallback()

        chunk_size = 8
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]
        self.last_stream_finish_reason = self._finish_reason


# ---------------------------------------------------------------------------
# LlmOrchestrator
# ---------------------------------------------------------------------------


class LlmOrchestrator:
    """Coordinates route resolution, prompt composition, and model execution.

    Constructor:
        model_executor:    Required.  Any object conforming to ModelExecutor
                           protocol.  No default — callers must inject one.
        route_resolver_fn: Optional callable with signature
                           ``(RouteRequest) -> RouteDecision``.  Defaults to
                           the module-level ``resolve_route`` from Part 1.
        prompt_resolver:   Optional PromptResolver instance.  Defaults to the
                           module-level singleton from Part 2 (reads real
                           app/prompts/).  Inject a PromptResolver(tmp_path)
                           in tests.
    """

    def __init__(
        self,
        *,
        model_executor: ModelExecutor,
        route_resolver_fn: Callable[[RouteRequest], RouteDecision] | None = None,
        prompt_resolver: PromptResolver | None = None,
    ) -> None:
        self._model_executor = model_executor
        self._route_resolver_fn: Callable[[RouteRequest], RouteDecision] = (
            route_resolver_fn if route_resolver_fn is not None else resolve_route
        )
        self._prompt_resolver: PromptResolver = (
            prompt_resolver if prompt_resolver is not None else get_prompt_resolver()
        )

    def _finalize_generator_content(
        self,
        *,
        request_id: str,
        content: str,
        finish_reason: str | None,
        policy: AnswerCompletionPolicy,
        quality_policy: AnswerQualityPolicy,
        route_decision: RouteDecision,
        route_request: RouteRequest,
        messages: list[LlmMessage],
        continuation_used: bool,
        continuation_attempts: int,
    ) -> tuple[str, bool]:
        """Validate, optionally rewrite, sanitize, and strip completion marker."""
        rewrite_used = False
        working = strip_duplicate_final_answer_section(content)
        marker_found = has_completion_marker(working, policy.marker)
        answer_complete = detect_final_answer(working)

        if marker_missing_but_answer_complete(working, policy):
            logger.info(
                "answer_completion  request_id=%s  "
                "completion_marker_missing_but_answer_complete=true  continuation_used=false",
                request_id,
            )

        quality = validate_answer_quality(
            working,
            subject=route_decision.subject,
            difficulty=route_decision.difficulty,
            intent=route_request.intent,
            policy=quality_policy,
        )
        if quality.severity == "error":
            log_answer_quality_validation(
                request_id=request_id,
                route_id=route_decision.route_id,
                subject=route_decision.subject,
                difficulty=route_decision.difficulty,
                intent=route_request.intent,
                result=quality,
                output_chars=len(working),
                rewrite_required=False,
                sanitized=False,
            )
            return generation_failure_message(), False
        rewrite_required = quality.severity in ("rewrite_required", "unsafe")
        log_answer_quality_validation(
            request_id=request_id,
            route_id=route_decision.route_id,
            subject=route_decision.subject,
            difficulty=route_decision.difficulty,
            intent=route_request.intent,
            result=quality,
            output_chars=len(working),
            rewrite_required=rewrite_required,
            sanitized=quality.sanitized_text is not None,
        )

        if (
            rewrite_required
            and quality_policy.rewrite_enabled
            and quality_policy.max_rewrite_attempts >= 1
        ):
            rewrite_used = True
            rewrite_route = route_decision.model_copy(
                update={
                    "max_tokens": rewrite_max_tokens(
                        difficulty=route_decision.difficulty,
                        route_subject=route_decision.subject,
                    )
                }
            )
            rewrite_messages = build_rewrite_messages(
                messages,
                draft_answer=working,
            )
            try:
                rewrite_result = self._model_executor.execute(
                    route_decision=rewrite_route,
                    messages=rewrite_messages,
                )
                working = rewrite_result.content
                finish_reason = rewrite_result.finish_reason
                quality = validate_answer_quality(
                    working,
                    subject=route_decision.subject,
                    difficulty=route_decision.difficulty,
                    intent=route_request.intent,
                    policy=quality_policy,
                )
                log_answer_quality_rewrite(
                    request_id=request_id,
                    used=True,
                    attempt_count=1,
                    success=quality.severity in ("clean", "minor"),
                    final_output_chars=len(working),
                )
            except (LlmOrchestrationError, Exception):
                log_answer_quality_rewrite(
                    request_id=request_id,
                    used=True,
                    attempt_count=1,
                    success=False,
                    final_output_chars=0,
                )
                working = apply_safe_sanitizer(working, marker=policy.marker)
                if not detect_final_answer(working):
                    working = plain_text_fallback(subject=route_decision.subject)
        elif quality.sanitized_text is not None:
            working = quality.sanitized_text

        if quality.severity in ("rewrite_required", "unsafe") and not rewrite_used:
            working = apply_safe_sanitizer(working, marker=policy.marker)
            if not detect_final_answer(working):
                working = plain_text_fallback(subject=route_decision.subject)

        marker_found = has_completion_marker(working, policy.marker)
        answer_complete = detect_final_answer(working)
        final_content = strip_completion_marker(working, policy.marker)
        log_answer_completion(
            request_id=request_id,
            finish_reason=finish_reason,
            completion_marker_found=marker_found,
            final_answer_detected=answer_complete,
            continuation_used=continuation_used,
            continuation_attempts=continuation_attempts,
            rewrite_used=rewrite_used,
            output_chars=len(final_content),
            marker_missing_but_answer_complete=marker_missing_but_answer_complete(
                working, policy
            ),
        )
        return final_content, rewrite_used

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        route_request: RouteRequest,
        query: str,
        classification: Any | None = None,
        context: str | None = None,
    ) -> OrchestrationResult:
        """Run the full orchestration pipeline and return a safe result.

        Steps:
            1. Validate query (non-empty, within MAX_QUERY_CHARS).
            2. Resolve route via route_resolver_fn.
            3. Build messages via prompt_resolver.resolve().
            4. Call model_executor.execute().
            5. Wrap executor exceptions as LlmExecutionError.
            6. Build and return OrchestrationResult.

        Safe logging:
            Logs request_id, route_id, subject, task_role, difficulty, model,
            fallback_used, and latency_ms only.  Never logs query, context,
            classification data, or prompt content.

        Raises:
            LlmOrchestratorError:    Invalid query (empty or over limit).
            LlmExecutionError:       model_executor.execute() raised an exception.
            LlmRouteNotFoundError:   route_resolver_fn could not find a route.
            LlmRouteResolutionError: Unexpected route resolution error.
            PromptResolverError:     Prompt file missing, unsafe path, or invalid
                                     content.
        """
        # --- 1. Input validation -------------------------------------------
        if not query or not query.strip():
            raise LlmOrchestratorError("Query must not be empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise LlmOrchestratorError(
                f"Query exceeds maximum allowed length of {MAX_QUERY_CHARS} characters."
            )

        start_ms = int(time.monotonic() * 1000)

        # --- 2. Route resolution -------------------------------------------
        route_decision: RouteDecision = self._route_resolver_fn(route_request)

        # --- 3. Prompt composition -----------------------------------------
        messages: list[LlmMessage] = self._prompt_resolver.resolve(
            route_decision,
            query,
            classification,
            context,
        )

        context_chars = len(context) if context else 0
        is_generator = is_answer_generation_route(
            task_role=route_decision.task_role,
            route_id=route_decision.route_id,
        )
        if is_generator:
            log_answer_generation_budget(
                request_id=route_request.request_id,
                route_id=route_decision.route_id,
                subject=route_decision.subject,
                difficulty=route_decision.difficulty,
                intent=route_request.intent,
                max_output_tokens=route_decision.max_tokens,
                context_chars=context_chars,
            )

        policy = AnswerCompletionPolicy.from_settings(get_settings())
        quality_policy = AnswerQualityPolicy.from_settings(get_settings())

        # --- 4. Model execution -------------------------------------------
        try:
            execution_result: ModelExecutionResult = self._model_executor.execute(
                route_decision=route_decision,
                messages=messages,
            )
        except LlmOrchestrationError:
            # Controlled orchestration-layer errors bubble unchanged.
            raise
        except Exception as exc:
            raise LlmExecutionError(
                f"Model executor raised an unexpected error for route "
                f"'{route_decision.route_id}': {type(exc).__name__}"
            ) from exc

        content = execution_result.content
        finish_reason = execution_result.finish_reason
        continuation_used = False
        continuation_attempts = 0

        if should_run_continuation(
            content,
            finish_reason,
            policy,
            provider=execution_result.provider,
            task_role=route_decision.task_role,
            route_id=route_decision.route_id,
        ):
            continuation_attempts = 1
            continuation_used = True
            cont_route = route_decision.model_copy(
                update={
                    "max_tokens": continuation_max_tokens(
                        difficulty=route_decision.difficulty,
                        route_subject=route_decision.subject,
                    )
                }
            )
            cont_messages = build_continuation_messages(
                messages,
                partial_content=content,
                policy=policy,
            )
            try:
                cont_result = self._model_executor.execute(
                    route_decision=cont_route,
                    messages=cont_messages,
                )
            except LlmOrchestrationError:
                raise
            except Exception as exc:
                raise LlmExecutionError(
                    f"Continuation execution failed for route "
                    f"'{route_decision.route_id}': {type(exc).__name__}"
                ) from exc
            content = content + cont_result.content
            finish_reason = cont_result.finish_reason

        rewrite_used = False
        if is_generator:
            if quality_policy.validation_enabled:
                final_content, rewrite_used = self._finalize_generator_content(
                    request_id=route_request.request_id,
                    content=content,
                    finish_reason=finish_reason,
                    policy=policy,
                    quality_policy=quality_policy,
                    route_decision=route_decision,
                    route_request=route_request,
                    messages=messages,
                    continuation_used=continuation_used,
                    continuation_attempts=continuation_attempts,
                )
            else:
                marker_found = has_completion_marker(content, policy.marker)
                final_content = strip_completion_marker(content, policy.marker)
                if not final_content.strip():
                    final_content = generation_failure_message()
                log_answer_completion(
                    request_id=route_request.request_id,
                    finish_reason=finish_reason,
                    completion_marker_found=marker_found,
                    final_answer_detected=detect_final_answer(content),
                    continuation_used=continuation_used,
                    continuation_attempts=continuation_attempts,
                    rewrite_used=False,
                    output_chars=len(final_content),
                    marker_missing_but_answer_complete=marker_missing_but_answer_complete(
                        content, policy
                    ),
                )
        else:
            final_content = content

        elapsed_ms = int(time.monotonic() * 1000) - start_ms

        # --- 5. Derive answer_source ---------------------------------------
        answer_source = _derive_answer_source(execution_result)

        # --- 6. Safe logging -----------------------------------------------
        logger.info(
            "llm_orchestrator.generate  request_id=%s  route_id=%s  subject=%s  "
            "task_role=%s  difficulty=%s  model=%s  model_config_source=yaml  "
            "fallback_used=%s  latency_ms=%d",
            route_request.request_id,
            route_decision.route_id,
            route_decision.subject,
            route_decision.task_role,
            route_decision.difficulty,
            route_decision.model,
            execution_result.fallback_used,
            elapsed_ms,
        )

        # --- 7. Build OrchestrationResult ----------------------------------
        return OrchestrationResult(
            content=final_content,
            route_decision=route_decision,
            model=execution_result.model,
            provider=execution_result.provider,
            fallback_used=execution_result.fallback_used,
            finish_reason=finish_reason,
            input_tokens=execution_result.input_tokens,
            output_tokens=execution_result.output_tokens,
            latency_ms=execution_result.latency_ms,
            answer_source=answer_source,
            metadata={},
        )

    def generate_stream(
        self,
        *,
        route_request: RouteRequest,
        query: str,
        classification: Any | None = None,
        context: str | None = None,
        on_before_fallback: Callable[[], None] | None = None,
        on_before_continuation: Callable[[], None] | None = None,
        verify_before_stream: bool = True,
    ) -> Iterator[str]:
        """Run orchestration and yield answer text chunks from the model executor."""
        if not query or not query.strip():
            raise LlmOrchestratorError("Query must not be empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise LlmOrchestratorError(
                f"Query exceeds maximum allowed length of {MAX_QUERY_CHARS} characters."
            )

        start_ms = int(time.monotonic() * 1000)
        route_decision: RouteDecision = self._route_resolver_fn(route_request)
        messages: list[LlmMessage] = self._prompt_resolver.resolve(
            route_decision,
            query,
            classification,
            context,
        )

        context_chars = len(context) if context else 0
        is_generator = is_answer_generation_route(
            task_role=route_decision.task_role,
            route_id=route_decision.route_id,
        )
        if is_generator:
            log_answer_generation_budget(
                request_id=route_request.request_id,
                route_id=route_decision.route_id,
                subject=route_decision.subject,
                difficulty=route_decision.difficulty,
                intent=route_request.intent,
                max_output_tokens=route_decision.max_tokens,
                context_chars=context_chars,
            )

        policy = AnswerCompletionPolicy.from_settings(get_settings())
        quality_policy = AnswerQualityPolicy.from_settings(get_settings())
        continuation_used = False
        continuation_attempts = 0
        finish_reason: str | None = None
        streamed_parts: list[str] = []
        buffer_for_quality = (
            verify_before_stream and is_generator and quality_policy.validation_enabled
        )
        visible_emitted = False

        chunk_count = 0
        marker_filter = StreamingMarkerFilter(policy.marker) if is_generator else None
        try:
            for chunk in self._model_executor.execute_stream(
                route_decision=route_decision,
                messages=messages,
                on_before_fallback=on_before_fallback,
            ):
                chunk_count += 1
                if marker_filter is not None:
                    clean = marker_filter.feed(chunk)
                else:
                    clean = chunk
                if clean:
                    if clean.strip():
                        visible_emitted = True
                    streamed_parts.append(clean)
                    if not buffer_for_quality:
                        yield clean
            if marker_filter is not None:
                tail = marker_filter.flush()
                if tail:
                    if tail.strip():
                        visible_emitted = True
                    streamed_parts.append(tail)
                    if not buffer_for_quality:
                        yield tail
            finish_reason = getattr(self._model_executor, "last_stream_finish_reason", None)
        except ProviderExecutionError as exc:
            if is_generator and not visible_emitted:
                logger.warning(
                    "llm_orchestrator.generate_stream  all_attempts_failed  route_id=%s  "
                    "model=%s",
                    route_decision.route_id,
                    route_decision.model,
                )
            raise LlmExecutionError(
                f"Model stream failed for route '{route_decision.route_id}': "
                f"{type(exc).__name__}"
            ) from exc
        except LlmOrchestrationError:
            raise
        except Exception as exc:
            raise LlmExecutionError(
                f"Model executor stream raised an unexpected error for route "
                f"'{route_decision.route_id}': {type(exc).__name__}"
            ) from exc

        partial_content = "".join(streamed_parts)
        if not visible_emitted:
            logger.warning(
                "llm_orchestrator.generate_stream  empty_stream  route_id=%s  model=%s  "
                "finish_reason=%s — no visible text chunks received from provider",
                route_decision.route_id,
                route_decision.model,
                finish_reason or "none",
            )
            if is_generator:
                raise LlmExecutionError(
                    f"Model stream returned no visible output for route "
                    f"'{route_decision.route_id}'."
                )
        provider_hint = "mock" if isinstance(self._model_executor, MockModelExecutor) else None
        if (not visible_emitted or buffer_for_quality) and should_run_continuation(
            partial_content,
            finish_reason,
            policy,
            provider=provider_hint,
            task_role=route_decision.task_role,
            route_id=route_decision.route_id,
        ):
            continuation_attempts = 1
            continuation_used = True
            if on_before_continuation is not None:
                on_before_continuation()
            cont_route = route_decision.model_copy(
                update={
                    "max_tokens": continuation_max_tokens(
                        difficulty=route_decision.difficulty,
                        route_subject=route_decision.subject,
                    )
                }
            )
            cont_messages = build_continuation_messages(
                messages,
                partial_content=partial_content,
                policy=policy,
            )
            cont_filter = StreamingMarkerFilter(policy.marker)
            try:
                for chunk in self._model_executor.execute_stream(
                    route_decision=cont_route,
                    messages=cont_messages,
                    on_before_fallback=None,
                ):
                    chunk_count += 1
                    clean = cont_filter.feed(chunk)
                    if clean:
                        if clean.strip():
                            visible_emitted = True
                        streamed_parts.append(clean)
                        if not buffer_for_quality:
                            yield clean
                cont_tail = cont_filter.flush()
                if cont_tail:
                    if cont_tail.strip():
                        visible_emitted = True
                    streamed_parts.append(cont_tail)
                    if not buffer_for_quality:
                        yield cont_tail
                finish_reason = getattr(
                    self._model_executor, "last_stream_finish_reason", finish_reason
                )
            except LlmOrchestrationError:
                raise
            except Exception as exc:
                raise LlmExecutionError(
                    f"Continuation stream failed for route "
                    f"'{route_decision.route_id}': {type(exc).__name__}"
                ) from exc

        if is_generator:
            raw_content = "".join(streamed_parts)
            if verify_before_stream and quality_policy.validation_enabled:
                final_content, _rewrite_used = self._finalize_generator_content(
                    request_id=route_request.request_id,
                    content=raw_content,
                    finish_reason=finish_reason,
                    policy=policy,
                    quality_policy=quality_policy,
                    route_decision=route_decision,
                    route_request=route_request,
                    messages=messages,
                    continuation_used=continuation_used,
                    continuation_attempts=continuation_attempts,
                )
                if buffer_for_quality:
                    if final_content.strip():
                        yield final_content
                    else:
                        yield generation_failure_message()
            else:
                final_content = strip_completion_marker(raw_content, policy.marker)
                log_answer_completion(
                    request_id=route_request.request_id,
                    finish_reason=finish_reason,
                    completion_marker_found=has_completion_marker(
                        raw_content, policy.marker
                    ),
                    final_answer_detected=detect_final_answer(raw_content),
                    continuation_used=continuation_used,
                    continuation_attempts=continuation_attempts,
                    rewrite_used=False,
                    output_chars=len(final_content),
                    marker_missing_but_answer_complete=marker_missing_but_answer_complete(
                        raw_content, policy
                    ),
                )
        else:
            final_content = "".join(streamed_parts)

        elapsed_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "llm_orchestrator.generate_stream  request_id=%s  route_id=%s  subject=%s  "
            "task_role=%s  difficulty=%s  model=%s  chunk_count=%d  latency_ms=%d",
            route_request.request_id,
            route_decision.route_id,
            route_decision.subject,
            route_decision.task_role,
            route_decision.difficulty,
            route_decision.model,
            chunk_count,
            elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_answer_source(
    execution_result: ModelExecutionResult,
) -> str:
    """Derive answer_source from the execution result's provenance."""
    if execution_result.fallback_used:
        return "fallback"
    if execution_result.provider is None or execution_result.provider == "mock":
        return "mock"
    return "llm"


# ---------------------------------------------------------------------------
# Test factory
# ---------------------------------------------------------------------------


def create_mock_orchestrator_for_tests(
    content: str = "Mock response. <ANSWER_DONE>",
    prompt_resolver: PromptResolver | None = None,
    route_resolver_fn: Callable[[RouteRequest], RouteDecision] | None = None,
    raise_on_execute: Exception | None = None,
) -> tuple[LlmOrchestrator, MockModelExecutor]:
    """Create a fully-injected LlmOrchestrator backed by a MockModelExecutor.

    Returns both the orchestrator and the executor so tests can inspect
    ``executor.last_messages``, ``executor.last_route_decision``, and
    ``executor.call_count`` after calling ``orchestrator.generate()``.

    Args:
        content:           Text the mock executor returns as generated content.
        prompt_resolver:   Optional PromptResolver.  Pass ``PromptResolver(tmp_path)``
                           in tests that need prompt file isolation.  If None,
                           the module-level singleton (reading real app/prompts/)
                           is used.
        route_resolver_fn: Optional callable that overrides route resolution.
                           Useful for returning a fixed RouteDecision without
                           needing a real registry YAML.
        raise_on_execute:  If provided, the mock executor raises this exception
                           on execute() — for testing LlmExecutionError wrapping.

    Returns:
        ``(LlmOrchestrator, MockModelExecutor)``
    """
    executor = MockModelExecutor(content=content, raise_on_execute=raise_on_execute)
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        route_resolver_fn=route_resolver_fn,
        prompt_resolver=prompt_resolver,
    )
    return orchestrator, executor
