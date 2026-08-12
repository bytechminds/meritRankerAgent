"""
app/services/llm_orchestration/model_execution.py
-------------------------------------------------
Model execution boundary backed by the LLM config registry.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from observability import (
    current_execution_context,
    current_llm_attempt_type,
    current_request_context,
    log_event,
    record_llm_call,
    update_request_summary,
)
from schemas.llm import LlmMessage
from schemas.llm_orchestration import (
    ModelExecutionResult,
    ProviderExecutionRequest,
    ResolvedModelConfig,
)
from schemas.llm_routing import PracticeGenerationWorkload, RouteDecision
from schemas.llm_usage import ProviderTokenUsage, UsageStatus
from services.llm.orchestration.errors import (
    ModelExecutionConfigError,
    ProviderExecutionError,
)
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.practice_generation_capacity import (
    PracticeGenerationCapacityPolicy,
)
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.finish_reasons import normalize_completion_outcome
from services.llm.providers.usage import clear_stream_usage, consume_stream_usage

if TYPE_CHECKING:
    from services.llm.providers.provider_factory import ProviderAdapterFactory
    from services.secrets.provider_credentials import ProviderCredentialResolver

logger = logging.getLogger(__name__)


def _configured_model_name(resolution: ResolvedModelConfig) -> str:
    config = resolution.model_config
    return config.deployment or config.model_id or resolution.model_alias


def _log_model_execution(
    *,
    route_decision: RouteDecision,
    resolution: ResolvedModelConfig,
    duration_ms: int,
    fallback_used: bool,
) -> None:
    configured_model = _configured_model_name(resolution)
    logger.info(
        "model_execution  route=%s  role=%s  provider=%s  model=%s  "
        "model_alias=%s  fallback_used=%s  duration_ms=%d",
        route_decision.route_id,
        route_decision.task_role,
        resolution.provider,
        configured_model,
        resolution.model_alias,
        str(fallback_used).lower(),
        duration_ms,
    )
    log_event(
        "model_execution_completed",
        component="llm.model_execution",
        stage=route_decision.task_role,
        status="completed",
        duration_ms=duration_ms,
        details={
            "route": route_decision.route_id,
            "role": route_decision.task_role,
            "provider": resolution.provider,
            "model": configured_model,
            "model_alias": resolution.model_alias,
            "fallback_used": fallback_used,
        },
    )


def _is_visible_text(chunk: str) -> bool:
    """True when a stream chunk contains user-visible answer text."""
    return bool(chunk and chunk.strip())


def _is_practice_generator_token_exhausted(
    *,
    route_decision: RouteDecision,
    finish_reason: str | None,
) -> bool:
    """Classify all truncated structured practice responses before parsing."""
    return (
        route_decision.task_role == "generator"
        and route_decision.intent == "practice"
        and normalize_completion_outcome(finish_reason) == "output_token_exhausted"
    )


def _provider_response_failure_kind(
    *,
    route_decision: RouteDecision,
    response_error: LlmProviderResponseError,
) -> str:
    outcome = getattr(
        response_error, "normalized_finish_reason", None
    ) or normalize_completion_outcome(getattr(response_error, "finish_reason", None))
    if _is_practice_generator_token_exhausted(
        route_decision=route_decision,
        finish_reason=getattr(response_error, "finish_reason", None),
    ):
        return "output_token_exhausted"
    if outcome == "content_filtered":
        return "safety_blocked"
    return response_error.failure_kind


def _model_result_failure_kind(
    *,
    route_decision: RouteDecision,
    result: ModelExecutionResult,
) -> str | None:
    outcome = result.normalized_finish_reason
    if outcome == "unknown":
        outcome = normalize_completion_outcome(result.finish_reason)
    if (
        outcome == "output_token_exhausted"
        and _is_practice_generator(route_decision)
    ):
        return "output_token_exhausted"
    if outcome == "content_filtered":
        return "safety_blocked"
    return None


def _log_practice_generator_token_exhausted(
    *,
    route_decision: RouteDecision,
    model_alias: str,
    content: str | None = None,
    finish_reason: str | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    actual_output_budget: int | None = None,
) -> None:
    execution_context = current_execution_context()
    requested_items = (
        len(execution_context.slot_ids)
        if execution_context is not None and execution_context.slot_ids
        else None
    )
    complete_items, structured_payload_complete = _structured_question_payload_status(
        content
    )
    answer_tokens = (
        max(output_tokens - reasoning_tokens, 0)
        if output_tokens is not None and reasoning_tokens is not None
        else None
    )
    log_event(
        "practice_generator_token_exhausted",
        component="llm.model_execution",
        stage="generator",
        status="failed",
        error_code="PRACTICE_GENERATOR_OUTPUT_TOKEN_EXHAUSTED",
        details={
            "route": route_decision.route_id,
            "modelAlias": model_alias,
            "failureClass": "output_token_exhausted",
            "finishReason": finish_reason,
            "normalizedFailureReason": normalize_completion_outcome(finish_reason),
            "requestedItems": requested_items,
            "completeItemsDetected": complete_items,
            "structuredPayloadComplete": structured_payload_complete,
            "structuredPayloadStartsWithObject": bool(
                content and content.lstrip().startswith("{")
            ),
            "structuredPayloadEndsMidObjectOrArray": _ends_mid_json_structure(content),
            "outputTokenLimit": actual_output_budget or route_decision.max_tokens,
            "outputTokens": output_tokens,
            "reasoningTokens": reasoning_tokens,
            "answerTokens": answer_tokens,
        },
        level=logging.WARNING,
    )


def _structured_question_payload_status(content: str | None) -> tuple[int, bool]:
    """Inspect structure only; this deliberately never emits provider content."""
    if not content:
        return 0, False
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return _count_complete_question_items(content), False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
        return 0, False
    return len(parsed["questions"]), True


def _count_complete_question_items(content: str) -> int:
    """Count fully decodable array items without retaining their values."""
    questions_key = content.find('"questions"')
    if questions_key < 0:
        return 0
    array_start = content.find("[", questions_key)
    if array_start < 0:
        return 0
    decoder = json.JSONDecoder()
    cursor = array_start + 1
    complete_items = 0
    while cursor < len(content):
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor >= len(content) or content[cursor] == "]":
            return complete_items
        try:
            item, cursor = decoder.raw_decode(content, cursor)
        except ValueError:
            return complete_items
        if not isinstance(item, dict):
            return complete_items
        complete_items += 1
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor < len(content) and content[cursor] == ",":
            cursor += 1
            continue
        return complete_items
    return complete_items


def _ends_mid_json_structure(content: str | None) -> bool:
    """Report unfinished object/array structure without logging raw output."""
    if not content:
        return False
    depth = 0
    in_string = False
    escaped = False
    for character in content:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth = max(depth - 1, 0)
    return depth > 0 or in_string


def _log_practice_model_fallback(
    *,
    event_name: str,
    route_decision: RouteDecision,
    model_alias: str,
) -> None:
    log_event(
        event_name,
        component="llm.model_execution",
        stage="generator",
        status="started" if event_name.endswith("started") else "completed",
        details={
            "route": route_decision.route_id,
            "modelAlias": model_alias,
            "failureClass": "output_token_exhausted",
        },
    )


def _is_practice_generator(route_decision: RouteDecision) -> bool:
    return (
        route_decision.task_role == "generator"
        and route_decision.intent == "practice"
    )


def _log_practice_generator_attempt(
    *,
    event_name: str,
    route_decision: RouteDecision,
    model_alias: str,
    provider: str | None,
    attempt_type: str,
    fallback_index: int,
    duration_ms: int,
    failure_kind: str | None = None,
    error_class: str | None = None,
    actual_output_budget: int | None = None,
    capacity_escalated: bool = False,
    status: str,
) -> None:
    """Emit bounded diagnostics for practice model attempts without content."""
    if not _is_practice_generator(route_decision):
        return
    fallback_eligible = (
        failure_kind in FALLBACK_ELIGIBLE_FAILURE_KINDS
        if failure_kind is not None
        else None
    )
    details: dict[str, object] = {
        "routeId": route_decision.route_id,
        "modelAlias": model_alias,
        "provider": provider or "",
        "attemptType": attempt_type,
        "fallbackIndex": fallback_index,
        "actualBudgetSent": actual_output_budget or route_decision.max_tokens,
        "capacityEscalated": capacity_escalated,
    }
    capacity = route_decision.practice_generation_capacity
    if capacity is not None:
        details.update(
            {
                "configuredInitialBudget": capacity.initial_max_output_tokens,
                "hardProductCap": capacity.product_hard_max_output_tokens,
                "complexity": capacity.complexity,
                "capacitySlotCount": capacity.slot_count,
            }
        )
    execution_context = current_execution_context()
    if execution_context is not None:
        if execution_context.activity_id is not None:
            details["activityId"] = execution_context.activity_id
        if execution_context.batch_id is not None:
            details["batchId"] = execution_context.batch_id
        if execution_context.slot_ids:
            details["slotCount"] = len(execution_context.slot_ids)
            details["slotIds"] = ",".join(execution_context.slot_ids)
    if error_class is not None:
        details["errorClass"] = error_class
    if failure_kind is not None:
        details["errorCode"] = failure_kind
    if fallback_eligible is not None:
        details["fallbackEligible"] = fallback_eligible
        details["retryEligible"] = fallback_eligible
    log_event(
        event_name,
        component="llm.model_execution",
        stage="generator",
        status=status,
        duration_ms=duration_ms,
        error_code=failure_kind,
        details=details,
        level=logging.WARNING if status == "failed" else logging.INFO,
    )


def _log_practice_generator_completion(
    *,
    route_decision: RouteDecision,
    resolution: ResolvedModelConfig,
    result: ModelExecutionResult,
    actual_output_budget: int,
    fallback_used: bool,
    capacity_escalated: bool,
) -> None:
    """Emit safe actual-usage telemetry; limits are never treated as usage."""
    if not _is_practice_generator(route_decision):
        return
    capacity = route_decision.practice_generation_capacity
    outcome = result.normalized_finish_reason
    if outcome == "unknown":
        outcome = normalize_completion_outcome(result.finish_reason)
    log_event(
        "practice_generator_model_completed",
        component="llm.model_execution",
        stage="generator",
        status="completed",
        details={
            "route": route_decision.route_id,
            "modelAlias": resolution.model_alias,
            "provider": resolution.provider,
            "subject": route_decision.subject,
            "difficulty": route_decision.difficulty,
            "complexity": capacity.complexity if capacity is not None else None,
            "slotCount": capacity.slot_count if capacity is not None else None,
            "configuredInitialBudget": (
                capacity.initial_max_output_tokens
                if capacity is not None
                else route_decision.max_tokens
            ),
            "actualBudgetSent": actual_output_budget,
            "hardProductCap": (
                capacity.product_hard_max_output_tokens if capacity is not None else None
            ),
            "inputTokens": result.input_tokens,
            "completionTokens": result.output_tokens,
            "reasoningTokens": result.reasoning_tokens,
            "visibleOutputTokens": (
                max(result.output_tokens - result.reasoning_tokens, 0)
                if result.output_tokens is not None and result.reasoning_tokens is not None
                else None
            ),
            "finishReason": result.finish_reason,
            "normalizedFailureReason": outcome,
            "fallbackUsed": fallback_used,
            "capacityEscalated": capacity_escalated,
            "contractValid": None,
            "verifierOutcome": None,
        },
    )


def _validate_model_role(
    route_decision: RouteDecision, *, allowed_task_roles: list[str]
) -> None:
    if allowed_task_roles and route_decision.task_role not in allowed_task_roles:
        raise ModelExecutionConfigError(
            f"Model alias '{route_decision.model}' is not allowed for task role "
            f"'{route_decision.task_role}'."
        )


def _log_generation_empty_output(
    *,
    route_id: str,
    model_alias: str,
    provider: str | None,
    deployment: str | None,
    finish_reason: str | None,
    visible_chunk_count: int,
    failure_kind: str,
    fallback_eligible: bool,
    fallback_attempted: bool = False,
) -> None:
    """Log empty generator output diagnostics (no chunk content)."""
    logger.warning(
        "generation_empty_output  route_id=%s  model_alias=%s  provider=%s  "
        "deployment=%s  finish_reason=%s  visible_chunk_count=%d  "
        "first_visible_chunk_emitted=false  failure_kind=%s  "
        "fallback_eligible=%s  fallback_attempted=%s",
        route_id,
        model_alias,
        provider or "",
        deployment or "",
        finish_reason or "none",
        visible_chunk_count,
        failure_kind,
        fallback_eligible,
        fallback_attempted,
    )


@runtime_checkable
class ProviderExecutor(Protocol):
    """Boundary for provider adapters used by RegistryBackedModelExecutor."""

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        """Execute a provider request and return a normalized result."""
        ...

    def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
        """Execute a provider request and yield answer text chunks."""
        ...


class FakeProviderExecutor:
    """Test-only ProviderExecutor that records the last request."""

    def __init__(
        self,
        *,
        content: str = "Fake provider response. <ANSWER_DONE>",
        raise_on_execute: Exception | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self._content = content
        self._raise_on_execute = raise_on_execute
        self._finish_reason = finish_reason
        self.last_request: ProviderExecutionRequest | None = None
        self.call_count: int = 0

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        self.last_request = request
        self.call_count += 1

        if self._raise_on_execute is not None:
            raise self._raise_on_execute

        return ModelExecutionResult(
            content=self._content,
            model=request.model_resolution.model_alias,
            provider=request.model_resolution.provider,
            finish_reason=self._finish_reason,
            metadata=request.model_resolution.safe_metadata,
        )

    def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
        self.last_request = request
        self.call_count += 1

        if self._raise_on_execute is not None:
            raise self._raise_on_execute

        chunk_size = 8
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]
        self.last_stream_finish_reason = self._finish_reason


class RegistryBackedModelExecutor:
    """Resolve model metadata and delegate execution to an injected provider."""

    def __init__(
        self,
        *,
        provider_executor: ProviderExecutor,
        model_config_resolver: ModelConfigResolver | None = None,
    ) -> None:
        if provider_executor is None:
            raise TypeError("provider_executor is required.")
        self._provider_executor = provider_executor
        self._model_config_resolver = (
            model_config_resolver
            if model_config_resolver is not None
            else ModelConfigResolver()
        )
        self.last_stream_finish_reason: str | None = None

    def _try_practice_capacity_escalation(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
        resolution: ResolvedModelConfig,
        started_at: float,
    ) -> tuple[ModelExecutionResult | None, str]:
        """Retry one exact Practice work unit once at a strictly larger safe cap."""
        capacity = route_decision.practice_generation_capacity
        if (
            capacity is None
            or capacity.escalation_max_output_tokens <= route_decision.max_tokens
        ):
            return None, "output_token_exhausted"

        escalation_budget = capacity.escalation_max_output_tokens
        escalation_request = ProviderExecutionRequest(
            route_decision=route_decision,
            model_resolution=resolution,
            messages=messages,
            temperature=route_decision.temperature,
            max_tokens=escalation_budget,
            provider_options=dict(route_decision.provider_options),
        )
        _log_practice_generator_attempt(
            event_name="generator_capacity_escalation_started",
            route_decision=route_decision,
            model_alias=resolution.model_alias,
            provider=resolution.provider,
            attempt_type="capacity_escalation",
            fallback_index=0,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            actual_output_budget=escalation_budget,
            capacity_escalated=True,
            status="started",
        )
        error_class: str | None = None
        try:
            result = self._provider_executor.execute(escalation_request)
        except LlmProviderResponseError as exc:
            failure_kind = _provider_response_failure_kind(
                route_decision=route_decision,
                response_error=exc,
            )
            error_class = type(exc).__name__
            if failure_kind == "output_token_exhausted":
                _log_practice_generator_token_exhausted(
                    route_decision=route_decision,
                    model_alias=resolution.model_alias,
                    finish_reason=getattr(exc, "finish_reason", None),
                    output_tokens=getattr(exc, "output_tokens", None),
                    reasoning_tokens=getattr(exc, "reasoning_tokens", None),
                    actual_output_budget=escalation_budget,
                )
        except LlmProviderExecutionError as exc:
            failure_kind = exc.failure_kind
            error_class = type(exc).__name__
        except Exception as exc:  # noqa: BLE001
            failure_kind = "unknown_provider_error"
            error_class = type(exc).__name__
        else:
            failure_kind = _model_result_failure_kind(
                route_decision=route_decision,
                result=result,
            )
            if failure_kind == "output_token_exhausted":
                _log_practice_generator_token_exhausted(
                    route_decision=route_decision,
                    model_alias=resolution.model_alias,
                    content=result.content,
                    finish_reason=result.finish_reason,
                    output_tokens=result.output_tokens,
                    reasoning_tokens=result.reasoning_tokens,
                    actual_output_budget=escalation_budget,
                )
            if failure_kind is None and _is_visible_text(result.content):
                _log_practice_generator_completion(
                    route_decision=route_decision,
                    resolution=resolution,
                    result=result,
                    actual_output_budget=escalation_budget,
                    fallback_used=False,
                    capacity_escalated=True,
                )
                _log_practice_generator_attempt(
                    event_name="generator_capacity_escalation_succeeded",
                    route_decision=route_decision,
                    model_alias=resolution.model_alias,
                    provider=resolution.provider,
                    attempt_type="capacity_escalation",
                    fallback_index=0,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    actual_output_budget=escalation_budget,
                    capacity_escalated=True,
                    status="completed",
                )
                return result, "completed"
            if failure_kind is None:
                failure_kind = "empty_answer"
            error_class = "EmptyProviderResponse" if not _is_visible_text(result.content) else None

        _log_practice_generator_attempt(
            event_name="generator_capacity_escalation_failed",
            route_decision=route_decision,
            model_alias=resolution.model_alias,
            provider=resolution.provider,
            attempt_type="capacity_escalation",
            fallback_index=0,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            failure_kind=failure_kind,
            error_class=error_class,
            actual_output_budget=escalation_budget,
            capacity_escalated=True,
            status="failed",
        )
        return None, failure_kind

    def execute(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
    ) -> ModelExecutionResult:
        started_at = time.monotonic()
        primary_alias = route_decision.model
        model_resolution = self._model_config_resolver.resolve(route_decision)
        if (
            _is_practice_generator(route_decision)
            and route_decision.practice_generation_capacity is None
        ):
            execution_context = current_execution_context()
            capacity = PracticeGenerationCapacityPolicy.resolve(
                route_decision=route_decision,
                model_config=model_resolution.model_config,
                workload=PracticeGenerationWorkload(
                    complexity="medium",
                    slot_count=(
                        len(execution_context.slot_ids)
                        if execution_context is not None and execution_context.slot_ids
                        else 1
                    ),
                ),
            )
            route_decision = route_decision.model_copy(
                update={
                    "max_tokens": capacity.initial_max_output_tokens,
                    "practice_generation_capacity": capacity,
                }
            )
        _validate_model_role(
            route_decision,
            allowed_task_roles=list(model_resolution.model_config.allowed_task_roles),
        )
        self._model_config_resolver.validate_provider_options(
            provider_options=route_decision.provider_options,
            model_config=model_resolution.model_config,
            model_alias=model_resolution.model_alias,
        )
        primary_request = ProviderExecutionRequest(
            route_decision=route_decision,
            model_resolution=model_resolution,
            messages=messages,
            temperature=route_decision.temperature,
            max_tokens=route_decision.max_tokens,
            provider_options=dict(route_decision.provider_options),
        )

        logger.debug(
            "registry_backed_model_executor.execute  model_alias=%s  provider=%s  "
            "supports_streaming=%s  supports_thinking=%s  timeout_seconds=%d",
            model_resolution.safe_metadata["model_alias"],
            model_resolution.safe_metadata["provider"],
            model_resolution.safe_metadata["supports_streaming"],
            model_resolution.safe_metadata["supports_thinking"],
            model_resolution.safe_metadata["timeout_seconds"],
        )

        # --- Try primary model ---
        primary_failure_kind: str | None = None
        _log_practice_generator_attempt(
            event_name="generator_model_attempt_started",
            route_decision=route_decision,
            model_alias=model_resolution.model_alias,
            provider=model_resolution.provider,
            attempt_type="primary",
            fallback_index=0,
            duration_ms=0,
            status="started",
        )
        try:
            raw_result = self._provider_executor.execute(primary_request)
            result_failure_kind = _model_result_failure_kind(
                route_decision=route_decision,
                result=raw_result,
            )
            if result_failure_kind is not None:
                if result_failure_kind == "output_token_exhausted":
                    _log_practice_generator_token_exhausted(
                        route_decision=route_decision,
                        model_alias=model_resolution.model_alias,
                        content=raw_result.content,
                        finish_reason=raw_result.finish_reason,
                        output_tokens=raw_result.output_tokens,
                        reasoning_tokens=raw_result.reasoning_tokens,
                    )
                raise LlmProviderExecutionError(
                    "Provider completion was not acceptable for structured generation.",
                    failure_kind=result_failure_kind,
                    provider=model_resolution.provider,
                    model_alias=primary_alias,
                )
            if not _is_visible_text(raw_result.content):
                _log_generation_empty_output(
                    route_id=route_decision.route_id,
                    model_alias=model_resolution.model_alias,
                    provider=model_resolution.provider,
                    deployment=model_resolution.model_config.deployment,
                    finish_reason=raw_result.finish_reason,
                    visible_chunk_count=0,
                    failure_kind="empty_answer",
                    fallback_eligible=True,
                )
                raise LlmProviderExecutionError(
                    "Provider returned empty answer content.",
                    failure_kind="empty_answer",
                    provider=model_resolution.provider,
                    model_alias=primary_alias,
                )
            if route_decision.task_role == "generator":
                update_request_summary(
                    generation_route=route_decision.route_id,
                    generation_model=model_resolution.model_alias,
                )
            _log_model_execution(
                route_decision=route_decision,
                resolution=model_resolution,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                fallback_used=False,
            )
            _log_practice_generator_completion(
                route_decision=route_decision,
                resolution=model_resolution,
                result=raw_result,
                actual_output_budget=route_decision.max_tokens,
                fallback_used=False,
                capacity_escalated=False,
            )
            return raw_result
        except LlmProviderResponseError as exc:
            primary_failure_kind = _provider_response_failure_kind(
                route_decision=route_decision,
                response_error=exc,
            )
            _log_practice_generator_attempt(
                event_name="generator_model_attempt_failed",
                route_decision=route_decision,
                model_alias=model_resolution.model_alias,
                provider=model_resolution.provider,
                attempt_type="primary",
                fallback_index=0,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                failure_kind=primary_failure_kind,
                error_class=type(exc).__name__,
                status="failed",
            )
            if primary_failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                raise ProviderExecutionError(
                    f"Provider response failed for model '{primary_alias}' "
                    f"(failure_kind={primary_failure_kind!r}): {type(exc).__name__}",
                    failure_kind=primary_failure_kind,
                    attempted_aliases=(primary_alias,),
                ) from exc
            if primary_failure_kind == "output_token_exhausted":
                _log_practice_generator_token_exhausted(
                    route_decision=route_decision,
                    model_alias=model_resolution.model_alias,
                    finish_reason=getattr(exc, "finish_reason", None),
                    output_tokens=getattr(exc, "output_tokens", None),
                    reasoning_tokens=getattr(exc, "reasoning_tokens", None),
                )
            logger.warning(
                "registry_backed_model_executor.execute  primary_response_failed  "
                "model_alias=%s  failure_kind=%s — attempting fallback",
                primary_alias,
                primary_failure_kind,
            )
        except LlmProviderExecutionError as exc:
            _log_practice_generator_attempt(
                event_name="generator_model_attempt_failed",
                route_decision=route_decision,
                model_alias=model_resolution.model_alias,
                provider=model_resolution.provider,
                attempt_type="primary",
                fallback_index=0,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                failure_kind=exc.failure_kind,
                error_class=type(exc).__name__,
                status="failed",
            )
            if exc.failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                # Not a retryable provider failure — wrap and raise immediately.
                raise ProviderExecutionError(
                    f"Provider execution failed for model '{primary_alias}' "
                    f"(failure_kind={exc.failure_kind!r}): {type(exc).__name__}",
                    failure_kind=exc.failure_kind,
                    attempted_aliases=(primary_alias,),
                ) from exc
            primary_failure_kind = exc.failure_kind
            logger.warning(
                "registry_backed_model_executor.execute  primary_failed  "
                "model_alias=%s  failure_kind=%s — attempting fallback",
                primary_alias,
                primary_failure_kind,
            )
        except ProviderExecutionError:
            raise
        except Exception as exc:
            _log_practice_generator_attempt(
                event_name="generator_model_attempt_failed",
                route_decision=route_decision,
                model_alias=model_resolution.model_alias,
                provider=model_resolution.provider,
                attempt_type="primary",
                fallback_index=0,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                failure_kind="unknown_provider_error",
                error_class=type(exc).__name__,
                status="failed",
            )
            raise ProviderExecutionError(
                f"Provider executor failed for model '{primary_alias}': "
                f"{type(exc).__name__}",
                attempted_aliases=(primary_alias,),
            ) from exc

        if primary_failure_kind == "output_token_exhausted":
            escalated_result, escalation_failure_kind = (
                self._try_practice_capacity_escalation(
                    route_decision=route_decision,
                    messages=messages,
                    resolution=model_resolution,
                    started_at=started_at,
                )
            )
            if escalated_result is not None:
                if route_decision.task_role == "generator":
                    update_request_summary(
                        generation_route=route_decision.route_id,
                        generation_model=model_resolution.model_alias,
                    )
                _log_model_execution(
                    route_decision=route_decision,
                    resolution=model_resolution,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    fallback_used=False,
                )
                return escalated_result
            primary_failure_kind = escalation_failure_kind

        capacity = route_decision.practice_generation_capacity
        if (
            primary_failure_kind == "output_token_exhausted"
            and (
                capacity is None
                or capacity.escalation_max_output_tokens <= route_decision.max_tokens
            )
        ):
            raise ProviderExecutionError(
                "Practice generator exhausted its maximum safe output capacity.",
                failure_kind="output_token_exhausted",
                attempted_aliases=(primary_alias,),
            )

        # --- Fallback loop ---
        fallback_aliases: list[str] = list(
            getattr(model_resolution.model_config, "fallback_models", None) or []
        )
        attempted: list[str] = [primary_alias]

        for fallback_index, fallback_alias in enumerate(fallback_aliases, start=1):
            try:
                fallback_resolution = self._model_config_resolver.resolve_for_alias(
                    fallback_alias
                )
            except Exception as cfg_exc:
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_config_error  "
                    "fallback_alias=%s  error=%s — skipping",
                    fallback_alias,
                    type(cfg_exc).__name__,
                )
                attempted.append(fallback_alias)
                _log_practice_generator_attempt(
                    event_name="generator_fallback_failed",
                    route_decision=route_decision,
                    model_alias=fallback_alias,
                    provider=None,
                    attempt_type="fallback",
                    fallback_index=fallback_index,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    failure_kind="unknown_provider_error",
                    error_class=type(cfg_exc).__name__,
                    status="failed",
                )
                continue
            _validate_model_role(
                route_decision,
                allowed_task_roles=list(
                    fallback_resolution.model_config.allowed_task_roles
                ),
            )

            fallback_budget = route_decision.max_tokens
            if primary_failure_kind == "output_token_exhausted":
                assert capacity is not None
                fallback_budget = capacity.escalation_max_output_tokens
                if (
                    fallback_resolution.model_config.model_hard_max_output_tokens
                    < fallback_budget
                ):
                    attempted.append(fallback_alias)
                    _log_practice_generator_attempt(
                        event_name="generator_fallback_skipped",
                        route_decision=route_decision,
                        model_alias=fallback_resolution.model_alias,
                        provider=fallback_resolution.provider,
                        attempt_type="fallback",
                        fallback_index=fallback_index,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                        failure_kind="output_token_exhausted",
                        error_class="FallbackCapacityInsufficient",
                        actual_output_budget=fallback_budget,
                        capacity_escalated=True,
                        status="failed",
                    )
                    continue

            fallback_request = ProviderExecutionRequest(
                route_decision=route_decision,
                model_resolution=fallback_resolution,
                messages=messages,  # same messages — reused safely
                temperature=route_decision.temperature,
                max_tokens=fallback_budget,
                provider_options={},  # strip thinking/stream options for fallback
            )

            logger.debug(
                "registry_backed_model_executor.execute  trying_fallback  "
                "fallback_alias=%s  provider=%s",
                fallback_alias,
                fallback_resolution.provider,
            )
            _log_practice_generator_attempt(
                event_name="generator_fallback_selected",
                route_decision=route_decision,
                model_alias=fallback_resolution.model_alias,
                provider=fallback_resolution.provider,
                attempt_type="fallback",
                fallback_index=fallback_index,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                failure_kind=primary_failure_kind,
                actual_output_budget=fallback_budget,
                capacity_escalated=(primary_failure_kind == "output_token_exhausted"),
                status="selected",
            )
            _log_practice_generator_attempt(
                event_name="generator_fallback_started",
                route_decision=route_decision,
                model_alias=fallback_resolution.model_alias,
                provider=fallback_resolution.provider,
                attempt_type="fallback",
                fallback_index=fallback_index,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                actual_output_budget=fallback_budget,
                capacity_escalated=(primary_failure_kind == "output_token_exhausted"),
                status="started",
            )
            if primary_failure_kind == "output_token_exhausted":
                _log_practice_model_fallback(
                    event_name="practice_model_fallback_started",
                    route_decision=route_decision,
                    model_alias=fallback_alias,
                )

            try:
                raw_result = self._provider_executor.execute(fallback_request)
                result_failure_kind = _model_result_failure_kind(
                    route_decision=route_decision,
                    result=raw_result,
                )
                if result_failure_kind is not None:
                    if result_failure_kind == "output_token_exhausted":
                        _log_practice_generator_token_exhausted(
                            route_decision=route_decision,
                            model_alias=fallback_alias,
                            content=raw_result.content,
                            finish_reason=raw_result.finish_reason,
                            output_tokens=raw_result.output_tokens,
                            reasoning_tokens=raw_result.reasoning_tokens,
                            actual_output_budget=fallback_budget,
                        )
                    raise LlmProviderExecutionError(
                        "Provider completion was not acceptable for structured generation.",
                        failure_kind=result_failure_kind,
                        provider=fallback_resolution.provider,
                        model_alias=fallback_alias,
                    )
                if not _is_visible_text(raw_result.content):
                    attempted.append(fallback_alias)
                    logger.warning(
                        "registry_backed_model_executor.execute  fallback_empty_answer  "
                        "fallback_alias=%s — skipping",
                        fallback_alias,
                    )
                    _log_practice_generator_attempt(
                        event_name="generator_fallback_failed",
                        route_decision=route_decision,
                        model_alias=fallback_resolution.model_alias,
                        provider=fallback_resolution.provider,
                        attempt_type="fallback",
                        fallback_index=fallback_index,
                        duration_ms=int((time.monotonic() - started_at) * 1000),
                        failure_kind="empty_answer",
                        error_class="EmptyProviderResponse",
                        status="failed",
                    )
                    continue
                # Build a new result that records the fallback provenance safely.
                result = ModelExecutionResult(
                    content=raw_result.content,
                    model=fallback_alias,
                    provider=raw_result.provider,
                    finish_reason=raw_result.finish_reason,
                    normalized_finish_reason=raw_result.normalized_finish_reason,
                    input_tokens=raw_result.input_tokens,
                    output_tokens=raw_result.output_tokens,
                    total_tokens=raw_result.total_tokens,
                    cached_input_tokens=raw_result.cached_input_tokens,
                    reasoning_tokens=raw_result.reasoning_tokens,
                    usage_source=raw_result.usage_source,
                    fallback_used=True,
                    metadata={
                        **{k: v for k, v in raw_result.metadata.items()},
                        "fallback_from": primary_alias,
                        "fallback_to": fallback_alias,
                        "failure_kind": primary_failure_kind,
                    },
                )
                logger.debug(
                    "registry_backed_model_executor.execute  fallback_succeeded  "
                    "fallback_alias=%s  provider=%s  failure_kind=%s",
                    fallback_alias,
                    raw_result.provider,
                    primary_failure_kind,
                )
                if route_decision.task_role == "generator":
                    update_request_summary(
                        generation_route=route_decision.route_id,
                        generation_model=fallback_alias,
                    )
                _log_model_execution(
                    route_decision=route_decision,
                    resolution=fallback_resolution,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    fallback_used=True,
                )
                _log_practice_generator_completion(
                    route_decision=route_decision,
                    resolution=fallback_resolution,
                    result=result,
                    actual_output_budget=fallback_budget,
                    fallback_used=True,
                    capacity_escalated=(primary_failure_kind == "output_token_exhausted"),
                )
                if primary_failure_kind == "output_token_exhausted":
                    _log_practice_model_fallback(
                        event_name="practice_model_fallback_succeeded",
                        route_decision=route_decision,
                        model_alias=fallback_alias,
                    )
                _log_practice_generator_attempt(
                    event_name="generator_fallback_succeeded",
                    route_decision=route_decision,
                    model_alias=fallback_resolution.model_alias,
                    provider=fallback_resolution.provider,
                    attempt_type="fallback",
                    fallback_index=fallback_index,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    actual_output_budget=fallback_budget,
                    capacity_escalated=(primary_failure_kind == "output_token_exhausted"),
                    status="completed",
                )
                return result
            except LlmProviderResponseError as exc:
                attempted.append(fallback_alias)
                failure_kind = _provider_response_failure_kind(
                    route_decision=route_decision,
                    response_error=exc,
                )
                _log_practice_generator_attempt(
                    event_name="generator_fallback_failed",
                    route_decision=route_decision,
                    model_alias=fallback_resolution.model_alias,
                    provider=fallback_resolution.provider,
                    attempt_type="fallback",
                    fallback_index=fallback_index,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    failure_kind=failure_kind,
                    error_class=type(exc).__name__,
                    status="failed",
                )
                if failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                    raise ProviderExecutionError(
                        f"Provider response failed for fallback model '{fallback_alias}' "
                        f"(failure_kind={failure_kind!r}): {type(exc).__name__}",
                        failure_kind=failure_kind,
                        attempted_aliases=tuple(attempted),
                    ) from exc
                if failure_kind == "output_token_exhausted":
                    _log_practice_generator_token_exhausted(
                        route_decision=route_decision,
                        model_alias=fallback_alias,
                        finish_reason=getattr(exc, "finish_reason", None),
                        output_tokens=getattr(exc, "output_tokens", None),
                        reasoning_tokens=getattr(exc, "reasoning_tokens", None),
                        actual_output_budget=fallback_budget,
                    )
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_response_failed  "
                    "fallback_alias=%s  failure_kind=%s",
                    fallback_alias,
                    failure_kind,
                )
            except LlmProviderExecutionError as exc:
                attempted.append(fallback_alias)
                _log_practice_generator_attempt(
                    event_name="generator_fallback_failed",
                    route_decision=route_decision,
                    model_alias=fallback_resolution.model_alias,
                    provider=fallback_resolution.provider,
                    attempt_type="fallback",
                    fallback_index=fallback_index,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    failure_kind=exc.failure_kind,
                    error_class=type(exc).__name__,
                    status="failed",
                )
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_failed  "
                    "fallback_alias=%s  failure_kind=%s",
                    fallback_alias,
                    exc.failure_kind,
                )
                if exc.failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                    raise ProviderExecutionError(
                        f"Provider execution failed for fallback model '{fallback_alias}' "
                        f"(failure_kind={exc.failure_kind!r}): {type(exc).__name__}",
                        failure_kind=exc.failure_kind,
                        attempted_aliases=tuple(attempted),
                    ) from exc
            except Exception as exc:
                attempted.append(fallback_alias)
                _log_practice_generator_attempt(
                    event_name="generator_fallback_failed",
                    route_decision=route_decision,
                    model_alias=fallback_resolution.model_alias,
                    provider=fallback_resolution.provider,
                    attempt_type="fallback",
                    fallback_index=fallback_index,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    failure_kind="unknown_provider_error",
                    error_class=type(exc).__name__,
                    status="failed",
                )
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_error  "
                    "fallback_alias=%s  error=%s",
                    fallback_alias,
                    type(exc).__name__,
                )

        # All attempts exhausted.
        _log_practice_generator_attempt(
            event_name="generator_fallback_exhausted",
            route_decision=route_decision,
            model_alias=attempted[-1],
            provider=None,
            attempt_type="fallback",
            fallback_index=len(fallback_aliases),
            duration_ms=int((time.monotonic() - started_at) * 1000),
            failure_kind=primary_failure_kind or "unknown_provider_error",
            status="failed",
        )
        raise ProviderExecutionError(
            f"All model execution attempts failed. "
            f"Attempted aliases: {attempted}. "
            f"Primary failure_kind: {primary_failure_kind!r}.",
            failure_kind=primary_failure_kind or "unknown_provider_error",
            attempted_aliases=tuple(attempted),
        )

    def execute_stream(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
        on_before_fallback: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        """Resolve model metadata and stream answer text chunks from the provider."""
        started_at = time.monotonic()
        primary_alias = route_decision.model
        model_resolution = self._model_config_resolver.resolve(route_decision)
        _validate_model_role(
            route_decision,
            allowed_task_roles=list(model_resolution.model_config.allowed_task_roles),
        )
        self._model_config_resolver.validate_provider_options(
            provider_options=route_decision.provider_options,
            model_config=model_resolution.model_config,
            model_alias=model_resolution.model_alias,
        )
        primary_request = ProviderExecutionRequest(
            route_decision=route_decision,
            model_resolution=model_resolution,
            messages=messages,
            temperature=route_decision.temperature,
            max_tokens=route_decision.max_tokens,
            provider_options=dict(route_decision.provider_options),
        )

        logger.debug(
            "registry_backed_model_executor.execute_stream  model_alias=%s  provider=%s",
            model_resolution.safe_metadata["model_alias"],
            model_resolution.safe_metadata["provider"],
        )

        primary_failure_kind: str | None = None
        self.last_stream_finish_reason = None
        visible_chunk_count = 0
        leading_chunks: list[str] = []
        try:
            for chunk in self._provider_executor.execute_stream(primary_request):
                if not chunk:
                    continue
                if visible_chunk_count == 0 and not _is_visible_text(chunk):
                    leading_chunks.append(chunk)
                    continue
                if visible_chunk_count == 0:
                    visible_chunk_count = 1
                    yield from leading_chunks
                    leading_chunks.clear()
                elif _is_visible_text(chunk):
                    visible_chunk_count += 1
                yield chunk
            self.last_stream_finish_reason = getattr(
                self._provider_executor, "last_stream_finish_reason", "stop"
            )
            if visible_chunk_count == 0:
                finish_reason = self.last_stream_finish_reason
                _log_generation_empty_output(
                    route_id=route_decision.route_id,
                    model_alias=model_resolution.model_alias,
                    provider=model_resolution.provider,
                    deployment=model_resolution.model_config.deployment,
                    finish_reason=finish_reason,
                    visible_chunk_count=0,
                    failure_kind="empty_stream",
                    fallback_eligible=True,
                )
                raise LlmProviderExecutionError(
                    "Provider stream returned no visible text chunks.",
                    failure_kind="empty_stream",
                    provider=model_resolution.provider,
                    model_alias=primary_alias,
                )
            if route_decision.task_role == "generator":
                update_request_summary(
                    generation_route=route_decision.route_id,
                    generation_model=model_resolution.model_alias,
                )
            _log_model_execution(
                route_decision=route_decision,
                resolution=model_resolution,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                fallback_used=False,
            )
            return
        except LlmProviderExecutionError as exc:
            if visible_chunk_count > 0:
                raise ProviderExecutionError(
                    f"Provider stream failed after visible output for model "
                    f"'{primary_alias}': {type(exc).__name__}"
                ) from exc
            if exc.failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                raise ProviderExecutionError(
                    f"Provider stream failed for model '{primary_alias}' "
                    f"(failure_kind={exc.failure_kind!r}): {type(exc).__name__}"
                ) from exc
            primary_failure_kind = exc.failure_kind
            logger.warning(
                "registry_backed_model_executor.execute_stream  primary_failed  "
                "model_alias=%s  failure_kind=%s — attempting fallback",
                primary_alias,
                primary_failure_kind,
            )
        except ProviderExecutionError:
            raise
        except Exception as exc:
            if visible_chunk_count > 0:
                raise ProviderExecutionError(
                    f"Provider executor stream failed after visible output for model "
                    f"'{primary_alias}': {type(exc).__name__}"
                ) from exc
            raise ProviderExecutionError(
                f"Provider executor stream failed for model '{primary_alias}': "
                f"{type(exc).__name__}"
            ) from exc

        fallback_aliases: list[str] = list(
            getattr(model_resolution.model_config, "fallback_models", None) or []
        )
        attempted: list[str] = [primary_alias]

        if fallback_aliases and on_before_fallback is not None:
            on_before_fallback()

        for fallback_alias in fallback_aliases:
            try:
                fallback_resolution = self._model_config_resolver.resolve_for_alias(
                    fallback_alias
                )
            except Exception as cfg_exc:
                logger.warning(
                    "registry_backed_model_executor.execute_stream  fallback_config_error  "
                    "fallback_alias=%s  error=%s — skipping",
                    fallback_alias,
                    type(cfg_exc).__name__,
                )
                attempted.append(fallback_alias)
                continue
            _validate_model_role(
                route_decision,
                allowed_task_roles=list(
                    fallback_resolution.model_config.allowed_task_roles
                ),
            )

            fallback_request = ProviderExecutionRequest(
                route_decision=route_decision,
                model_resolution=fallback_resolution,
                messages=messages,
                temperature=route_decision.temperature,
                max_tokens=route_decision.max_tokens,
                provider_options={},
            )
            attempted.append(fallback_alias)

            logger.debug(
                "registry_backed_model_executor.execute_stream  trying_fallback  "
                "fallback_alias=%s  provider=%s",
                fallback_alias,
                fallback_resolution.provider,
            )

            try:
                fallback_visible = 0
                if hasattr(self._provider_executor, "execute_stream"):
                    fallback_leading_chunks: list[str] = []
                    for chunk in self._provider_executor.execute_stream(fallback_request):
                        if not chunk:
                            continue
                        if fallback_visible == 0 and not _is_visible_text(chunk):
                            fallback_leading_chunks.append(chunk)
                            continue
                        if fallback_visible == 0:
                            fallback_visible = 1
                            yield from fallback_leading_chunks
                            fallback_leading_chunks.clear()
                        elif _is_visible_text(chunk):
                            fallback_visible += 1
                        yield chunk
                    self.last_stream_finish_reason = getattr(
                        self._provider_executor, "last_stream_finish_reason", "stop"
                    )
                else:
                    result = self._provider_executor.execute(fallback_request)
                    if _is_visible_text(result.content):
                        fallback_visible = 1
                        yield result.content
                    self.last_stream_finish_reason = result.finish_reason
                if fallback_visible == 0:
                    logger.warning(
                        "registry_backed_model_executor.execute_stream  fallback_empty_stream  "
                        "fallback_alias=%s — skipping",
                        fallback_alias,
                    )
                    continue
                logger.debug(
                    "registry_backed_model_executor.execute_stream  fallback_succeeded  "
                    "fallback_alias=%s  provider=%s  failure_kind=%s",
                    fallback_alias,
                    fallback_resolution.provider,
                    primary_failure_kind,
                )
                if route_decision.task_role == "generator":
                    update_request_summary(
                        generation_route=route_decision.route_id,
                        generation_model=fallback_alias,
                    )
                _log_model_execution(
                    route_decision=route_decision,
                    resolution=fallback_resolution,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    fallback_used=True,
                )
                return
            except LlmProviderExecutionError as exc:
                if fallback_visible > 0:
                    raise ProviderExecutionError(
                        f"Fallback provider stream failed after visible output for model "
                        f"'{fallback_alias}': {type(exc).__name__}"
                    ) from exc
                logger.warning(
                    "registry_backed_model_executor.execute_stream  fallback_failed  "
                    "fallback_alias=%s  failure_kind=%s",
                    fallback_alias,
                    exc.failure_kind,
                )
            except Exception as exc:
                if fallback_visible > 0:
                    raise ProviderExecutionError(
                        f"Fallback provider executor stream failed after visible output for "
                        f"model '{fallback_alias}': {type(exc).__name__}"
                    ) from exc
                logger.warning(
                    "registry_backed_model_executor.execute_stream  fallback_error  "
                    "fallback_alias=%s  error=%s",
                    fallback_alias,
                    type(exc).__name__,
                )

        raise ProviderExecutionError(
            f"All model stream attempts failed. "
            f"Attempted aliases: {attempted}. "
            f"Primary failure_kind: {primary_failure_kind!r}."
        )


# ---------------------------------------------------------------------------
# Part 6: ProviderAdapterExecutor
# ---------------------------------------------------------------------------


class ProviderAdapterExecutor:
    """Part 6 bridge that implements the ProviderExecutor protocol using the
    ProviderAdapterFactory and ProviderCredentialResolver.

    Flow:
        1. Resolve ProviderCredentials via credential_resolver.resolve(profile).
        2. Obtain the matching ProviderAdapter via provider_factory.get_provider(provider).
        3. Call adapter.generate(request=request, credentials=credentials).

    Error propagation:
        - SecretResolverError subclasses propagate unchanged (credential resolution failed).
        - LlmProviderAdapterError subclasses propagate unchanged (adapter failed).
        - No exceptions are swallowed or wrapped by this executor.

    No fallback logic. No graph dependency. No AWS calls.
    """

    def __init__(
        self,
        *,
        credential_resolver: ProviderCredentialResolver,
        provider_factory: ProviderAdapterFactory,
    ) -> None:
        """
        Args:
            credential_resolver: Resolves ProviderProfile env var references to
                                  ProviderCredentials.  Required.
            provider_factory:    Maps provider names to ProviderAdapter instances.
                                  Required.
        """
        if credential_resolver is None:
            raise TypeError("credential_resolver is required.")
        if provider_factory is None:
            raise TypeError("provider_factory is required.")
        self._credential_resolver = credential_resolver
        self._provider_factory = provider_factory
        self.last_stream_finish_reason: str | None = None

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        """Execute the provider request and return a normalized result.

        Args:
            request: The resolved provider execution request.

        Returns:
            ModelExecutionResult from the selected provider adapter.

        Raises:
            SecretResolverError:        Credential resolution failed.
            LlmProviderAdapterError:    Adapter execution failed.
        """
        started_at = time.monotonic()
        try:
            profile = request.model_resolution.provider_profile
            credentials = self._credential_resolver.resolve(profile)
            adapter = self._provider_factory.get_provider(
                request.model_resolution.provider
            )
            logger.debug(
                "provider_adapter_executor.execute  model_alias=%s  provider=%s",
                request.model_resolution.model_alias,
                request.model_resolution.provider,
            )
            result = adapter.generate(request=request, credentials=credentials)
        except Exception as exc:
            provider_usage = getattr(exc, "provider_usage", None)
            self._record_usage(
                request=request,
                usage=(
                    provider_usage
                    if isinstance(provider_usage, ProviderTokenUsage)
                    else ProviderTokenUsage()
                ),
                started_at=started_at,
                streaming=False,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise
        self._record_usage(
            request=request,
            usage=self._usage_from_result(result),
            started_at=started_at,
            streaming=False,
            status="succeeded",
        )
        return result

    def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
        """Execute the provider request and yield answer text chunks."""
        started_at = time.monotonic()
        usage = ProviderTokenUsage()
        status: UsageStatus = "succeeded"
        error_type: str | None = None
        clear_stream_usage()
        try:
            profile = request.model_resolution.provider_profile
            credentials = self._credential_resolver.resolve(profile)
            adapter = self._provider_factory.get_provider(
                request.model_resolution.provider
            )
            logger.debug(
                "provider_adapter_executor.execute_stream  model_alias=%s  provider=%s",
                request.model_resolution.model_alias,
                request.model_resolution.provider,
            )

            if hasattr(adapter, "generate_stream"):
                finish_reason: str | None = None
                for chunk in adapter.generate_stream(
                    request=request,
                    credentials=credentials,
                ):
                    if hasattr(adapter, "last_stream_finish_reason"):
                        finish_reason = adapter.last_stream_finish_reason
                    yield chunk
                self.last_stream_finish_reason = finish_reason or "stop"
                return

            result = adapter.generate(request=request, credentials=credentials)
            usage = self._usage_from_result(result)
            if result.content:
                yield result.content
            self.last_stream_finish_reason = result.finish_reason or "stop"
        except GeneratorExit:
            status = "cancelled"
            error_type = "GeneratorExit"
            raise
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            provider_usage = consume_stream_usage()
            if provider_usage.available:
                usage = provider_usage
            self._record_usage(
                request=request,
                usage=usage,
                started_at=started_at,
                streaming=True,
                status=status,
                error_type=error_type,
            )

    @staticmethod
    def _usage_from_result(result: ModelExecutionResult) -> ProviderTokenUsage:
        return ProviderTokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            cached_input_tokens=result.cached_input_tokens,
            reasoning_tokens=result.reasoning_tokens,
        )

    @staticmethod
    def _record_usage(
        *,
        request: ProviderExecutionRequest,
        usage: ProviderTokenUsage,
        started_at: float,
        streaming: bool,
        status: UsageStatus,
        error_type: str | None = None,
    ) -> None:
        resolution = request.model_resolution
        config = resolution.model_config
        context = current_request_context()
        base_attempt = current_llm_attempt_type()
        is_fallback = resolution.model_alias != request.route_decision.model
        attempt_type = (
            f"{base_attempt}_fallback"
            if is_fallback and "fallback" not in base_attempt
            else base_attempt
        )
        record_llm_call(
            request_id=context.request_id if context else "unknown",
            role=request.route_decision.route_id,
            provider=resolution.provider,
            model=config.model_id or config.deployment or resolution.model_alias,
            deployment=config.deployment or None,
            model_alias=resolution.model_alias,
            attempt_type=attempt_type,
            streaming=streaming,
            usage=usage,
            duration_ms=max(int((time.monotonic() - started_at) * 1000), 0),
            status=status,
            error_type=error_type,
        )
