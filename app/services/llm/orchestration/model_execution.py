"""
app/services/llm_orchestration/model_execution.py
-------------------------------------------------
Model execution boundary backed by the LLM config registry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from observability import update_request_summary
from schemas.llm import LlmMessage
from schemas.llm_orchestration import (
    ModelExecutionResult,
    ProviderExecutionRequest,
)
from schemas.llm_routing import RouteDecision
from services.llm.orchestration.errors import (
    ModelExecutionConfigError,
    ProviderExecutionError,
)
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderExecutionError,
)

if TYPE_CHECKING:
    from services.llm.providers.provider_factory import ProviderAdapterFactory
    from services.secrets.provider_credentials import ProviderCredentialResolver

logger = logging.getLogger(__name__)


def _is_visible_text(chunk: str) -> bool:
    """True when a stream chunk contains user-visible answer text."""
    return bool(chunk and chunk.strip())


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

    def execute(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
    ) -> ModelExecutionResult:
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
        try:
            raw_result = self._provider_executor.execute(primary_request)
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
            return raw_result
        except LlmProviderExecutionError as exc:
            if exc.failure_kind not in FALLBACK_ELIGIBLE_FAILURE_KINDS:
                # Not a retryable provider failure — wrap and raise immediately.
                raise ProviderExecutionError(
                    f"Provider execution failed for model '{primary_alias}' "
                    f"(failure_kind={exc.failure_kind!r}): {type(exc).__name__}"
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
            raise ProviderExecutionError(
                f"Provider executor failed for model '{primary_alias}': "
                f"{type(exc).__name__}"
            ) from exc

        # --- Fallback loop ---
        fallback_aliases: list[str] = list(
            getattr(model_resolution.model_config, "fallback_models", None) or []
        )
        attempted: list[str] = [primary_alias]

        for fallback_alias in fallback_aliases:
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
                messages=messages,  # same messages — reused safely
                temperature=route_decision.temperature,
                max_tokens=route_decision.max_tokens,
                provider_options={},  # strip thinking/stream options for fallback
            )

            logger.debug(
                "registry_backed_model_executor.execute  trying_fallback  "
                "fallback_alias=%s  provider=%s",
                fallback_alias,
                fallback_resolution.provider,
            )

            try:
                raw_result = self._provider_executor.execute(fallback_request)
                if not _is_visible_text(raw_result.content):
                    attempted.append(fallback_alias)
                    logger.warning(
                        "registry_backed_model_executor.execute  fallback_empty_answer  "
                        "fallback_alias=%s — skipping",
                        fallback_alias,
                    )
                    continue
                # Build a new result that records the fallback provenance safely.
                result = ModelExecutionResult(
                    content=raw_result.content,
                    model=fallback_alias,
                    provider=raw_result.provider,
                    finish_reason=raw_result.finish_reason,
                    input_tokens=raw_result.input_tokens,
                    output_tokens=raw_result.output_tokens,
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
                return result
            except LlmProviderExecutionError as exc:
                attempted.append(fallback_alias)
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_failed  "
                    "fallback_alias=%s  failure_kind=%s",
                    fallback_alias,
                    exc.failure_kind,
                )
            except Exception as exc:
                attempted.append(fallback_alias)
                logger.warning(
                    "registry_backed_model_executor.execute  fallback_error  "
                    "fallback_alias=%s  error=%s",
                    fallback_alias,
                    type(exc).__name__,
                )

        # All attempts exhausted.
        raise ProviderExecutionError(
            f"All model execution attempts failed. "
            f"Attempted aliases: {attempted}. "
            f"Primary failure_kind: {primary_failure_kind!r}."
        )

    def execute_stream(
        self,
        *,
        route_decision: RouteDecision,
        messages: list[LlmMessage],
        on_before_fallback: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        """Resolve model metadata and stream answer text chunks from the provider."""
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
        profile = request.model_resolution.provider_profile
        credentials = self._credential_resolver.resolve(profile)

        adapter = self._provider_factory.get_provider(request.model_resolution.provider)

        logger.debug(
            "provider_adapter_executor.execute  model_alias=%s  provider=%s",
            request.model_resolution.model_alias,
            request.model_resolution.provider,
        )

        return adapter.generate(request=request, credentials=credentials)

    def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
        """Execute the provider request and yield answer text chunks."""
        profile = request.model_resolution.provider_profile
        credentials = self._credential_resolver.resolve(profile)

        adapter = self._provider_factory.get_provider(request.model_resolution.provider)

        logger.debug(
            "provider_adapter_executor.execute_stream  model_alias=%s  provider=%s",
            request.model_resolution.model_alias,
            request.model_resolution.provider,
        )

        if hasattr(adapter, "generate_stream"):
            finish_reason: str | None = None
            for chunk in adapter.generate_stream(request=request, credentials=credentials):
                if hasattr(adapter, "last_stream_finish_reason"):
                    finish_reason = adapter.last_stream_finish_reason
                yield chunk
            self.last_stream_finish_reason = finish_reason or "stop"
            return

        # Buffered fallback when adapter lacks native streaming.
        result = adapter.generate(request=request, credentials=credentials)
        if result.content:
            yield result.content
        self.last_stream_finish_reason = result.finish_reason or "stop"
