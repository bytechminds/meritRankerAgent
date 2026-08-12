"""
app/services/llm_providers/mock_provider.py
---------------------------------------------
Mock provider implementations.

Legacy class (model_router / BaseLlmProvider path):
    MockProvider — echoes the last user message, yields word-level stream chunks.

Part 6 adapter class (ProviderAdapterExecutor path):
    MockProviderAdapter — deterministic adapter conforming to ProviderAdapter
                          protocol; no network, no env, no credentials required.

No network calls. No API keys required. No env reads.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from schemas.llm import LlmRequest, LlmResponse, LlmRoleConfig, LlmStreamChunk
from services.llm.providers.base import BaseLlmProvider
from services.llm.providers.finish_reasons import normalize_completion_outcome

if TYPE_CHECKING:
    from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
    from services.secrets.provider_credentials import ProviderCredentials

logger = logging.getLogger(__name__)


class MockProvider(BaseLlmProvider):
    """Deterministic mock provider — always returns predictable content."""

    def generate(self, request: LlmRequest, config: LlmRoleConfig) -> LlmResponse:
        """Return a deterministic response that echoes the last user message.

        Content format: ``[mock] echo: <last user message content>``
        """
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        content = f"[mock] echo: {last_user}"
        logger.debug(
            "mock_provider.generate  role=%s  model_label=%s",
            request.role,
            config.model_label,
        )
        return LlmResponse(
            role=request.role,
            provider="mock",
            model_label=config.model_label,
            content=content,
            finish_reason="stop",
        )

    def stream(self, request: LlmRequest, config: LlmRoleConfig) -> Iterator[LlmStreamChunk]:
        """Yield one LlmStreamChunk per word of the mock response.

        The last word's chunk has is_final=True.
        """
        response = self.generate(request, config)
        words = response.content.split()
        if not words:
            yield LlmStreamChunk(
                role=request.role,
                provider="mock",
                model_label=config.model_label,
                content_delta="",
                is_final=True,
            )
            return

        for i, word in enumerate(words):
            is_final = i == len(words) - 1
            yield LlmStreamChunk(
                role=request.role,
                provider="mock",
                model_label=config.model_label,
                content_delta=word + ("" if is_final else " "),
                is_final=is_final,
            )


# ---------------------------------------------------------------------------
# Part 6 adapter — ProviderAdapterExecutor path
# ---------------------------------------------------------------------------


class MockProviderAdapter:
    """Part 6 deterministic mock adapter conforming to the ProviderAdapter protocol.

    - No network calls.
    - No env reads.
    - No credentials required (credentials parameter is accepted but ignored).
    - Useful for execution boundary tests without any external dependency.

    Records ``last_request`` and ``call_count`` for test assertions.
    """

    def __init__(self, *, content: str = "Mock provider response.") -> None:
        """
        Args:
            content: The fixed response string to return on every generate() call.
        """
        self._content = content
        self.last_request: ProviderExecutionRequest | None = None
        self.call_count: int = 0

    def generate(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> ModelExecutionResult:
        """Return a deterministic ModelExecutionResult.

        Args:
            request:     The resolved provider execution request.
            credentials: Accepted for protocol conformance; ignored by mock.

        Returns:
            ModelExecutionResult with provider="mock" and the configured content.
        """
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        self.last_request = request
        self.call_count += 1

        logger.debug(
            "mock_provider_adapter.generate  model_alias=%s",
            request.model_resolution.model_alias,
        )

        return ModelExecutionResult(
            content=self._content,
            model=request.route_decision.model,
            provider="mock",
            finish_reason="stop",
            normalized_finish_reason=normalize_completion_outcome("stop"),
            metadata={
                "model_alias": request.model_resolution.model_alias,
                "model_label": request.model_resolution.model_config.model_label,
            },
        )

    def generate_stream(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> Iterator[str]:
        """Yield deterministic text chunks for the configured mock content."""
        self.last_request = request
        self.call_count += 1

        logger.debug(
            "mock_provider_adapter.generate_stream  model_alias=%s",
            request.model_resolution.model_alias,
        )

        if not self._content:
            yield ""
            return

        chunk_size = 8
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]
