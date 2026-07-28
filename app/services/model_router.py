"""
app/services/model_router.py
------------------------------
Legacy LLM routing service (role-based ``LLM_ROLE_CONFIG_JSON`` path).

Active when ``ENABLE_ORCHESTRATED_DOUBT_SOLVER=false``:
  - ``answer_generator_service.generate_answer`` (legacy 7-node graph)
  - ``query_classifier_service.classify_query`` legacy branch

The orchestrated doubt solver path uses ``services.llm.orchestration`` and
``services.doubt_solver.answer_generation_adapter`` instead — not this module.

Rules:
- Graph nodes must not import provider modules directly.
- Provider selection is driven entirely by role config + ENABLE_REAL_LLM flag.
- Secrets (API keys, endpoints) are read inside provider modules — never here.
- Log role, provider, model_label only — never keys or full request content.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from observability import current_request_context, record_llm_call
from schemas.llm import LlmMessage, LlmRequest, LlmResponse, LlmStreamChunk
from schemas.llm_usage import ProviderTokenUsage, UsageStatus
from services.llm.providers.azure_openai_provider import AzureOpenAIProvider
from services.llm.providers.base import BaseLlmProvider
from services.llm.providers.errors import LlmConfigurationError
from services.llm.providers.mock_provider import MockProvider
from services.llm.providers.openai_provider import OpenAIProvider
from services.llm.providers.usage import clear_stream_usage, consume_stream_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type[BaseLlmProvider]] = {
    "mock": MockProvider,
    "azure_openai": AzureOpenAIProvider,
    "openai": OpenAIProvider,
}


def _get_provider(provider_name: str) -> BaseLlmProvider:
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise LlmConfigurationError(
            f"Unknown provider {provider_name!r}. "
            f"Supported: {list(_PROVIDER_MAP.keys())}"
        )
    return cls()


def _coerce_messages(messages: list[LlmMessage] | list[dict]) -> list[LlmMessage]:
    """Accept either LlmMessage objects or plain dicts and return LlmMessage list."""
    result: list[LlmMessage] = []
    for m in messages:
        if isinstance(m, LlmMessage):
            result.append(m)
        else:
            result.append(LlmMessage.model_validate(m))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    role: str,
    messages: list[LlmMessage] | list[dict],
) -> LlmResponse:
    """Generate a complete LLM response for the given role and messages.

    When ENABLE_REAL_LLM=false (default), the mock provider is always used
    regardless of the role config.  Set ENABLE_REAL_LLM=true to use the
    provider configured in LLM_ROLE_CONFIG_JSON.

    Args:
        role:     Named role (maps to a config entry in LLM_ROLE_CONFIG_JSON).
        messages: Conversation messages as LlmMessage objects or plain dicts.

    Returns:
        Provider-neutral LlmResponse.

    Raises:
        LlmConfigurationError: If config is missing or the provider is unknown.
        LlmGenerationError:    If the provider call fails.
    """
    # Import deferred to ensure dotenv has loaded before config is read.
    from config import get_llm_role_config  # noqa: PLC0415

    config = get_llm_role_config(role)
    provider = _get_provider(config.provider)
    coerced = _coerce_messages(messages)
    request = LlmRequest(role=role, messages=coerced)

    logger.info(
        "model_router.generate  role=%s  provider=%s  model_label=%s",
        role,
        config.provider,
        config.model_label,
    )
    started_at = time.monotonic()
    try:
        response = provider.generate(request, config)
    except Exception as exc:
        _record_usage(
            role=role,
            config=config,
            usage=ProviderTokenUsage(),
            started_at=started_at,
            streaming=False,
            status="failed",
            error_type=type(exc).__name__,
        )
        raise
    _record_usage(
        role=role,
        config=config,
        usage=ProviderTokenUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cached_input_tokens=response.cached_input_tokens,
            reasoning_tokens=response.reasoning_tokens,
        ),
        started_at=started_at,
        streaming=False,
        status="succeeded",
    )
    return response


def stream(
    role: str,
    messages: list[LlmMessage] | list[dict],
) -> Iterator[LlmStreamChunk]:
    """Stream LLM response chunks for the given role and messages.

    Same provider-selection rules as generate().

    Args:
        role:     Named role (maps to a config entry in LLM_ROLE_CONFIG_JSON).
        messages: Conversation messages as LlmMessage objects or plain dicts.

    Yields:
        LlmStreamChunk instances. The last chunk has is_final=True.

    Raises:
        LlmProviderError:      If the chosen provider does not support streaming.
        LlmConfigurationError: If config is missing or provider is unknown.
        LlmGenerationError:    If the provider call fails during streaming.
    """
    from config import get_llm_role_config  # noqa: PLC0415

    config = get_llm_role_config(role)
    provider = _get_provider(config.provider)
    coerced = _coerce_messages(messages)
    request = LlmRequest(role=role, messages=coerced)

    logger.info(
        "model_router.stream  role=%s  provider=%s  model_label=%s",
        role,
        config.provider,
        config.model_label,
    )
    started_at = time.monotonic()
    status: UsageStatus = "succeeded"
    error_type: str | None = None
    clear_stream_usage()
    try:
        yield from provider.stream(request, config)
    except GeneratorExit:
        status = "cancelled"
        error_type = "GeneratorExit"
        raise
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        _record_usage(
            role=role,
            config=config,
            usage=consume_stream_usage(),
            started_at=started_at,
            streaming=True,
            status=status,
            error_type=error_type,
        )


def _record_usage(
    *,
    role: str,
    config: object,
    usage: ProviderTokenUsage,
    started_at: float,
    streaming: bool,
    status: UsageStatus,
    error_type: str | None = None,
) -> None:
    context = current_request_context()
    provider = str(getattr(config, "provider", "unknown"))
    model = (
        getattr(config, "model", None)
        or getattr(config, "deployment", None)
        or getattr(config, "model_label", "unknown")
    )
    deployment = getattr(config, "deployment", None)
    record_llm_call(
        request_id=context.request_id if context else "unknown",
        role=role,
        provider=provider,
        model=str(model),
        deployment=str(deployment) if deployment else None,
        attempt_type="primary",
        streaming=streaming,
        usage=usage,
        duration_ms=max(int((time.monotonic() - started_at) * 1000), 0),
        status=status,
        error_type=error_type,
    )
