"""
app/services/llm/providers/payload_shaping.py
---------------------------------------------
Provider/model capability-driven request shaping for LLM adapters.

Converts neutral ProviderExecutionRequest + ModelConfig into provider-safe
chat completion kwargs. Drops unsupported parameters and None values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemas.llm_orchestration import ProviderExecutionRequest
    from schemas.llm_routing import ModelConfig

logger = logging.getLogger(__name__)

_BLOCKED_THINKING_KEYS = frozenset({"thinking", "provider_options"})


def effective_token_budget_param(model_config: ModelConfig) -> str:
    """Resolve output token budget field for provider payloads."""
    if model_config.token_budget_param != "max_tokens":
        return model_config.token_budget_param
    if model_config.provider == "azure_openai" and model_config.supports_reasoning:
        return "max_completion_tokens"
    return "max_tokens"


def effective_supports_temperature(model_config: ModelConfig) -> bool:
    """Resolve whether temperature may be sent for provider payloads."""
    if model_config.provider == "azure_openai" and model_config.supports_reasoning:
        return False
    return model_config.supports_temperature


@dataclass(frozen=True)
class PayloadShapingMetadata:
    """Safe payload-shaping diagnostics (no messages, secrets, or bodies)."""

    route_id: str
    model_alias: str
    provider: str
    deployment: str
    token_budget_param_used: str
    dropped_params: tuple[str, ...]
    supports_streaming: bool
    supports_reasoning: bool
    reasoning_param_sent: bool


def _resolve_send_reasoning_effort(model_config: ModelConfig) -> bool:
    """Return whether reasoning_effort may be sent for this model."""
    if not model_config.send_reasoning_effort:
        return False
    if model_config.reasoning_effort == "none":
        return False
    try:
        from config import get_settings  # noqa: PLC0415
    except ImportError:
        return False
    return get_settings().azure_openai_send_reasoning_effort


def build_azure_openai_chat_completion_kwargs(
    *,
    request: ProviderExecutionRequest,
    deployment: str,
    stream: bool = False,
) -> tuple[dict[str, Any], PayloadShapingMetadata]:
    """Build Azure OpenAI chat.completions.create kwargs from model capabilities."""
    model_config = request.model_resolution.model_config
    dropped: list[str] = []
    payload: dict[str, Any] = {
        "model": deployment,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }

    token_budget = request.max_tokens
    budget_param = effective_token_budget_param(model_config)
    if budget_param == "max_completion_tokens":
        payload["max_completion_tokens"] = token_budget
        dropped.append("max_tokens")
    elif budget_param == "max_output_tokens":
        payload["max_output_tokens"] = token_budget
        dropped.append("max_tokens")
    else:
        payload["max_tokens"] = token_budget

    if effective_supports_temperature(model_config):
        payload["temperature"] = request.temperature
    else:
        dropped.append("temperature")

    if not model_config.supports_top_p:
        dropped.extend(("top_p",))
    if not model_config.supports_penalties:
        dropped.extend(("presence_penalty", "frequency_penalty"))
    if not model_config.supports_logprobs:
        dropped.extend(("logprobs", "top_logprobs", "logit_bias"))

    for blocked in _BLOCKED_THINKING_KEYS:
        if blocked in request.provider_options:
            dropped.append(blocked)

    reasoning_param_sent = False
    if _resolve_send_reasoning_effort(model_config):
        payload["reasoning_effort"] = model_config.reasoning_effort
        reasoning_param_sent = True
    else:
        dropped.append("reasoning_effort")

    if stream:
        payload["stream"] = True

    payload = {key: value for key, value in payload.items() if value is not None}

    metadata = PayloadShapingMetadata(
        route_id=request.route_decision.route_id,
        model_alias=request.model_resolution.model_alias,
        provider=str(model_config.provider),
        deployment=deployment,
        token_budget_param_used=budget_param,
        dropped_params=tuple(dropped),
        supports_streaming=model_config.supports_streaming,
        supports_reasoning=model_config.supports_reasoning,
        reasoning_param_sent=reasoning_param_sent,
    )
    log_payload_shaped(metadata)
    return payload, metadata


def log_payload_shaped(metadata: PayloadShapingMetadata) -> None:
    """Emit safe payload-shaping diagnostics."""
    logger.info(
        "llm_payload_shaped  route_id=%s  model_alias=%s  provider=%s  deployment=%s  "
        "token_budget_param=%s  dropped_params=%s  supports_streaming=%s  "
        "supports_reasoning=%s  reasoning_param_sent=%s",
        metadata.route_id,
        metadata.model_alias,
        metadata.provider,
        metadata.deployment,
        metadata.token_budget_param_used,
        ",".join(metadata.dropped_params) if metadata.dropped_params else "",
        metadata.supports_streaming,
        metadata.supports_reasoning,
        metadata.reasoning_param_sent,
    )
