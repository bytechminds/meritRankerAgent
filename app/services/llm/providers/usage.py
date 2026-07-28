"""Provider-local token usage extraction with no prompt or response logging."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar

from schemas.llm_usage import ProviderTokenUsage

_stream_usage: ContextVar[ProviderTokenUsage | None] = ContextVar(
    "llm_provider_stream_usage", default=None
)


def _read(value: object, *names: str) -> object | None:
    current = value
    for name in names:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


def _count(value: object | None) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None


def _first_non_none(*values: object | None) -> object | None:
    return next((value for value in values if value is not None), None)


def _usage(
    *,
    input_tokens: object | None,
    output_tokens: object | None,
    total_tokens: object | None,
    cached_input_tokens: object | None,
    reasoning_tokens: object | None,
) -> ProviderTokenUsage:
    input_count = _count(input_tokens)
    output_count = _count(output_tokens)
    total_count = _count(total_tokens)
    if total_count is None and input_count is not None and output_count is not None:
        total_count = input_count + output_count
    return ProviderTokenUsage(
        input_tokens=input_count,
        output_tokens=output_count,
        total_tokens=total_count,
        cached_input_tokens=_count(cached_input_tokens),
        reasoning_tokens=_count(reasoning_tokens),
    )


def extract_openai_usage(response: object) -> ProviderTokenUsage:
    """Extract OpenAI/Azure/OpenAI-compatible usage fields when present."""
    usage = _read(response, "usage")
    return _usage(
        input_tokens=_first_non_none(
            _read(usage, "prompt_tokens"),
            _read(usage, "input_tokens"),
        ),
        output_tokens=_first_non_none(
            _read(usage, "completion_tokens"),
            _read(usage, "output_tokens"),
        ),
        total_tokens=_read(usage, "total_tokens"),
        cached_input_tokens=_first_non_none(
            _read(usage, "prompt_tokens_details", "cached_tokens"),
            _read(usage, "input_tokens_details", "cached_tokens"),
        ),
        reasoning_tokens=_first_non_none(
            _read(usage, "completion_tokens_details", "reasoning_tokens"),
            _read(usage, "output_tokens_details", "reasoning_tokens"),
        ),
    )


def extract_gemini_usage(response: object) -> ProviderTokenUsage:
    """Extract native Gemini metadata, falling back to OpenAI-compatible usage."""
    metadata = _first_non_none(
        _read(response, "usage_metadata"),
        _read(response, "usageMetadata"),
    )
    if metadata is None:
        return extract_openai_usage(response)
    return _usage(
        input_tokens=_first_non_none(
            _read(metadata, "prompt_token_count"),
            _read(metadata, "promptTokenCount"),
        ),
        output_tokens=_first_non_none(
            _read(metadata, "candidates_token_count"),
            _read(metadata, "candidatesTokenCount"),
        ),
        total_tokens=_first_non_none(
            _read(metadata, "total_token_count"),
            _read(metadata, "totalTokenCount"),
        ),
        cached_input_tokens=_first_non_none(
            _read(metadata, "cached_content_token_count"),
            _read(metadata, "cachedContentTokenCount"),
        ),
        reasoning_tokens=_first_non_none(
            _read(metadata, "thoughts_token_count"),
            _read(metadata, "thoughtsTokenCount"),
        ),
    )


def extract_bedrock_usage(response_or_event: object) -> ProviderTokenUsage:
    """Extract Bedrock Converse or ConverseStream metadata usage."""
    usage = _read(response_or_event, "usage")
    if usage is None:
        usage = _read(response_or_event, "metadata", "usage")
    return _usage(
        input_tokens=_first_non_none(
            _read(usage, "inputTokens"),
            _read(usage, "input_tokens"),
        ),
        output_tokens=_first_non_none(
            _read(usage, "outputTokens"),
            _read(usage, "output_tokens"),
        ),
        total_tokens=_first_non_none(
            _read(usage, "totalTokens"),
            _read(usage, "total_tokens"),
        ),
        cached_input_tokens=_first_non_none(
            _read(usage, "cacheReadInputTokens"),
            _read(usage, "cache_read_input_tokens"),
        ),
        reasoning_tokens=None,
    )


def clear_stream_usage() -> None:
    _stream_usage.set(None)


def set_stream_usage(usage: ProviderTokenUsage) -> None:
    _stream_usage.set(usage)


def consume_stream_usage() -> ProviderTokenUsage:
    usage = _stream_usage.get()
    _stream_usage.set(None)
    return usage or ProviderTokenUsage()
