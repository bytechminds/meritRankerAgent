"""Embedding boundary errors."""

from __future__ import annotations


class EmbeddingConfigurationError(Exception):
    """Raised when the fixed Titan V2 / S3 Vector dimension contract is invalid."""


class QueryEmbeddingError(Exception):
    """Raised when Bedrock cannot return a valid query embedding."""
