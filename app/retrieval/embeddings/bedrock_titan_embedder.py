"""Bedrock Titan Text Embeddings V2 query embedder."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from botocore.exceptions import ClientError

from config import Settings, get_settings
from retrieval.embeddings.base import EmbeddingConfigurationError, QueryEmbeddingError
from services.aws_client_factory import get_bedrock_runtime_client

_MODEL_ID = "amazon.titan-embed-text-v2:0"
_DIMENSIONS = 1024


class BedrockTitanEmbedder:
    """Embed a normalized student retrieval query using the approved Titan model."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client_factory: Callable[[str | None], Any] = get_bedrock_runtime_client,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory
        self._validate_contract()

    def embed_query(self, text: str) -> list[float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise QueryEmbeddingError("Retrieval query must not be empty.")

        client = self._client_factory(self._settings.bedrock_embedding_region or None)
        try:
            response = client.invoke_model(
                modelId=_MODEL_ID,
                body=json.dumps(
                    {
                        "inputText": normalized,
                        "dimensions": _DIMENSIONS,
                        "normalize": True,
                    }
                ),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as exc:
            raise QueryEmbeddingError("Bedrock query embedding request failed.") from exc

        body = response.get("body")
        try:
            raw_body = body.read() if hasattr(body, "read") else body
            payload = raw_body.decode("utf-8") if isinstance(raw_body, bytes) else raw_body
            parsed = json.loads(payload)
        except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryEmbeddingError("Bedrock query embedding response was invalid.") from exc

        embedding = parsed.get("embedding") if isinstance(parsed, dict) else None
        if not isinstance(embedding, list) or len(embedding) != _DIMENSIONS:
            raise QueryEmbeddingError("Bedrock query embedding dimension did not equal 1024.")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise QueryEmbeddingError("Bedrock query embedding contained non-numeric values.")
        return [float(value) for value in embedding]

    def _validate_contract(self) -> None:
        if self._settings.bedrock_embedding_provider != "bedrock":
            raise EmbeddingConfigurationError("BEDROCK_EMBEDDING_PROVIDER must be 'bedrock'.")
        if self._settings.bedrock_embedding_model_id != _MODEL_ID:
            raise EmbeddingConfigurationError(
                "BEDROCK_EMBEDDING_MODEL_ID must be 'amazon.titan-embed-text-v2:0'."
            )
        if self._settings.bedrock_embedding_dimensions != _DIMENSIONS:
            raise EmbeddingConfigurationError("BEDROCK_EMBEDDING_DIMENSIONS must equal 1024.")
        if self._settings.s3_vector_dimensions != _DIMENSIONS:
            raise EmbeddingConfigurationError("S3_VECTOR_DIMENSIONS must equal 1024.")
        if not self._settings.bedrock_embedding_normalize:
            raise EmbeddingConfigurationError("BEDROCK_EMBEDDING_NORMALIZE must be true.")
