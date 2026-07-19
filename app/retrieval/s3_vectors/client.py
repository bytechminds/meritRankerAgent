"""Direct S3 Vectors QueryVectors client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botocore.exceptions import ClientError

from config import Settings, get_settings
from retrieval.models import RetrievedCandidate
from retrieval.s3_vectors.query_builder import pattern_filter, runtime_filter
from retrieval.s3_vectors.result_mapper import map_query_vectors_response
from services.aws_client_factory import get_s3_vectors_client


class S3VectorQueryError(Exception):
    """Raised when S3 Vectors cannot return candidate matches."""


class S3VectorClient:
    """Query the runtime and pattern S3 Vector indexes directly."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client_factory: Callable[[str | None], Any] = get_s3_vectors_client,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory

    def query_runtime(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        return self._query(
            index_name=self._settings.s3_vector_runtime_index_name,
            query_vector=query_vector,
            top_k=self._settings.s3_vector_top_k_runtime,
            metadata_filter=runtime_filter(subject),
        )

    def query_pattern(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        return self._query(
            index_name=self._settings.s3_vector_pattern_index_name,
            query_vector=query_vector,
            top_k=self._settings.s3_vector_top_k_pattern,
            metadata_filter=pattern_filter(subject),
        )

    def _query(
        self,
        *,
        index_name: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, object],
    ) -> list[RetrievedCandidate]:
        if len(query_vector) != self._settings.s3_vector_dimensions:
            raise S3VectorQueryError("Query vector dimension does not match S3_VECTOR_DIMENSIONS.")
        if not self._settings.s3_vector_bucket_name or not index_name:
            raise S3VectorQueryError("S3 Vector bucket and index name must be configured.")

        client = self._client_factory(self._settings.s3_vector_region or None)
        try:
            response = client.query_vectors(
                vectorBucketName=self._settings.s3_vector_bucket_name,
                indexName=index_name,
                queryVector={"float32": query_vector},
                topK=max(1, top_k),
                returnDistance=True,
                returnMetadata=True,
                filter=metadata_filter,
            )
        except ClientError as exc:
            raise S3VectorQueryError("S3 Vector candidate query failed.") from exc
        return map_query_vectors_response(response if isinstance(response, dict) else {})
