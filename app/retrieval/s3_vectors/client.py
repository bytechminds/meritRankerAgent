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

    def query_pattern_intelligence_candidates(
        self,
        *,
        query_vector: list[float],
        subject: str | None,
        top_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        """Return bounded canonical Pattern candidates for authoritative hydration.

        This method intentionally applies no legacy runtime approval metadata.
        The canonical Pattern record remains the only authority for status,
        quality, and compatibility after DynamoDB hydration.
        """
        maximum_candidates = self._settings.pattern_intelligence_max_candidates
        requested_top_k = maximum_candidates if top_k is None else top_k
        effective_top_k = min(max(1, requested_top_k), maximum_candidates)
        normalized_subject = subject.strip() if isinstance(subject, str) else ""
        metadata_filter = {"subject": normalized_subject} if normalized_subject else None
        return self._query(
            index_name=self._settings.s3_vector_pattern_index_name,
            query_vector=query_vector,
            top_k=effective_top_k,
            metadata_filter=metadata_filter,
        )

    def query_question_page(
        self,
        *,
        query_vector: list[float],
        metadata_filter: dict[str, object] | None,
        top_k: int,
        next_token: str | None = None,
    ) -> tuple[list[RetrievedCandidate], str | None]:
        """Fetch one page of questions-v1 candidates plus its continuation token.

        Discovery only: every candidate is re-validated against the authoritative
        QuestionBank row before it may be served.  Paging policy lives with the
        caller so the bound stays a product decision, not a transport detail.
        """
        index_arn = self._settings.s3_vector_question_index_arn
        if not index_arn:
            raise S3VectorQueryError("S3_VECTOR_QUESTION_INDEX_ARN must be configured.")
        if len(query_vector) != self._settings.s3_vector_dimensions:
            raise S3VectorQueryError("Query vector dimension does not match S3_VECTOR_DIMENSIONS.")
        request: dict[str, object] = {
            "indexArn": index_arn,
            "queryVector": {"float32": query_vector},
            "topK": max(1, top_k),
            "returnDistance": True,
            "returnMetadata": True,
        }
        if metadata_filter:
            request["filter"] = metadata_filter
        if next_token:
            request["nextToken"] = next_token
        client = self._client_factory(self._settings.s3_vector_region or None)
        try:
            response = client.query_vectors(**request)
        except ClientError as exc:
            raise S3VectorQueryError("questions-v1 candidate query failed.") from exc
        payload = response if isinstance(response, dict) else {}
        token = payload.get("nextToken")
        return (
            map_query_vectors_response(payload),
            str(token) if isinstance(token, str) and token else None,
        )

    def _query(
        self,
        *,
        index_name: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, object] | None,
    ) -> list[RetrievedCandidate]:
        if len(query_vector) != self._settings.s3_vector_dimensions:
            raise S3VectorQueryError("Query vector dimension does not match S3_VECTOR_DIMENSIONS.")
        index_arn = (
            self._settings.s3_vector_pattern_index_arn
            if index_name == self._settings.s3_vector_pattern_index_name
            else ""
        )
        if not index_arn and (not self._settings.s3_vector_bucket_name or not index_name):
            raise S3VectorQueryError(
                "S3 Vector index ARN or bucket/index names must be configured."
            )

        client = self._client_factory(self._settings.s3_vector_region or None)
        request: dict[str, object] = {
            "queryVector": {"float32": query_vector},
            "topK": max(1, top_k),
            "returnDistance": True,
            "returnMetadata": True,
        }
        if index_arn:
            request["indexArn"] = index_arn
        else:
            request["vectorBucketName"] = self._settings.s3_vector_bucket_name
            request["indexName"] = index_name
        if metadata_filter:
            request["filter"] = metadata_filter
        try:
            response = client.query_vectors(**request)
        except ClientError as exc:
            raise S3VectorQueryError("S3 Vector candidate query failed.") from exc
        return map_query_vectors_response(response if isinstance(response, dict) else {})
