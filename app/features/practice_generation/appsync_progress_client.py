"""IAM-authenticated transport for the deployed practice-progress mutation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from features.practice_generation.progress_contract import PRACTICE_PROGRESS_ALLOWED_META_KEYS

logger = logging.getLogger(__name__)

_MUTATION = """
mutation UpdatePracticeGenerationProgress(
  $testId: String!
  $userId: String!
  $status: String!
  $meta: AWSJSON!
  $live: Boolean!
  $expectedUpdatedAt: String!
) {
  updatePracticeGenerationProgress(
    testId: $testId
    userId: $userId
    status: $status
    meta: $meta
    live: $live
    expectedUpdatedAt: $expectedUpdatedAt
  ) {
    testId
    userId
    status
    totalQuestions
    live
    meta
    updatedAt
  }
}
""".strip()

_NON_RETRYABLE_CODES = (
    "UNAUTHORIZED",
    "OWNER_MISMATCH",
    "INVALID_",
    "UNKNOWN_META_FIELD",
    "META_TOO_LARGE",
    "CONFLICT",
    "GRAPHQL_VALIDATION_FAILED",
)

# AppSync errorType / extensions.code are controlled classification labels
# ("Unauthorized", "DynamoDB:ConditionalCheckFailedException", "Lambda:Unhandled").
# The pattern admits only that shape, so a GraphQL message carrying request data
# can never reach a log through this field.
_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}")


def _json_compatible(value: object) -> object:
    """Convert DynamoDB number values without changing their JSON meaning."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal")
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _serialize_meta(meta: dict[str, object]) -> str:
    try:
        return json.dumps(
            _json_compatible(meta),
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PracticeProgressError("PRACTICE_PROGRESS_SERIALIZATION_FAILED") from exc


def _validate_meta_keys(meta: dict[str, object]) -> None:
    if any(
        not isinstance(key, str) or key not in PRACTICE_PROGRESS_ALLOWED_META_KEYS
        for key in meta
    ):
        raise PracticeProgressError("PRACTICE_PROGRESS_UNKNOWN_META_FIELD")


class PracticeProgressError(RuntimeError):
    """Controlled AppSync progress-publication failure."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        safe_detail: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.safe_detail = safe_detail
        self.error_type = error_type


class PracticeProgressResult(BaseModel):
    """Typed safe selection returned by the deployed mutation."""

    test_id: str = Field(alias="testId")
    user_id: str = Field(alias="userId")
    status: str
    total_questions: int = Field(alias="totalQuestions")
    live: bool
    meta: dict[str, Any]
    updated_at: str = Field(alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("meta", mode="before")
    @classmethod
    def _parse_meta(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid AppSync meta") from exc
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("invalid AppSync meta")


class AppSyncPracticeProgressClient:
    """Sign and send only ``updatePracticeGenerationProgress`` requests."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        timeout_seconds: float = 8.0,
        max_retries: int = 2,
        session: boto3.Session | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if not endpoint.strip():
            raise PracticeProgressError("PRACTICE_APPSYNC_ENDPOINT_MISSING")
        if not region.strip():
            raise PracticeProgressError("PRACTICE_APPSYNC_REGION_MISSING")
        self._endpoint = endpoint.strip()
        self._region = region.strip()
        self._timeout = timeout_seconds
        self._max_retries = min(max(max_retries, 0), 2)
        self._session = session or boto3.Session(region_name=self._region)
        self._http_client = http_client
        self._sleep = sleep

    async def update_progress(
        self,
        *,
        test_id: str,
        user_id: str,
        status: str,
        meta: dict[str, object],
        live: bool | None,
        expected_updated_at: str,
    ) -> PracticeProgressResult:
        if live is None:
            raise PracticeProgressError("PRACTICE_PROGRESS_LIVE_REQUIRED")
        current_meta = dict(meta)
        _validate_meta_keys(current_meta)
        for attempt in range(self._max_retries + 1):
            try:
                return await self._send(
                    test_id=test_id,
                    user_id=user_id,
                    status=status,
                    meta=current_meta,
                    live=live,
                    expected_updated_at=expected_updated_at,
                )
            except PracticeProgressError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                await self._sleep(0.1 * (2**attempt))
        raise PracticeProgressError("PRACTICE_PROGRESS_UPDATE_FAILED")

    async def _send(
        self,
        *,
        test_id: str,
        user_id: str,
        status: str,
        meta: dict[str, object],
        live: bool,
        expected_updated_at: str,
    ) -> PracticeProgressResult:
        body = json.dumps(
            {
                "query": _MUTATION,
                "variables": {
                    "testId": test_id,
                    "userId": user_id,
                    "status": status,
                    "meta": _serialize_meta(meta),
                    "live": live,
                    "expectedUpdatedAt": expected_updated_at,
                },
            },
            separators=(",", ":"),
        ).encode()
        headers = self._signed_headers(body)
        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    self._endpoint,
                    content=body,
                    headers=headers,
                    timeout=self._timeout,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self._endpoint,
                        content=body,
                        headers=headers,
                        timeout=self._timeout,
                    )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise PracticeProgressError(
                "PRACTICE_PROGRESS_TRANSPORT_FAILED", retryable=True
            ) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise PracticeProgressError("PRACTICE_PROGRESS_TRANSPORT_FAILED", retryable=True)
        if response.status_code >= 400:
            raise PracticeProgressError("PRACTICE_PROGRESS_HTTP_REJECTED")
        try:
            payload = response.json()
        except ValueError as exc:
            raise PracticeProgressError("PRACTICE_PROGRESS_INVALID_RESPONSE") from exc
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if isinstance(errors, list) and errors:
            code, safe_detail, error_type = self._graphql_error(errors)
            raise PracticeProgressError(
                code,
                retryable=False,
                safe_detail=safe_detail,
                error_type=error_type,
            )
        try:
            result = payload["data"]["updatePracticeGenerationProgress"]
            return PracticeProgressResult.model_validate(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise PracticeProgressError("PRACTICE_PROGRESS_INVALID_RESPONSE") from exc

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        try:
            credentials = self._session.get_credentials()
            if credentials is None:
                raise PracticeProgressError(
                    "PRACTICE_PROGRESS_CREDENTIALS_UNAVAILABLE", retryable=True
                )
            frozen = credentials.get_frozen_credentials()
            request = AWSRequest(
                method="POST",
                url=self._endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            SigV4Auth(frozen, "appsync", self._region).add_auth(request)
            return {str(key): str(value) for key, value in request.headers.items()}
        except PracticeProgressError:
            raise
        except Exception as exc:  # temporary refresh failures are retried by the caller
            logger.warning(
                "practice AppSync credential refresh failed error_type=%s",
                type(exc).__name__,
            )
            raise PracticeProgressError(
                "PRACTICE_PROGRESS_CREDENTIALS_UNAVAILABLE", retryable=True
            ) from exc

    @staticmethod
    def _safe_error_type(error: dict[str, Any]) -> str | None:
        """Sanitized AppSync classification label, never the GraphQL message."""
        candidates = [error.get("errorType")]
        extensions = error.get("extensions")
        if isinstance(extensions, dict):
            candidates.extend((extensions.get("errorType"), extensions.get("code")))
        for candidate in candidates:
            if isinstance(candidate, str) and _SAFE_ERROR_TYPE.fullmatch(candidate):
                return candidate
        return None

    @staticmethod
    def _graphql_error(errors: list[Any]) -> tuple[str, str | None, str | None]:
        for error in errors:
            if not isinstance(error, dict):
                continue
            message = str(error.get("message") or "")
            for token in message.replace(":", " ").split():
                if token.startswith("PRACTICE_PROGRESS_"):
                    code = token.rstrip(".,")
                    extensions = error.get("extensions")
                    safe_detail = None
                    if isinstance(extensions, dict):
                        candidate = extensions.get("unknownField")
                        if isinstance(candidate, str) and re.fullmatch(
                            r"[A-Za-z][A-Za-z0-9]{0,63}",
                            candidate,
                        ):
                            safe_detail = candidate
                    error_type = AppSyncPracticeProgressClient._safe_error_type(error)
                    if any(marker in code for marker in _NON_RETRYABLE_CODES):
                        return code, safe_detail, error_type
                    return code, safe_detail, error_type
            # Gate stays on raw presence, exactly as before: the sanitizer decides
            # what may be logged, never which error terminates the scan.
            if str(error.get("errorType") or ""):
                return (
                    "PRACTICE_PROGRESS_GRAPHQL_REJECTED",
                    None,
                    AppSyncPracticeProgressClient._safe_error_type(error),
                )
        return "PRACTICE_PROGRESS_GRAPHQL_REJECTED", None, None
