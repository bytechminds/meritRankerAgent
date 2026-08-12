"""
app/services/dynamodb_service.py
----------------------------------
Generic DynamoDB low-level client service.

Public surface:
    get_item(table_name, key) -> dict | None
    batch_get_items(table_name, keys) -> list[dict]
    query_by_partition_key(table_name, key_name, key_value, *, index_name, limit)
    query_by_index(table_name, index_name, key_name, key_value, *, limit)

Architecture invariants:
    - No Scan support by design.  Scans are unbounded cost risks; add only after explicit review.
    - Query limit is capped at _MAX_QUERY_LIMIT (50) to prevent unbounded calls.
    - Batch size is capped at _MAX_BATCH_SIZE (100 — DynamoDB API maximum).
    - Full item payloads are NEVER logged.
    - Graph nodes MUST NOT import this module directly.
    - All boto3 calls go through aws_client_factory.get_dynamodb_client().

[DYNAMODB SERVICE FOUNDATION — not yet wired into the Doubt Solver graph]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from botocore.exceptions import ClientError

from config import get_settings
from services.aws_client_factory import get_dynamodb_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard cap for Query operations.  Never query more than this per call.
_MAX_QUERY_LIMIT = 50

# DynamoDB BatchGetItem allows at most 100 keys per request.
_MAX_BATCH_SIZE = 100

# Opt-in retries are deliberately small to keep request latency bounded.
_MAX_UNPROCESSED_KEY_RETRIES = 2
_UNPROCESSED_KEY_RETRY_BASE_SECONDS = 0.05

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DynamoDbServiceError(Exception):
    """Raised when a DynamoDB API call fails unexpectedly."""


class DynamoDbConfigurationError(Exception):
    """Raised when DynamoDB is enabled but required configuration is missing."""


# ---------------------------------------------------------------------------
# AttributeValue conversion helpers  (pure, no I/O)
# ---------------------------------------------------------------------------


def _parse_number(n: str) -> int | float:
    """Parse a DynamoDB number string to int or float."""
    if "." in n or "e" in n.lower():
        return float(n)
    return int(n)


def _from_dynamodb_value(value: dict[str, Any]) -> Any:
    """Convert a single DynamoDB AttributeValue dict to a plain Python value.

    Supported types: S, N, BOOL, NULL, L, M, SS, NS, B.
    Unknown type codes return None rather than raising — defensive parsing.
    """
    if "S" in value:
        return value["S"]
    if "N" in value:
        return _parse_number(value["N"])
    if "BOOL" in value:
        return value["BOOL"]
    if "NULL" in value:
        return None
    if "L" in value:
        return [_from_dynamodb_value(v) for v in value["L"]]
    if "M" in value:
        return {k: _from_dynamodb_value(v) for k, v in value["M"].items()}
    if "SS" in value:
        return list(value["SS"])
    if "NS" in value:
        return [_parse_number(n) for n in value["NS"]]
    if "B" in value:
        return value["B"]  # bytes — return as-is
    return None  # unknown type code


def _from_dynamodb_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an entire DynamoDB item (dict of AttributeValues) to plain Python."""
    return {k: _from_dynamodb_value(v) for k, v in item.items()}


def _to_dynamodb_key_value(value: Any) -> dict[str, Any]:
    """Convert a plain Python value to DynamoDB AttributeValue for key lookups.

    DynamoDB primary keys must be S (string) or N (number).  Binary keys are
    not supported here.

    Raises:
        TypeError: If the value type is not supported as a DynamoDB key.
    """
    # bool must be checked before int since bool is a subclass of int in Python.
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid DynamoDB key types")
    if isinstance(value, (int, float)):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    raise TypeError(f"Unsupported DynamoDB key value type: {type(value).__name__!r}")


def _key_to_dynamodb(key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert a plain Python key dict to DynamoDB AttributeValue key format."""
    return {k: _to_dynamodb_key_value(v) for k, v in key.items()}


# ---------------------------------------------------------------------------
# Internal client helper
# ---------------------------------------------------------------------------


def _get_client() -> Any:
    """Return a DynamoDB client using the region from settings."""
    region = get_settings().dynamodb_region or None
    return get_dynamodb_client(region_name=region)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_item(
    table_name: str,
    key: dict[str, Any],
) -> dict[str, Any] | None:
    """Fetch a single item by primary key.

    Args:
        table_name: DynamoDB table name (from env/settings — never hardcoded).
        key:        Plain Python dict mapping key attribute names to values.
                    Example: ``{"question_id": "q-001"}``.

    Returns:
        Plain Python dict of the item attributes, or None if not found.

    Raises:
        DynamoDbServiceError: On any ClientError from DynamoDB.
    """
    client = _get_client()
    t0 = time.monotonic()
    try:
        response = client.get_item(
            TableName=table_name,
            Key=_key_to_dynamodb(key),
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise DynamoDbServiceError(
            f"DynamoDB GetItem failed (table={table_name!r}, code={error_code})"
        ) from exc

    duration_ms = int((time.monotonic() - t0) * 1000)
    item = response.get("Item")
    logger.info(
        "DynamoDB get_item",
        extra={
            "table": table_name,
            "found": item is not None,
            "duration_ms": duration_ms,
        },
    )
    if item is None:
        return None
    return _from_dynamodb_item(item)


def batch_get_items(
    table_name: str,
    keys: list[dict[str, Any]],
    *,
    max_unprocessed_retries: int = 0,
) -> list[dict[str, Any]]:
    """Fetch multiple items by primary key in a single batch call.

    Args:
        table_name: DynamoDB table name.
        keys:       List of plain Python key dicts.
                    Capped at _MAX_BATCH_SIZE (100) silently.
        max_unprocessed_retries: Opt-in bounded retry count for DynamoDB
                    ``UnprocessedKeys`` responses. Defaults to ``0`` to
                    preserve the legacy single-call behavior. Maximum ``2``.

    Returns:
        List of plain Python dicts.  Missing items are simply absent (not an error).

    Raises:
        DynamoDbServiceError: On any ClientError from DynamoDB.

    When retries are enabled, only the unprocessed keys are retried with a
    short bounded backoff. Items returned before a partial response are kept.
    """
    if not keys:
        return []
    if (
        isinstance(max_unprocessed_retries, bool)
        or not isinstance(max_unprocessed_retries, int)
        or not 0 <= max_unprocessed_retries <= _MAX_UNPROCESSED_KEY_RETRIES
    ):
        raise ValueError(
            "max_unprocessed_retries must be an integer between 0 and "
            f"{_MAX_UNPROCESSED_KEY_RETRIES}."
        )

    capped_keys = keys[:_MAX_BATCH_SIZE]
    dynamo_keys = [_key_to_dynamodb(k) for k in capped_keys]

    client = _get_client()
    t0 = time.monotonic()
    pending_keys = dynamo_keys
    raw_items: list[dict[str, Any]] = []
    retry_count = 0
    while pending_keys:
        try:
            response = client.batch_get_item(
                RequestItems={table_name: {"Keys": pending_keys}}
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            raise DynamoDbServiceError(
                f"DynamoDB BatchGetItem failed (table={table_name!r}, code={error_code})"
            ) from exc

        response_items = response.get("Responses", {}).get(table_name, [])
        if isinstance(response_items, list):
            raw_items.extend(item for item in response_items if isinstance(item, dict))

        unprocessed = response.get("UnprocessedKeys", {}).get(table_name, {})
        unprocessed_keys = unprocessed.get("Keys", []) if isinstance(unprocessed, dict) else []
        if (
            not isinstance(unprocessed_keys, list)
            or not unprocessed_keys
            or retry_count >= max_unprocessed_retries
        ):
            break

        retry_count += 1
        pending_keys = [key for key in unprocessed_keys if isinstance(key, dict)]
        if not pending_keys:
            break
        time.sleep(_UNPROCESSED_KEY_RETRY_BASE_SECONDS * retry_count)

    duration_ms = int((time.monotonic() - t0) * 1000)
    result_count = len(raw_items)
    logger.info(
        "DynamoDB batch_get_items",
        extra={
            "table": table_name,
            "requested": len(capped_keys),
            "result_count": result_count,
            "unprocessed_retry_count": retry_count,
            "duration_ms": duration_ms,
        },
    )
    return [_from_dynamodb_item(item) for item in raw_items]


def query_by_partition_key(
    table_name: str,
    key_name: str,
    key_value: str,
    *,
    index_name: str | None = None,
    limit: int = 10,
    projection_fields: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Query items by a partition key condition.

    Optionally directs the query at a GSI/LSI via *index_name*.

    Args:
        table_name:  DynamoDB table name.
        key_name:    Partition key attribute name (e.g. ``"question_id"``).
        key_value:   Partition key value.
        index_name:  GSI or LSI name.  None means query the base table.
        limit:       Maximum number of items to return.  Capped at _MAX_QUERY_LIMIT.
        projection_fields: Optional safe attributes to fetch.  When supplied,
                    DynamoDB omits all non-projected fields from the response.

    Returns:
        List of plain Python dicts, up to *limit* items.

    Raises:
        DynamoDbServiceError: On any ClientError from DynamoDB.
    """
    effective_limit = min(max(limit, 1), _MAX_QUERY_LIMIT)

    kwargs: dict[str, Any] = {
        "TableName": table_name,
        "KeyConditionExpression": "#pk = :pk_val",
        "ExpressionAttributeNames": {"#pk": key_name},
        "ExpressionAttributeValues": {":pk_val": _to_dynamodb_key_value(key_value)},
        "Limit": effective_limit,
    }
    if index_name:
        kwargs["IndexName"] = index_name
    if projection_fields:
        safe_fields = tuple(
            dict.fromkeys(
                field.strip()
                for field in projection_fields
                if isinstance(field, str) and field.strip()
            )
        )
        if safe_fields:
            projection_names = {
                f"#projection_{index}": field
                for index, field in enumerate(safe_fields)
            }
            kwargs["ExpressionAttributeNames"].update(projection_names)
            kwargs["ProjectionExpression"] = ", ".join(projection_names)

    client = _get_client()
    t0 = time.monotonic()
    try:
        response = client.query(**kwargs)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise DynamoDbServiceError(
            f"DynamoDB Query failed (table={table_name!r}, code={error_code})"
        ) from exc

    duration_ms = int((time.monotonic() - t0) * 1000)
    items = response.get("Items", [])
    logger.info(
        "DynamoDB query_by_partition_key",
        extra={
            "table": table_name,
            "result_count": len(items),
            "duration_ms": duration_ms,
        },
    )
    return [_from_dynamodb_item(item) for item in items]


def query_by_index(
    table_name: str,
    index_name: str,
    key_name: str,
    key_value: str,
    *,
    limit: int = 10,
    projection_fields: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Query items via a named GSI or LSI.

    Thin wrapper around :func:`query_by_partition_key` that makes the
    *index_name* argument required.

    Args:
        table_name:  DynamoDB table name.
        index_name:  GSI or LSI name (required).
        key_name:    Index partition key attribute name.
        key_value:   Index partition key value.
        limit:       Maximum items to return.  Capped at _MAX_QUERY_LIMIT.
        projection_fields: Optional safe attributes to fetch.

    Returns:
        List of plain Python dicts.

    Raises:
        DynamoDbServiceError: On any ClientError from DynamoDB.
    """
    return query_by_partition_key(
        table_name,
        key_name,
        key_value,
        index_name=index_name,
        limit=limit,
        projection_fields=projection_fields,
    )
