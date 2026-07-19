"""DynamoDB source-of-truth PatternGraph / SolveFlow bundle store."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from config import Settings, get_settings
from retrieval.models import PatternRuntimeBundle
from services.dynamodb_service import DynamoDbServiceError, get_item


class PatternBundleStoreError(Exception):
    """Raised when DynamoDB cannot safely provide a runtime bundle."""


class DynamoDbPatternStore:
    """Fetch one bundle by the approved configurable PatternGraph primary key."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        get_item_fn: Callable[[str, dict[str, Any]], dict[str, Any] | None] = get_item,
    ) -> None:
        self._settings = settings or get_settings()
        self._get_item = get_item_fn

    def fetch(self, pattern_id: str) -> PatternRuntimeBundle | None:
        normalized_id = pattern_id.strip()
        if not normalized_id:
            return None
        if not self._settings.dynamodb_pattern_table:
            raise PatternBundleStoreError("DYNAMODB_PATTERN_TABLE must be configured.")
        if not self._settings.dynamodb_pattern_pk:
            raise PatternBundleStoreError("DYNAMODB_PATTERN_PK must be configured.")

        try:
            item = self._get_item(
                self._settings.dynamodb_pattern_table,
                {self._settings.dynamodb_pattern_pk: normalized_id},
            )
        except DynamoDbServiceError as exc:
            raise PatternBundleStoreError("DynamoDB pattern bundle fetch failed.") from exc

        if item is None:
            return None
        try:
            return PatternRuntimeBundle.model_validate(item)
        except ValidationError as exc:
            raise PatternBundleStoreError(
                "DynamoDB pattern bundle did not match the runtime contract."
            ) from exc
