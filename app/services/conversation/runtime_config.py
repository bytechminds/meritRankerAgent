"""Cached SSM-backed conversation runtime configuration."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from services.aws_client_factory import get_ssm_client

_HISTORY_ROOT = "/meritranker/agent-runtime/v1/conversation-history"
_SESSION_ROOT = "/meritranker/agent-runtime/v1/conversation-session"
_TABLE_NAME_PATH = f"{_HISTORY_ROOT}/table-name"
_TABLE_ARN_PATH = f"{_HISTORY_ROOT}/table-arn"
_SESSION_TABLE_NAME_PATH = f"{_SESSION_ROOT}/table-name"
_SESSION_TABLE_ARN_PATH = f"{_SESSION_ROOT}/table-arn"
_MEMORY_ENV = "MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID"
_lock = threading.Lock()
_cached: ConversationRuntimeConfig | None = None


class ConversationConfigurationError(RuntimeError):
    """Required conversation infrastructure configuration is unavailable."""


@dataclass(frozen=True)
class ConversationRuntimeConfig:
    table_name: str
    table_arn: str
    conversation_session_table_name: str | None
    conversation_session_table_arn: str | None
    memory_id: str | None
    region_name: str


def load_conversation_runtime_config(
    *, ssm_client: Any | None = None, region_name: str | None = None
) -> ConversationRuntimeConfig:
    global _cached  # noqa: PLW0603
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        memory_id = os.getenv(_MEMORY_ENV, "").strip() or None
        resolved_region = (
            region_name
            or os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
        )
        if not resolved_region:
            raise ConversationConfigurationError(
                "Missing conversation runtime configuration: AWS_REGION"
            )
        client = ssm_client or get_ssm_client(resolved_region)
        try:
            response = client.get_parameters(
                Names=[
                    _TABLE_NAME_PATH,
                    _TABLE_ARN_PATH,
                    _SESSION_TABLE_NAME_PATH,
                    _SESSION_TABLE_ARN_PATH,
                ],
                WithDecryption=False,
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "unknown")
            raise ConversationConfigurationError(
                f"Conversation runtime configuration read failed: {code}"
            ) from exc
        values = {
            str(item.get("Name")): str(item.get("Value", "")).strip()
            for item in response.get("Parameters", [])
        }
        required_paths = (_TABLE_NAME_PATH, _TABLE_ARN_PATH)
        missing = [path for path in required_paths if not values.get(path)]
        if missing:
            raise ConversationConfigurationError(
                f"Missing conversation runtime configuration: {', '.join(missing)}"
            )
        _cached = ConversationRuntimeConfig(
            table_name=values[_TABLE_NAME_PATH],
            table_arn=values[_TABLE_ARN_PATH],
            conversation_session_table_name=values.get(_SESSION_TABLE_NAME_PATH) or None,
            conversation_session_table_arn=values.get(_SESSION_TABLE_ARN_PATH) or None,
            memory_id=memory_id,
            region_name=resolved_region,
        )
        return _cached


def reset_conversation_runtime_config_for_tests() -> None:
    global _cached  # noqa: PLW0603
    _cached = None
