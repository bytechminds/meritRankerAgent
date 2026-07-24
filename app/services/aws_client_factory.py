"""
app/services/aws_client_factory.py
-----------------------------------
Lazy factory for AWS SDK clients.

Rules:
- Uses boto3 default credential chain only (IAM role, env vars, ~/.aws/credentials).
- No credentials in code, no hardcoded regions.
- Each client is created lazily on first use.
- Graph nodes MUST NOT import this module directly.
"""

from __future__ import annotations

import threading
from typing import Any

import boto3
from botocore.config import Config

# Per-service, per-region cache protected by a lock.
# Key format: "<service>::<region_or___default__>"
# Using a composite key prevents collisions between different service types
# sharing the same region (e.g. bedrock-agent-runtime and dynamodb).
_lock = threading.Lock()
_clients: dict[str, Any] = {}
_CONVERSATION_AWS_CONFIG = Config(
    connect_timeout=1,
    read_timeout=1,
    retries={"max_attempts": 1, "mode": "standard"},
)


def _get_or_create_client(service: str, region_name: str | None) -> Any:
    """Return a cached boto3 client for *service* in *region_name*.

    Args:
        service:     AWS service identifier (e.g. ``"bedrock-agent-runtime"``).
        region_name: AWS region.  When None or empty, boto3 uses its default
                     region resolution chain.

    Returns:
        A boto3 low-level client for the requested service.
    """
    cache_key = f"{service}::{region_name or '__default__'}"

    with _lock:
        if cache_key not in _clients:
            kwargs: dict[str, Any] = {"service_name": service}
            if region_name:
                kwargs["region_name"] = region_name
            _clients[cache_key] = boto3.client(**kwargs)
        return _clients[cache_key]


def get_bedrock_agent_runtime_client(region_name: str | None = None) -> Any:
    """Return a cached ``bedrock-agent-runtime`` boto3 client."""
    return _get_or_create_client("bedrock-agent-runtime", region_name)


def get_dynamodb_client(region_name: str | None = None) -> Any:
    """Return a cached ``dynamodb`` boto3 low-level client."""
    return _get_or_create_client("dynamodb", region_name)


def get_bedrock_runtime_client(region_name: str | None = None) -> Any:
    """Return a cached ``bedrock-runtime`` boto3 low-level client."""
    return _get_or_create_client("bedrock-runtime", region_name)


def get_s3_vectors_client(region_name: str | None = None) -> Any:
    """Return a cached ``s3vectors`` boto3 low-level client."""
    return _get_or_create_client("s3vectors", region_name)


def get_ssm_client(region_name: str | None = None) -> Any:
    """Return a cached ``ssm`` client."""
    return _get_or_create_client("ssm", region_name)


def get_bedrock_agentcore_client(region_name: str | None = None) -> Any:
    """Return a cached ``bedrock-agentcore`` control-plane data client."""
    cache_key = f"bedrock-agentcore-conversation::{region_name or '__default__'}"
    with _lock:
        if cache_key not in _clients:
            kwargs: dict[str, Any] = {
                "service_name": "bedrock-agentcore",
                "config": _CONVERSATION_AWS_CONFIG,
            }
            if region_name:
                kwargs["region_name"] = region_name
            _clients[cache_key] = boto3.client(**kwargs)
        return _clients[cache_key]


def get_conversation_dynamodb_client(region_name: str | None = None) -> Any:
    """Return a bounded-timeout DynamoDB client for conversation persistence."""
    cache_key = f"dynamodb-conversation::{region_name or '__default__'}"
    with _lock:
        if cache_key not in _clients:
            kwargs: dict[str, Any] = {
                "service_name": "dynamodb",
                "config": _CONVERSATION_AWS_CONFIG,
            }
            if region_name:
                kwargs["region_name"] = region_name
            _clients[cache_key] = boto3.client(**kwargs)
        return _clients[cache_key]
