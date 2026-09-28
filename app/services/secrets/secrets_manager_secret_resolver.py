"""
app/services/secrets/secrets_manager_secret_resolver.py
---------------------------------------------------------
SecretsManagerSecretResolver — the Secrets Manager backend of ``SecretResolver``.

Contract:
- One JSON secret whose keys are the existing provider ENV names
  (e.g. ``AZURE_OPENAI_API_KEY``) and whose values are the credentials.
- Fetched once, on first use, through the shared AWS client factory, then
  cached for the process lifetime; no Secrets Manager call per LLM request.
- A name absent from the secret falls back to ``EnvSecretResolver`` so the
  existing ENV contract keeps working unchanged.
- An unreadable or malformed secret raises ``SecretResolverError`` (never
  ``SecretNotFoundError``) so no caller can mistake it for an optional absence.
- Never logs or includes secret values in error messages.

``PROVIDER_CREDENTIALS_SECRET_ID`` selects this backend. When it is unset
(local development) ``get_runtime_secret_resolver`` returns ``EnvSecretResolver``.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from services.aws_client_factory import get_secretsmanager_client
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.errors import SecretResolverConfigError, SecretResolverError
from services.secrets.secret_resolver import SecretResolver

PROVIDER_CREDENTIALS_SECRET_ID_ENV = "PROVIDER_CREDENTIALS_SECRET_ID"


class SecretsManagerSecretResolver:
    """Resolves provider credentials from one cached Secrets Manager JSON secret."""

    def __init__(
        self,
        secret_id: str,
        *,
        region_name: str | None = None,
        client: Any | None = None,
        fallback: SecretResolver | None = None,
    ) -> None:
        if not secret_id.strip():
            raise SecretResolverConfigError("Provider secret id must not be empty.")
        self._secret_id = secret_id.strip()
        self._region_name = region_name or None
        self._client = client
        self._fallback = fallback or EnvSecretResolver()
        self._values: dict[str, str] | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Fetch and cache the secret once; raise without exposing any value."""
        with self._lock:
            if self._values is not None:
                return
            client = self._client or get_secretsmanager_client(self._region_name)
            try:
                response = client.get_secret_value(SecretId=self._secret_id)
            except Exception as exc:
                raise SecretResolverError(
                    f"Provider secret {self._secret_id!r} could not be read "
                    f"(error_type={type(exc).__name__})."
                ) from exc
            raw = response.get("SecretString")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else None
            except json.JSONDecodeError:
                parsed = None
            if not isinstance(parsed, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in parsed.items()
            ):
                raise SecretResolverConfigError(
                    f"Provider secret {self._secret_id!r} must be a JSON object "
                    "of string values keyed by ENV name."
                )
            for key in parsed:
                EnvSecretResolver._validate_name(key)
            self._values = {key: value for key, value in parsed.items() if value.strip()}

    def get_secret(self, name: str) -> str:
        EnvSecretResolver._validate_name(name)
        self.load()
        assert self._values is not None
        value = self._values.get(name)
        if value is not None:
            return value
        return self._fallback.get_secret(name)


_runtime_resolvers: dict[str, SecretsManagerSecretResolver] = {}
_runtime_lock = threading.Lock()


def get_runtime_secret_resolver() -> SecretResolver:
    """Return the process-wide resolver selected by ``PROVIDER_CREDENTIALS_SECRET_ID``."""
    secret_id = os.getenv(PROVIDER_CREDENTIALS_SECRET_ID_ENV, "").strip()
    if not secret_id:
        return EnvSecretResolver()
    with _runtime_lock:
        resolver = _runtime_resolvers.get(secret_id)
        if resolver is None:
            resolver = SecretsManagerSecretResolver(
                secret_id, region_name=os.getenv("AWS_REGION") or None
            )
            _runtime_resolvers[secret_id] = resolver
        return resolver
