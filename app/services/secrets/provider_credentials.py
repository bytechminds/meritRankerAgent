"""
app/services/secrets/provider_credentials.py
----------------------------------------------
ProviderCredentials model and ProviderCredentialResolver.

Design:
- ``ProviderCredentials`` holds resolved credential values for one provider call.
  Fields are ``None`` when not needed by the provider.
- ``safe_metadata()`` returns only boolean presence flags — never actual values.
  Use this method for any logging or metadata propagation.
- ``ProviderCredentialResolver`` accepts a ``SecretResolver`` and resolves the
  env var references in a ``ProviderProfile`` into a ``ProviderCredentials``
  instance.  No env reads happen at construction time.
- ``credential_ref`` is recognized but not resolved in Part 5.
  It raises ``SecretResolverUnsupportedError``.

Security:
- Never log or include ``ProviderCredentials`` in orchestration metadata directly.
- Use ``safe_metadata()`` for any observability.
- ``__repr__`` is overridden to show only boolean flags, not values.

Deferred:
- [DEFER] credential_ref runtime resolution (Secrets Manager or AgentCore Identity).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.llm_routing import ProviderProfile
from services.secrets.errors import SecretNotFoundError, SecretResolverUnsupportedError
from services.secrets.secret_resolver import SecretResolver


class ProviderCredentials(BaseModel):
    """Resolved provider credentials.

    Security contract:
    - Never include this model directly in logs, responses, or metadata.
    - Use ``safe_metadata()`` for any observability purpose.
    - ``__repr__`` intentionally omits credential values.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    provider: str = Field(..., min_length=1)
    api_key: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    base_url: str | None = None
    # For azure_openai provider: the API mode selected via ProviderProfile.
    # None is treated as "azure_deployment_chat_completions" by the adapter.
    # This is NOT a secret — it is safe to log.
    azure_api_mode: str | None = None

    @field_validator("api_key", "endpoint", "api_version", "base_url", mode="after")
    @classmethod
    def validate_not_blank(cls, v: str | None) -> str | None:
        """Reject empty strings — field must be non-empty if present."""
        if v is not None and not v:
            raise ValueError(
                "Credential field must not be an empty string when present. "
                "Use None to indicate an absent field."
            )
        return v

    def safe_metadata(self) -> dict[str, object]:
        """Return only boolean presence flags — never the actual credential values.

        Use this method for logging, audit trails, and orchestration metadata.
        """
        return {
            "provider": self.provider,
            "has_api_key": self.api_key is not None,
            "has_endpoint": self.endpoint is not None,
            "has_api_version": self.api_version is not None,
            "has_base_url": self.base_url is not None,
            "azure_api_mode": self.azure_api_mode,
        }

    def __repr__(self) -> str:
        """Safe repr — shows presence flags only, not credential values."""
        return (
            f"ProviderCredentials("
            f"provider={self.provider!r}, "
            f"has_api_key={self.api_key is not None}, "
            f"has_endpoint={self.endpoint is not None}, "
            f"has_api_version={self.api_version is not None}, "
            f"has_base_url={self.base_url is not None}, "
            f"azure_api_mode={self.azure_api_mode!r}"
            f")"
        )


class ProviderCredentialResolver:
    """Resolve ``ProviderProfile`` env var references into ``ProviderCredentials``.

    Each ``*_env`` field in ``ProviderProfile`` is an environment variable name.
    The actual value is fetched via the injected ``SecretResolver`` only when
    ``resolve()`` is called — never at construction time or import time.

    ``credential_ref`` is recognized but not resolved in Part 5.

    Usage:
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        creds = resolver.resolve(provider_profile)
        logger.info("resolved creds: %s", creds.safe_metadata())
    """

    def __init__(self, *, secret_resolver: SecretResolver) -> None:
        self._secret_resolver = secret_resolver

    def resolve(self, profile: ProviderProfile) -> ProviderCredentials:
        """Resolve all env var references in a ``ProviderProfile``.

        Args:
            profile: The provider profile from the compiled LLM config registry.

        Returns:
            ``ProviderCredentials`` with resolved values for all present env fields.

        Raises:
            SecretResolverUnsupportedError: If ``credential_ref`` is present
                (deferred — see [DEFER]).
            SecretNotFoundError: If a referenced env var is not set or blank.
            SecretResolverConfigError: If an env var name fails validation.
        """
        if profile.credential_ref is not None:
            raise SecretResolverUnsupportedError(
                "credential_ref resolution is deferred. "
                "Use env var references (api_key_env, endpoint_env, etc.) "
                "for local development. "
                "[DEFER] Secrets Manager and AgentCore Identity support is planned."
            )

        api_key = self._resolve_optional_secret(
            profile.api_key_env,
            optional=profile.optional_api_key,
        )
        endpoint = self._resolve_optional_secret(
            profile.endpoint_env,
            optional=profile.optional_endpoint,
        )
        api_version = self._resolve_optional_secret(
            profile.api_version_env,
            optional=profile.optional_api_version,
        )
        base_url = self._resolve_optional_secret(
            profile.base_url_env,
            optional=profile.optional_base_url,
        )

        return ProviderCredentials(
            provider=profile.provider,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            base_url=base_url,
            azure_api_mode=profile.azure_api_mode,
        )

    def _resolve_optional_secret(
        self,
        env_name: str | None,
        *,
        optional: bool,
    ) -> str | None:
        """Resolve an env var reference, optionally returning None when absent."""
        if env_name is None:
            return None
        try:
            return self._secret_resolver.get_secret(env_name)
        except SecretNotFoundError:
            if optional:
                return None
            raise
