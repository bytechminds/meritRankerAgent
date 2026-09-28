"""
app/services/secrets
---------------------
Secret resolver foundation (Part 5).

Public API:

    SecretResolver                  — Protocol for secret resolution backends
    EnvSecretResolver               — reads from os.environ (local dev)
    ProviderCredentialResolver      — resolves ProviderProfile env refs
    ProviderCredentials             — resolved credentials (use safe_metadata() for logging)
    SecretResolverError             — base error
    SecretResolverConfigError       — invalid resolver config / bad name
    SecretNotFoundError             — secret not found or blank
    SecretResolverUnsupportedError  — operation deferred (credential_ref, Secrets Manager)

Production backend (selected by PROVIDER_CREDENTIALS_SECRET_ID):

    SecretsManagerSecretResolver    — services.secrets.secrets_manager_secret_resolver

Deferred:

    AgentCoreIdentitySecretResolver — [DEFER] AgentCore Identity backend
"""

from __future__ import annotations

from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.errors import (
    SecretNotFoundError,
    SecretResolverConfigError,
    SecretResolverError,
    SecretResolverUnsupportedError,
)
from services.secrets.provider_credentials import ProviderCredentialResolver, ProviderCredentials
from services.secrets.secret_resolver import SecretResolver

__all__ = [
    "SecretResolver",
    "EnvSecretResolver",
    "ProviderCredentialResolver",
    "ProviderCredentials",
    "SecretResolverError",
    "SecretResolverConfigError",
    "SecretNotFoundError",
    "SecretResolverUnsupportedError",
]
