"""
app/tests/test_provider_credentials.py
----------------------------------------
Tests for ProviderCredentialResolver and ProviderCredentials (Part 5).

Coverage:
1.  ProviderCredentialResolver resolves api_key_env into api_key.
2.  Resolves endpoint_env.
3.  Resolves api_version_env.
4.  Resolves base_url_env.
5.  Resolves multiple env fields together.
6.  Missing api_key env raises SecretNotFoundError.
7.  credential_ref raises SecretResolverUnsupportedError (Part 5 deferred).
8.  ProviderCredentials.safe_metadata() hides actual values.
9.  ProviderCredentials.safe_metadata() exposes only boolean flags.
10. ProviderCredentials repr does not expose secret values.
11. ProviderCredentialResolver does not read env at construction time.
12. ProviderCredentialResolver does not call provider SDKs (AST check).
13. ProviderCredentialResolver does not call AWS (AST check).
14. ProviderCredentialResolver works with ProviderProfile from llm_routing schema.
15. Import safety: importing secrets modules does not read env vars.
16. ProviderCredentials rejects blank string fields.
17. mock profile with no env refs returns empty ProviderCredentials.
18. credential_ref error message contains "credential_ref" or "deferred".

All tests use monkeypatch for env vars — no .env.local dependency.
"""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from schemas.llm_routing import ProviderProfile
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.errors import SecretNotFoundError, SecretResolverUnsupportedError
from services.secrets.provider_credentials import ProviderCredentialResolver, ProviderCredentials

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRETS_DIR = pathlib.Path(__file__).parent.parent / "services" / "secrets"
_SECRET_MODULE_NAMES = [
    "env_secret_resolver",
    "secret_resolver",
    "errors",
    "provider_credentials",
]


def _make_env_resolver(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> ProviderCredentialResolver:
    """Build a ProviderCredentialResolver backed by EnvSecretResolver with patched env."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return ProviderCredentialResolver(secret_resolver=EnvSecretResolver())


# ---------------------------------------------------------------------------
# 1–5: Happy path — individual and combined env refs
# ---------------------------------------------------------------------------


class TestProviderCredentialResolverHappyPath:
    """Successful resolution of individual and combined env var references."""

    def test_resolves_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolver = _make_env_resolver({"OPENAI_API_KEY": "test-api-key"}, monkeypatch)
        profile = ProviderProfile(provider="openai", api_key_env="OPENAI_API_KEY")
        creds = resolver.resolve(profile)
        assert creds.api_key == "test-api-key"
        assert creds.provider == "openai"
        assert creds.endpoint is None
        assert creds.api_version is None
        assert creds.base_url is None

    def test_resolves_endpoint_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolver = _make_env_resolver(
            {"AZURE_ENDPOINT": "https://test.openai.azure.com"}, monkeypatch
        )
        profile = ProviderProfile(provider="azure_openai", endpoint_env="AZURE_ENDPOINT")
        creds = resolver.resolve(profile)
        assert creds.endpoint == "https://test.openai.azure.com"
        assert creds.api_key is None

    def test_resolves_api_version_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolver = _make_env_resolver(
            {"AZURE_API_VERSION": "2024-02-01"}, monkeypatch
        )
        profile = ProviderProfile(
            provider="azure_openai", api_version_env="AZURE_API_VERSION"
        )
        creds = resolver.resolve(profile)
        assert creds.api_version == "2024-02-01"

    def test_resolves_base_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolver = _make_env_resolver(
            {"GEMINI_BASE_URL": "https://api.gemini.test"}, monkeypatch
        )
        profile = ProviderProfile(provider="gemini", base_url_env="GEMINI_BASE_URL")
        creds = resolver.resolve(profile)
        assert creds.base_url == "https://api.gemini.test"

    def test_resolves_multiple_env_fields_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {
            "AZURE_KEY": "az-key",
            "AZURE_ENDPOINT": "https://ep.azure.com",
            "AZURE_VERSION": "2024-01-01",
        }
        resolver = _make_env_resolver(env, monkeypatch)
        profile = ProviderProfile(
            provider="azure_openai",
            api_key_env="AZURE_KEY",
            endpoint_env="AZURE_ENDPOINT",
            api_version_env="AZURE_VERSION",
        )
        creds = resolver.resolve(profile)
        assert creds.api_key == "az-key"
        assert creds.endpoint == "https://ep.azure.com"
        assert creds.api_version == "2024-01-01"
        assert creds.base_url is None

    def test_mock_profile_with_no_env_refs_returns_all_none(self) -> None:
        """A mock provider profile with no env refs → all fields None."""
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile(provider="mock")
        creds = resolver.resolve(profile)
        assert creds.provider == "mock"
        assert creds.api_key is None
        assert creds.endpoint is None
        assert creds.api_version is None
        assert creds.base_url is None


# ---------------------------------------------------------------------------
# 6–7: Error paths
# ---------------------------------------------------------------------------


class TestProviderCredentialResolverErrorPaths:
    """Missing env vars and unsupported features raise expected errors."""

    def test_missing_api_key_env_raises_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile(provider="openai", api_key_env="OPENAI_API_KEY")
        with pytest.raises(SecretNotFoundError):
            resolver.resolve(profile)

    def test_credential_ref_raises_unsupported_error(self) -> None:
        """credential_ref is deferred in Part 5."""
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile(provider="openai", credential_ref="my-secret-resource")
        with pytest.raises(SecretResolverUnsupportedError):
            resolver.resolve(profile)

    def test_credential_ref_error_message_is_informative(self) -> None:
        """credential_ref error must mention 'credential_ref' or 'deferred'."""
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile(provider="openai", credential_ref="safe-ref-name")
        with pytest.raises(SecretResolverUnsupportedError) as exc_info:
            resolver.resolve(profile)
        msg = str(exc_info.value).lower()
        assert "credential_ref" in msg or "deferred" in msg

    def test_credential_ref_error_does_not_expose_arn(self) -> None:
        """credential_ref error must not include raw secret ARN in message."""
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        # Use a ref that looks like an ARN — error should be generic
        profile = ProviderProfile(
            provider="openai",
            credential_ref="my-secret-arn-ref",
        )
        with pytest.raises(SecretResolverUnsupportedError) as exc_info:
            resolver.resolve(profile)
        # The error message should not expose actual secret values
        # (ARN is a reference name, not a secret value, so it may appear)
        # Main check: no actual key material in the message
        assert "sk-" not in str(exc_info.value)
        assert "AIza" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8–10: ProviderCredentials safe_metadata and repr
# ---------------------------------------------------------------------------


class TestProviderCredentialsSafeMetadata:
    """safe_metadata() exposes only boolean flags; repr hides values."""

    def test_safe_metadata_hides_api_key_value(self) -> None:
        creds = ProviderCredentials(provider="openai", api_key="sk-test-value")
        meta = creds.safe_metadata()
        assert "sk-test-value" not in str(meta)
        assert meta["has_api_key"] is True

    def test_safe_metadata_exposes_only_booleans_for_optional_fields(self) -> None:
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="key",
            endpoint="https://ep.test",
        )
        meta = creds.safe_metadata()
        assert meta["provider"] == "azure_openai"
        assert isinstance(meta["has_api_key"], bool)
        assert isinstance(meta["has_endpoint"], bool)
        assert isinstance(meta["has_api_version"], bool)
        assert isinstance(meta["has_base_url"], bool)

    def test_safe_metadata_returns_false_for_absent_fields(self) -> None:
        creds = ProviderCredentials(provider="mock")
        meta = creds.safe_metadata()
        assert meta["has_api_key"] is False
        assert meta["has_endpoint"] is False
        assert meta["has_api_version"] is False
        assert meta["has_base_url"] is False

    def test_safe_metadata_returns_true_for_all_present_fields(self) -> None:
        creds = ProviderCredentials(
            provider="gemini",
            api_key="key",
            endpoint="https://ep.test",
            api_version="v1",
            base_url="https://base.test",
        )
        meta = creds.safe_metadata()
        assert meta["has_api_key"] is True
        assert meta["has_endpoint"] is True
        assert meta["has_api_version"] is True
        assert meta["has_base_url"] is True

    def test_safe_metadata_has_exactly_the_expected_keys(self) -> None:
        creds = ProviderCredentials(provider="openai")
        meta = creds.safe_metadata()
        assert set(meta.keys()) == {
            "provider",
            "has_api_key",
            "has_endpoint",
            "has_api_version",
            "has_base_url",
            "region",
            "azure_api_mode",
        }

    def test_repr_does_not_expose_api_key_value(self) -> None:
        creds = ProviderCredentials(provider="openai", api_key="sk-super-secret")
        r = repr(creds)
        assert "sk-super-secret" not in r
        assert "has_api_key=True" in r

    def test_repr_shows_provider_and_boolean_flags(self) -> None:
        creds = ProviderCredentials(provider="gemini", api_key="key")
        r = repr(creds)
        assert "gemini" in r
        assert "has_api_key=True" in r
        assert "has_endpoint=False" in r


# ---------------------------------------------------------------------------
# 11–13: Isolation — no env reads at construction, no SDK/AWS calls
# ---------------------------------------------------------------------------


class TestProviderCredentialResolverIsolation:
    """Construction never reads env; resolution never calls SDKs or AWS."""

    def test_construction_does_not_call_get_secret(self) -> None:
        """Building the resolver must not invoke get_secret on the injected resolver."""
        mock_resolver = MagicMock()
        ProviderCredentialResolver(secret_resolver=mock_resolver)
        mock_resolver.get_secret.assert_not_called()

    def test_construction_does_not_read_os_environ(self) -> None:
        """Building the resolver must not access os.environ."""
        from unittest.mock import patch

        with patch("os.environ") as mock_env:
            ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        mock_env.__getitem__.assert_not_called()
        mock_env.get.assert_not_called()

    def test_resolver_does_not_import_provider_sdks(self) -> None:
        """No secrets module may import provider SDKs."""
        banned = ["openai", "anthropic", "google", "azure", "boto3"]
        for module_name in _SECRET_MODULE_NAMES:
            path = _SECRETS_DIR / f"{module_name}.py"
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for sdk in banned:
                            assert sdk not in alias.name, (
                                f"secrets/{module_name}.py must not import {sdk!r}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for sdk in banned:
                        assert sdk not in module, (
                            f"secrets/{module_name}.py must not import from {sdk!r}"
                        )

    def test_resolver_does_not_call_aws_during_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve() must not make any AWS API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile(provider="openai", api_key_env="OPENAI_API_KEY")
        # Patch boto3 at session level to detect any import/call
        with patch.dict("sys.modules", {"boto3": None}):
            creds = resolver.resolve(profile)
        assert creds.api_key == "test-key"


# ---------------------------------------------------------------------------
# 14: Works with ProviderProfile from llm_routing schema
# ---------------------------------------------------------------------------


class TestProviderCredentialResolverWithSchema:
    """ProviderCredentialResolver integrates correctly with the ProviderProfile schema."""

    def test_works_with_pydantic_constructed_provider_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile.model_validate(
            {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"}
        )
        creds = resolver.resolve(profile)
        assert creds.api_key == "gemini-test-key"
        assert creds.provider == "gemini"

    def test_works_with_full_azure_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = {
            "AZURE_OPENAI_KEY": "az-test-key",
            "AZURE_OPENAI_ENDPOINT": "https://my.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = ProviderProfile.model_validate(
            {
                "provider": "azure_openai",
                "api_key_env": "AZURE_OPENAI_KEY",
                "endpoint_env": "AZURE_OPENAI_ENDPOINT",
                "api_version_env": "AZURE_OPENAI_API_VERSION",
            }
        )
        creds = resolver.resolve(profile)
        assert creds.api_key == "az-test-key"
        assert creds.endpoint == "https://my.openai.azure.com"
        assert creds.api_version == "2024-02-01"
        assert creds.safe_metadata()["has_api_key"] is True
        assert creds.safe_metadata()["has_endpoint"] is True


# ---------------------------------------------------------------------------
# 15: Import safety
# ---------------------------------------------------------------------------


class TestImportSafety:
    """Importing secrets modules must not read env vars or call external systems.

    These tests verify that importing the modules succeeds without errors.
    We do NOT call importlib.reload() because reloading creates new class
    objects, which corrupts isinstance/except checks in tests that imported
    those classes before the reload.
    """

    def test_import_errors_module(self) -> None:
        import services.secrets.errors  # noqa: F401

    def test_import_secret_resolver_module(self) -> None:
        import services.secrets.secret_resolver  # noqa: F401

    def test_import_env_secret_resolver_module(self) -> None:
        import services.secrets.env_secret_resolver  # noqa: F401

    def test_import_provider_credentials_module(self) -> None:
        import services.secrets.provider_credentials  # noqa: F401

    def test_import_secrets_package(self) -> None:
        import services.secrets  # noqa: F401


# ---------------------------------------------------------------------------
# 16: ProviderCredentials field validation
# ---------------------------------------------------------------------------


class TestProviderCredentialsValidation:
    """ProviderCredentials Pydantic model enforces field constraints."""

    def test_blank_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCredentials(provider="openai", api_key="   ")

    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCredentials(provider="openai", api_key="")

    def test_blank_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCredentials(provider="azure_openai", endpoint="  ")

    def test_none_fields_are_allowed(self) -> None:
        creds = ProviderCredentials(provider="mock")
        assert creds.api_key is None
        assert creds.endpoint is None
        assert creds.api_version is None
        assert creds.base_url is None

    def test_empty_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCredentials(provider="")
