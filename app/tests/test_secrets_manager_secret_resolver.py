"""Production provider-secret path: Secrets Manager behind the existing SecretResolver seam."""

from __future__ import annotations

import json
import logging

import pytest

import config as cfg_module
from config import ConfigurationError
from services.secrets import secrets_manager_secret_resolver as sm_module
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.errors import (
    SecretNotFoundError,
    SecretResolverConfigError,
    SecretResolverError,
)
from services.secrets.secrets_manager_secret_resolver import (
    PROVIDER_CREDENTIALS_SECRET_ID_ENV,
    SecretsManagerSecretResolver,
    get_runtime_secret_resolver,
)

SECRET_ID = "meritranker/agent-runtime/prod/providers"
SECRET_VALUES = {
    "AZURE_OPENAI_ENDPOINT": "https://example-endpoint.invalid/",
    "AZURE_OPENAI_API_KEY": "azure-value-must-not-leak",
    "GEMINI_API_KEY": "gemini-value-must-not-leak",
    "GOOGLE_GEMINI_API_KEY": "image-value-must-not-leak",
    "TAVILY_API_KEY": "tavily-value-must-not-leak",
}


class FakeSecretsManager:
    def __init__(self, secret_string: object = None, error: Exception | None = None) -> None:
        self.secret_string = json.dumps(SECRET_VALUES) if secret_string is None else secret_string
        self.error = error
        self.calls: list[str] = []

    def get_secret_value(self, *, SecretId: str) -> dict:  # noqa: N803 - boto3 casing
        self.calls.append(SecretId)
        if self.error is not None:
            raise self.error
        return {"SecretString": self.secret_string}


@pytest.fixture(autouse=True)
def _isolated_runtime_resolver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(PROVIDER_CREDENTIALS_SECRET_ID_ENV, raising=False)
    monkeypatch.setattr(sm_module, "_runtime_resolvers", {})
    yield


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, fake: FakeSecretsManager) -> list:
    regions: list = []

    def factory(region_name=None):
        regions.append(region_name)
        return fake

    monkeypatch.setattr(sm_module, "get_secretsmanager_client", factory)
    return regions


class TestSecretsManagerSecretResolver:
    def test_values_come_from_one_cached_fetch(self) -> None:
        fake = FakeSecretsManager()
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=fake)
        for _ in range(5):
            assert resolver.get_secret("AZURE_OPENAI_API_KEY") == SECRET_VALUES["AZURE_OPENAI_API_KEY"]
            assert resolver.get_secret("GEMINI_API_KEY") == SECRET_VALUES["GEMINI_API_KEY"]
        assert fake.calls == [SECRET_ID]

    def test_uses_shared_aws_client_factory_with_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeSecretsManager()
        regions = _install_fake_client(monkeypatch, fake)
        resolver = SecretsManagerSecretResolver(SECRET_ID, region_name="ap-south-1")
        assert resolver.get_secret("TAVILY_API_KEY") == SECRET_VALUES["TAVILY_API_KEY"]
        assert regions == ["ap-south-1"]

    def test_absent_key_falls_back_to_existing_env_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_BASE_URL", "https://env-configured.invalid")
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=FakeSecretsManager())
        assert resolver.get_secret("GEMINI_BASE_URL") == "https://env-configured.invalid"
        monkeypatch.delenv("GEMINI_BASE_URL")
        with pytest.raises(SecretNotFoundError):
            resolver.get_secret("GEMINI_BASE_URL")

    def test_blank_secret_value_is_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        fake = FakeSecretsManager(json.dumps({"DEEPSEEK_API_KEY": "   "}))
        with pytest.raises(SecretNotFoundError):
            SecretsManagerSecretResolver(SECRET_ID, client=fake).get_secret("DEEPSEEK_API_KEY")

    def test_unreadable_secret_is_a_hard_error_not_an_optional_absence(self) -> None:
        fake = FakeSecretsManager(error=RuntimeError("AccessDeniedException"))
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=fake)
        with pytest.raises(SecretResolverError) as excinfo:
            resolver.get_secret("AZURE_OPENAI_API_KEY")
        assert not isinstance(excinfo.value, SecretNotFoundError)
        assert "error_type=RuntimeError" in str(excinfo.value)

    @pytest.mark.parametrize(
        "secret_string",
        ["not json azure-value-must-not-leak", json.dumps(["azure-value-must-not-leak"]),
         json.dumps({"AZURE_OPENAI_API_KEY": 123}), json.dumps({"bad-name": "azure-value-must-not-leak"})],
    )
    def test_malformed_secret_fails_without_echoing_content(self, secret_string: str) -> None:
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=FakeSecretsManager(secret_string))
        with pytest.raises(SecretResolverConfigError) as excinfo:
            resolver.get_secret("AZURE_OPENAI_API_KEY")
        assert "must-not-leak" not in str(excinfo.value)

    def test_rejects_invalid_names_like_env_resolver(self) -> None:
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=FakeSecretsManager())
        with pytest.raises(SecretResolverConfigError):
            resolver.get_secret("sk-looks-like-a-value")

    def test_empty_secret_id_is_rejected(self) -> None:
        with pytest.raises(SecretResolverConfigError):
            SecretsManagerSecretResolver("  ")

    def test_secret_values_are_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)
        resolver = SecretsManagerSecretResolver(SECRET_ID, client=FakeSecretsManager())
        for name in SECRET_VALUES:
            resolver.get_secret(name)
        assert "must-not-leak" not in caplog.text
        assert "must-not-leak" not in repr(resolver)


class TestRuntimeSelection:
    def test_unset_secret_id_keeps_local_env_resolver(self) -> None:
        assert isinstance(get_runtime_secret_resolver(), EnvSecretResolver)

    def test_secret_id_selects_one_process_wide_secrets_manager_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PROVIDER_CREDENTIALS_SECRET_ID_ENV, SECRET_ID)
        first = get_runtime_secret_resolver()
        assert isinstance(first, SecretsManagerSecretResolver)
        assert get_runtime_secret_resolver() is first


class TestSettingsTimeCredentials:
    def test_without_secret_id_matches_previous_env_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", " raw-env-value ")
        assert cfg_module._settings_time_provider_secret("TAVILY_API_KEY") == " raw-env-value "
        monkeypatch.setenv("TAVILY_API_KEY", "   ")
        assert cfg_module._settings_time_provider_secret("TAVILY_API_KEY") == "   "
        monkeypatch.delenv("TAVILY_API_KEY")
        assert cfg_module._settings_time_provider_secret("TAVILY_API_KEY") == ""

    def test_production_settings_read_image_and_web_search_keys_from_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
        _install_fake_client(monkeypatch, FakeSecretsManager())
        monkeypatch.setenv(PROVIDER_CREDENTIALS_SECRET_ID_ENV, SECRET_ID)
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        cfg_module._settings = None
        settings = cfg_module.get_settings()
        assert settings.tavily_api_key == SECRET_VALUES["TAVILY_API_KEY"]
        assert settings.image_classifier_api_key == SECRET_VALUES["GOOGLE_GEMINI_API_KEY"]

    def test_unreadable_secret_fails_settings_instead_of_disabling_features(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_client(monkeypatch, FakeSecretsManager(error=RuntimeError("denied")))
        monkeypatch.setenv(PROVIDER_CREDENTIALS_SECRET_ID_ENV, SECRET_ID)
        cfg_module._settings = None
        with pytest.raises(ConfigurationError, match="could not be read"):
            cfg_module.get_settings()


class _MissingEverything:
    def get_secret(self, name: str) -> str:
        raise SecretNotFoundError(f"Environment variable '{name}' is not set.")


class _HasEverything:
    def get_secret(self, name: str) -> str:
        return "configured"


class TestProductionStartupCredentialCheck:
    def _settings(self, monkeypatch: pytest.MonkeyPatch, app_env: str):
        monkeypatch.setenv("APP_ENV", app_env)
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        cfg_module._settings = None
        return cfg_module.get_settings()

    def test_production_fails_fast_when_active_provider_credentials_are_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from services.llm.runtime_factory import build_model_executor

        settings = self._settings(monkeypatch, "production")
        with pytest.raises(ConfigurationError, match="provider_profile="):
            build_model_executor(settings, secret_resolver=_MissingEverything())

    def test_production_starts_when_active_credentials_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.llm.runtime_factory import build_model_executor

        settings = self._settings(monkeypatch, "production")
        assert build_model_executor(settings, secret_resolver=_HasEverything()) is not None

    def test_non_production_startup_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.llm.runtime_factory import build_model_executor

        settings = self._settings(monkeypatch, "local")
        assert build_model_executor(settings, secret_resolver=_MissingEverything()) is not None

    def test_checks_only_profiles_of_active_routes(self) -> None:
        from services.llm.orchestration.config_registry import get_registry

        assert set(get_registry().active_route_provider_profiles()) == {"azure_foundry_v1", "gemini_primary"}


def test_classifier_orchestrator_uses_runtime_secret_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.query_classifier_service as classifier_module

    sentinel = _HasEverything()
    monkeypatch.setattr(sm_module, "get_runtime_secret_resolver", lambda: sentinel)
    monkeypatch.setattr(classifier_module, "_classifier_orchestrator", None)
    orchestrator = classifier_module._get_classifier_orchestrator()
    credential_resolver = orchestrator._model_executor._provider_executor._credential_resolver
    assert credential_resolver._secret_resolver is sentinel
    monkeypatch.setattr(classifier_module, "_classifier_orchestrator", None)
