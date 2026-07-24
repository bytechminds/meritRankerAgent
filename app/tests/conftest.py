"""
app/tests/conftest.py
----------------------
Pytest session configuration and autouse fixtures.

Purpose
-------
Force safe environment defaults for every unit test so tests never
accidentally attempt real provider calls when .env.local has
ENABLE_REAL_LLM=true, ENABLE_KB_RETRIEVAL=true, or ENABLE_DYNAMODB_FETCH=true.

Also resets the config._settings singleton before/after every test to prevent
cross-test contamination from the module-level singleton.

Design notes
------------
* The ``monkeypatch`` fixture used here is function-scoped, so every
  override is automatically reverted after each test.
* Tests that legitimately need non-default config (e.g. test_config_validation.py)
  call ``monkeypatch.setenv()`` + ``_reset_settings()`` themselves.  This works
  because their ``setenv()`` call overrides the value set by this autouse
  fixture, and monkeypatch restores everything at teardown.
* No test should ever attempt a real call to:
    - OpenAI / Azure OpenAI / Gemini / Bedrock model APIs
    - Bedrock Knowledge Base (ENABLE_KB_RETRIEVAL guard)
    - DynamoDB (ENABLE_DYNAMODB_FETCH guard)
    - V1 LLM orchestration path via real provider (ENABLE_ORCHESTRATED_DOUBT_SOLVER guard)
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Set safe environment defaults before any test module is imported.

    This hook runs at the very start of pytest initialization, before test
    collection and before any ``import main`` at module level in test files.
    It must run before ``config.py``'s ``load_dotenv(override=False)`` call so
    that dotenv does not overwrite the safe defaults with .env.local values.

    Tests that legitimately need a non-default value (e.g. ENABLE_REAL_LLM=true)
    use ``monkeypatch.setenv`` to override for that specific test only; the
    ``autouse`` fixture below resets ``config._settings`` before each test so
    ``get_settings()`` always re-reads the active env.
    """
    os.environ.setdefault("ENABLE_REAL_LLM", "false")
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("ENABLE_KB_RETRIEVAL", "false")
    os.environ.setdefault("ENABLE_DYNAMODB_FETCH", "false")
    os.environ.setdefault("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
    os.environ.setdefault("IMAGE_CLASSIFIER_ENABLED", "false")
    os.environ.setdefault("LLM_ROLE_CONFIG_JSON", "{}")


@pytest.fixture(autouse=True)
def _unit_test_env_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enforce safe environment defaults for every unit test.

    Sets:
        ENABLE_REAL_LLM=false              — prevents real LLM provider calls
        ENABLE_KB_RETRIEVAL=false          — prevents real Bedrock KB queries
        ENABLE_DYNAMODB_FETCH=false        — prevents real DynamoDB queries
        ENABLE_ORCHESTRATED_DOUBT_SOLVER=false  — keeps existing test_main_routing.py
                                             tests on the legacy graph path

    Resets config._settings to None so get_settings() always rebuilds
    from the patched environment, not from a stale singleton.
    """
    import config as cfg_module  # noqa: PLC0415

    monkeypatch.setenv("ENABLE_REAL_LLM", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
    monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
    monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
    monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "false")
    # Empty role map so get_llm_role_config() returns the safe mock default
    # for any role, rather than the real provider config from .env.local.
    monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")

    # Reset before the test so get_settings() re-reads the safe values above.
    cfg_module._settings = None

    yield  # test runs here

    # Reset after the test so the next test gets a clean singleton.
    cfg_module._settings = None
