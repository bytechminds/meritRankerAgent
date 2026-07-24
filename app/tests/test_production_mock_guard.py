"""
tests/test_production_mock_guard.py
------------------------------------
Part 8.2.1 — Production Mock-Mode Safety Guard.

Verifies that main.py raises a ConfigurationError at module-import time when
ENABLE_ORCHESTRATED_DOUBT_SOLVER=true AND ENABLE_REAL_LLM=false AND
APP_ENV=production — preventing fake mock answers from ever reaching production
users silently.

Guard logic (in main.py):
    - APP_ENV=production + ENABLE_ORCHESTRATED_DOUBT_SOLVER=true + ENABLE_REAL_LLM=false
      → ConfigurationError (fail fast at startup)
    - Same as above BUT ENABLE_ORCHESTRATED_MOCK_LLM=true
      → allowed (explicit internal-testing override)
    - Any non-production APP_ENV with ENABLE_REAL_LLM=false
      → allowed (mock is safe for local/dev/test)
    - ENABLE_REAL_LLM=true in any APP_ENV
      → not checked (real provider chain — guard doesn't apply)
    - ENABLE_ORCHESTRATED_DOUBT_SOLVER=false (default)
      → guard not reached (legacy path)

Why subprocess-based?
    main.py builds graphs at module-import time.  The ConfigurationError is
    raised during that construction, not inside invoke().  Subprocess isolation
    is required so we can set APP_ENV before main.py is imported.
    importlib.reload() is explicitly forbidden by task spec.

[NOT VERIFIED] AgentCore HTTP runtime end-to-end.
[NOT VERIFIED] Real provider streaming.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]  # app/

# Error message fragments that must appear in the ConfigurationError text.
_GUARD_MSG_FRAGMENTS = {
    "Unsafe configuration",
    "ENABLE_REAL_LLM=true",
}

# Sentinel for successful imports.
_IMPORT_OK_SENTINEL = "IMPORT_OK:"


def _run_import_subprocess(
    env_overrides: dict[str, str],
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run a subprocess that only imports main (no invoke call).

    This is the right isolation strategy for testing startup-time guards, since
    the ConfigurationError is raised during module-level construction, not
    inside invoke().

    The subprocess prints IMPORT_OK:<json> on success so we can distinguish
    clean imports from non-zero exits caused by the guard.
    """
    base_env: dict[str, str] = {k: v for k, v in os.environ.items()}
    safe_defaults: dict[str, str] = {
        "ENABLE_REAL_LLM": "false",
        "ENABLE_KB_RETRIEVAL": "false",
        "ENABLE_DYNAMODB_FETCH": "false",
        "LLM_ROLE_CONFIG_JSON": "{}",
        "PYTHONPATH": str(APP_DIR),
    }
    env = {**base_env, **safe_defaults, **env_overrides}

    script = textwrap.dedent(f"""\
        import sys, json
        sys.path.insert(0, {str(APP_DIR)!r})
        import services.conversation.bootstrap as conversation_bootstrap
        conversation_bootstrap.build_conversation_persistence_service = lambda: None
        import main  # triggers module-level graph construction
        print({_IMPORT_OK_SENTINEL!r} + json.dumps({{"app_env": main.settings.app_env}}))
    """)

    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _import_succeeded(proc: subprocess.CompletedProcess) -> bool:
    """Return True if the subprocess printed the import-OK sentinel."""
    return any(
        line.startswith(_IMPORT_OK_SENTINEL) for line in proc.stdout.splitlines()
    )


def _guard_error_in_stderr(proc: subprocess.CompletedProcess) -> bool:
    """Return True if the subprocess stderr contains ConfigurationError text."""
    return "ConfigurationError" in proc.stderr


# ---------------------------------------------------------------------------
# Tests — guard fires in production
# ---------------------------------------------------------------------------


class TestProductionGuardFires:
    """In APP_ENV=production, orchestrated + mock must be rejected at startup."""

    def test_production_mock_raises_config_error(self) -> None:
        """The subprocess must exit non-zero with ConfigurationError in stderr."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert proc.returncode != 0, (
            "Expected non-zero exit (ConfigurationError) but subprocess succeeded.\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    def test_production_mock_error_is_config_error_type(self) -> None:
        """The raised exception must be ConfigurationError (not a generic error)."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert _guard_error_in_stderr(proc), (
            "Expected 'ConfigurationError' in stderr.\n"
            f"STDERR: {proc.stderr}"
        )

    def test_production_mock_error_message_mentions_fix(self) -> None:
        """Error message must tell the operator how to fix the problem."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        for fragment in _GUARD_MSG_FRAGMENTS:
            assert fragment in proc.stderr, (
                f"Expected fragment {fragment!r} in stderr.\n"
                f"STDERR: {proc.stderr}"
            )

    def test_production_mock_does_not_invoke_model(self) -> None:
        """The guard must fire before any model or AWS call is attempted."""
        # If the guard fires (returncode != 0) and the error is ConfigurationError,
        # it means we never reached the MockModelExecutor — proved by the guard test above.
        # This test additionally checks that no provider/AWS error appears before
        # the guard message.
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert _guard_error_in_stderr(proc), (
            "ConfigurationError not found — guard may not have fired before model call."
        )
        # No other error classes should appear before the guard fires.
        for unexpected in ("LlmExecutionError", "LlmProviderError", "BotocoreError"):
            assert unexpected not in proc.stderr, (
                f"Unexpected error {unexpected!r} appeared — guard fired too late."
            )


# ---------------------------------------------------------------------------
# Tests — guard passes (allowed cases)
# ---------------------------------------------------------------------------


class TestGuardAllowedCases:
    """Cases where the guard must NOT raise."""

    def test_production_real_llm_true_with_real_deployments_succeeds(self) -> None:
        """APP_ENV=production + ENABLE_REAL_LLM=true succeeds when deployments are real.

        model_registry.yaml now has real deployment names (gpt-4o, gpt-4o-mini).
        The preflight guard must NOT fire.  The service must start successfully.

        NOTE: Previously this test asserted returncode != 0 because model_registry.yaml
        had YOUR_* placeholder names.  Now that real names are configured, startup
        must succeed.  If YOUR_* placeholder names are re-introduced, startup will
        again fail with a ConfigurationError and this test should be reverted.
        """
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "true",
            }
        )
        # Startup must SUCCEED — real deployment names are configured.
        assert proc.returncode == 0, (
            "Expected startup success with real deployment names, but it failed.\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc), (
            f"IMPORT_OK not found in output.\nSTDOUT: {proc.stdout}"
        )

    def test_local_env_mock_allowed(self) -> None:
        """APP_ENV=local (default) + ENABLE_REAL_LLM=false must be allowed."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "local",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert proc.returncode == 0, (
            f"Mock executor must be allowed in APP_ENV=local.\n"
            f"STDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc)

    def test_dev_env_mock_allowed(self) -> None:
        """APP_ENV=dev + ENABLE_REAL_LLM=false must be allowed."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "dev",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert proc.returncode == 0, (
            f"Mock executor must be allowed in APP_ENV=dev.\nSTDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc)

    def test_test_env_mock_allowed(self) -> None:
        """APP_ENV=test + ENABLE_REAL_LLM=false must be allowed."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "test",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert proc.returncode == 0, (
            f"Mock executor must be allowed in APP_ENV=test.\nSTDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc)

    def test_escape_hatch_overrides_production_guard(self) -> None:
        """ENABLE_ORCHESTRATED_MOCK_LLM=true must override the production guard.

        This escape hatch is for controlled internal testing only.
        It must never be set in normal production deployments.
        """
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
                "ENABLE_ORCHESTRATED_MOCK_LLM": "true",
            }
        )
        assert proc.returncode == 0, (
            "ENABLE_ORCHESTRATED_MOCK_LLM=true must override the production guard.\n"
            f"STDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc), (
            f"Import sentinel not found.\nSTDOUT: {proc.stdout}"
        )


# ---------------------------------------------------------------------------
# Tests — legacy path unaffected
# ---------------------------------------------------------------------------


class TestLegacyPathUnaffected:
    """The guard must not interfere with the legacy graph path."""

    def test_legacy_path_production_env_allowed(self) -> None:
        """APP_ENV=production + ENABLE_ORCHESTRATED_DOUBT_SOLVER=false (default) must work."""
        proc = _run_import_subprocess(
            {
                "APP_ENV": "production",
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "false",
                "ENABLE_REAL_LLM": "false",
            }
        )
        assert proc.returncode == 0, (
            "Legacy path must not be blocked by the production mock guard.\n"
            f"STDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc)

    def test_default_settings_unaffected(self) -> None:
        """Default config (ENABLE_ORCHESTRATED_DOUBT_SOLVER=false) must always succeed."""
        proc = _run_import_subprocess(
            {
                # Explicit defaults — guard is not reached
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "false",
            }
        )
        assert proc.returncode == 0, (
            f"Default config caused unexpected startup failure.\nSTDERR: {proc.stderr}"
        )
        assert _import_succeeded(proc)


# ---------------------------------------------------------------------------
# Tests — config schema and static guards
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Settings schema must expose the new fields; ConfigurationError must be importable."""

    def test_config_error_is_importable(self) -> None:
        """ConfigurationError must be importable from config."""
        from config import ConfigurationError  # noqa: PLC0415 (deferred for test isolation)

        assert issubclass(ConfigurationError, Exception)

    def test_config_has_enable_orchestrated_mock_llm_field(self) -> None:
        """Settings dataclass must have the enable_orchestrated_mock_llm field."""
        from config import Settings

        assert "enable_orchestrated_mock_llm" in Settings.__dataclass_fields__, (
            "Settings must have enable_orchestrated_mock_llm field"
        )

    def test_enable_orchestrated_mock_llm_defaults_false(self) -> None:
        """ENABLE_ORCHESTRATED_MOCK_LLM must default to False (safe-by-default)."""
        import os

        # Ensure the env var is absent
        env_backup = os.environ.pop("ENABLE_ORCHESTRATED_MOCK_LLM", None)
        try:
            import config as cfg

            cfg._settings = None  # reset singleton for this test
            s = cfg.get_settings()
            assert s.enable_orchestrated_mock_llm is False, (
                "enable_orchestrated_mock_llm must default to False"
            )
        finally:
            if env_backup is not None:
                os.environ["ENABLE_ORCHESTRATED_MOCK_LLM"] = env_backup
            # Restore settings singleton so other tests are unaffected
            import config as cfg2
            cfg2._settings = None

    def test_guard_not_triggered_when_orchestrated_is_false(self) -> None:
        """Guard must only check when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true."""
        main_src = (APP_DIR / "main.py").read_text()
        # The guard must be inside the enable_orchestrated_doubt_solver block.
        orchestrated_pos = main_src.find("if settings.enable_orchestrated_doubt_solver")
        guard_pos = main_src.find("_is_production")
        assert orchestrated_pos != -1
        assert guard_pos != -1
        assert guard_pos > orchestrated_pos, (
            "Production guard must be nested inside the enable_orchestrated_doubt_solver block"
        )

    def test_guard_is_inside_real_llm_false_branch(self) -> None:
        """Guard must only check when ENABLE_REAL_LLM=false (mock path)."""
        main_src = (APP_DIR / "main.py").read_text()
        real_llm_false_pos = main_src.find("if not settings.enable_real_llm")
        guard_pos = main_src.find("_is_production")
        assert real_llm_false_pos != -1
        assert guard_pos != -1
        assert guard_pos > real_llm_false_pos, (
            "Production guard must be nested inside the 'if not enable_real_llm' block"
        )
