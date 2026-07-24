"""
tests/test_main_orchestrated_entrypoint.py
-------------------------------------------
Phase A — Orchestrated Entrypoint Verification.

Proves that app.main.invoke routes through the orchestrated doubt solver graph
when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true, and through the legacy graph when
the flag is false.

Why subprocess-based?
    main.py builds graphs at module-import time by reading os.environ.  The
    conftest autouse fixture always sets ENABLE_ORCHESTRATED_DOUBT_SOLVER=false
    before tests run.  The only reliable way to test the flag=true branch is to
    run a *fresh* Python process where the env var is set before any import of
    main.py.

Rules:
    - No importlib.reload() used anywhere.
    - No monkeypatching of settings *after* app.main import.
    - Each test sets ENABLE_REAL_LLM=false so no real provider call is made.
    - Each test sets ENABLE_KB_RETRIEVAL=false and ENABLE_DYNAMODB_FETCH=false
      so no AWS call is made.
    - The orchestrated mock answer is identified by the "[orchestrated-mock]"
      content prefix injected by MockModelExecutor in main.py.
    - The legacy mock answer is identified by "[Mock]" content injected by
      answer_generator_service._mock_answer.

[NOT VERIFIED] AgentCore HTTP runtime end-to-end (POST /invocations) is not
               tested here — only the Python invoke() boundary is exercised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parents[1]  # app/

# Content marker set in main.py's MockModelExecutor when ENABLE_REAL_LLM=false.
ORCHESTRATED_MOCK_MARKER = "[orchestrated-mock]"

# Keys present in the legacy DoubtSolverResponse that are absent in the
# orchestrated response dict (which only has: success, request_id, mode,
# answer, classification).
_LEGACY_ONLY_KEYS = {"needs_review", "answer_source", "is_truncated", "used_retrieval"}

# Keys that must NEVER appear in any response (secrets / internals).
_FORBIDDEN_RESPONSE_KEYS = {
    "prompt",
    "messages",
    "context_text",
    "api_key",
    "secret",
    "credential",
    "authorization",
    "raw_response",
    "system_prompt",
    "user_prompt",
}

_DOUBT_SOLVER_PAYLOAD = {
    "mode": "doubt_solver",
    "query": "What is 20% of 500?",
    "user_id": "local-user",
    "conversation_id": "conversation-entrypoint",
    "turn_id": "turn-entrypoint",
}


# Sentinel prefix used to locate the result JSON in stdout, which also
# contains Rich log output (RichHandler defaults to stdout).
_RESULT_SENTINEL = "INVOKE_RESULT:"


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def _run_invoke_subprocess(
    env_overrides: dict[str, str],
    payload: dict,
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run a subprocess that imports main.invoke and calls it with *payload*.

    The subprocess:
        1. Prepends APP_DIR to sys.path so app/ modules are importable.
        2. Imports main.invoke — this triggers module-level graph construction.
        3. Calls invoke(payload) and writes the result with a sentinel prefix.

    Environment:
        Starts from the current process env (contains PATH, PYTHONPATH, venv
        paths, etc.), overrides unsafe/test-specific vars with safe defaults,
        then applies *env_overrides* on top.

    Note on stdout:
        Rich's RichHandler writes log output to stdout by default.  To
        distinguish the result JSON from log lines we print a unique sentinel
        prefix on the result line.
    """
    # Build a clean base env: start from current process, apply safe defaults,
    # then let caller overrides take precedence.
    base_env: dict[str, str] = {
        k: v for k, v in os.environ.items()
    }
    safe_defaults: dict[str, str] = {
        "ENABLE_REAL_LLM": "false",
        "ENABLE_KB_RETRIEVAL": "false",
        "ENABLE_DYNAMODB_FETCH": "false",
        "LLM_ROLE_CONFIG_JSON": "{}",
        # Ensure the subprocess can import app/ modules.
        "PYTHONPATH": str(APP_DIR),
    }
    env = {**base_env, **safe_defaults, **env_overrides}

    payload_json = json.dumps(payload)
    script = (
        f"import sys, json\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        f"from main import invoke\n"
        f"result = invoke(json.loads({payload_json!r}))\n"
        # Use sentinel so the result line is distinguishable from Rich log output.
        f"print({_RESULT_SENTINEL!r} + json.dumps(result))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    """Extract the sentinel line from stdout and parse it as JSON.

    stdout contains both Rich log lines and the sentinel result line.
    We find the line prefixed with _RESULT_SENTINEL.
    """
    assert proc.returncode == 0, (
        f"Subprocess failed (exit {proc.returncode}):\n"
        f"STDOUT: {proc.stdout}\n"
        f"STDERR: {proc.stderr}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL):])
    raise AssertionError(
        f"Result sentinel {_RESULT_SENTINEL!r} not found in stdout.\n"
        f"STDOUT: {proc.stdout}\n"
        f"STDERR: {proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Tests — Phase A
# ---------------------------------------------------------------------------


class TestFlagFalseUsesLegacyPath:
    """ENABLE_ORCHESTRATED_DOUBT_SOLVER=false must use the legacy 7-node graph."""

    def test_flag_false_returns_success(self) -> None:
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "false"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert data["success"] is True

    def test_flag_false_response_has_legacy_keys(self) -> None:
        """Legacy path returns full DoubtSolverResponse including needs_review."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "false"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        # Legacy DoubtSolverResponse always includes needs_review.
        assert "needs_review" in data, (
            "Legacy path must return a full DoubtSolverResponse with needs_review"
        )

    def test_flag_false_answer_not_orchestrated(self) -> None:
        """Legacy answer must NOT contain the orchestrated mock marker."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "false"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert ORCHESTRATED_MOCK_MARKER not in data.get("answer", ""), (
            "Legacy path must not produce an orchestrated mock answer"
        )


class TestFlagTrueUsesOrchestratedPath:
    """ENABLE_ORCHESTRATED_DOUBT_SOLVER=true must use the orchestrated 3-node graph."""

    def test_flag_true_returns_success(self) -> None:
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert data["success"] is True

    def test_flag_true_uses_orchestrated_graph_answer(self) -> None:
        """Orchestrated path answer must contain the '[orchestrated-mock]' marker.

        This marker is injected by MockModelExecutor in main.py when
        ENABLE_REAL_LLM=false, proving the orchestrated code path was taken.
        """
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert ORCHESTRATED_MOCK_MARKER in data.get("answer", ""), (
            f"Expected orchestrated mock marker in answer.  Got: {data.get('answer')!r}"
        )

    def test_flag_true_response_lacks_legacy_fields(self) -> None:
        """Orchestrated response must not include legacy DoubtSolverResponse fields."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        present_legacy = _LEGACY_ONLY_KEYS.intersection(data.keys())
        assert not present_legacy, (
            f"Orchestrated response must not have legacy fields: {present_legacy}"
        )

    def test_main_invoke_returns_answer(self) -> None:
        """invoke() must return a non-empty answer string."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert data.get("answer"), "Answer must be a non-empty string"

    def test_request_id_preserved(self) -> None:
        """The response must include a non-empty request_id UUID string."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert data.get("request_id"), "request_id must be present and non-empty"
        # UUIDs have 4 hyphens — just verify it looks like one.
        assert data["request_id"].count("-") == 4, (
            f"request_id does not look like a UUID: {data['request_id']!r}"
        )

    def test_response_contains_no_sensitive_fields(self) -> None:
        """Response must not expose prompt, messages, context_text, or credentials."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        leaked = _FORBIDDEN_RESPONSE_KEYS.intersection(data.keys())
        assert not leaked, (
            f"Response must not contain sensitive fields: {leaked}\n"
            f"Full response keys: {list(data.keys())}"
        )

    def test_no_real_provider_call(self) -> None:
        """ENABLE_REAL_LLM=false must prevent any real provider call.

        Verified by confirming the subprocess exits cleanly and returns the
        mock answer (not an API error or network error).
        """
        proc = _run_invoke_subprocess(
            {
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_REAL_LLM": "false",
            },
            _DOUBT_SOLVER_PAYLOAD,
        )
        assert proc.returncode == 0, (
            f"Subprocess crashed — likely a real provider call failed:\n"
            f"STDERR: {proc.stderr}"
        )
        data = _parse_result(proc)
        assert data["success"] is True
        # Mock marker present means MockModelExecutor was used, not real provider.
        assert ORCHESTRATED_MOCK_MARKER in data.get("answer", "")

    def test_no_aws_call(self) -> None:
        """ENABLE_KB_RETRIEVAL=false + ENABLE_DYNAMODB_FETCH=false must prevent AWS calls."""
        proc = _run_invoke_subprocess(
            {
                "ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true",
                "ENABLE_KB_RETRIEVAL": "false",
                "ENABLE_DYNAMODB_FETCH": "false",
            },
            _DOUBT_SOLVER_PAYLOAD,
        )
        assert proc.returncode == 0, (
            f"Subprocess crashed — possibly an unexpected AWS call:\n"
            f"STDERR: {proc.stderr}"
        )
        data = _parse_result(proc)
        assert data["success"] is True

    def test_subprocess_exits_cleanly(self) -> None:
        """Subprocess must exit with code 0 (no unhandled exception)."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        assert proc.returncode == 0, (
            f"Subprocess exited with code {proc.returncode}.\n"
            f"STDERR:\n{proc.stderr}"
        )

    def test_mode_field_in_response(self) -> None:
        """Response must carry mode='doubt_solver'."""
        proc = _run_invoke_subprocess(
            {"ENABLE_ORCHESTRATED_DOUBT_SOLVER": "true"},
            _DOUBT_SOLVER_PAYLOAD,
        )
        data = _parse_result(proc)
        assert data.get("mode") == "doubt_solver"


class TestStaticGuards:
    """Static code inspections that do not require subprocess execution."""

    def test_no_importlib_reload_in_main(self) -> None:
        """main.py must not use importlib.reload() — forbidden by task spec."""
        main_src = (APP_DIR / "main.py").read_text()
        assert "importlib.reload" not in main_src, (
            "main.py must not use importlib.reload()"
        )

    def test_flag_env_var_is_loaded_before_graph_build(self) -> None:
        """main.py must read ENABLE_ORCHESTRATED_DOUBT_SOLVER before building graphs.

        Verified by checking that settings is assigned at module level before the
        orchestrated graph construction block.
        """
        main_src = (APP_DIR / "main.py").read_text()
        # settings = get_settings() must appear before the orchestrated graph block.
        settings_pos = main_src.find("settings = get_settings()")
        orchestrated_pos = main_src.find("enable_orchestrated_doubt_solver")
        assert settings_pos != -1, "main.py must call get_settings() at module level"
        assert orchestrated_pos != -1, (
            "main.py must check enable_orchestrated_doubt_solver"
        )
        assert settings_pos < orchestrated_pos, (
            "get_settings() must be called BEFORE checking enable_orchestrated_doubt_solver"
        )

    def test_env_var_override_wins_over_env_local(self) -> None:
        """config.py must load .env.local with override=False so real env vars win.

        This guarantees subprocess tests set via os.environ are not shadowed by
        any local .env.local file.
        """
        config_src = (APP_DIR / "config.py").read_text()
        assert "override=False" in config_src, (
            "config.py must use load_dotenv(override=False) so real env vars always win"
        )
