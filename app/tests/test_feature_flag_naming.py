"""
tests/test_feature_flag_naming.py
----------------------------------
Naming-guard tests: verify production code under app/ contains no versioned
(V1/V2) symbols or obsolete env-var names.

Covers:
    - No class/function/variable name contains V1, v1, V2, or v2.
    - No ENABLE_LLM_ORCHESTRATION_V2 env-var reference.
    - No enable_llm_orchestration_v2 attribute reference.
    - No DoubtSolverV1State, V1QueryClassification, build_doubt_solver_v1_graph.
    - Allowed exception: this guard file itself and skills/docs (not scanned).
    - Allowed exception: test files may mention old names only in docstrings/comments.

[NOT VERIFIED]: the scan does not parse AST; it uses substring matching.
A symbol inside a string literal would be a false positive — reviewed manually.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parents[1]  # app/

# Files that are allowed to contain the banned tokens:
#   - This file itself (contains the token list as string literals).
#   - skills/ (documentation, not production code).
#   - .venv/ (third-party packages).
_EXCLUDE_DIRS = {".venv", "__pycache__", "skills", "docs"}
_EXCLUDE_FILES = {Path(__file__).resolve()}

_BANNED_TOKENS: list[str] = [
    "V1QueryClassification",
    "DoubtSolverV1State",
    "build_doubt_solver_v1_graph",
    "ENABLE_LLM_ORCHESTRATION_V2",
    "enable_llm_orchestration_v2",
    "_map_to_v1_classification",
    "_v1_classify_node",
    "_v1_collect_context_node",
]

# Pattern for bare V1/v1/V2/v2 as identifiers (word boundaries).
# Matches: V1State, v1_classify, DoubtSolverV1, V2Flag, etc.
# Does NOT match: revision1, level2, etc. (no V/v prefix before the digit).
_VERSIONED_IDENT_RE = re.compile(r"\b(?:[A-Z_a-z][A-Za-z_]*(?:V1|v1|V2|v2)[A-Za-z_]*)\b")

# Allowed versioned tokens: external names / Azure API path conventions that
# happen to contain V1/v1 but are NOT versioned app identifiers.
# azure_openai_v1: Azure OpenAI /openai/v1 compatible API mode name.
# _V1_SUFFIX / _OPENAI_V1_SUFFIX: constant holding the "/openai/v1" path string.
_ALLOWED_VERSIONED_TOKENS: frozenset[str] = frozenset({
    "azure_openai_v1",
    "_V1_SUFFIX",
    "AzureApiMode",         # type alias, not a versioned state
    "_normalize_v1_base_url",  # Azure /openai/v1 path normalization helper
    "_build_client_v1",     # Azure v1-mode client builder
    "_validate_and_build_v1",  # Azure v1-mode credential validator
    # Structured-output schema name sent to the provider on the wire. It versions
    # a wire contract, not an app identifier, so it moves only when that contract
    # genuinely changes shape.
    "practice_request_intelligence_v4",
    # Real prompt filename (prompts/practice_generation/question_verifier_v2.md) that
    # names the schema-v2 Practice verifier wire contract, not a versioned app state.
    "question_verifier_v2",
})

# Env-var / attribute pattern: must not appear in Python source.
_ENVVAR_RE = re.compile(r"ENABLE_LLM_ORCHESTRATION_V2|enable_llm_orchestration_v2")


def _collect_python_files() -> list[Path]:
    """Yield all .py files under app/ excluding venv and pycache."""
    files: list[Path] = []
    for path in _APP_DIR.rglob("*.py"):
        if any(excl in path.parts for excl in _EXCLUDE_DIRS):
            continue
        if path.resolve() in _EXCLUDE_FILES:
            continue
        files.append(path)
    return files


def _is_comment_or_docstring_line(line: str) -> bool:
    """Rough heuristic: line is a pure comment or string-only line."""
    stripped = line.strip()
    return (
        stripped.startswith("#")
        or stripped.startswith('"""')
        or stripped.startswith("'''")
        or stripped.startswith('"')
        or stripped.startswith("'")
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_banned_production_symbols() -> None:
    """No .py file under app/ (except this guard) may contain banned V1/V2 tokens."""
    violations: list[str] = []

    for path in _collect_python_files():
        rel = path.relative_to(_APP_DIR)
        lines = path.read_text().splitlines()
        for lineno, line in enumerate(lines, start=1):
            for token in _BANNED_TOKENS:
                if token in line:
                    violations.append(f"  {rel}:{lineno}: {token!r}  →  {line.rstrip()}")

    assert not violations, (
        "Banned versioned tokens found in production code:\n" + "\n".join(violations)
    )


def test_no_versioned_env_var_references() -> None:
    """No .py file under app/ may reference ENABLE_LLM_ORCHESTRATION_V2."""
    violations: list[str] = []

    for path in _collect_python_files():
        rel = path.relative_to(_APP_DIR)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _ENVVAR_RE.search(line):
                violations.append(f"  {rel}:{lineno}: {line.rstrip()}")

    assert not violations, (
        "Obsolete env-var name found in production code:\n" + "\n".join(violations)
    )


def test_no_v1_v2_in_production_identifiers() -> None:
    """No production .py file under app/services/ or app/graphs/ or app/schemas/
    or app/main.py or app/config.py may define a symbol with V1/v1/V2/v2 in its name.
    Test files are excluded (they may reference old names in comments only).
    """
    prod_dirs = [
        _APP_DIR / "services",
        _APP_DIR / "graphs",
        _APP_DIR / "schemas",
    ]
    prod_files = [
        _APP_DIR / "main.py",
        _APP_DIR / "config.py",
        _APP_DIR / "logging_config.py",
    ]

    violations: list[str] = []

    def _scan(path: Path) -> None:
        rel = path.relative_to(_APP_DIR)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Skip pure comments / docstring-only lines.
            if _is_comment_or_docstring_line(line):
                continue
            for m in _VERSIONED_IDENT_RE.finditer(line):
                token = m.group(0)
                if token in _ALLOWED_VERSIONED_TOKENS:
                    continue
                violations.append(f"  {rel}:{lineno}: {token!r}  →  {line.rstrip()}")

    for d in prod_dirs:
        for path in d.rglob("*.py"):
            if any(excl in path.parts for excl in _EXCLUDE_DIRS):
                continue
            _scan(path)

    for path in prod_files:
        if path.exists():
            _scan(path)

    assert not violations, (
        "Versioned identifier (V1/v1/V2/v2) found in production code:\n"
        + "\n".join(violations)
    )
