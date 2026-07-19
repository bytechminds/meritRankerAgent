"""
app/services/prompt_loader.py
------------------------------
Safe cached prompt loader for app/prompts/ Markdown files.

Design:
  - An explicit allowlist (_ALLOWED_PROMPTS) is the only defence against
    path traversal.  The loader rejects any name not in the set.
  - functools.lru_cache caches the file content per prompt_name so that
    repeated calls within the same process do not touch the file system.
  - PromptLoadError is raised for unknown names or missing files so callers
    can degrade gracefully (fallback to deterministic path).

Public API:
    load_prompt(prompt_name: str) -> str
    PromptLoadError
"""

from __future__ import annotations

import functools
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Allowlist of bare prompt names (no extension, no path components).
# Add a new name here when a new prompt file is created in app/prompts/.
_ALLOWED_PROMPTS: frozenset[str] = frozenset(
    {
        "query_classifier",
        "answer_generator",
        "image_question_classifier",
    }
)


class PromptLoadError(Exception):
    """Raised when a prompt file cannot be loaded."""


@functools.cache
def load_prompt(prompt_name: str) -> str:
    """Return the text content of a named prompt file.

    Caches the result per ``prompt_name`` after the first successful read so
    that repeated calls within the same process lifetime do not hit the file
    system.

    Note: Uses ``functools.cache`` (equivalent to ``lru_cache(maxsize=None)``).
    Clear with ``load_prompt.cache_clear()`` in tests.

    Args:
        prompt_name: Bare file name without extension (e.g. ``"query_classifier"``).
                     Must appear in the ``_ALLOWED_PROMPTS`` allowlist.

    Returns:
        Prompt file content as a str.

    Raises:
        PromptLoadError: If ``prompt_name`` is not in the allowlist or the file
                         is missing from ``app/prompts/``.
    """
    if prompt_name not in _ALLOWED_PROMPTS:
        raise PromptLoadError(
            f"Unknown prompt name {prompt_name!r}. "
            f"Allowed names: {sorted(_ALLOWED_PROMPTS)}"
        )
    path = _PROMPTS_DIR / f"{prompt_name}.md"
    if not path.exists():
        raise PromptLoadError(f"Prompt file not found: {path.name}")
    return path.read_text(encoding="utf-8")
