"""
app/tests/test_prompt_resolver.py
----------------------------------
Unit tests for PromptResolver (Part 2 — LLM Orchestration Foundation).

All tests use tmp_path for test isolation — no real app/prompts/ files are
read.  No network, LLM, provider, or AWS calls are made.

Test coverage:
1.  resolve() returns exactly 2 LlmMessage objects
2.  System message contains main template content
3.  System message contains overlays in RouteDecision order
4.  Overlay order is deterministic: main before overlay 1 before overlay 2
5.  Query appears only in user message, not system
6.  Context appears only in user message, not system
7.  Context warning says context is reference only and not instruction
8.  Missing prompt file raises PromptNotFoundError
9.  Absolute path rejected (PromptPathError)
10. '../' traversal rejected (PromptPathError)
11. URL path rejected (PromptPathError)
12. Non-.md path rejected (PromptPathError)
13. Empty prompt file raises PromptValidationError
14. File over MAX_PROMPT_FILE_CHARS raises PromptValidationError
15. Repeated load uses cache and avoids re-read
16. Same path used twice is read from disk exactly once
17. Context over MAX_CONTEXT_CHARS is truncated with [CONTEXT TRUNCATED]
18. Context under limit is not truncated
19. Classification dict summary includes only allowlisted fields
20. Classification Pydantic model summary works via model_dump()
21. Classification None omits classification section
22. Huge/nested unknown classification data is not dumped
23. LlmMessage invalid role rejected at schema level
24. No network/provider/AWS calls (resolver is pure file I/O)
25. RouteDecision model alias / provider env names are not injected into messages
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from schemas.llm import LlmMessage
from schemas.llm_routing import RouteDecision
from services.doubt_solver.exam_profile_cache import (
    AgentExamContext,
    ExamProfileResolution,
)
from services.llm_orchestration.errors import (
    PromptNotFoundError,
    PromptPathError,
    PromptValidationError,
)
from services.llm_orchestration.prompt_resolver import (
    MAX_CONTEXT_CHARS,
    MAX_PROMPT_FILE_CHARS,
    PromptResolver,
    reset_prompt_resolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_route(
    prompt: str = "main.md",
    overlays: list[str] | None = None,
    model: str = "gemini_flash_light",
    intent: str | None = None,
    exam: str | None = None,
) -> RouteDecision:
    """Construct a minimal RouteDecision for testing."""
    return RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        intent=intent,
        exam=exam,
        model=model,
        prompt=prompt,
        overlays=overlays or [],
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )


def _write(tmp_path: Path, rel_path: str, content: str) -> None:
    """Write a file under tmp_path, creating parent directories as needed."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


class _ExamProfileRuntime:
    def resolve(self, **_: object) -> ExamProfileResolution:
        return ExamProfileResolution(
            context=AgentExamContext(
                exam_profile_id="CAT#2026",
                exam_id="CAT",
                exam_name="CAT 2026",
                stage="2026",
                description="Admin-managed context.",
                sections=(
                    {
                        "sectionId": "QA",
                        "subject": "MATH",
                        "level": "ADVANCED",
                        "questionStyles": ("mcq",),
                        "excludeTopics": (),
                    },
                ),
                totals=None,
            ),
            source="dynamodb_cache",
            reason=None,
            cache_hit=True,
            duration_us=1,
        )


class _LegacyExamProfileResolver:
    def resolve(self, *_: object, **__: object) -> object:
        raise AssertionError("legacy resolver must not run after a cache hit")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the module singleton before every test."""
    reset_prompt_resolver()


# ---------------------------------------------------------------------------
# Test 1 — resolve() returns exactly 2 LlmMessage objects
# ---------------------------------------------------------------------------


def test_resolve_returns_two_messages(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main prompt\nContent here.")
    resolver = PromptResolver(prompt_root=tmp_path)
    messages = resolver.resolve(_make_route("main.md"), query="What is 2+2?")
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"


# ---------------------------------------------------------------------------
# Test 2 — system message contains main template content
# ---------------------------------------------------------------------------


def test_system_message_contains_main_content(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main\nMy unique main content ABC.")
    resolver = PromptResolver(prompt_root=tmp_path)
    msgs = resolver.resolve(_make_route("main.md"), query="q")
    assert "My unique main content ABC." in msgs[0].content


def test_cached_exam_profile_precedes_the_legacy_exam_resolver(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    route = _make_route("main.md", exam="CAT").model_copy(
        update={"exam_profile_id": "CAT#2026", "exam_stage": "2026"}
    )
    resolver = PromptResolver(
        prompt_root=tmp_path,
        exam_profile_runtime=_ExamProfileRuntime(),
        exam_profile_resolver=_LegacyExamProfileResolver(),  # type: ignore[arg-type]
    )

    messages = resolver.resolve(route, query="Solve this", request_id="request-1")

    assert '"examProfileId":"CAT#2026"' in messages[0].content
    assert "Admin-managed context." in messages[0].content


def test_resolve_structured_uses_safe_cached_prompts_without_answer_contract(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "structured.md", "STRUCTURED_ROLE")
    _write(tmp_path, "shared.md", "SHARED_CONTRACT")
    _write(tmp_path, "generator_answer_contract.md", "ANSWER_CONTRACT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route("structured.md", overlays=["shared.md"])

    messages = resolver.resolve_structured(route, '{"count":1}')

    assert "STRUCTURED_ROLE" in messages[0].content
    assert "SHARED_CONTRACT" in messages[0].content
    assert "ANSWER_CONTRACT" not in messages[0].content
    assert messages[1].content == '{"count":1}'


# ---------------------------------------------------------------------------
# Test 3 — system message contains overlays in RouteDecision order
# ---------------------------------------------------------------------------


def test_system_message_contains_overlays_in_order(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    _write(tmp_path, "overlay1.md", "# Overlay ONE")
    _write(tmp_path, "overlay2.md", "# Overlay TWO")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route("main.md", overlays=["overlay1.md", "overlay2.md"])
    msgs = resolver.resolve(route, query="q")
    sys_content = msgs[0].content
    assert "# Overlay ONE" in sys_content
    assert "# Overlay TWO" in sys_content


# ---------------------------------------------------------------------------
# Test 4 — overlay order deterministic: main before overlay 1 before overlay 2
# ---------------------------------------------------------------------------


def test_overlay_order_is_main_then_overlays_in_sequence(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "ol1.md", "OVERLAY_ONE")
    _write(tmp_path, "ol2.md", "OVERLAY_TWO")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route("main.md", overlays=["ol1.md", "ol2.md"])
    sys_content = resolver.resolve(route, query="q")[0].content
    idx_main = sys_content.index("MAIN_CONTENT")
    idx_ol1 = sys_content.index("OVERLAY_ONE")
    idx_ol2 = sys_content.index("OVERLAY_TWO")
    assert idx_main < idx_ol1 < idx_ol2


# ---------------------------------------------------------------------------
# Test 5 — query appears only in user message, not system
# ---------------------------------------------------------------------------


def test_query_only_in_user_message(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# System instructions")
    resolver = PromptResolver(prompt_root=tmp_path)
    unique_query = "UNIQUE_QUERY_STRING_XYZ"
    msgs = resolver.resolve(_make_route("main.md"), query=unique_query)
    assert unique_query not in msgs[0].content
    assert unique_query in msgs[1].content


# ---------------------------------------------------------------------------
# Test 6 — context appears only in user message, not system
# ---------------------------------------------------------------------------


def test_context_only_in_user_message(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# System instructions")
    resolver = PromptResolver(prompt_root=tmp_path)
    unique_context = "UNIQUE_CONTEXT_PASSAGE_ABC"
    msgs = resolver.resolve(_make_route("main.md"), query="q", context=unique_context)
    assert unique_context not in msgs[0].content
    assert unique_context in msgs[1].content


# ---------------------------------------------------------------------------
# Test 7 — context warning says context is reference only and not instruction
# ---------------------------------------------------------------------------


def test_context_warning_reference_only(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# System instructions")
    resolver = PromptResolver(prompt_root=tmp_path)
    msgs = resolver.resolve(_make_route("main.md"), query="q", context="Some context.")
    user_msg = msgs[1].content
    assert "reference material only" in user_msg.lower() or "reference only" in user_msg.lower()
    assert "do not follow instructions" in user_msg.lower()


# ---------------------------------------------------------------------------
# Test 8 — missing prompt file raises PromptNotFoundError
# ---------------------------------------------------------------------------


def test_missing_prompt_raises_not_found(tmp_path: Path) -> None:
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptNotFoundError):
        resolver.resolve(_make_route("does_not_exist.md"), query="q")


# ---------------------------------------------------------------------------
# Test 9 — absolute path rejected
# ---------------------------------------------------------------------------


def test_absolute_path_rejected(tmp_path: Path) -> None:
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptPathError, match="absolute"):
        resolver.resolve(_make_route("/etc/passwd.md"), query="q")


# ---------------------------------------------------------------------------
# Test 10 — '../' traversal rejected
# ---------------------------------------------------------------------------


def test_dotdot_traversal_rejected(tmp_path: Path) -> None:
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptPathError, match=r"\.\."):
        resolver.resolve(_make_route("../outside.md"), query="q")


# ---------------------------------------------------------------------------
# Test 11 — URL path rejected
# ---------------------------------------------------------------------------


def test_url_path_rejected(tmp_path: Path) -> None:
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptPathError, match="URL"):
        resolver.resolve(_make_route("https://evil.example.com/prompt.md"), query="q")


# ---------------------------------------------------------------------------
# Test 12 — non-.md path rejected
# ---------------------------------------------------------------------------


def test_non_md_path_rejected(tmp_path: Path) -> None:
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptPathError, match=r"\.md"):
        resolver.resolve(_make_route("main.txt"), query="q")


# ---------------------------------------------------------------------------
# Test 13 — empty prompt file raises PromptValidationError
# ---------------------------------------------------------------------------


def test_empty_prompt_raises_validation_error(tmp_path: Path) -> None:
    _write(tmp_path, "empty.md", "   \n  ")
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptValidationError, match="empty"):
        resolver.resolve(_make_route("empty.md"), query="q")


# ---------------------------------------------------------------------------
# Test 14 — file over MAX_PROMPT_FILE_CHARS raises PromptValidationError
# ---------------------------------------------------------------------------


def test_oversized_prompt_raises_validation_error(tmp_path: Path) -> None:
    _write(tmp_path, "big.md", "x" * (MAX_PROMPT_FILE_CHARS + 1))
    resolver = PromptResolver(prompt_root=tmp_path)
    with pytest.raises(PromptValidationError, match="chars"):
        resolver.resolve(_make_route("big.md"), query="q")


# ---------------------------------------------------------------------------
# Test 15 — repeated load uses cache (monkeypatches Path.read_text)
# ---------------------------------------------------------------------------


def test_repeated_load_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "cached.md", "# Cached content")
    resolver = PromptResolver(prompt_root=tmp_path)

    read_count = 0
    original_read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal read_count
        read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    # Load the same file twice via _load_prompt
    resolver._load_prompt("cached.md")
    resolver._load_prompt("cached.md")

    assert read_count == 1, f"Expected 1 disk read but got {read_count}"


# ---------------------------------------------------------------------------
# Test 16 — same path used twice in resolve() reads disk exactly once
# ---------------------------------------------------------------------------


def test_same_path_in_overlays_reads_disk_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "shared.md", "# Shared overlay")
    resolver = PromptResolver(prompt_root=tmp_path)

    read_count = 0
    original_read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal read_count
        read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    route = _make_route("shared.md", overlays=["shared.md"])
    resolver.resolve(route, query="q")

    # main prompt + overlay — both point to shared.md; second call must use cache
    assert read_count == 1, f"Expected 1 disk read but got {read_count}"


# ---------------------------------------------------------------------------
# Test 17 — context over MAX_CONTEXT_CHARS is truncated with [CONTEXT TRUNCATED]
# ---------------------------------------------------------------------------


def test_context_over_limit_is_truncated(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    long_context = "A" * (MAX_CONTEXT_CHARS + 500)
    msgs = resolver.resolve(_make_route("main.md"), query="q", context=long_context)
    user_msg = msgs[1].content
    assert "[CONTEXT TRUNCATED]" in user_msg
    # The context in the message should not exceed limit + marker overhead by much
    assert "A" * MAX_CONTEXT_CHARS in user_msg
    assert "A" * (MAX_CONTEXT_CHARS + 1) not in user_msg


# ---------------------------------------------------------------------------
# Test 18 — context under limit is not truncated
# ---------------------------------------------------------------------------


def test_context_under_limit_not_truncated(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    short_context = "B" * (MAX_CONTEXT_CHARS - 1)
    msgs = resolver.resolve(_make_route("main.md"), query="q", context=short_context)
    user_msg = msgs[1].content
    assert "[CONTEXT TRUNCATED]" not in user_msg
    assert short_context in user_msg


# ---------------------------------------------------------------------------
# Test 19 — classification dict summary includes only allowlisted fields
# ---------------------------------------------------------------------------


def test_classification_dict_allowlisted_fields_only(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    classification = {
        "subject": "math",
        "intent": "solve",
        "difficulty": "basic",
        "secret_key": "should_not_appear",
        "provider_endpoint": "https://bad.example.com",
    }
    msgs = resolver.resolve(_make_route("main.md"), query="q", classification=classification)
    user_msg = msgs[1].content
    assert "math" in user_msg
    assert "solve" in user_msg
    assert "should_not_appear" not in user_msg
    assert "https://bad.example.com" not in user_msg


# ---------------------------------------------------------------------------
# Test 20 — classification Pydantic model works via model_dump()
# ---------------------------------------------------------------------------


class _SampleClassification(BaseModel):
    subject: str = "reasoning"
    intent: str = "explain"
    difficulty: str = "intermediate"
    confidence: float = 0.9


def test_classification_pydantic_model(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    cls_model = _SampleClassification()
    msgs = resolver.resolve(_make_route("main.md"), query="q", classification=cls_model)
    user_msg = msgs[1].content
    assert "reasoning" in user_msg
    assert "explain" in user_msg
    assert "intermediate" in user_msg


# ---------------------------------------------------------------------------
# Test 21 — classification None omits classification section
# ---------------------------------------------------------------------------


def test_classification_none_omits_section(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    msgs = resolver.resolve(_make_route("main.md"), query="q", classification=None)
    user_msg = msgs[1].content
    assert "Classification:" not in user_msg


# ---------------------------------------------------------------------------
# Test 22 — huge/nested unknown classification data is not dumped
# ---------------------------------------------------------------------------


def test_huge_nested_classification_not_dumped(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "# Main")
    resolver = PromptResolver(prompt_root=tmp_path)
    nested_cls = {
        "subject": "math",
        "deeply_nested": {"a": {"b": {"c": "should_not_appear_in_message"}}},
        "huge_list": ["item"] * 1000,
        "unknown_field": "also_should_not_appear",
    }
    msgs = resolver.resolve(_make_route("main.md"), query="q", classification=nested_cls)
    user_msg = msgs[1].content
    assert "should_not_appear_in_message" not in user_msg
    assert "also_should_not_appear" not in user_msg


# ---------------------------------------------------------------------------
# Test 23 — LlmMessage invalid role rejected at schema level
# ---------------------------------------------------------------------------


def test_llm_message_invalid_role_rejected() -> None:
    with pytest.raises((ValidationError, ValueError)):
        LlmMessage(role="invalid_role", content="some content")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 24 — no network/provider/AWS calls
# ---------------------------------------------------------------------------


def test_no_network_or_provider_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm resolver does not import or call boto3, httpx, requests, etc."""
    _write(tmp_path, "main.md", "# Main content")
    resolver = PromptResolver(prompt_root=tmp_path)

    # Patch socket.socket to raise if any network call is attempted
    import socket

    original_socket = socket.socket

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network call attempted inside PromptResolver")

    monkeypatch.setattr(socket, "socket", no_network)

    # This must succeed without any network I/O
    msgs = resolver.resolve(_make_route("main.md"), query="What is 2+2?")
    assert len(msgs) == 2

    monkeypatch.setattr(socket, "socket", original_socket)


# ---------------------------------------------------------------------------
# Test 25 — RouteDecision model alias / provider env names not in messages
# ---------------------------------------------------------------------------


def test_route_credentials_not_injected(tmp_path: Path) -> None:
    """Provider model alias and route internals must not leak into prompt messages."""
    _write(tmp_path, "main.md", "# Main content")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route("main.md", model="gemini_flash_reasoning_light")
    msgs = resolver.resolve(route, query="q")
    # Model alias is internal metadata — it must not appear in system or user message
    assert "gemini_flash_reasoning_light" not in msgs[0].content
    assert "gemini_flash_reasoning_light" not in msgs[1].content


# ---------------------------------------------------------------------------
# Helpers for intent overlay tests
# ---------------------------------------------------------------------------


def _make_route_with_intent_overlays(
    prompt: str = "main.md",
    overlays: list[str] | None = None,
    intent: str | None = None,
    intent_overlays: dict[str, list[str]] | None = None,
) -> RouteDecision:
    """Construct a RouteDecision with explicit intent_overlays for testing."""
    return RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        intent=intent,
        model="math_basic_generator",
        prompt=prompt,
        overlays=overlays or [],
        intent_overlays=intent_overlays or {},
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )


# ---------------------------------------------------------------------------
# Test 26 — solve intent appends solve overlay
# ---------------------------------------------------------------------------


def test_solve_intent_appends_solve_overlay(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "intents/solve.md", "SOLVE_OVERLAY_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="solve",
        intent_overlays={"solve": ["intents/solve.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    assert "MAIN_CONTENT" in sys_content
    assert "SOLVE_OVERLAY_CONTENT" in sys_content


# ---------------------------------------------------------------------------
# Test 27 — explain intent appends explain overlay
# ---------------------------------------------------------------------------


def test_explain_intent_appends_explain_overlay(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "intents/explain.md", "EXPLAIN_OVERLAY_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="explain",
        intent_overlays={"explain": ["intents/explain.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    assert "EXPLAIN_OVERLAY_CONTENT" in sys_content


# ---------------------------------------------------------------------------
# Test 28 — practice intent appends practice overlay
# ---------------------------------------------------------------------------


def test_practice_intent_appends_practice_overlay(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "intents/practice.md", "PRACTICE_OVERLAY_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="practice",
        intent_overlays={"practice": ["intents/practice.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    assert "PRACTICE_OVERLAY_CONTENT" in sys_content


# ---------------------------------------------------------------------------
# Test 29 — visualize intent appends visualize overlay
# ---------------------------------------------------------------------------


def test_visualize_intent_appends_visualize_overlay(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "intents/visualize.md", "VISUALIZE_OVERLAY_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="visualize",
        intent_overlays={"visualize": ["intents/visualize.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    assert "VISUALIZE_OVERLAY_CONTENT" in sys_content


# ---------------------------------------------------------------------------
# Test 30 — overlay order: route overlays first, intent overlays second
# ---------------------------------------------------------------------------


def test_overlay_order_route_overlays_before_intent_overlays(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN")
    _write(tmp_path, "level.md", "LEVEL_OVERLAY")
    _write(tmp_path, "intents/solve.md", "INTENT_OVERLAY")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        overlays=["level.md"],
        intent="solve",
        intent_overlays={"solve": ["intents/solve.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    idx_level = sys_content.index("LEVEL_OVERLAY")
    idx_intent = sys_content.index("INTENT_OVERLAY")
    assert idx_level < idx_intent, "Route overlays must come before intent overlays"


# ---------------------------------------------------------------------------
# Test 31 — unconfigured intent does not fail (intent not in intent_overlays)
# ---------------------------------------------------------------------------


def test_unconfigured_intent_does_not_fail(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="solve",  # intent set but intent_overlays is empty
        intent_overlays={},
    )
    msgs = resolver.resolve(route, query="q")
    assert len(msgs) == 2
    assert "MAIN_CONTENT" in msgs[0].content


# ---------------------------------------------------------------------------
# Test 32 — no intent set → intent overlays not applied
# ---------------------------------------------------------------------------


def test_no_intent_skips_intent_overlays(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN_CONTENT")
    _write(tmp_path, "intents/solve.md", "SOLVE_OVERLAY_CONTENT")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent=None,  # no intent
        intent_overlays={"solve": ["intents/solve.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    assert "SOLVE_OVERLAY_CONTENT" not in sys_content


# ---------------------------------------------------------------------------
# Test 33 — duplicate overlay paths not added twice
# ---------------------------------------------------------------------------


def test_duplicate_overlay_not_added_twice(tmp_path: Path) -> None:
    _write(tmp_path, "main.md", "MAIN")
    _write(tmp_path, "intents/solve.md", "SOLVE_UNIQUE_MARKER")
    resolver = PromptResolver(prompt_root=tmp_path)
    # Route overlays already include the same file that intent_overlays would add.
    route = _make_route_with_intent_overlays(
        "main.md",
        overlays=["intents/solve.md"],  # already in route overlays
        intent="solve",
        intent_overlays={"solve": ["intents/solve.md"]},
    )
    sys_content = resolver.resolve(route, query="q")[0].content
    # The marker should appear exactly once
    assert sys_content.count("SOLVE_UNIQUE_MARKER") == 1


# ---------------------------------------------------------------------------
# Test 34 — no prompt content in logs (security)
# ---------------------------------------------------------------------------


def test_no_prompt_content_in_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret_prompt_content = "TOP_SECRET_PROMPT_CONTENT_XYZ"
    _write(tmp_path, "main.md", secret_prompt_content)
    _write(tmp_path, "intents/solve.md", "OVERLAY_CONTENT_ABC")
    resolver = PromptResolver(prompt_root=tmp_path)
    route = _make_route_with_intent_overlays(
        "main.md",
        intent="solve",
        intent_overlays={"solve": ["intents/solve.md"]},
    )
    import logging

    with caplog.at_level(logging.DEBUG):
        resolver.resolve(route, query="What is 2+2?")

    for record in caplog.records:
        assert secret_prompt_content not in record.getMessage()
        assert "OVERLAY_CONTENT_ABC" not in record.getMessage()
        assert "What is 2+2?" not in record.getMessage()
