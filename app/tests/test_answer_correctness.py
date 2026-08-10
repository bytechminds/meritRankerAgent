"""Focused reliability tests for the existing independent answer verifier."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import services.doubt_solver.answer_correctness as verifier_module
from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_correctness import AnswerCorrectnessVerifier
from services.llm.orchestration.route_resolver import resolve_route

_VALID_OUTPUT = (
    '{"status":"MATCH","independent_answer":"Option A",'
    '"single_defensible_answer":true,"reason":"The independently derived option agrees."}'
)


class _FakeOrchestrator:
    def __init__(self, content: str = _VALID_OUTPUT, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    def generate(self, **_: object) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


def _verify(orchestrator: _FakeOrchestrator):
    return AnswerCorrectnessVerifier(orchestrator=orchestrator).verify(  # type: ignore[arg-type]
        request_id="verifier-test",
        query="Solve the quadratic and select the correct option.",
        candidate_answer="**Final Answer:** Option A",
        subject="math",
        difficulty="intermediate",
        language="english",
    )


def test_valid_structured_verifier_response_is_approved() -> None:
    result = _verify(_FakeOrchestrator())

    assert result.approved is True
    assert result.status == "match"
    assert result.method == "model"


def test_valid_response_does_not_require_usage_metadata() -> None:
    result = _verify(_FakeOrchestrator())

    assert result.approved is True


def test_lowercase_status_is_normalized() -> None:
    result = _verify(
        _FakeOrchestrator(
            _VALID_OUTPUT.replace('"MATCH"', '"match"'),
        )
    )

    assert result.status == "match"


@pytest.mark.parametrize("content", ["", " ", "not json"])
def test_empty_or_malformed_response_becomes_typed_unavailable(content: str) -> None:
    result = _verify(_FakeOrchestrator(content=content))

    assert result.status == "unavailable"
    assert result.approved is False
    assert result.reason == "ANSWER_VERIFICATION_UNAVAILABLE"


def test_none_parsed_output_becomes_typed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verifier_module,
        "parse_classifier_json_strict",
        lambda _content: (None, ""),
    )

    result = _verify(_FakeOrchestrator())

    assert result.status == "unavailable"
    assert result.approved is False


@pytest.mark.parametrize(
    "content",
    [
        '{"status":"MATCH","independent_answer":"A","reason":"missing bool"}',
        '{"status":"APPROVED","independent_answer":"A",'
        '"single_defensible_answer":true,"reason":"bad enum"}',
        '{"status":"MATCH","independent_answer":"A",'
        '"single_defensible_answer":true,"reason":"ok","extra":"forbidden"}',
    ],
)
def test_invalid_schema_never_approves(content: str) -> None:
    result = _verify(_FakeOrchestrator(content=content))

    assert result.status == "unavailable"
    assert result.approved is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("provider refusal"),
        TimeoutError("provider timeout"),
        TypeError("provider parser error"),
    ],
)
def test_provider_errors_are_normalized_without_raw_exception(error: Exception) -> None:
    result = _verify(_FakeOrchestrator(error=error))

    assert result.status == "unavailable"
    assert result.approved is False
    assert result.reason == "ANSWER_VERIFICATION_UNAVAILABLE"


def test_unavailable_logging_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("DEBUG"):
        result = _verify(_FakeOrchestrator(error=RuntimeError("usage unavailable")))

    assert result.status == "unavailable"
    assert any(
        "answer_correctness_verification unavailable" in message
        for message in caplog.messages
    )


def test_verifier_route_has_reasoning_and_json_output_budget() -> None:
    route = resolve_route(
        RouteRequest(
            request_id="verifier-budget",
            subject="general",
            task_role="verifier",
            difficulty="default",
            intent="solve",
            language="english",
        )
    )

    assert route.model == "openai_o4_mini"
    assert route.max_tokens == 5000
    assert route.provider_options == {"reasoning_effort": "medium"}
    assert route.fallback_attempts == []


def test_verifier_prompt_requires_boolean_single_defensible_answer() -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "answer_correctness_verifier.md"
    ).read_text(encoding="utf-8")

    assert "`single_defensible_answer` must be the JSON boolean" in prompt
