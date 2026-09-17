"""Focused reliability tests for the existing independent answer verifier."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import services.doubt_solver.answer_correctness as verifier_module
import services.llm.structured_output as structured_output_module
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
    # The syntax parser now lives behind the shared structured-output boundary,
    # so the seam moved with it. The guarded behaviour is unchanged: a payload
    # that is not an object must still fail closed as typed unavailable.
    monkeypatch.setattr(
        structured_output_module,
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

    # Doubt Solver keeps its own verifier route; retired o4-mini is replaced by Terra
    # with the same prompt, budget and reasoning effort.
    assert route.model == "openai_gpt_5_6_terra"
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


_DIRECTION_QUERY = (
    "A person walks 10m North, turns right and walks 15m, then turns left and walks 10m. "
    "What is the shortest distance from the starting point?"
)


def _verify_direction(content: str, candidate_answer: str):
    return AnswerCorrectnessVerifier(orchestrator=_FakeOrchestrator(content)).verify(  # type: ignore[arg-type]
        request_id="verifier-m1",
        query=_DIRECTION_QUERY,
        candidate_answer=candidate_answer,
        subject="math",
        difficulty="intermediate",
        language="english",
    )


def test_match_contradicting_its_own_independent_answer_is_not_approved() -> None:
    """Incident M1: a wrong 15 m answer was approved under status=MATCH.

    The contract defines MATCH as the result agreeing with the candidate, so a MATCH
    whose own independent answer says 25 m is self-contradictory and must not ship.
    """
    verification = _verify_direction(
        '{"status":"MATCH","independent_answer":"25 m",'
        '"single_defensible_answer":true,"reason":"checked"}',
        "**Final Answer:** 15 m",
    )

    assert verification.approved is False
    assert verification.status == "mismatch"


def test_match_agreeing_with_its_independent_answer_is_approved() -> None:
    verification = _verify_direction(
        '{"status":"MATCH","independent_answer":"25 m",'
        '"single_defensible_answer":true,"reason":"checked"}',
        "**Final Answer:** 25 m",
    )

    assert verification.approved is True
    assert verification.status == "match"


@pytest.mark.parametrize("candidate", ["25", "25 m", "25 meters"])
def test_supported_answer_forms_still_match_the_independent_answer(candidate: str) -> None:
    """Existing normalization forms must keep matching; the guard adds no strictness."""
    verification = _verify_direction(
        '{"status":"MATCH","independent_answer":"25 m",'
        '"single_defensible_answer":true,"reason":"checked"}',
        f"**Final Answer:** {candidate}",
    )

    assert verification.approved is True


@pytest.mark.parametrize(
    ("independent", "candidate"),
    (
        ("Option A", "**Final Answer:** Option B"),
        ("3/4", "**Final Answer:** 0.75"),
        ("3:4", "**Final Answer:** 6:8"),
        ("x = 2y + 1", "**Final Answer:** x = 2y + 1"),
        ("25 m", "The distance is 25 m."),
    ),
)
def test_guard_stays_silent_when_either_side_is_not_a_single_number(
    independent: str,
    candidate: str,
) -> None:
    """No numeric opinion on options, fractions, ratios, symbolic or unlabelled answers.

    Downgrading these would convert model judgement into false mismatches, so the
    guard defers and the model's status stands.
    """
    verification = _verify_direction(
        f'{{"status":"MATCH","independent_answer":"{independent}",'
        '"single_defensible_answer":true,"reason":"checked"}',
        candidate,
    )

    assert verification.approved is True
    assert verification.status == "match"


def test_percentage_answers_compare_on_their_numeric_value() -> None:
    verification = _verify_direction(
        '{"status":"MATCH","independent_answer":"15%",'
        '"single_defensible_answer":true,"reason":"checked"}',
        "**Final Answer:** 15%",
    )

    assert verification.approved is True


# ---------------------------------------------------------------------------
# Shared structured-output boundary adoption
# ---------------------------------------------------------------------------


def _verify_generic(content: str, candidate_answer: str, query: str = "Resolve the item."):
    return AnswerCorrectnessVerifier(orchestrator=_FakeOrchestrator(content)).verify(  # type: ignore[arg-type]
        request_id="verifier-boundary",
        query=query,
        candidate_answer=candidate_answer,
        subject="math",
        difficulty="intermediate",
        language="english",
    )


def test_numeric_independent_answer_completes_normal_verification() -> None:
    """Regression for the proven incident: a numeric verdict must not be discarded.

    One fixture only. The behaviour it proves is representation handling, not
    anything about counting digits, so the architecture is exercised by the
    non-numeric cases below in exactly the same way.
    """
    verification = _verify_generic(
        '{"status":"MATCH","independent_answer":55,'
        '"single_defensible_answer":true,"reason":"independently derived"}',
        "**Answer:** 55",
    )

    assert verification.status == "match"
    assert verification.approved is True
    assert verification.independent_answer == "55"
    assert verification.method == "model"


@pytest.mark.parametrize(
    ("independent", "candidate"),
    [
        ("Option B", "**Answer:** Option B"),
        ("noun", "**Answer:** noun"),
        ("50 km/h", "**Answer:** 50 km/h"),
        ("Article 21", "**Answer:** Article 21"),
        ("photosynthesis", "**Answer:** photosynthesis"),
        ("1991", "**Answer:** 1991"),
    ],
)
def test_non_numeric_answers_pass_through_the_same_architecture(
    independent: str, candidate: str
) -> None:
    verification = _verify_generic(
        f'{{"status":"MATCH","independent_answer":"{independent}",'
        '"single_defensible_answer":true,"reason":"independently derived"}',
        candidate,
    )

    assert verification.status == "match"
    assert verification.approved is True
    assert verification.independent_answer == independent


@pytest.mark.parametrize(("raw", "expected"), [("12.5", "12.5"), ("55.0", "55"), ("-4", "-4")])
def test_decimal_and_signed_numeric_verdicts_canonicalize(raw: str, expected: str) -> None:
    verification = _verify_generic(
        f'{{"status":"MATCH","independent_answer":{raw},'
        '"single_defensible_answer":true,"reason":"independently derived"}',
        f"**Answer:** {expected}",
    )

    assert verification.approved is True
    assert verification.independent_answer == expected


def test_self_consistency_guard_still_rejects_a_numeric_false_match() -> None:
    """The guard caught a real o4-mini false positive; normalization must not blunt it."""
    verification = _verify_generic(
        '{"status":"MATCH","independent_answer":50,'
        '"single_defensible_answer":true,"reason":"independently derived"}',
        "**Answer:** 45 km/h",
    )

    assert verification.status == "mismatch"
    assert verification.approved is False
    assert verification.reason == "verifier_self_contradiction"


@pytest.mark.parametrize(
    "raw",
    ["true", "false", "null", "[]", "{}", '["55"]'],
)
def test_non_scalar_independent_answer_is_still_rejected(raw: str) -> None:
    verification = _verify_generic(
        f'{{"status":"MATCH","independent_answer":{raw},'
        '"single_defensible_answer":true,"reason":"r"}',
        "**Answer:** 55",
    )

    assert verification.status == "unavailable"
    assert verification.approved is False
    assert verification.reason == "ANSWER_VERIFICATION_UNAVAILABLE"


@pytest.mark.parametrize(
    ("content", "error", "expected_kind", "expected_stage"),
    [
        (_VALID_OUTPUT, RuntimeError("down"), "provider_failure", None),
        ("not json", None, "structured_output_parse_failure", "parse"),
        (
            '{"status":"NOPE","independent_answer":"A",'
            '"single_defensible_answer":true,"reason":"r"}',
            None,
            "structured_output_schema_failure",
            "schema",
        ),
    ],
)
def test_failure_kind_is_recorded_without_changing_the_client_reason_code(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    error: Exception | None,
    expected_kind: str,
    expected_stage: str | None,
) -> None:
    captured: dict = {}

    def _capture(_name: str, **kwargs: object) -> None:
        captured.update(kwargs.get("details") or {})  # type: ignore[arg-type]

    monkeypatch.setattr(verifier_module, "log_event", _capture)
    verification = AnswerCorrectnessVerifier(
        orchestrator=_FakeOrchestrator(content=content, error=error)  # type: ignore[arg-type]
    ).verify(
        request_id="verifier-taxonomy",
        query="Resolve the item.",
        candidate_answer="**Answer:** 55",
        subject="math",
        difficulty="intermediate",
        language="english",
    )

    # External contract unchanged for every internal cause.
    assert verification.status == "unavailable"
    assert verification.reason == "ANSWER_VERIFICATION_UNAVAILABLE"
    # Internally the cause is now distinguishable.
    assert captured["failureKind"] == expected_kind
    if expected_stage is not None:
        assert captured["validationStage"] == expected_stage
        assert captured["schemaName"] == "_VerifierOutput"


# ---------------------------------------------------------------------------
# Candidate extraction reads the declared answer, never an unrelated number
# ---------------------------------------------------------------------------

_BASE_QUERY = "The number 2006! is written in base 22. How many zeroes are there at the end?"
_REMAINDER_QUERY = (
    "The numbers 400, 536 and 645, when divided by a number N, give the remainders "
    "of 22, 23 and 24 respectively. Find the greatest such number N."
)


@pytest.mark.parametrize(
    ("answer_line", "query", "expected"),
    [
        ("**Answer:** 199 zeroes at the end when written in base 22.", _BASE_QUERY, 199.0),
        (
            "**Answer:** 27 is the greatest number that leaves remainders 22, 23 and 24.",
            _REMAINDER_QUERY,
            27.0,
        ),
        ("**Answer:** The remainder when 22004 is divided by 7 is 3.",
         "What is the remainder when 22004 is divided by 7?", 3.0),
        ("**Answer:** 50 km/h", "", 50.0),
        # an answer expression declares the result after its last "="
        (r"**Answer:** 30% of 300 = \(\frac{30}{100} \times 300 = 90\)",
         "What is 30% of 300?", 90.0),
        ("**Answer:** 12.5", "", 12.5),
        ("**Answer:** -4", "", -4.0),
        ("**Answer:** 15%", "", 15.0),
        (r"**Answer:** \(1,250\)", "", 1250.0),
        # a single stated number is declared even when the question also contains it
        ("**Answer:** The greatest such number is 24.", _REMAINDER_QUERY, 24.0),
    ],
)
def test_candidate_number_is_the_declared_answer(
    answer_line: str, query: str, expected: float
) -> None:
    assert verifier_module._candidate_number(answer_line, query=query) == expected


@pytest.mark.parametrize(
    ("answer_line", "query"),
    [
        ("**Answer:** photosynthesis", ""),
        ("**Answer:** Option C", ""),
        # two candidate numbers and no question to set either aside
        ("**Answer:** 199 zeroes at the end when written in base 22.", ""),
        # every number left over is still ambiguous
        (r"**Answer:** \(\frac{3}{8}\)", ""),
        ("**Answer:** 1 (since 2^2004 mod 7 cycles)", "What is 22004 divided by 7?"),
        ("No answer heading here: 42", ""),
    ],
)
def test_candidate_number_is_none_when_not_provable(answer_line: str, query: str) -> None:
    assert verifier_module._candidate_number(answer_line, query=query) is None


def _guard(candidate: str, independent: str, query: str):
    return AnswerCorrectnessVerifier(
        orchestrator=_FakeOrchestrator(
            f'{{"status":"MATCH","independent_answer":"{independent}",'
            '"single_defensible_answer":true,"reason":"checked"}'
        )  # type: ignore[arg-type]
    ).verify(
        request_id="verifier-extraction",
        query=query,
        candidate_answer=candidate,
        subject="math",
        difficulty="advanced",
        language="english",
    )


@pytest.mark.parametrize(
    ("candidate", "independent", "query"),
    [
        ("**Answer:** 199 zeroes at the end when written in base 22.", "199", _BASE_QUERY),
        (
            "**Answer:** 27 is the greatest number that leaves remainders 22, 23 and 24.",
            "27",
            _REMAINDER_QUERY,
        ),
    ],
)
def test_correct_answer_is_no_longer_rejected_for_a_trailing_context_number(
    candidate: str, independent: str, query: str
) -> None:
    verification = _guard(candidate, independent, query)

    assert verification.status == "match"
    assert verification.approved is True


@pytest.mark.parametrize(
    ("candidate", "independent", "query"),
    [
        # captured live: the generator stated 90; the verifier matched against 199
        (
            "**Answer:** 2006! has **90 zeroes** at the end when written in base 22.",
            "199",
            _BASE_QUERY,
        ),
        # a wrong single number that also appears in the question must still be caught
        ("**Answer:** The greatest such number is 24.", "27", _REMAINDER_QUERY),
        ("**Answer:** 89", "27", _REMAINDER_QUERY),
    ],
)
def test_self_consistency_guard_still_rejects_wrong_candidates(
    candidate: str, independent: str, query: str
) -> None:
    verification = _guard(candidate, independent, query)

    assert verification.status == "mismatch"
    assert verification.reason == "verifier_self_contradiction"
    assert verification.approved is False
