"""Presentation-only post-rewrite quality recovery.

After the single bounded rewrite, an answer whose remaining quality reasons are all
presentation-only (too many display-math blocks / visible steps) continues to the
existing correctness verifier instead of failing terminally. Every other reason keeps
ANSWER_QUALITY_FAILED, the verifier stays authoritative, and nothing continues where
the verifier does not run.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
from observability.llm_usage import (
    begin_llm_usage_collection,
    record_llm_call,
    reset_llm_usage_collection,
)
from schemas.llm_routing import RouteRequest
from schemas.llm_usage import ProviderTokenUsage
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.answer_quality import (
    GENERATION_FAILURE_MESSAGE,
    PRESENTATION_ONLY_REASON_CODES,
    is_presentation_only_failure,
    validate_answer_quality,
)
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor

_QUERY = (
    "A boat goes 30 km downstream in 2 hours and returns the same distance upstream in "
    "3 hours. What is the speed of the stream in km/h?"
)
_STEPS = (
    ("Downstream speed", "30 \\div 2 = 15"),
    ("Upstream speed", "30 \\div 3 = 10"),
    ("Boat plus stream", "b + s = 15"),
    ("Boat minus stream", "b - s = 10"),
    ("Adding both", "2b = 25"),
    ("Boat speed", "b = 12.5"),
    ("Stream speed", "s = 15 - 12.5 = 2.5"),
)
DISPLAY_BLOCKS_ONLY = (
    "\n\n".join(f"{label}:\n\n\\[ {expression} \\]" for label, expression in _STEPS)
    + "\n\n**Answer:** 2.5 km/h"
)
VISIBLE_STEPS_ONLY = (
    "\n".join(
        f"{index}. {label} is {expression.replace(chr(92) + 'div', '/')}."
        for index, (label, expression) in enumerate(_STEPS + _STEPS[:2], start=1)
    )
    + "\n\n**Answer:** 2.5 km/h"
)
DISPLAY_PLUS_MALFORMED_MATH = DISPLAY_BLOCKS_ONLY + "\n\nCheck: \\( s + b = 15"
DISPLAY_PLUS_CONFLICT = "**Answer:** 3 km/h\n\n" + DISPLAY_BLOCKS_ONLY
DISPLAY_PLUS_UNSAFE_MARKUP = DISPLAY_BLOCKS_ONLY + "\n\n<script>alert(1)</script>"


def _reasons(content: str, *, language: str = "english") -> set[str]:
    return set(
        validate_answer_quality(
            content,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            query=_QUERY,
            language=language,  # type: ignore[arg-type]
        ).reason_codes
    )


def test_fixtures_carry_exactly_the_intended_reason_codes() -> None:
    assert _reasons(DISPLAY_BLOCKS_ONLY) == {"too_many_display_math_blocks"}
    assert _reasons(VISIBLE_STEPS_ONLY) == {"too_many_visible_steps"}
    assert "math_unbalanced_inline" in _reasons(DISPLAY_PLUS_MALFORMED_MATH)
    assert "conflicting_answer_values" in _reasons(DISPLAY_PLUS_CONFLICT)
    assert "raw_html_script" in _reasons(DISPLAY_PLUS_UNSAFE_MARKUP)
    assert "language_mismatch" in _reasons(DISPLAY_BLOCKS_ONLY, language="hindi")
    assert PRESENTATION_ONLY_REASON_CODES == {
        "too_many_display_math_blocks",
        "too_many_visible_steps",
    }


@pytest.mark.parametrize(
    ("content", "language", "expected"),
    [
        (DISPLAY_BLOCKS_ONLY, "english", True),
        (VISIBLE_STEPS_ONLY, "english", True),
        (DISPLAY_PLUS_MALFORMED_MATH, "english", False),
        (DISPLAY_PLUS_CONFLICT, "english", False),
        (DISPLAY_PLUS_UNSAFE_MARKUP, "english", False),
        (DISPLAY_BLOCKS_ONLY, "hindi", False),
    ],
)
def test_presentation_only_classification(content: str, language: str, expected: bool) -> None:
    result = validate_answer_quality(
        content,
        subject="math",
        difficulty="intermediate",
        intent="solve",
        query=_QUERY,
        language=language,  # type: ignore[arg-type]
    )
    assert is_presentation_only_failure(result) is expected


# ---------------------------------------------------------------------------
# Streaming verified-replay path
# ---------------------------------------------------------------------------

_INTERMEDIATE = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "intermediate",
    "need_web_search": False,
    "classifier_confidence": 0.99,
    "classification_source": "llm",
}


class _AdapterAfterRewrite:
    """Returns ``content`` as the orchestrator's post-rewrite output would be returned."""

    def __init__(self, content: str, verdict: CorrectnessVerification, *, rewrite: bool = True):
        self._content = content
        self._rewrite = rewrite
        self.generate_calls = 0
        self.correctness_verifier = _Verifier(verdict)

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        attempts = ["primary", "rewrite"] if self._rewrite else ["primary"]
        for attempt in attempts:
            record_llm_call(
                request_id=str(kwargs["request_id"]),
                role="math.generator.intermediate",
                provider="mock",
                model="mock",
                deployment=None,
                attempt_type=attempt,
                streaming=False,
                usage=ProviderTokenUsage(),
                duration_ms=1,
                status="succeeded",
            )
        return self._content


class _Verifier:
    def __init__(self, verdict: CorrectnessVerification) -> None:
        self.verdict = verdict
        self.calls = 0

    def verify(self, **_: object) -> CorrectnessVerification:
        self.calls += 1
        return self.verdict


_MATCH = CorrectnessVerification(
    status="match",
    independent_answer="2.5",
    single_defensible_answer=True,
    reason="agrees",
    method="model",
)
_MISMATCH = CorrectnessVerification(
    status="mismatch",
    independent_answer="3",
    single_defensible_answer=True,
    reason="differs",
    method="model",
)
_AMBIGUOUS = CorrectnessVerification(
    status="ambiguous",
    independent_answer="2.5",
    single_defensible_answer=False,
    reason="two readings",
    method="model",
)


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "true")
    cfg_module._settings = None
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    yield
    cfg_module._settings = None


@pytest.fixture
def logged_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    original = streaming_module.log_event

    def _record(name: str, **kwargs: object) -> None:
        events.append((name, dict(kwargs.get("details") or {})))
        original(name, **kwargs)

    monkeypatch.setattr(streaming_module, "log_event", _record)
    return events


def _stream(
    adapter: _AdapterAfterRewrite, *, language: str = "english", classification: dict | None = None
) -> list:
    classification = classification or dict(_INTERMEDIATE)
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="presentation-only",
                query=_QUERY,
                language=language,  # type: ignore[arg-type]
                classification=classification,
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


@pytest.mark.parametrize("content", [DISPLAY_BLOCKS_ONLY, VISIBLE_STEPS_ONLY])
def test_presentation_only_after_rewrite_reaches_the_verifier_and_delivers_on_match(
    content: str, logged_events: list[tuple[str, dict]]
) -> None:
    adapter = _AdapterAfterRewrite(content, _MATCH)

    events = _stream(adapter)

    assert adapter.correctness_verifier.calls == 1
    assert adapter.generate_calls == 1
    assert events[-1].type == "complete"
    delivered = "".join(event.content or "" for event in events if event.type == "chunk")
    assert delivered == content
    assert events[-1].response.answer == content
    continued = [
        details for name, details in logged_events if name == "QUALITY_PRESENTATION_ONLY_CONTINUED"
    ]
    assert len(continued) == 1
    assert set(continued[0]["reason_codes"].split(",")) <= PRESENTATION_ONLY_REASON_CODES
    assert content not in str(continued)


@pytest.mark.parametrize("verdict", [_MISMATCH, _AMBIGUOUS])
def test_presentation_only_answer_still_fails_closed_when_the_verifier_rejects(
    verdict: CorrectnessVerification,
) -> None:
    adapter = _AdapterAfterRewrite(DISPLAY_BLOCKS_ONLY, verdict)

    events = _stream(adapter)

    assert adapter.correctness_verifier.calls == 1
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"
    assert not any(event.type in {"chunk", "complete"} for event in events)


@pytest.mark.parametrize(
    ("content", "language"),
    [
        (DISPLAY_PLUS_MALFORMED_MATH, "english"),
        (DISPLAY_PLUS_CONFLICT, "english"),
        (DISPLAY_PLUS_UNSAFE_MARKUP, "english"),
        (DISPLAY_BLOCKS_ONLY, "hindi"),
    ],
)
def test_hard_quality_defects_keep_terminal_quality_failure_without_verification(
    content: str, language: str, logged_events: list[tuple[str, dict]]
) -> None:
    adapter = _AdapterAfterRewrite(content, _MATCH)

    events = _stream(adapter, language=language)

    assert adapter.correctness_verifier.calls == 0
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_QUALITY_FAILED"
    assert not any(event.type in {"chunk", "complete"} for event in events)
    assert not any(name == "QUALITY_PRESENTATION_ONLY_CONTINUED" for name, _ in logged_events)


def test_presentation_only_without_a_prior_rewrite_is_unchanged() -> None:
    adapter = _AdapterAfterRewrite(DISPLAY_BLOCKS_ONLY, _MATCH, rewrite=False)

    events = _stream(adapter)

    assert adapter.correctness_verifier.calls == 0
    assert events[-1].metadata["code"] == "ANSWER_QUALITY_FAILED"


def test_presentation_only_does_not_continue_where_no_verifier_runs() -> None:
    adapter = _AdapterAfterRewrite(DISPLAY_BLOCKS_ONLY, _MATCH)

    events = _stream(adapter, classification={**_INTERMEDIATE, "difficulty": "basic"})

    assert adapter.correctness_verifier.calls == 0
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_QUALITY_FAILED"


@pytest.mark.parametrize("difficulty", ["basic", "intermediate"])
def test_rejected_presentation_only_text_is_never_streamed_with_the_answer_verifier_disabled(
    difficulty: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "false")
    cfg_module._settings = None
    adapter = _AdapterAfterRewrite(DISPLAY_BLOCKS_ONLY, _MATCH)

    events = _stream(adapter, classification={**_INTERMEDIATE, "difficulty": difficulty})

    assert not any(event.type in {"chunk", "complete"} for event in events)
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] in {"ANSWER_QUALITY_FAILED", "ANSWER_VERIFICATION_FAILED"}


# ---------------------------------------------------------------------------
# Orchestrator: one rewrite, content retained only for presentation-only results
# ---------------------------------------------------------------------------


def _orchestrator(*contents: str) -> tuple[LlmOrchestrator, list[int]]:
    calls: list[int] = []

    class _Sequenced:
        last_stream_finish_reason = "stop"

        def execute(self, *, route_decision, messages):  # noqa: ANN001, ANN202
            calls.append(len(messages))
            content = contents[min(len(calls), len(contents)) - 1]
            return MockModelExecutor(content=content, finish_reason="stop").execute(
                route_decision=route_decision, messages=messages
            )

    return LlmOrchestrator(model_executor=_Sequenced()), calls


_GENERATOR_REQUEST = RouteRequest(
    request_id="presentation-only-orchestrator",
    subject="math",
    task_role="generator",
    difficulty="intermediate",
    intent="solve",
)


def test_orchestrator_keeps_a_presentation_only_rewrite_as_failed_quality_with_one_rewrite() -> (
    None
):
    orchestrator, calls = _orchestrator(
        DISPLAY_BLOCKS_ONLY + "\n<ANSWER_DONE>", DISPLAY_BLOCKS_ONLY + "\n<ANSWER_DONE>"
    )

    result = orchestrator.generate(route_request=_GENERATOR_REQUEST, query=_QUERY)

    assert len(calls) == 2  # draft + exactly one rewrite
    assert result.final_answer.quality_status == "failed_quality_gate"
    assert result.final_answer.content == DISPLAY_BLOCKS_ONLY


def test_orchestrator_keeps_the_draft_when_the_rewrite_fails_quality() -> None:
    """A failed reformat must not cost the student an otherwise usable draft.

    The rewrite used to replace the draft whether or not it was accepted, so a draft that
    failed only on presentation was discarded because its reformat came back malformed.
    The draft is kept instead, still `failed_quality_gate`, so only a caller that runs the
    correctness verifier may continue it.
    """
    orchestrator, calls = _orchestrator(
        DISPLAY_BLOCKS_ONLY + "\n<ANSWER_DONE>",
        DISPLAY_PLUS_MALFORMED_MATH + "\n<ANSWER_DONE>",
    )

    result = orchestrator.generate(route_request=_GENERATOR_REQUEST, query=_QUERY)

    assert len(calls) == 2
    assert result.final_answer.quality_status == "failed_quality_gate"
    assert result.final_answer.content == DISPLAY_BLOCKS_ONLY


def test_orchestrator_still_replaces_a_hard_quality_failure_after_rewrite() -> None:
    """When the draft itself is unsafe to show, the placeholder still substitutes."""
    orchestrator, calls = _orchestrator(
        DISPLAY_PLUS_MALFORMED_MATH + "\n<ANSWER_DONE>",
        DISPLAY_PLUS_MALFORMED_MATH + "\n<ANSWER_DONE>",
    )

    result = orchestrator.generate(route_request=_GENERATOR_REQUEST, query=_QUERY)

    assert len(calls) == 2
    assert result.final_answer.quality_status == "failed_quality_gate"
    assert result.final_answer.content == GENERATION_FAILURE_MESSAGE


def test_orchestrator_does_not_retain_presentation_only_content_without_a_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_QUALITY_REWRITE_ENABLED", "false")
    cfg_module._settings = None
    orchestrator, calls = _orchestrator(DISPLAY_BLOCKS_ONLY + "\n<ANSWER_DONE>")

    result = orchestrator.generate(route_request=_GENERATOR_REQUEST, query=_QUERY)

    assert len(calls) == 1
    assert result.final_answer.quality_status == "failed_quality_gate"
    assert result.final_answer.content != DISPLAY_BLOCKS_ONLY


# ---------------------------------------------------------------------------
# Non-streaming graph path follows the same rule
# ---------------------------------------------------------------------------


class _GraphAdapter:
    def __init__(self, content: str, verdict: CorrectnessVerification, *, rewrite: bool = True):
        self._content = content
        self._rewrite = rewrite
        self.correctness_verifier = _Verifier(verdict)

    def generate_final(self, **kwargs: object):  # noqa: ANN201
        for attempt in ["primary", "rewrite"] if self._rewrite else ["primary"]:
            record_llm_call(
                request_id=str(kwargs["request_id"]),
                role="math.generator.intermediate",
                provider="mock",
                model="mock",
                deployment=None,
                attempt_type=attempt,
                streaming=False,
                usage=ProviderTokenUsage(),
                duration_ms=1,
                status="succeeded",
            )
        return build_final_answer_result(
            content=self._content, language="english", quality_status="failed_quality_gate"
        )


def _graph_state(difficulty: str = "intermediate") -> dict:
    return {
        "request_id": "presentation-only-graph",
        "actor_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": _QUERY,
        "original_query": _QUERY,
        "language": "english",
        "exam_id": "CAT",
        "exam_stage": None,
        "classification": {
            **_INTERMEDIATE,
            "difficulty": difficulty,
            "topic": "boats",
            "retrieval_required": False,
        },
        "query_classification": None,
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
        "final_answer": None,
        "conversation_context": "",
        "conversation_relation": None,
        "conversation_preparation": None,
        "source_modality": "text",
    }


@pytest.mark.parametrize(
    ("verdict", "difficulty", "rewrite", "delivered", "verifier_calls"),
    [
        (_MATCH, "intermediate", True, True, 1),
        (_MISMATCH, "intermediate", True, False, 1),
        (_MATCH, "basic", True, False, 0),
        (_MATCH, "intermediate", False, False, 0),
    ],
)
def test_non_streaming_graph_applies_the_same_rule(
    verdict: CorrectnessVerification,
    difficulty: str,
    rewrite: bool,
    delivered: bool,
    verifier_calls: int,
) -> None:
    adapter = _GraphAdapter(DISPLAY_BLOCKS_ONLY, verdict, rewrite=rewrite)
    token = begin_llm_usage_collection()
    try:
        result = build_orchestrated_doubt_solver_graph(adapter).invoke(_graph_state(difficulty))
    finally:
        reset_llm_usage_collection(token)

    assert adapter.correctness_verifier.calls == verifier_calls
    if delivered:
        assert result["answer"] == DISPLAY_BLOCKS_ONLY
        assert result["final_answer"]["quality_status"] == "checked"
    else:
        assert result["answer"] != DISPLAY_BLOCKS_ONLY
        assert result["final_answer"]["quality_status"] == "failed_quality_gate"
