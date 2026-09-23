"""End-to-end delivery-path tests without provider or network calls."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from schemas.doubt_solver import CanonicalLanguage
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)

_REQUEST_ID = "adaptive-stream-001"
_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "basic",
    "need_web_search": False,
    "classifier_confidence": 0.99,
    "classification_source": "llm",
}
_VALID_ANSWER = (
    "**Final Answer:**\n\\(20\\)\n\nUsing percentage = part per hundred, "
    "\\(20\\% \\times 100 = 20\\)."
)


class _FakeAdapter:
    def __init__(
        self,
        *,
        stream_chunks: list[str] | None = None,
        generated_answers: list[str] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.stream_chunks = stream_chunks or []
        self.generated_answers = list(generated_answers or [_VALID_ANSWER])
        self.stream_error = stream_error
        self.stream_calls = 0
        self.generate_calls = 0
        self.last_stream_kwargs: dict[str, object] = {}
        self.last_generate_kwargs: dict[str, object] = {}

    def generate_stream(self, **kwargs: object) -> Iterator[str]:
        self.stream_calls += 1
        self.last_stream_kwargs = kwargs
        yield from self.stream_chunks
        if self.stream_error is not None:
            raise self.stream_error

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        self.last_generate_kwargs = kwargs
        if not self.generated_answers:
            raise AssertionError("unexpected generation")
        return self.generated_answers.pop(0)


class _FakeCorrectnessVerifier:
    def __init__(self, result: CorrectnessVerification) -> None:
        self.result = result
        self.calls = 0

    def verify(self, **_: object) -> CorrectnessVerification:
        self.calls += 1
        return self.result


class _NoPersistenceExpected:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected persistence call: {name}")


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("ANSWER_REPLAY_MAX_CHUNK_CHARS", "20")
    cfg_module._settings = None
    yield
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _fresh_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )


def _events(
    adapter: _FakeAdapter,
    *,
    classification: dict | None = None,
    should_cancel=None,
    language: CanonicalLanguage = "english",
) -> list:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                query="What is 20 percent of 100?",
                language=language,
                classification=classification or dict(_CLASSIFICATION),
                classifier_confidence=(classification or _CLASSIFICATION).get(
                    "classifier_confidence"
                ),
                classifier_fallback=(classification or _CLASSIFICATION).get(
                    "classification_source"
                )
                == "fallback",
                should_cancel=should_cancel,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


def test_low_risk_request_streams_live_and_final_response_is_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "adaptive")
    adapter = _FakeAdapter(
        stream_chunks=[
            "**Final ",
            "Answer:**\n\\(20\\)\n\nUsing percentage = part per hundred, ",
            "\\(20\\% \\times 100 = 20\\).",
        ]
    )

    events = _events(adapter)

    assert adapter.stream_calls == 1
    assert adapter.generate_calls == 0
    chunks = "".join(event.content or "" for event in events if event.type == "chunk")
    complete = events[-1]
    assert chunks == _VALID_ANSWER
    assert complete.type == "complete"
    assert complete.response is not None
    assert complete.response.schema_version == "1"
    assert complete.response.content.format == "markdown"
    assert complete.response.content.value == chunks
    assert complete.response.final_answer is not None
    assert complete.response.final_answer.content == chunks
    assert "final_answer" not in complete.response.model_dump()
    assert complete.response.answer == chunks
    assert "provider" not in str(complete.metadata).lower()


def test_language_reaches_live_stream_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "adaptive")
    adapter = _FakeAdapter(generated_answers=["**Answer:** Sahi option B hai."])

    events = _events(adapter, language="hinglish")

    assert adapter.stream_calls == 0
    assert adapter.last_generate_kwargs["language"] == "hinglish"
    assert events[-1].type == "complete"


def test_language_reaches_verified_replay_generation() -> None:
    adapter = _FakeAdapter(
        generated_answers=["**Final Answer:**\nसही विकल्प B है क्योंकि वेग स्थिर है।"]
    )
    classification = dict(_CLASSIFICATION)
    classification["classifier_confidence"] = 0.2
    classification["classification_source"] = "fallback"

    events = _events(adapter, language="hindi", classification=classification)

    assert adapter.last_generate_kwargs["language"] == "hindi"
    assert events[-1].response is not None
    assert events[-1].response.final_answer is not None
    assert events[-1].response.final_answer.language_compliant is True


def test_language_mismatch_is_rewritten_once_before_hindi_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "adaptive")
    adapter = _FakeAdapter(
        generated_answers=[
            "**Answer:** The correct option is B because velocity is constant.",
            "**Answer:** सही विकल्प B है क्योंकि वेग स्थिर है।",
        ]
    )

    events = _events(adapter, language="hindi")

    rendered = "".join(event.content or "" for event in events if event.type == "chunk")
    assert adapter.generate_calls == 2
    assert adapter.stream_calls == 0
    assert rendered == "**Answer:** सही विकल्प B है क्योंकि वेग स्थिर है।"
    assert events[-1].type == "complete"


def test_live_stream_preserves_spaces_at_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "adaptive")
    expected = (
        "**Answer:** Adopted on 26 November 1949. India as a sovereign, socialist, "
        "secular, democratic republic."
    )
    adapter = _FakeAdapter(
        stream_chunks=[
            "**Answer:** Adopted on ",
            "26 November 1949. India as",
            " a sovereign, socialist, ",
            "secular, democratic republic.",
        ]
    )

    events = _events(adapter)

    emitted = "".join(event.content or "" for event in events if event.type == "chunk")
    complete = events[-1]
    assert emitted == expected
    assert complete.response is not None
    assert complete.response.answer == expected
    assert complete.response.content.value == expected


def test_low_confidence_request_generates_privately_then_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "adaptive")
    adapter = _FakeAdapter(stream_chunks=["must not stream"], generated_answers=[_VALID_ANSWER])

    events = _events(adapter, classification={**_CLASSIFICATION, "classifier_confidence": 0.5})

    assert adapter.stream_calls == 0
    assert adapter.generate_calls == 1
    verifying_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "status" and event.stage == "verifying"
    )
    first_chunk_index = next(index for index, event in enumerate(events) if event.type == "chunk")
    assert verifying_index < first_chunk_index
    assert "must not stream" not in "".join(
        event.content or "" for event in events if event.type == "chunk"
    )


def test_verification_failure_uses_at_most_one_private_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    adapter = _FakeAdapter(
        generated_answers=[
            "Actually, check the setup again. Final Answer: 2",
            _VALID_ANSWER,
        ]
    )

    events = _events(adapter)

    assert adapter.generate_calls == 2
    assert events[-1].type == "complete"
    rendered = "".join(event.content or "" for event in events if event.type == "chunk")
    assert "Actually" not in rendered
    assert rendered == _VALID_ANSWER


def test_conflicting_percentage_is_repaired_once_and_never_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    corrected = "**Answer:** 70%\n\nThe percentage formula confirms this result."
    adapter = _FakeAdapter(
        generated_answers=[
            "**Final Answer:** 7%\n\n**Final Answer:** 70%",
            corrected,
        ]
    )

    events = _events(adapter)

    rendered = "".join(event.content or "" for event in events if event.type == "chunk")
    assert adapter.generate_calls == 2
    assert events[-1].type == "complete"
    assert rendered == corrected
    assert "7%" not in rendered
    assert rendered.count("70%") == 1


def test_live_stream_failure_after_partial_output_ends_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
    adapter = _FakeAdapter(stream_chunks=["partial"], stream_error=RuntimeError("boom"))

    events = _events(adapter)

    assert [event.content for event in events if event.type == "chunk"] == ["partial"]
    assert events[-1].type == "error"
    assert not any(event.type == "complete" for event in events)
    assert adapter.stream_calls == 1
    assert adapter.generate_calls == 0


def test_cancellation_during_verified_replay_has_no_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    adapter = _FakeAdapter(generated_answers=[_VALID_ANSWER])
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 6

    events = _events(adapter, should_cancel=should_cancel)

    assert not any(event.type in {"complete", "error"} for event in events)
    assert not any(event.type == "complete" for event in events)


def test_correctness_infrastructure_failure_is_controlled_without_regeneration_or_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    adapter = _FakeAdapter(generated_answers=[_VALID_ANSWER])
    correctness = _FakeCorrectnessVerifier(
        CorrectnessVerification(
            status="unavailable",
            reason="ANSWER_VERIFICATION_UNAVAILABLE",
        )
    )
    adapter.correctness_verifier = correctness  # type: ignore[attr-defined]
    classification = {
        **_CLASSIFICATION,
        "difficulty": "intermediate",
        "classifier_confidence": 0.99,
    }

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                query="Solve the quadratic and select the correct option.",
                language="english",
                classification=classification,
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
            conversation_persistence=_NoPersistenceExpected(),
        )
    )

    assert adapter.generate_calls == 1
    assert correctness.calls == 1
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_UNAVAILABLE"
    assert not any(event.type == "complete" for event in events)


def test_correctness_rejection_still_reports_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 4: the genuine correctness failure keeps ANSWER_VERIFICATION_FAILED.

    The quality-stage taxonomy correction must not weaken the code that actually
    means "the independent verifier rejected this answer".
    """
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    adapter = _FakeAdapter(generated_answers=[_VALID_ANSWER])
    correctness = _FakeCorrectnessVerifier(
        CorrectnessVerification(
            status="mismatch",
            independent_answer="25 m",
            single_defensible_answer=True,
            reason="independent result differs",
            method="model",
        )
    )
    adapter.correctness_verifier = correctness  # type: ignore[attr-defined]
    classification = {
        **_CLASSIFICATION,
        "difficulty": "intermediate",
        "classifier_confidence": 0.99,
    }

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                query="Solve the quadratic and select the correct option.",
                language="english",
                classification=classification,
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
            conversation_persistence=_NoPersistenceExpected(),
        )
    )

    assert correctness.calls == 1
    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"
