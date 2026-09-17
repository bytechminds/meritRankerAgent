"""Terminal errors separate system retry semantics from the student's manual Retry.

`retryable` keeps its meaning (the operation itself may be retried or fall back);
`user_retryable` says whether the student may start a new attempt. Only the three
policy codes carry the new field; every other code omits it so clients keep deciding
from `retryable`. Nothing here adds an automatic resubmission.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from services.doubt_solver.answer_correctness import CorrectnessVerification
from services.doubt_solver.stream_transport import StreamCancellation, stream_events_as_sse
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    _error_event,
    stream_doubt_solver,
)


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("ANSWER_VERIFICATION_FAILED", False),
        ("ANSWER_QUALITY_FAILED", False),
        ("ANSWER_PROVIDER_FAILED", True),
    ],
)
def test_policy_codes_are_user_retryable_without_changing_retryable(
    code: str, retryable: bool
) -> None:
    event = _error_event("r", code=code, retryable=retryable)
    assert event.metadata == {"retryable": retryable, "code": code, "user_retryable": True}


@pytest.mark.parametrize(
    "code",
    [
        "ANSWER_VERIFICATION_UNAVAILABLE",
        "ANSWER_PARTIAL_STREAM_FAILED",
        "ANSWER_REPAIR_FAILED",
        "INSUFFICIENT_CREDITS",
        "INVALID_PRACTICE_REQUEST",
    ],
)
def test_other_codes_omit_user_retryable(code: str) -> None:
    event = _error_event("r", code=code, retryable=False)
    assert "user_retryable" not in event.metadata


def test_transport_generated_errors_are_unchanged() -> None:
    def failing() -> Iterator:
        raise RuntimeError("boom")
        yield  # pragma: no cover

    async def collect() -> list[bytes]:
        return [
            frame
            async for frame in stream_events_as_sse(
                failing(),
                request_id="t",
                cancellation=StreamCancellation(),
                heartbeat_interval_seconds=5,
            )
        ]

    frames = [
        json.loads(frame[5:]) for frame in asyncio.run(collect()) if frame.startswith(b"data:")
    ]
    assert frames[-1]["metadata"] == {"retryable": True, "code": "ANSWER_STREAM_FAILED"}


# ---------------------------------------------------------------------------
# End-to-end through the streaming service
# ---------------------------------------------------------------------------

_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "intermediate",
    "need_web_search": False,
    "classifier_confidence": 0.99,
    "classification_source": "llm",
}
_VALID = "Speed = 240 / 16 = 15 m/s, and 15 × 18/5 = 54.\n\n**Answer:** 54 km/h"


class _Adapter:
    def __init__(
        self,
        *,
        answer: str | None = _VALID,
        error: Exception | None = None,
        verdict: CorrectnessVerification | None = None,
    ) -> None:
        self._answer, self._error = answer, error
        self.generate_calls = 0
        verifier_verdict = verdict or CorrectnessVerification(
            status="match",
            independent_answer="54",
            single_defensible_answer=True,
            reason="ok",
            method="model",
        )

        class _Verifier:
            calls = 0

            def verify(self_inner, **_kwargs: object) -> CorrectnessVerification:  # noqa: N805
                type(self_inner).calls += 1
                return verifier_verdict

        self.correctness_verifier = _Verifier()

    def generate(self, **_kwargs: object) -> str:
        self.generate_calls += 1
        if self._error is not None:
            raise self._error
        return self._answer or ""


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
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


def _terminal(adapter: _Adapter) -> dict:
    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="user-retryable",
                query="A train 240 m long passes a pole in 16 s.",
                language="english",
                classification=dict(_CLASSIFICATION),
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )
    assert events[-1].type == "error"
    return events[-1].metadata


def test_verification_failure_is_not_system_retryable_but_user_retryable() -> None:
    adapter = _Adapter(
        verdict=CorrectnessVerification(
            status="mismatch",
            independent_answer="50",
            single_defensible_answer=True,
            reason="differs",
            method="model",
        )
    )
    assert _terminal(adapter) == {
        "retryable": False,
        "code": "ANSWER_VERIFICATION_FAILED",
        "user_retryable": True,
    }
    assert adapter.generate_calls == 1  # no automatic new attempt


def test_quality_failure_is_not_system_retryable_but_user_retryable() -> None:
    adapter = _Adapter(answer="<script>alert(1)</script>")
    metadata = _terminal(adapter)
    assert metadata == {"retryable": False, "code": "ANSWER_QUALITY_FAILED", "user_retryable": True}


def test_provider_failure_stays_system_retryable_and_is_user_retryable() -> None:
    adapter = _Adapter(error=RuntimeError("provider down"))
    assert _terminal(adapter) == {
        "retryable": True,
        "code": "ANSWER_PROVIDER_FAILED",
        "user_retryable": True,
    }
    assert adapter.generate_calls == 1
