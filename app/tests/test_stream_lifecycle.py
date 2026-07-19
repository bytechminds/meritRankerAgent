"""Streaming lifecycle, heartbeat, disconnect, and replay regression tests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import config as cfg_module
import services.doubt_solver.stream_transport as transport_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from schemas.doubt_solver import (
    DoubtSolverFinalResponse,
    DoubtSolverStreamEvent,
    ResponseContent,
)
from services.doubt_solver.markdown_replay import iter_markdown_replay_chunks
from services.doubt_solver.stream_transport import (
    StreamCancellation,
    stream_events_as_sse,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)

_REQUEST_ID = "lifecycle-request-001"
_ANSWER = "**Final Answer:**\n\\(20\\)"
_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "basic",
    "need_web_search": False,
    "classifier_confidence": 0.5,
    "classification_source": "llm",
}


class _Adapter:
    def __init__(
        self,
        *,
        generated: list[str] | None = None,
        chunks: list[str] | None = None,
        generate_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.generated = list(generated or [_ANSWER])
        self.chunks = list(chunks or [])
        self.generate_error = generate_error
        self.stream_error = stream_error

    def generate(self, **_: object) -> str:
        if self.generate_error is not None:
            error = self.generate_error
            self.generate_error = None
            raise error
        if not self.generated:
            raise RuntimeError("repair failed")
        return self.generated.pop(0)

    def generate_stream(self, **_: object) -> Iterator[str]:
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error


@pytest.fixture(autouse=True)
def _settings_and_retrieval(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_REPLAY_MAX_CHUNK_CHARS", "12")
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    cfg_module._settings = None
    yield
    cfg_module._settings = None


def _service_events(adapter: _Adapter) -> list[DoubtSolverStreamEvent]:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                query="What is 20 percent of 100?",
                classification=dict(_CLASSIFICATION),
                classifier_confidence=0.5,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )


def _complete_event() -> DoubtSolverStreamEvent:
    response = DoubtSolverFinalResponse(
        request_id=_REQUEST_ID,
        content=ResponseContent(value=_ANSWER),
        answer=_ANSWER,
    )
    return DoubtSolverStreamEvent(
        type="complete",
        request_id=_REQUEST_ID,
        stage="complete",
        label="Done",
        metadata={"request_id": _REQUEST_ID},
        response=response,
    )


def _parse_data_frame(frame: bytes) -> dict | None:
    if frame.startswith(b":"):
        return None
    assert frame.startswith(b"data: ")
    assert frame.endswith(b"\n\n")
    return json.loads(frame[6:-2].decode())


async def _collect_sse(
    events: Iterator[DoubtSolverStreamEvent],
    *,
    heartbeat_interval: float = 0.01,
) -> tuple[list[bytes], StreamCancellation]:
    cancellation = StreamCancellation()
    frames = [
        frame
        async for frame in stream_events_as_sse(
            events,
            request_id=_REQUEST_ID,
            cancellation=cancellation,
            heartbeat_interval_seconds=heartbeat_interval,
        )
    ]
    return frames, cancellation


def test_successful_service_stream_has_exactly_one_complete() -> None:
    events = _service_events(_Adapter())

    assert sum(event.type == "complete" for event in events) == 1
    assert events[-1].type == "complete"
    assert not any(event.type == "error" for event in events)


def test_provider_error_before_content_is_controlled_and_retryable() -> None:
    events = _service_events(_Adapter(generate_error=RuntimeError("private")))

    assert events[-1].type == "error"
    assert events[-1].metadata == {
        "retryable": True,
        "code": "ANSWER_PROVIDER_FAILED",
    }
    assert not any(event.type == "complete" for event in events)


def test_provider_error_after_content_is_typed_partial_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
    cfg_module._settings = None

    events = _service_events(
        _Adapter(chunks=["partial"], stream_error=RuntimeError("stream"))
    )

    assert events[-1].metadata == {
        "retryable": False,
        "code": "ANSWER_PARTIAL_STREAM_FAILED",
    }
    assert not any(event.type == "complete" for event in events)


def test_verifier_exception_is_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        streaming_module,
        "validate_answer_quality",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verifier")),
    )

    events = _service_events(_Adapter())

    assert events[-1].metadata["code"] == "ANSWER_VERIFICATION_FAILED"
    assert events[-1].response is None


def test_repair_failure_is_controlled() -> None:
    events = _service_events(
        _Adapter(generated=["Actually, inspect this. Final Answer: 2"])
    )

    assert events[-1].metadata["code"] == "ANSWER_REPAIR_FAILED"
    assert not any(event.type == "complete" for event in events)


def test_silent_source_exit_becomes_terminal_error() -> None:
    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage="generating",
            label="Preparing answer...",
        )

    frames, _ = asyncio.run(_collect_sse(source()))
    payloads = [payload for frame in frames if (payload := _parse_data_frame(frame))]

    assert payloads[-1]["type"] == "error"
    assert payloads[-1]["metadata"]["code"] == "ANSWER_STREAM_FAILED"
    assert sum(item["type"] in {"complete", "error"} for item in payloads) == 1


def test_unexpected_source_exception_after_chunk_is_partial_error() -> None:
    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="chunk", request_id=_REQUEST_ID, content="partial"
        )
        raise RuntimeError("unexpected")

    frames, _ = asyncio.run(_collect_sse(source()))
    payloads = [payload for frame in frames if (payload := _parse_data_frame(frame))]

    assert payloads[-1]["metadata"] == {
        "retryable": False,
        "code": "ANSWER_PARTIAL_STREAM_FAILED",
    }


def test_duplicate_terminal_event_is_suppressed() -> None:
    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield _complete_event()
        yield DoubtSolverStreamEvent(
            type="error",
            request_id=_REQUEST_ID,
            stage="failed",
            label="Unable to complete",
        )

    frames, _ = asyncio.run(_collect_sse(source()))
    payloads = [payload for frame in frames if (payload := _parse_data_frame(frame))]

    assert [item["type"] for item in payloads] == ["complete"]


def test_close_after_complete_does_not_reclassify_as_disconnect() -> None:
    cancellation = StreamCancellation()

    async def close_after_terminal() -> None:
        body = stream_events_as_sse(
            iter([_complete_event()]),
            request_id=_REQUEST_ID,
            cancellation=cancellation,
            heartbeat_interval_seconds=0.01,
        )
        assert _parse_data_frame(await anext(body))["type"] == "complete"
        await body.aclose()

    asyncio.run(close_after_terminal())

    assert cancellation.reason is None


@pytest.mark.parametrize("stage", ["generating", "verifying"])
def test_heartbeat_is_sse_comment_and_stops_after_complete(stage: str) -> None:
    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage=stage,
            label="Working...",
        )
        time.sleep(0.04)
        yield _complete_event()

    frames, _ = asyncio.run(_collect_sse(source(), heartbeat_interval=0.005))

    assert any(frame == b": heartbeat\n\n" for frame in frames)
    assert _parse_data_frame(frames[-1])["type"] == "complete"
    assert all(frame == b": heartbeat\n\n" for frame in frames if frame.startswith(b":"))


def test_heartbeat_stops_after_error() -> None:
    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="error",
            request_id=_REQUEST_ID,
            stage="failed",
            label="Unable to complete",
            metadata={"retryable": True, "code": "ANSWER_STREAM_FAILED"},
        )
        time.sleep(0.03)

    frames, _ = asyncio.run(_collect_sse(source(), heartbeat_interval=0.005))

    assert len(frames) == 1
    assert _parse_data_frame(frames[0])["type"] == "error"


def test_serialization_failure_emits_controlled_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = transport_module._serialize_event
    calls = 0

    def fail_once(event: DoubtSolverStreamEvent) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("serialization")
        return original(event)

    monkeypatch.setattr(transport_module, "_serialize_event", fail_once)
    frames, _ = asyncio.run(_collect_sse(iter([_complete_event()])))
    payload = _parse_data_frame(frames[-1])

    assert payload["type"] == "error"
    assert payload["metadata"]["code"] == "ANSWER_SERIALIZATION_FAILED"


def test_client_disconnect_cancels_worker_and_logs_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cancellation = StreamCancellation()

    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage="verifying",
            label="Verifying answer...",
        )
        time.sleep(0.1)
        yield _complete_event()

    async def disconnect() -> None:
        body = stream_events_as_sse(
            source(),
            request_id=_REQUEST_ID,
            cancellation=cancellation,
            heartbeat_interval_seconds=0.5,
        )
        assert _parse_data_frame(await anext(body))["type"] == "status"
        pending = asyncio.create_task(anext(body))
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    with caplog.at_level(logging.INFO):
        asyncio.run(disconnect())

    assert cancellation.reason == "client_disconnected"
    assert "terminal_reason=client_disconnected" in " ".join(
        record.message for record in caplog.records
    )


@pytest.mark.parametrize(
    ("stage", "visible"),
    [
        ("understanding", False),
        ("generating", False),
        ("verifying", False),
        ("generating", True),
        ("verifying", True),
    ],
)
def test_disconnect_in_each_stream_phase_suppresses_later_output(
    stage: str,
    visible: bool,
) -> None:
    cancellation = StreamCancellation()

    def source() -> Iterator[DoubtSolverStreamEvent]:
        yield DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage=stage,
            label="Working...",
        )
        if visible:
            yield DoubtSolverStreamEvent(
                type="chunk", request_id=_REQUEST_ID, content="partial"
            )
        time.sleep(0.1)
        yield _complete_event()

    async def disconnect() -> list[dict]:
        payloads: list[dict] = []
        body = stream_events_as_sse(
            source(),
            request_id=_REQUEST_ID,
            cancellation=cancellation,
            heartbeat_interval_seconds=0.5,
        )
        payloads.append(_parse_data_frame(await anext(body)))
        if visible:
            payloads.append(_parse_data_frame(await anext(body)))
        pending = asyncio.create_task(anext(body))
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        return payloads

    payloads = asyncio.run(disconnect())

    assert cancellation.reason == "client_disconnected"
    assert not any(payload["type"] == "complete" for payload in payloads)


def test_heartbeat_write_cancellation_is_classified_as_disconnect() -> None:
    cancellation = StreamCancellation()

    def source() -> Iterator[DoubtSolverStreamEvent]:
        time.sleep(0.1)
        yield _complete_event()

    async def disconnect_after_heartbeat() -> None:
        body = stream_events_as_sse(
            source(),
            request_id=_REQUEST_ID,
            cancellation=cancellation,
            heartbeat_interval_seconds=0.005,
        )
        assert await anext(body) == b": heartbeat\n\n"
        pending = asyncio.create_task(anext(body))
        await asyncio.sleep(0.001)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(disconnect_after_heartbeat())

    assert cancellation.reason == "client_disconnected"


def test_cancellation_callback_failure_becomes_controlled_error() -> None:
    def broken_cancellation_check() -> bool:
        raise RuntimeError("cancellation state unavailable")

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                query="Question",
                classification=dict(_CLASSIFICATION),
                should_cancel=broken_cancellation_check,
            ),
            adapter=_Adapter(),  # type: ignore[arg-type]
        )
    )

    assert events[-1].type == "error"
    assert events[-1].metadata["code"] == "ANSWER_STREAM_FAILED"


@pytest.mark.parametrize(
    "protected",
    [
        "\\[\na^2+b^2=c^2\n\\]",
        "\\(a+b\\)",
        "$$\na+b\n$$",
        "$a+b$",
        "```python\nprint('x')\n```",
        "```mermaid\ngraph TD\nA-->B\n```",
        "[reference](https://example.com/a_(b))",
        "![diagram](https://example.com/image_(1).png)",
        "<details>\n<summary>Work</summary>\nAnswer\n</details>",
        "| A | B |\n|---|---|\n| 1 | 2 |\n",
    ],
)
def test_replay_never_splits_protected_markdown(protected: str) -> None:
    content = f"Before.\n{protected}\nAfter."
    chunks = list(iter_markdown_replay_chunks(content, max_chunk_chars=8))

    assert "".join(chunks) == content
    assert any(protected in chunk for chunk in chunks)


def test_verified_sequence_and_complete_response_match_replayed_chunks() -> None:
    events = _service_events(_Adapter())
    stages = [event.stage for event in events if event.type == "status"]
    chunks = "".join(event.content or "" for event in events if event.type == "chunk")
    complete = events[-1]

    assert stages == [
        "understanding",
        "thinking",
        "retrieving",
        "generating",
        "verifying",
        "finalizing",
    ]
    assert complete.response is not None
    assert complete.response.answer == chunks
    assert complete.response.content.value == chunks


def test_agentcore_invocations_preserves_event_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    import main

    def stream_source(input, *, adapter) -> Iterator[DoubtSolverStreamEvent]:
        del adapter
        response = DoubtSolverFinalResponse(
            request_id=input.request_id,
            content=ResponseContent(value=_ANSWER),
            answer=_ANSWER,
        )
        yield DoubtSolverStreamEvent(
            type="status",
            request_id=input.request_id,
            stage="generating",
            label="Preparing answer...",
        )
        yield DoubtSolverStreamEvent(
            type="chunk",
            request_id=input.request_id,
            content=_ANSWER,
        )
        yield DoubtSolverStreamEvent(
            type="complete",
            request_id=input.request_id,
            stage="complete",
            label="Done",
            metadata={"request_id": input.request_id},
            response=response,
        )

    monkeypatch.setattr(main, "orchestrated_doubt_solver_graph", object())
    monkeypatch.setattr(main, "orchestrated_adapter", object())
    monkeypatch.setattr(main, "stream_doubt_solver", stream_source)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            enable_orchestrated_doubt_solver=True,
            answer_stream_heartbeat_interval_seconds=0.01,
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/invocations",
            json={
                "mode": "doubt_solver",
                "query": "What is 20 percent of 100?",
                "stream": True,
            },
        )

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [payload["type"] for payload in payloads] == [
        "status",
        "chunk",
        "complete",
    ]
    assert set(payloads[-1]) == {
        "type",
        "request_id",
        "stage",
        "label",
        "content",
        "metadata",
        "response",
    }
