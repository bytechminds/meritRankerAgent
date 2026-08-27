"""AgentCore SSE lifecycle, heartbeat, and cancellation boundary."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import copy_context
from dataclasses import dataclass
from typing import Literal

from schemas.doubt_solver import DoubtSolverStreamEvent

logger = logging.getLogger(__name__)

CancellationReason = Literal["client_disconnected", "user_cancelled"]
TerminalFrameCallback = Callable[[DoubtSolverStreamEvent], None]

_HEARTBEAT_FRAME = b": heartbeat\n\n"
_WORKER_DONE = object()


class StreamCancellation:
    """Thread-safe cooperative cancellation shared with the sync stream worker."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: CancellationReason | None = None

    @property
    def reason(self) -> CancellationReason | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: CancellationReason) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class _WorkerFailure:
    error: BaseException


class _TerminalFrameCallbacks:
    def __init__(
        self,
        *,
        request_id: str,
        committed: TerminalFrameCallback | None,
        abandoned: TerminalFrameCallback | None,
    ) -> None:
        self._request_id = request_id
        self._committed = committed
        self._abandoned = abandoned
        self._pending_event: DoubtSolverStreamEvent | None = None
        self._resolved = False
        self._lock = threading.Lock()

    def register(self, event: DoubtSolverStreamEvent) -> None:
        with self._lock:
            if self._resolved or self._pending_event is not None:
                return
            self._pending_event = event

    def commit_pending(self) -> None:
        self._resolve_pending(callback=self._committed, outcome="commit")

    def abandon_pending(self) -> None:
        self._resolve_pending(callback=self._abandoned, outcome="abandon")

    def _resolve_pending(
        self,
        *,
        callback: TerminalFrameCallback | None,
        outcome: str,
    ) -> None:
        with self._lock:
            if self._resolved or self._pending_event is None:
                return
            self._resolved = True
            event = self._pending_event
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "terminal_frame_%s_callback_failed request_id=%s error_type=%s",
                outcome,
                self._request_id,
                type(exc).__name__,
            )


def _error_event(
    request_id: str,
    *,
    code: str,
    retryable: bool,
) -> DoubtSolverStreamEvent:
    return DoubtSolverStreamEvent(
        type="error",
        request_id=request_id,
        stage="failed",
        label="Unable to complete",
        metadata={"retryable": retryable, "code": code},
    )


def _terminal_reason(event: DoubtSolverStreamEvent, *, visible: bool) -> str:
    if event.type == "complete":
        return "completed"
    if event.type == "practice_generation_started":
        return "practice_generation_started"
    code = str(event.metadata.get("code") or "")
    return {
        "ANSWER_PROVIDER_FAILED": (
            "provider_failed_after_content" if visible else "provider_failed_before_content"
        ),
        "ANSWER_PARTIAL_STREAM_FAILED": "provider_failed_after_content",
        "ANSWER_VERIFICATION_FAILED": "verification_failed",
        "ANSWER_QUALITY_FAILED": "quality_failed",
        "ANSWER_REPAIR_FAILED": "repair_failed",
        "ANSWER_SERIALIZATION_FAILED": "serialization_failed",
    }.get(code, "unexpected_internal_error")


def _serialize_event(event: DoubtSolverStreamEvent) -> bytes:
    payload = event.model_dump(mode="json", by_alias=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode()


def _run_worker(
    events: Iterator[DoubtSolverStreamEvent],
    output: queue.Queue[object],
    cancellation: StreamCancellation,
    stop: threading.Event,
    terminal_callbacks: _TerminalFrameCallbacks,
) -> None:
    def put(item: object) -> bool:
        while not cancellation.is_cancelled() and not stop.is_set():
            try:
                output.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    try:
        for event in events:
            if event.type in {"complete", "error", "practice_generation_started"}:
                terminal_callbacks.register(event)
            if cancellation.is_cancelled() or stop.is_set():
                if event.type in {"complete", "error", "practice_generation_started"}:
                    terminal_callbacks.abandon_pending()
                break
            if not put(event):
                if event.type in {"complete", "error", "practice_generation_started"}:
                    terminal_callbacks.abandon_pending()
                break
            if event.type in {"complete", "error", "practice_generation_started"}:
                break
    except BaseException as exc:  # the async boundary converts this to a safe event
        if not cancellation.is_cancelled() and not stop.is_set():
            put(_WorkerFailure(exc))
    finally:
        try:
            events.close()
        except (AttributeError, RuntimeError):
            pass
        if cancellation.is_cancelled() or stop.is_set():
            try:
                output.put_nowait(_WORKER_DONE)
            except queue.Full:
                pass
        else:
            while True:
                try:
                    output.put(_WORKER_DONE, timeout=0.1)
                    break
                except queue.Full:
                    continue


async def stream_events_as_sse(
    events: Iterator[DoubtSolverStreamEvent],
    *,
    request_id: str,
    cancellation: StreamCancellation,
    heartbeat_interval_seconds: float,
    on_terminal_frame_committed: TerminalFrameCallback | None = None,
    on_terminal_frame_abandoned: TerminalFrameCallback | None = None,
) -> AsyncIterator[bytes]:
    """Frame application events as SSE and enforce one observable terminal outcome."""
    started_at = time.monotonic()
    output: queue.Queue[object] = queue.Queue(maxsize=1)
    worker_stop = threading.Event()
    terminal_callbacks = _TerminalFrameCallbacks(
        request_id=request_id,
        committed=on_terminal_frame_committed,
        abandoned=on_terminal_frame_abandoned,
    )
    worker = threading.Thread(
        target=copy_context().run,
        args=(_run_worker, events, output, cancellation, worker_stop, terminal_callbacks),
        name=f"stream-{request_id[:12]}",
        daemon=True,
    )
    worker.start()

    sequence = 0
    chunk_count = 0
    heartbeat_count = 0
    visible = False
    terminal_sent = False
    terminal_reason: str | None = None
    pending_terminal_event: DoubtSolverStreamEvent | None = None
    terminal_frame_serialization_failed = False
    terminal_frame_replaced = False
    current_stage = "started"
    pending_get: asyncio.Future[object] | None = None

    logger.debug("stream_started request_id=%s stage=started", request_id)
    try:
        while not terminal_sent:
            if cancellation.is_cancelled():
                terminal_reason = cancellation.reason or "user_cancelled"
                logger.debug(
                    "request_cancelled request_id=%s stage=%s terminal_reason=%s",
                    request_id,
                    current_stage,
                    terminal_reason,
                )
                return

            if pending_get is None:
                loop = asyncio.get_running_loop()
                pending_get = loop.run_in_executor(None, output.get)
            try:
                item = await asyncio.wait_for(
                    asyncio.shield(pending_get),
                    timeout=heartbeat_interval_seconds,
                )
                pending_get = None
            except TimeoutError:
                if cancellation.is_cancelled() or terminal_sent:
                    continue
                heartbeat_count += 1
                logger.debug(
                    "heartbeat_sent request_id=%s stage=%s sequence=%d elapsed_ms=%d",
                    request_id,
                    current_stage,
                    heartbeat_count,
                    int((time.monotonic() - started_at) * 1000),
                )
                yield _HEARTBEAT_FRAME
                continue

            if item is _WORKER_DONE:
                if cancellation.is_cancelled():
                    terminal_reason = cancellation.reason or "user_cancelled"
                    return
                if terminal_sent:
                    return
                event = _error_event(
                    request_id,
                    code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
                    retryable=not visible,
                )
                terminal_reason = "unexpected_internal_error"
            elif isinstance(item, _WorkerFailure):
                event = _error_event(
                    request_id,
                    code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
                    retryable=not visible,
                )
                terminal_reason = "unexpected_internal_error"
            else:
                if not isinstance(item, DoubtSolverStreamEvent):
                    event = _error_event(
                        request_id,
                        code="ANSWER_STREAM_FAILED",
                        retryable=True,
                    )
                    terminal_reason = "unexpected_internal_error"
                else:
                    event = item

            if event.request_id != request_id:
                if event.type in {"complete", "error", "practice_generation_started"}:
                    terminal_callbacks.abandon_pending()
                    terminal_frame_replaced = True
                event = _error_event(
                    request_id,
                    code="ANSWER_STREAM_FAILED",
                    retryable=True,
                )
                terminal_reason = "unexpected_internal_error"

            if event.type in {"complete", "error", "practice_generation_started"}:
                if terminal_sent:
                    return
                terminal_sent = True
                terminal_reason = terminal_reason or _terminal_reason(event, visible=visible)
            elif terminal_sent:
                return

            try:
                frame = _serialize_event(event)
            except Exception:  # noqa: BLE001
                failed_terminal_event = event if terminal_sent else None
                event = _error_event(
                    request_id,
                    code="ANSWER_SERIALIZATION_FAILED",
                    retryable=not visible,
                )
                frame = _serialize_event(event)
                terminal_sent = True
                terminal_reason = "serialization_failed"
                if failed_terminal_event is not None:
                    pending_terminal_event = failed_terminal_event
                    terminal_frame_serialization_failed = True
                    terminal_callbacks.abandon_pending()

            if event.type in {"complete", "error", "practice_generation_started"}:
                pending_terminal_event = pending_terminal_event or event

            sequence += 1
            if event.stage:
                current_stage = event.stage
            if event.type == "status":
                logger.debug(
                    "status_sent request_id=%s stage=%s event_type=status sequence=%d",
                    request_id,
                    current_stage,
                    sequence,
                )
            elif event.type == "chunk":
                chunk_count += 1
                if not visible:
                    logger.debug(
                        "first_chunk_sent request_id=%s stage=%s event_type=chunk sequence=%d",
                        request_id,
                        current_stage,
                        sequence,
                    )
                visible = True
                logger.debug(
                    "chunk_sent request_id=%s stage=%s event_type=chunk sequence=%d chunk_count=%d",
                    request_id,
                    current_stage,
                    sequence,
                    chunk_count,
                )
            elif event.type == "complete":
                logger.debug(
                    "complete_sent request_id=%s stage=complete event_type=complete sequence=%d",
                    request_id,
                    sequence,
                )
            elif event.type == "practice_generation_started":
                logger.debug(
                    "practice_generation_started_sent request_id=%s sequence=%d",
                    request_id,
                    sequence,
                )
            else:
                logger.debug(
                    "error_sent request_id=%s stage=failed event_type=error sequence=%d "
                    "terminal_reason=%s",
                    request_id,
                    sequence,
                    terminal_reason,
                )
            yield frame
            if (
                pending_terminal_event is not None
                and not terminal_frame_serialization_failed
                and not terminal_frame_replaced
            ):
                terminal_callbacks.commit_pending()
    except asyncio.CancelledError:
        if not terminal_sent:
            cancellation.cancel("client_disconnected")
            terminal_reason = "client_disconnected"
            logger.debug(
                "client_disconnected request_id=%s stage=%s terminal_reason=%s",
                request_id,
                current_stage,
                terminal_reason,
            )
        raise
    except GeneratorExit:
        if not terminal_sent:
            cancellation.cancel("client_disconnected")
            terminal_reason = "client_disconnected"
            logger.debug(
                "client_disconnected request_id=%s stage=%s terminal_reason=%s",
                request_id,
                current_stage,
                terminal_reason,
            )
        raise
    except (BrokenPipeError, ConnectionError):
        cancellation.cancel("client_disconnected")
        terminal_reason = "transport_failed"
        logger.debug(
            "client_disconnected request_id=%s stage=%s terminal_reason=%s",
            request_id,
            current_stage,
            terminal_reason,
        )
        return
    finally:
        terminal_callbacks.abandon_pending()
        worker_stop.set()
        if terminal_reason is None and cancellation.is_cancelled():
            terminal_reason = cancellation.reason
        logger.debug(
            "stream_closed request_id=%s stage=%s terminal_reason=%s chunk_count=%d "
            "heartbeat_count=%d elapsed_ms=%d",
            request_id,
            current_stage,
            terminal_reason or "unexpected_internal_error",
            chunk_count,
            heartbeat_count,
            int((time.monotonic() - started_at) * 1000),
        )
