"""Per-request streaming status deduplication and safe logging."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from schemas.doubt_solver import DoubtSolverStreamEvent

logger = logging.getLogger(__name__)


@dataclass
class StreamStatusTracker:
    """Emit each lifecycle stage at most once per request."""

    request_id: str
    pending_events: list[DoubtSolverStreamEvent] = field(default_factory=list)
    _emitted_labels: set[str] = field(default_factory=set)
    _emitted_stages: set[str] = field(default_factory=set)

    def emit_direct(
        self,
        *,
        stage: str,
        label: str,
        reason_code: str,
    ) -> DoubtSolverStreamEvent | None:
        """Emit immediately (not queued) if this label has not been emitted yet."""
        if stage in self._emitted_stages:
            logger.debug(
                "stream_status_reason  request_id=%s  status_label=%s  "
                "reason_code=%s  emitted=false",
                self.request_id,
                label,
                reason_code,
            )
            return None

        self._emitted_labels.add(label)
        self._emitted_stages.add(stage)
        logger.debug(
            "stream_status_reason  request_id=%s  status_label=%s  "
            "reason_code=%s  emitted=true",
            self.request_id,
            label,
            reason_code,
        )
        return DoubtSolverStreamEvent(
            type="status",
            request_id=self.request_id,
            stage=stage,
            label=label,
        )

    def emit_if_new(
        self,
        *,
        stage: str,
        label: str,
        reason_code: str,
    ) -> None:
        """Queue a status event if this label has not been emitted yet."""
        if stage in self._emitted_stages:
            logger.debug(
                "stream_status_reason  request_id=%s  status_label=%s  "
                "reason_code=%s  emitted=false",
                self.request_id,
                label,
                reason_code,
            )
            return

        self._emitted_labels.add(label)
        self._emitted_stages.add(stage)
        logger.debug(
            "stream_status_reason  request_id=%s  status_label=%s  "
            "reason_code=%s  emitted=true",
            self.request_id,
            label,
            reason_code,
        )
        self.pending_events.append(
            DoubtSolverStreamEvent(
                type="status",
                request_id=self.request_id,
                stage=stage,
                label=label,
            )
        )

    def hook(
        self,
        *,
        stage: str,
        label: str,
        reason_code: str,
    ) -> Callable[[], None]:
        """Return a zero-arg callback suitable for wiring into services."""
        return lambda: self.emit_if_new(
            stage=stage,
            label=label,
            reason_code=reason_code,
        )
