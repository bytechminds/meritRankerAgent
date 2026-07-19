"""
app/services/doubt_solver/answer_generation_adapter.py
------------------------------------------------------
Adapter that translates orchestrated graph classification + context into a
RouteRequest and delegates to LlmOrchestrator.

Responsibilities:
- Accept query, subject, intent, difficulty, context_text.
- Build a RouteRequest with task_role="generator".
- Call LlmOrchestrator.generate() or generate_stream().
- Return only the answer string or text chunks.

Design invariants:
- Does NOT read env vars.
- Does NOT fetch secrets.
- Does NOT call any provider SDK directly.
- Does NOT know model IDs, provider profiles, or deployments.
- Does NOT mutate graph state.
- Constructor requires an LlmOrchestrator — no implicit default.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_completion import resolve_generator_route_subject
from services.llm.orchestration.orchestrator import LlmOrchestrator

logger = logging.getLogger(__name__)


class AnswerGenerationAdapter:
    """Translate orchestrated graph state into a RouteRequest and call LlmOrchestrator."""

    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        if orchestrator is None:
            raise TypeError("orchestrator is required.")
        self._orchestrator = orchestrator

    def generate(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        intent: str,
        difficulty: str,
        context: str,
        web_search_reason: str | None = None,
    ) -> str:
        """Build a RouteRequest and call the orchestrator."""
        route_subject = resolve_generator_route_subject(
            subject=subject,
            intent=intent,
            web_search_reason=web_search_reason,
        )
        route_request = RouteRequest(
            request_id=request_id,
            subject=route_subject,
            task_role="generator",
            difficulty=difficulty,
            intent=intent,
        )

        result = self._orchestrator.generate(
            route_request=route_request,
            query=query,
            context=context if context else None,
        )

        logger.info(
            "answer_generation_adapter.generate  request_id=%s  subject=%s  "
            "difficulty=%s  intent=%s  model=%s",
            request_id,
            subject,
            difficulty,
            intent,
            result.model,
        )

        return result.content

    def generate_stream(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        intent: str,
        difficulty: str,
        context_text: str,
        web_search_reason: str | None = None,
        on_before_generator_fallback: Callable[[], None] | None = None,
        on_before_continuation: Callable[[], None] | None = None,
        verify_before_stream: bool = True,
    ) -> Iterator[str]:
        """Yield answer text chunks from the orchestrator stream path.

        Yields answer text only — no status events, prompts, messages, or
        provider metadata.
        """
        route_subject = resolve_generator_route_subject(
            subject=subject,
            intent=intent,
            web_search_reason=web_search_reason,
        )
        route_request = RouteRequest(
            request_id=request_id,
            subject=route_subject,
            task_role="generator",
            difficulty=difficulty,
            intent=intent,
        )

        logger.info(
            "answer_generation_adapter.generate_stream  request_id=%s  subject=%s  "
            "difficulty=%s  intent=%s  stream_started=true",
            request_id,
            subject,
            difficulty,
            intent,
        )

        yield from self._orchestrator.generate_stream(
            route_request=route_request,
            query=query,
            context=context_text if context_text else None,
            on_before_fallback=on_before_generator_fallback,
            on_before_continuation=on_before_continuation,
            verify_before_stream=verify_before_stream,
        )
