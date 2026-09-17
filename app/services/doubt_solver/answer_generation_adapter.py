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

from observability import log_event
from retrieval.pattern_intelligence import DoubtPatternContext
from schemas.doubt_solver import CanonicalLanguage, FinalAnswerResult
from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_completion import resolve_generator_route_subject
from services.doubt_solver.answer_correctness import AnswerCorrectnessVerifier
from services.doubt_solver.answer_diagnosis import AnswerDiagnoser
from services.doubt_solver.final_answer import build_final_answer_result
from services.llm.orchestration.orchestrator import LlmOrchestrator

logger = logging.getLogger(__name__)


class AnswerGenerationAdapter:
    """Translate orchestrated graph state into a RouteRequest and call LlmOrchestrator."""

    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        if orchestrator is None:
            raise TypeError("orchestrator is required.")
        self._orchestrator = orchestrator
        self._correctness_verifier = AnswerCorrectnessVerifier(orchestrator=orchestrator)
        self._semantic_diagnoser = AnswerDiagnoser(orchestrator=orchestrator)

    @property
    def correctness_verifier(self) -> AnswerCorrectnessVerifier:
        return self._correctness_verifier

    @property
    def semantic_diagnoser(self) -> AnswerDiagnoser:
        return self._semantic_diagnoser

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
        exam_id: str | None = None,
        exam_stage: str | None = None,
        exam_profile_id: str | None = None,
        language: CanonicalLanguage = "english",
        conversation_context: str | None = None,
        doubt_pattern_context: DoubtPatternContext | None = None,
        recovery_instruction: str | None = None,
    ) -> str:
        """Return the authoritative answer content for compatibility callers."""
        return self.generate_final(
            request_id=request_id,
            query=query,
            subject=subject,
            intent=intent,
            difficulty=difficulty,
            context=context,
            web_search_reason=web_search_reason,
            exam_id=exam_id,
            exam_stage=exam_stage,
            exam_profile_id=exam_profile_id,
            language=language,
            conversation_context=conversation_context,
            doubt_pattern_context=doubt_pattern_context,
            recovery_instruction=recovery_instruction,
        ).content

    def generate_final(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        intent: str,
        difficulty: str,
        context: str,
        web_search_reason: str | None = None,
        exam_id: str | None = None,
        exam_stage: str | None = None,
        exam_profile_id: str | None = None,
        language: CanonicalLanguage = "english",
        conversation_context: str | None = None,
        doubt_pattern_context: DoubtPatternContext | None = None,
        recovery_instruction: str | None = None,
    ) -> FinalAnswerResult:
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
            exam=exam_id,
            exam_stage=exam_stage,
            exam_profile_id=exam_profile_id,
            language=language,
        )

        generation_kwargs = {
            "route_request": route_request,
            "query": query,
            "context": context if context else None,
            "conversation_context": conversation_context,
        }
        if doubt_pattern_context is not None:
            generation_kwargs["doubt_pattern_context"] = doubt_pattern_context
        if recovery_instruction:
            generation_kwargs["recovery_instruction"] = recovery_instruction
        result = self._orchestrator.generate(**generation_kwargs)

        logger.debug(
            "answer_generation_adapter.generate  request_id=%s  subject=%s  "
            "difficulty=%s  intent=%s  model=%s",
            request_id,
            subject,
            difficulty,
            intent,
            result.model,
        )
        route_decision = getattr(result, "route_decision", None)
        if route_decision is not None:
            log_event(
                "generation_completed",
                component="doubt_solver.generator",
                stage="generate",
                status="completed",
                duration_ms=getattr(result, "latency_ms", None),
                details={
                    "route_id": route_decision.route_id,
                    "task_role": route_decision.task_role,
                    "model_alias": result.model,
                    "provider": result.provider,
                    "deployment": getattr(result, "execution_deployment", None),
                    "fallback_used": getattr(result, "fallback_used", False),
                },
            )

        final_answer = getattr(result, "final_answer", None)
        if final_answer is not None:
            return final_answer
        return build_final_answer_result(
            content=result.content,
            language=language,
            quality_status="checked",
        )

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
        exam_id: str | None = None,
        exam_stage: str | None = None,
        exam_profile_id: str | None = None,
        language: CanonicalLanguage = "english",
        conversation_context: str | None = None,
        doubt_pattern_context: DoubtPatternContext | None = None,
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
            exam=exam_id,
            exam_stage=exam_stage,
            exam_profile_id=exam_profile_id,
            language=language,
        )

        logger.debug(
            "answer_generation_adapter.generate_stream  request_id=%s  subject=%s  "
            "difficulty=%s  intent=%s  stream_started=true",
            request_id,
            subject,
            difficulty,
            intent,
        )

        stream_kwargs = {
            "route_request": route_request,
            "query": query,
            "context": context_text if context_text else None,
            "conversation_context": conversation_context,
            "on_before_fallback": on_before_generator_fallback,
            "on_before_continuation": on_before_continuation,
            "verify_before_stream": verify_before_stream,
        }
        if doubt_pattern_context is not None:
            stream_kwargs["doubt_pattern_context"] = doubt_pattern_context
        yield from self._orchestrator.generate_stream(**stream_kwargs)
