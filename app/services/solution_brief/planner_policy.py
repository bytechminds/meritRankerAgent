"""Deterministic SolutionBrief gating; no Planner model is configured."""

from __future__ import annotations

from dataclasses import dataclass

from services.context_retrieval.context_models import ContextRetrievalRequest
from services.solution_brief.models import SolutionBrief

# Minimal schema fields an future LLM planner may populate (same as SolutionBrief).
PLANNER_BRIEF_FIELDS: frozenset[str] = frozenset(SolutionBrief.model_fields.keys())


@dataclass(frozen=True)
class PlannerNeedDecision:
    """Internal decision for deterministic context briefing only."""

    use_solution_brief: bool
    reason: str


class PlannerNeedPolicy:
    """Keep deterministic briefing proportional to request complexity.

    This is deliberately not an LLM Planner gate.  The deployed routing
    configuration has no ``planner`` route, prompt, or model alias, so this
    policy must not create an additional model call.
    """

    def decide(
        self,
        request: ContextRetrievalRequest,
        *,
        kb_item_count: int = 0,
        web_item_count: int = 0,
    ) -> PlannerNeedDecision:
        difficulty = request.difficulty.strip().lower()
        source_count = max(kb_item_count, 0) + max(web_item_count, 0)

        if difficulty == "advanced":
            return PlannerNeedDecision(True, "advanced_task")
        if difficulty == "intermediate" and source_count > 0:
            return PlannerNeedDecision(True, "intermediate_grounded_context")
        if difficulty == "default" and source_count > 1:
            return PlannerNeedDecision(True, "multi_source_synthesis")
        if source_count == 0:
            return PlannerNeedDecision(False, "no_grounded_context")
        return PlannerNeedDecision(False, "direct_generator_sufficient")


def should_run_llm_planner(
    request: ContextRetrievalRequest,
    *,
    kb_item_count: int = 0,
    web_item_count: int = 0,
    conflicting_web_sources: bool = False,
    image_ocr_confidence: str | None = None,
) -> bool:
    """Return True only when a future conditional LLM planner should run.

    This patch never calls an LLM planner — the function documents policy and
    supports tests confirming planner stays off by default.
    """
    _ = (
        request,
        kb_item_count,
        web_item_count,
        conflicting_web_sources,
        image_ocr_confidence,
    )
    return False
