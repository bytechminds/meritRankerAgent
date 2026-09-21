"""Central, workload-aware completion-token policy for Practice generation."""

from __future__ import annotations

from dataclasses import dataclass

from schemas.llm_routing import (
    ModelConfig,
    PracticeGenerationCapacity,
    PracticeGenerationWorkload,
    RouteDecision,
)


@dataclass(frozen=True)
class _CapacityBand:
    initial_floor: int
    escalation_floor: int
    product_hard_cap: int
    max_slots_per_batch: int


class PracticeGenerationCapacityPolicy:
    """Resolve one bounded cap plan from broad workload characteristics only.

    The values are product ceilings rather than expected consumption. Provider
    billing continues to use provider-reported usage.
    """

    _BANDS: dict[tuple[str, bool], _CapacityBand] = {
        ("basic", False): _CapacityBand(900, 1500, 2200, 5),
        ("basic", True): _CapacityBand(1600, 2400, 3000, 4),
        ("intermediate", False): _CapacityBand(1600, 2500, 3400, 4),
        ("intermediate", True): _CapacityBand(2600, 3600, 4400, 3),
        ("advanced", False): _CapacityBand(2600, 3600, 4400, 2),
        ("advanced", True): _CapacityBand(4000, 5600, 5600, 2),
    }

    @classmethod
    def resolve(
        cls,
        *,
        route_decision: RouteDecision,
        model_config: ModelConfig,
        workload: PracticeGenerationWorkload,
    ) -> PracticeGenerationCapacity:
        """Return the initial/recovery caps for one exact Practice work unit."""
        if route_decision.task_role != "generator" or route_decision.intent != "practice":
            raise ValueError("Practice capacity is only valid for Practice generators.")

        difficulty = route_decision.difficulty
        if difficulty not in {"basic", "intermediate", "advanced"}:
            difficulty = "basic"
        if route_decision.subject not in {"math", "practice_math", "reasoning"}:
            difficulty = "basic"
        is_reasoning_model = model_config.supports_reasoning
        band = cls._BANDS[(difficulty, is_reasoning_model)]
        band = cls._stabilized_band(
            band, subject=route_decision.subject, difficulty=difficulty,
            is_reasoning_model=is_reasoning_model,
        )
        visible_output_floor = cls._visible_output_floor(
            subject=route_decision.subject,
            complexity=workload.complexity,
            slot_count=workload.slot_count,
        )
        # The reserve records how many completion tokens a model actually spends on
        # hidden reasoning, which is independent of whether the model exposes reasoning
        # controls: gemini-3.7-flash declares no reasoning support yet spends the budget
        # on every call. Gating it on that flag left the reserve unused and over-packed
        # the batch.
        reasoning_reserve = model_config.structured_output_reasoning_reserve_tokens
        model_hard_cap = model_config.model_hard_max_output_tokens
        product_hard_cap = min(band.product_hard_cap, model_hard_cap)
        initial_cap = min(
            max(band.initial_floor, visible_output_floor + reasoning_reserve),
            product_hard_cap,
        )
        escalation_cap = min(
            max(band.escalation_floor, int(initial_cap * 1.35)),
            product_hard_cap,
        )
        if escalation_cap < initial_cap:
            escalation_cap = initial_cap

        max_slots = min(
            band.max_slots_per_batch,
            cls._complexity_slot_cap(workload.complexity),
        )
        if (
            route_decision.subject in {"math", "practice_math", "reasoning"}
            and difficulty == "advanced"
            and workload.complexity == "high"
        ):
            max_slots = 1

        return PracticeGenerationCapacity(
            initial_max_output_tokens=initial_cap,
            escalation_max_output_tokens=escalation_cap,
            product_hard_max_output_tokens=product_hard_cap,
            model_hard_max_output_tokens=model_hard_cap,
            max_slots_per_batch=max_slots,
            reasoning_effort=(
                model_config.reasoning_effort if is_reasoning_model else "none"
            ),
            complexity=workload.complexity,
            slot_count=workload.slot_count,
        )

    @staticmethod
    def _stabilized_band(
        band: _CapacityBand,
        *,
        subject: str,
        difficulty: str,
        is_reasoning_model: bool,
    ) -> _CapacityBand:
        """Start advanced reasoning at the cap that actually completes.

        Every observed advanced-reasoning attempt at the 4000 initial floor
        exhausted output tokens and was immediately retried at 5600, which then
        succeeded. Starting at 5600 removes that known-failing call. It buys no
        extra capacity: the escalation floor and the product hard cap are already
        5600, so the executor's existing "escalation adds nothing" guard now skips
        the second attempt and proceeds straight to the unchanged fallback.

        Scoped to reasoning: math advanced shares this band but uses a different
        provider with a measured reasoning reserve and is deliberately untouched.
        """
        if subject != "reasoning" or difficulty != "advanced" or not is_reasoning_model:
            return band
        return _CapacityBand(
            initial_floor=band.escalation_floor,
            escalation_floor=band.escalation_floor,
            product_hard_cap=band.product_hard_cap,
            max_slots_per_batch=band.max_slots_per_batch,
        )

    @staticmethod
    def _visible_output_floor(*, subject: str, complexity: str, slot_count: int) -> int:
        base = {
            "math": 300,
            "practice_math": 300,
            "reasoning": 310,
            "english": 280,
        }.get(subject, 260)
        complexity_adjustment = {"low": 0, "medium": 50, "high": 160}[complexity]
        return (base + complexity_adjustment) * slot_count + 160

    @staticmethod
    def _complexity_slot_cap(complexity: str) -> int:
        return {"low": 5, "medium": 4, "high": 2}[complexity]
