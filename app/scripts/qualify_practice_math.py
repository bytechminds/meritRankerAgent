#!/usr/bin/env python3
"""Run an opt-in, no-persistence Math author qualification sample.

The harness compares the configured Math author with one existing candidate
alias. It uses the production prompt/schema/slot and qualified Authority paths,
but it never constructs an assessment, invokes repositories, or emits question
text in its public report. Student-credit enforcement must be explicitly
disabled for this standalone local process.

An explicitly gated, mode-0600 private review capture can be requested under
``/private/tmp`` for independent human/deterministic mathematical adjudication.
The post-processing mode reads that private capture and an adjudication file,
then emits an aggregate-only semantic report without making LLM calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from features.practice_generation.schemas import (
        DemandBucket,
        GeneratedQuestion,
        GenerationGroup,
        PlannerSlot,
        PracticeGenerationRequest,
    )
    from services.llm.orchestration.orchestrator import ModelExecutor


_RUN_FLAG = "RUN_PRACTICE_MATH_QUALIFICATION"
_PRIVATE_REVIEW_FLAG = "ALLOW_PRIVATE_MATH_QUALIFICATION_REVIEW"
_PRIVATE_ROOT = Path("/private/tmp").resolve()
_REVIEW_CAPTURE_FORMAT = "practice_math_private_review_v1"
_ADJUDICATION_FORMAT = "practice_math_manual_adjudication_v1"
_SEMANTIC_BUCKETS: frozenset[str] = frozenset(
    {
        "VALID_CORRECT_KEY",
        "VALID_WRONG_KEY_ONLY",
        "NO_VALID_OPTION",
        "MULTIPLE_VALID_OPTIONS",
        "CONTRADICTORY_DATA",
        "AMBIGUOUS",
        "UNSOLVABLE",
        "STRUCTURAL_INVALID",
    }
)


@dataclass(frozen=True)
class QualificationTopic:
    topic: str
    description: str
    difficulty: Literal["basic", "intermediate", "advanced"]


# Five samples per slice produce a 60-question screen spanning the twelve required
# families, with basic, intermediate, and advanced route coverage. The harness still
# permits smaller focused screens through --topics and --samples-per-topic.
_TOPICS: tuple[QualificationTopic, ...] = (
    QualificationTopic("average", "arithmetic mean and weighted-average calculations", "basic"),
    QualificationTopic("percentage", "percentage increase, decrease, and comparison", "basic"),
    QualificationTopic(
        "ratio_proportion", "ratio, proportion, and partnership arithmetic", "basic"
    ),
    QualificationTopic(
        "number_system", "number-system divisibility and remainder calculations", "basic"
    ),
    QualificationTopic(
        "profit_loss", "profit, loss, discount, and markup calculations", "intermediate"
    ),
    QualificationTopic(
        "simple_compound_interest", "simple and compound interest calculations", "intermediate"
    ),
    QualificationTopic("time_and_work", "work-rate and combined-work calculations", "intermediate"),
    QualificationTopic(
        "time_speed_distance", "time, speed, and distance calculations", "intermediate"
    ),
    QualificationTopic(
        "boats_and_streams", "upstream, downstream, and relative-speed calculations", "intermediate"
    ),
    QualificationTopic("algebra", "equations and algebraic simplification", "advanced"),
    QualificationTopic("geometry", "geometry theorem and formula application", "advanced"),
    QualificationTopic("mensuration", "area, volume, and mensuration calculations", "advanced"),
)


@dataclass(frozen=True)
class AuthorityControl:
    """A hand-adjudicated, non-persistent Math Authority control."""

    control_id: str
    topic: str
    difficulty: Literal["basic", "intermediate", "advanced"]
    semantic_bucket: str
    expected_reason_code: str
    stem: str
    options: tuple[str, str, str, str]
    author_option_id: Literal["0", "1", "2", "3"]
    true_option_id: Literal["0", "1", "2", "3"] | None


# These are independently solvable, plausible competitive-exam controls, not generated
# captures. The 30-item mix is 21 normal valid controls (70%), five contradiction
# controls (16.7%), and one each of wrong-key-only, no-valid-option, multiple-valid,
# and ambiguous cases. The public report contains aggregates only.
_AUTHORITY_CONTROLS: tuple[AuthorityControl, ...] = (
    AuthorityControl(
        "average-mean",
        "average",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The arithmetic mean of 12, 18 and 24 is:",
        ("16", "18", "20", "22"),
        "1",
        "1",
    ),
    AuthorityControl(
        "average-even",
        "average",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The average of the first five positive even integers is:",
        ("4", "6", "8", "10"),
        "1",
        "1",
    ),
    AuthorityControl(
        "percentage-discount",
        "percentage",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "A price of Rs. 800 is reduced by 15%. The new price is:",
        ("Rs. 650", "Rs. 680", "Rs. 700", "Rs. 720"),
        "1",
        "1",
    ),
    AuthorityControl(
        "percentage-original",
        "percentage",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "A quantity increases by 20% to become 240. Its original value was:",
        ("180", "200", "220", "240"),
        "1",
        "1",
    ),
    AuthorityControl(
        "profit-loss-profit",
        "profit_loss",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "An article costing Rs. 500 is sold at a profit of 20%. Its selling price is:",
        ("Rs. 550", "Rs. 600", "Rs. 625", "Rs. 650"),
        "1",
        "1",
    ),
    AuthorityControl(
        "profit-loss-discount",
        "profit_loss",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        (
            "An article marked Rs. 800 is sold at a 10% discount. If its cost price is "
            "Rs. 600, the profit percentage is:"
        ),
        ("15%", "18%", "20%", "25%"),
        "2",
        "2",
    ),
    AuthorityControl(
        "ratio-larger",
        "ratio_proportion",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "Two numbers are in the ratio 5:7 and their sum is 144. The larger number is:",
        ("60", "72", "84", "96"),
        "2",
        "2",
    ),
    AuthorityControl(
        "ratio-total",
        "ratio_proportion",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "Two numbers are in the ratio 3:5 and differ by 64. Their sum is:",
        ("192", "224", "256", "288"),
        "2",
        "2",
    ),
    AuthorityControl(
        "simple-interest",
        "simple_compound_interest",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The simple interest on Rs. 2000 at 5% per annum for 3 years is:",
        ("Rs. 250", "Rs. 300", "Rs. 350", "Rs. 400"),
        "1",
        "1",
    ),
    AuthorityControl(
        "compound-interest",
        "simple_compound_interest",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The compound amount on Rs. 2000 at 10% per annum for 2 years is:",
        ("Rs. 2200", "Rs. 2400", "Rs. 2420", "Rs. 2440"),
        "2",
        "2",
    ),
    AuthorityControl(
        "time-work",
        "time_and_work",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        (
            "A can finish a job in 10 days and B can finish it in 15 days. Working "
            "together, they finish it in:"
        ),
        ("5 days", "6 days", "7 days", "8 days"),
        "1",
        "1",
    ),
    AuthorityControl(
        "time-work-fraction",
        "time_and_work",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        (
            "A can finish a job in 12 days and B can finish it in 18 days. Working "
            "together, they finish it in:"
        ),
        ("6 days", "7 days", "36/5 days", "8 days"),
        "2",
        "2",
    ),
    AuthorityControl(
        "time-speed-distance",
        "time_speed_distance",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "A car travels at 60 km/h for 2.5 hours. The distance travelled is:",
        ("120 km", "135 km", "150 km", "180 km"),
        "2",
        "2",
    ),
    AuthorityControl(
        "boats-downstream",
        "boats_and_streams",
        "intermediate",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        (
            "A boat has a speed of 12 km/h in still water and the stream speed is 3 km/h. "
            "Its downstream speed is:"
        ),
        ("12 km/h", "15 km/h", "18 km/h", "9 km/h"),
        "1",
        "1",
    ),
    AuthorityControl(
        "ages-sum",
        "ages",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "A is 5 years older than B. If their present ages add to 31 years, A's age is:",
        ("13 years", "16 years", "18 years", "20 years"),
        "2",
        "2",
    ),
    AuthorityControl(
        "number-remainder",
        "number_system",
        "basic",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The remainder when 1234 is divided by 9 is:",
        ("0", "1", "2", "3"),
        "1",
        "1",
    ),
    AuthorityControl(
        "algebra-linear",
        "algebra",
        "advanced",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "If 3x - 7 = 11, then x is:",
        ("4", "5", "6", "7"),
        "2",
        "2",
    ),
    AuthorityControl(
        "algebra-system",
        "algebra",
        "advanced",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "If x + y = 10 and x - y = 4, then x is:",
        ("5", "6", "7", "8"),
        "2",
        "2",
    ),
    AuthorityControl(
        "geometry-right-triangle",
        "geometry",
        "advanced",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "A right triangle has legs 6 cm and 8 cm. Its hypotenuse is:",
        ("9 cm", "10 cm", "11 cm", "12 cm"),
        "1",
        "1",
    ),
    AuthorityControl(
        "geometry-area",
        "geometry",
        "advanced",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "The area of a triangle with base 10 cm and height 6 cm is:",
        ("25 sq cm", "30 sq cm", "35 sq cm", "40 sq cm"),
        "1",
        "1",
    ),
    AuthorityControl(
        "mensuration-circumference",
        "mensuration",
        "advanced",
        "VALID_CORRECT_KEY",
        "SINGLE_VALID_OPTION",
        "Using pi = 22/7, the circumference of a circle of radius 7 cm is:",
        ("22 cm", "36 cm", "44 cm", "49 cm"),
        "2",
        "2",
    ),
    AuthorityControl(
        "wrong-key-average",
        "average",
        "basic",
        "VALID_WRONG_KEY_ONLY",
        "SINGLE_VALID_OPTION",
        "The average of 40, 50, 60 and 70 is:",
        ("45", "50", "55", "60"),
        "0",
        "2",
    ),
    AuthorityControl(
        "contradiction-percentage",
        "percentage",
        "basic",
        "CONTRADICTORY_DATA",
        "CONTRADICTORY_DATA",
        (
            "A price after a 20% discount is Rs. 800, and the discount amount is Rs. 100. "
            "The original price is:"
        ),
        ("Rs. 900", "Rs. 1000", "Rs. 1100", "Rs. 1200"),
        "1",
        None,
    ),
    AuthorityControl(
        "contradiction-interest",
        "simple_compound_interest",
        "intermediate",
        "CONTRADICTORY_DATA",
        "CONTRADICTORY_DATA",
        (
            "For simple interest, principal is Rs. 1000, rate is 10% per annum, time is "
            "2 years, and the amount is Rs. 1100. The simple interest is:"
        ),
        ("Rs. 100", "Rs. 150", "Rs. 200", "Rs. 250"),
        "2",
        None,
    ),
    AuthorityControl(
        "contradiction-work",
        "time_and_work",
        "intermediate",
        "CONTRADICTORY_DATA",
        "CONTRADICTORY_DATA",
        (
            "A finishes a job in 10 days and B in 15 days. They work together without "
            "interruption and complete it in exactly 3 days. Their combined completion time is:"
        ),
        ("3 days", "5 days", "6 days", "7 days"),
        "0",
        None,
    ),
    AuthorityControl(
        "contradiction-algebra",
        "algebra",
        "advanced",
        "CONTRADICTORY_DATA",
        "CONTRADICTORY_DATA",
        "x + y = 10, x - y = 4, and x = 8. The value of y is:",
        ("1", "2", "3", "4"),
        "1",
        None,
    ),
    AuthorityControl(
        "contradiction-geometry",
        "geometry",
        "advanced",
        "CONTRADICTORY_DATA",
        "CONTRADICTORY_DATA",
        "A triangle has side lengths 3 cm, 4 cm and 10 cm. Its perimeter is:",
        ("15 cm", "16 cm", "17 cm", "18 cm"),
        "0",
        None,
    ),
    AuthorityControl(
        "no-valid-profit",
        "profit_loss",
        "intermediate",
        "NO_VALID_OPTION",
        "NO_VALID_OPTION",
        "An article costing Rs. 500 is sold at a profit of 20%. Its selling price is:",
        ("Rs. 550", "Rs. 575", "Rs. 625", "Rs. 650"),
        "0",
        None,
    ),
    AuthorityControl(
        "multiple-valid-ratio",
        "ratio_proportion",
        "basic",
        "MULTIPLE_VALID_OPTIONS",
        "MULTIPLE_VALID_OPTIONS",
        "Which option is equivalent to the ratio 3:6?",
        ("1:2", "2:4", "3:6", "3:5"),
        "0",
        None,
    ),
    AuthorityControl(
        "ambiguous-boats",
        "boats_and_streams",
        "intermediate",
        "AMBIGUOUS",
        "AMBIGUOUS",
        (
            "A boat's downstream speed is 12 km/h. The stream may be 2 km/h or 4 km/h. "
            "Its speed in still water is:"
        ),
        ("8 km/h", "9 km/h", "10 km/h", "11 km/h"),
        "0",
        None,
    ),
)


@dataclass(frozen=True)
class CandidateOutcome:
    run_id: str
    arm: str
    topic: str
    difficulty: str
    outcome: str
    model: str | None = None
    finish_reason: str | None = None
    verifier_decision: str | None = None
    authority_valid_option_ids: tuple[str, ...] = ()
    verifier_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthorityControlEvaluation:
    control: AuthorityControl
    outcome: CandidateOutcome


@dataclass(frozen=True)
class CandidateEvaluation:
    outcome: CandidateOutcome
    question: GeneratedQuestion | None = None


@dataclass(frozen=True)
class ExecutionMetric:
    route_id: str
    task_role: str
    configured_model: str
    observed_model: str
    provider: str | None
    finish_reason: str | None
    fallback_used: bool
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int | None
    estimated_cost_usd: float | None


class _RecordingGeneratorOverride:
    """Optionally replace the author alias and retain safe per-call metering only."""

    def __init__(
        self,
        delegate: ModelExecutor,
        *,
        alias: str | None,
        model_configs: dict[str, Any],
        pricing_config: Any,
    ) -> None:
        self._delegate = delegate
        self._alias = alias
        self._model_configs = model_configs
        self._pricing_config = pricing_config
        self.metrics: list[ExecutionMetric] = []

    def _target_route(self, route_decision: Any) -> Any:
        if self._alias is not None and route_decision.task_role == "generator":
            return route_decision.model_copy(update={"model": self._alias})
        return route_decision

    def _estimated_cost(self, *, target: Any, result: Any) -> float | None:
        from schemas.llm_usage import ProviderTokenUsage
        from services.llm.pricing import estimate_cost_usd, find_model_pricing

        config = self._model_configs.get(result.model) or self._model_configs.get(target.model)
        if config is None:
            return None
        pricing = find_model_pricing(
            self._pricing_config,
            provider=result.provider or config.provider,
            model=config.model_id or "",
            deployment=config.deployment,
        )
        if pricing is None:
            return None
        return estimate_cost_usd(
            ProviderTokenUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                cached_input_tokens=result.cached_input_tokens,
                reasoning_tokens=result.reasoning_tokens,
            ),
            pricing,
        )

    def _record(self, *, target: Any, result: Any) -> None:
        self.metrics.append(
            ExecutionMetric(
                route_id=target.route_id,
                task_role=target.task_role,
                configured_model=target.model,
                observed_model=result.model,
                provider=result.provider,
                finish_reason=result.finish_reason,
                fallback_used=result.fallback_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_input_tokens=result.cached_input_tokens,
                reasoning_tokens=result.reasoning_tokens,
                latency_ms=result.latency_ms,
                estimated_cost_usd=self._estimated_cost(target=target, result=result),
            )
        )

    def execute(
        self,
        *,
        route_decision: Any,
        messages: list[Any],
        attempt_guard: Any = None,
    ) -> Any:
        target = self._target_route(route_decision)
        result = self._delegate.execute(
            route_decision=target,
            messages=messages,
            attempt_guard=attempt_guard,
        )
        self._record(target=target, result=result)
        return result

    def execute_stream(self, *, route_decision: Any, messages: list[Any]) -> Any:
        target = self._target_route(route_decision)
        return self._delegate.execute_stream(route_decision=target, messages=messages)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-model",
        default="openai_gpt_5_6_terra",
        help="Existing registry alias to substitute for the generator only.",
    )
    parser.add_argument(
        "--samples-per-topic",
        default=4,
        type=int,
        help="Questions generated in one bounded group per topic (1..5).",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated canonical topic IDs; defaults to all qualification topics.",
    )
    parser.add_argument(
        "--arms",
        default="current,candidate",
        help="Comma-separated arms to run: current,candidate (default: both).",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional new aggregate-only JSON path below /private/tmp.",
    )
    parser.add_argument(
        "--review-capture-path",
        default=None,
        help=(
            "Optional new private review JSON path below /private/tmp. Requires "
            "ALLOW_PRIVATE_MATH_QUALIFICATION_REVIEW=true and is never printed."
        ),
    )
    parser.add_argument(
        "--summarize-adjudication",
        action="store_true",
        help="Read a private review capture and manual adjudication file without LLM calls.",
    )
    parser.add_argument(
        "--reverify-unavailable",
        action="store_true",
        help=(
            "Recheck only unavailable Authority results from private review captures using the "
            "normal subject-qualified verifier."
        ),
    )
    parser.add_argument(
        "--adjudication-path",
        default=None,
        help="Existing private manual-adjudication JSON path below /private/tmp.",
    )
    parser.add_argument(
        "--reverified-capture-path",
        default=None,
        help=("New private merged review JSON path below /private/tmp for --reverify-unavailable."),
    )
    parser.add_argument(
        "--authority-controls",
        action="store_true",
        help=(
            "Run the fixed, no-persistence 30-item Math Authority safety corpus. "
            "It calls only the normally qualified Math verifier."
        ),
    )
    return parser.parse_args()


def _selected_topics(raw_topics: str | None) -> tuple[QualificationTopic, ...]:
    if raw_topics is None:
        return _TOPICS
    requested = tuple(
        value.strip().casefold().replace("-", "_")
        for value in raw_topics.split(",")
        if value.strip()
    )
    available = {item.topic: item for item in _TOPICS}
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unsupported qualification topics: {', '.join(unknown)}")
    if not requested:
        raise ValueError("At least one qualification topic is required.")
    return tuple(available[topic] for topic in requested)


def _selected_arms(raw_arms: str) -> tuple[Literal["current", "candidate"], ...]:
    requested = tuple(value.strip().casefold() for value in raw_arms.split(",") if value.strip())
    unknown = sorted(set(requested).difference({"current", "candidate"}))
    if unknown:
        raise ValueError(f"Unsupported qualification arms: {', '.join(unknown)}")
    if not requested:
        raise ValueError("At least one qualification arm is required.")
    if len(set(requested)) != len(requested):
        raise ValueError("Qualification arms must not repeat.")
    return tuple(requested)  # type: ignore[return-value]


def _build_request(
    *,
    run_id: str,
    topic: str,
    difficulty: str,
    expected_count: int,
) -> PracticeGenerationRequest:
    from features.practice_generation.schemas import PracticeGenerationRequest

    return PracticeGenerationRequest(
        request_id=f"math-qualification-{run_id}-{topic}",
        user_id="qualification-runner",
        conversation_id=f"qualification-{run_id}",
        turn_id=f"qualification-{topic}",
        original_query=f"Create {difficulty.title()} Math practice for {topic}.",
        practice_type="QUICK_PRACTICE",
        requested_count=expected_count,
        accepted_count=expected_count,
        subject="math",
        topic=topic,
        difficulty=difficulty,
        explicit_difficulty_requested=True,
        language="english",
        include_solutions=False,
        assessment_title=f"Math qualification: {topic} ({difficulty})",
    )


def _build_bucket(
    *,
    topic: str,
    description: str,
    difficulty: str,
    expected_count: int,
) -> DemandBucket:
    from features.practice_generation.schemas import DemandBucket

    return DemandBucket(
        bucket_id=f"math-{topic}",
        subject="math",
        topic=topic,
        difficulty=difficulty,
        question_type="mcq",
        required_count=expected_count,
        keywords=[topic],
        question_intent=description,
        verification_policy="MANDATORY",
        solution_required=False,
        generation_group_hint=expected_count,
    )


def _build_slots(
    *,
    topic: str,
    description: str,
    difficulty: str,
    count: int,
) -> tuple[PlannerSlot, ...]:
    from features.practice_generation.schemas import PlannerSlot

    return tuple(
        PlannerSlot(
            slot_id=f"slot-{index:03d}",
            subject_id="math",
            topic_id=topic,
            category_id=topic,
            difficulty=difficulty,
            complexity="medium",
            exam_ids=["QUALIFICATION"],
            question_type="mcq",
            target_skill=description,
            variation_hint=(
                f"Use distinct values and answer forms for {difficulty} "
                f"qualification sample {index}."
            ),
            generator_route_hint="math.generator.intermediate",
            reasoning_target="Show internally consistent arithmetic and one valid option.",
            not_same_when=["numbers", "story_context"],
            generation_group_hint=count,
        )
        for index in range(1, count + 1)
    )


def _build_group(*, topic: str, slots: tuple[PlannerSlot, ...]) -> GenerationGroup:
    from features.practice_generation.schemas import GenerationGroup

    return GenerationGroup(
        group_id=f"qualification-{topic}",
        bucket_id=f"math-{topic}",
        required_count=len(slots),
        slot_ids=[slot.slot_id for slot in slots],
    )


def _outcome_for_question(
    *,
    run_id: str,
    verifier: Any,
    request: PracticeGenerationRequest,
    bucket: DemandBucket,
    slot: PlannerSlot,
    question: GeneratedQuestion,
    arm: str,
    model: str,
    finish_reason: str | None,
) -> CandidateOutcome:
    try:
        verdict = verifier.verify_slot(
            request=request,
            bucket=bucket,
            slot=slot,
            question=question,
        )
    except Exception as exc:  # noqa: BLE001 - aggregate only, never expose provider details.
        return CandidateOutcome(
            run_id=run_id,
            arm=arm,
            topic=bucket.topic,
            difficulty=bucket.difficulty.value,
            outcome=f"verifier_error:{type(exc).__name__}",
            model=model,
            finish_reason=finish_reason,
        )

    valid_ids = tuple(verdict.valid_option_ids)
    if verdict.decision != "ACCEPT" or len(valid_ids) != 1:
        outcome = "authority_rejected"
    elif valid_ids[0] == question.correct_option_id:
        outcome = "authority_agreement"
    else:
        outcome = "authority_key_mismatch"
    return CandidateOutcome(
        run_id=run_id,
        arm=arm,
        topic=bucket.topic,
        difficulty=bucket.difficulty.value,
        outcome=outcome,
        model=model,
        finish_reason=finish_reason,
        verifier_decision=str(verdict.decision),
        authority_valid_option_ids=valid_ids,
        verifier_reason_codes=tuple(
            str(code) for code in (getattr(verdict, "reason_codes", ()) or ())
        ),
    )


def _evaluate_arm(
    *,
    arm: str,
    generator: Any,
    verifier: Any,
    run_id: str,
    topics: tuple[QualificationTopic, ...],
    samples_per_topic: int,
) -> list[CandidateEvaluation]:
    from features.practice_generation.schemas import GenerationEnvelope

    evaluations: list[CandidateEvaluation] = []
    for topic_spec in topics:
        topic = topic_spec.topic
        description = topic_spec.description
        difficulty = topic_spec.difficulty
        request = _build_request(
            run_id=run_id,
            topic=topic,
            difficulty=difficulty,
            expected_count=samples_per_topic,
        )
        bucket = _build_bucket(
            topic=topic,
            description=description,
            difficulty=difficulty,
            expected_count=samples_per_topic,
        )
        slots = _build_slots(
            topic=topic,
            description=description,
            difficulty=difficulty,
            count=samples_per_topic,
        )
        group = _build_group(topic=topic, slots=slots)
        try:
            batch = generator.generate_slots(
                request=request,
                bucket=bucket,
                group=group,
                slots=slots,
                exclude_normalized_texts=(),
                replacement_wave=0,
            )
            envelope = GenerationEnvelope.model_validate_json(batch.content)
        except Exception as exc:  # noqa: BLE001 - aggregate only, never expose provider details.
            evaluations.extend(
                CandidateEvaluation(
                    outcome=CandidateOutcome(
                        run_id=run_id,
                        arm=arm,
                        topic=topic,
                        difficulty=difficulty,
                        outcome=f"generator_error:{type(exc).__name__}",
                    )
                )
                for _ in slots
            )
            continue

        by_slot = {question.slot_id: question for question in envelope.questions}
        for slot in slots:
            question = by_slot.get(slot.slot_id)
            if question is None:
                evaluations.append(
                    CandidateEvaluation(
                        outcome=CandidateOutcome(
                            run_id=run_id,
                            arm=arm,
                            topic=topic,
                            difficulty=difficulty,
                            outcome="missing_or_unbound_generated_item",
                            model=batch.model,
                            finish_reason=batch.finish_reason,
                        )
                    )
                )
                continue
            evaluations.append(
                CandidateEvaluation(
                    outcome=_outcome_for_question(
                        run_id=run_id,
                        verifier=verifier,
                        request=request,
                        bucket=bucket,
                        slot=slot,
                        question=question,
                        arm=arm,
                        model=batch.model,
                        finish_reason=batch.finish_reason,
                    ),
                    question=question,
                )
            )
    return evaluations


def _authority_control_question(
    *,
    control: AuthorityControl,
    slot: PlannerSlot,
) -> GeneratedQuestion:
    from features.practice_generation.schemas import GeneratedQuestion

    return GeneratedQuestion.model_validate(
        {
            "schema_version": "2",
            "generation_item_id": f"authority-control-{control.control_id}",
            "bucket_id": f"authority-{control.topic}",
            "slot_id": slot.slot_id,
            "question": control.stem,
            "question_type": "mcq",
            "options": [
                {"option_id": str(index), "value": value}
                for index, value in enumerate(control.options)
            ],
            "correct_option_id": control.author_option_id,
            "subject": "math",
            "topic": control.topic,
            "difficulty": control.difficulty,
        }
    )


def _evaluate_authority_controls(
    *,
    verifier: Any,
    run_id: str,
) -> list[AuthorityControlEvaluation]:
    evaluations: list[AuthorityControlEvaluation] = []
    for control in _AUTHORITY_CONTROLS:
        request = _build_request(
            run_id=run_id,
            topic=control.topic,
            difficulty=control.difficulty,
            expected_count=1,
        )
        bucket = _build_bucket(
            topic=control.topic,
            description="Independent Authority safety control.",
            difficulty=control.difficulty,
            expected_count=1,
        )
        slot = _build_slots(
            topic=control.topic,
            description="Independent Authority safety control.",
            difficulty=control.difficulty,
            count=1,
        )[0]
        question = _authority_control_question(control=control, slot=slot)
        evaluations.append(
            AuthorityControlEvaluation(
                control=control,
                outcome=_outcome_for_question(
                    run_id=run_id,
                    verifier=verifier,
                    request=request,
                    bucket=bucket,
                    slot=slot,
                    question=question,
                    arm="authority_control",
                    model="static_authority_control",
                    finish_reason=None,
                ),
            )
        )
    return evaluations


def _authority_control_report(
    *,
    evaluations: list[AuthorityControlEvaluation],
    metrics: list[ExecutionMetric],
) -> dict[str, object]:
    confusion: Counter[str] = Counter()
    reason_metrics: dict[str, Counter[str]] = {}
    topic_outcomes: dict[str, Counter[str]] = {}
    valid_control_count = 0
    responded_valid_control_count = 0
    contradiction_controls = 0
    contradiction_rejections = 0
    contradiction_reason_matches = 0

    for evaluation in evaluations:
        control = evaluation.control
        outcome = evaluation.outcome
        valid_control = control.true_option_id is not None
        accepted = (
            outcome.verifier_decision == "ACCEPT" and len(outcome.authority_valid_option_ids) == 1
        )
        unavailable = outcome.outcome.startswith("verifier_error:")
        topic_counts = topic_outcomes.setdefault(control.topic, Counter())
        if valid_control:
            valid_control_count += 1
        if unavailable:
            confusion["technical_unavailable"] += 1
            topic_counts["technical_unavailable"] += 1
        elif valid_control:
            responded_valid_control_count += 1
            if accepted and outcome.authority_valid_option_ids[0] == control.true_option_id:
                confusion["true_approvals"] += 1
                topic_counts["true_approvals"] += 1
            else:
                confusion["false_rejections"] += 1
                topic_counts["false_rejections"] += 1
        elif accepted:
            confusion["false_approvals"] += 1
            topic_counts["false_approvals"] += 1
        else:
            confusion["true_rejections"] += 1
            topic_counts["true_rejections"] += 1

        if control.true_option_id is None:
            metric = reason_metrics.setdefault(control.semantic_bucket, Counter())
            metric["control_count"] += 1
            if not unavailable and not accepted:
                metric["rejected_count"] += 1
            if control.expected_reason_code in outcome.verifier_reason_codes:
                metric["expected_reason_count"] += 1
        if control.semantic_bucket == "CONTRADICTORY_DATA":
            contradiction_controls += 1
            if not unavailable and not accepted:
                contradiction_rejections += 1
            if control.expected_reason_code in outcome.verifier_reason_codes:
                contradiction_reason_matches += 1

    false_rejection_rate = (
        round(confusion["false_rejections"] / responded_valid_control_count, 4)
        if responded_valid_control_count
        else None
    )
    decision = "MATH_AUTHORITY_LAUNCH_SAFETY_VALIDATED"
    if confusion["technical_unavailable"]:
        decision = "NOT_ELIGIBLE_PENDING_AUTHORITY_COMPLETENESS"
    elif confusion["false_approvals"]:
        decision = "NOT_ELIGIBLE_FALSE_APPROVAL"
    elif contradiction_rejections != contradiction_controls:
        decision = "NOT_ELIGIBLE_CONTRADICTION_CONTROL_FAILURE"
    elif false_rejection_rate is None or false_rejection_rate > 0.10:
        decision = "NOT_ELIGIBLE_FALSE_REJECTION_RATE"

    return {
        "run_kind": "local_no_persistence_practice_math_authority_controls",
        "student_credit_enforcement": "disabled",
        "production_route_changed": False,
        "authority": "normal_subject_qualified_route",
        "control_count": len(evaluations),
        "composition": {
            "normal_valid_count": sum(
                control.semantic_bucket == "VALID_CORRECT_KEY" for control in _AUTHORITY_CONTROLS
            ),
            "wrong_key_only_count": sum(
                control.semantic_bucket == "VALID_WRONG_KEY_ONLY" for control in _AUTHORITY_CONTROLS
            ),
            "contradictory_data_count": contradiction_controls,
            "no_valid_option_count": sum(
                control.semantic_bucket == "NO_VALID_OPTION" for control in _AUTHORITY_CONTROLS
            ),
            "multiple_valid_options_count": sum(
                control.semantic_bucket == "MULTIPLE_VALID_OPTIONS"
                for control in _AUTHORITY_CONTROLS
            ),
            "ambiguous_count": sum(
                control.semantic_bucket == "AMBIGUOUS" for control in _AUTHORITY_CONTROLS
            ),
        },
        "authority_confusion": {
            "true_approvals": confusion["true_approvals"],
            "true_rejections": confusion["true_rejections"],
            "false_approvals": confusion["false_approvals"],
            "false_rejections": confusion["false_rejections"],
            "technical_unavailable": confusion["technical_unavailable"],
            "false_rejection_rate": false_rejection_rate,
            "valid_control_count": valid_control_count,
        },
        "semantic_reason_metrics": {
            bucket: {
                "control_count": counts["control_count"],
                "rejected_count": counts["rejected_count"],
                "expected_reason_count": counts["expected_reason_count"],
            }
            for bucket, counts in sorted(reason_metrics.items())
        },
        "contradiction_controls": {
            "control_count": contradiction_controls,
            "rejected_count": contradiction_rejections,
            "expected_reason_count": contradiction_reason_matches,
        },
        "topic_outcomes": {
            topic: dict(sorted(counts.items())) for topic, counts in sorted(topic_outcomes.items())
        },
        "execution_metrics": _summarize_metrics(metrics),
        "launch_safety_decision": decision,
    }


def _summarize(evaluations: list[CandidateEvaluation]) -> dict[str, object]:
    outcomes = [evaluation.outcome for evaluation in evaluations]
    counts = Counter(outcome.outcome for outcome in outcomes)
    total = len(outcomes)
    agreement = counts["authority_agreement"]
    return {
        "candidate_count": total,
        "authority_agreement_count": agreement,
        "authority_agreement_rate": round(agreement / total, 4) if total else 0.0,
        "outcomes": dict(sorted(counts.items())),
        "models_observed": sorted(
            {outcome.model for outcome in outcomes if outcome.model is not None}
        ),
        "finish_reasons": dict(
            sorted(
                Counter(
                    outcome.finish_reason for outcome in outcomes if outcome.finish_reason
                ).items()
            )
        ),
        "topic_outcomes": {
            topic: dict(
                sorted(
                    Counter(
                        outcome.outcome for outcome in outcomes if outcome.topic == topic
                    ).items()
                )
            )
            for topic in sorted({outcome.topic for outcome in outcomes})
        },
        "difficulty_outcomes": {
            difficulty: dict(
                sorted(
                    Counter(
                        outcome.outcome for outcome in outcomes if outcome.difficulty == difficulty
                    ).items()
                )
            )
            for difficulty in sorted({outcome.difficulty for outcome in outcomes})
        },
    }


def _validate_private_path(raw_path: str, *, require_new: bool) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_relative_to(_PRIVATE_ROOT):
        raise ValueError("private qualification paths must be below /private/tmp.")
    if path == _PRIVATE_ROOT or not path.parent.is_dir():
        raise ValueError("private qualification path parent must already exist.")
    if require_new and path.exists():
        raise ValueError("private qualification output path must not already exist.")
    if not require_new and not path.is_file():
        raise ValueError("private qualification input path must be an existing file.")
    return path


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    file_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, separators=(",", ":"), sort_keys=True)
        output_file.write("\n")


def _read_private_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("private qualification JSON could not be read.") from exc
    if not isinstance(raw, dict):
        raise ValueError("private qualification JSON must be an object.")
    return raw


def _combined_review_capture(paths: tuple[Path, ...]) -> dict[str, object]:
    records: list[object] = []
    for path in paths:
        capture = _read_private_json(path)
        if capture.get("format") != _REVIEW_CAPTURE_FORMAT:
            raise ValueError("private review capture format is not supported.")
        raw_records = capture.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("private review capture records must be a list.")
        records.extend(raw_records)
    return {"format": _REVIEW_CAPTURE_FORMAT, "records": records}


def _review_id(evaluation: CandidateEvaluation) -> str | None:
    question = evaluation.question
    if question is None or question.slot_id is None:
        return None
    outcome = evaluation.outcome
    return ":".join(
        (
            outcome.run_id,
            outcome.arm,
            outcome.topic,
            outcome.difficulty,
            question.slot_id,
        )
    )


def _private_review_payload(
    *,
    run_id: str,
    candidate_model_alias: str,
    evaluations_by_arm: dict[str, list[CandidateEvaluation]],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for arm, evaluations in sorted(evaluations_by_arm.items()):
        for evaluation in evaluations:
            review_id = _review_id(evaluation)
            if review_id is None or evaluation.question is None:
                continue
            outcome = evaluation.outcome
            question = evaluation.question
            records.append(
                {
                    "review_id": review_id,
                    "arm": arm,
                    "topic": outcome.topic,
                    "difficulty": outcome.difficulty,
                    "model": outcome.model,
                    "question": {
                        "schema_version": question.schema_version,
                        "generation_item_id": question.generation_item_id,
                        "slot_id": question.slot_id,
                        "stem": question.question,
                        "options": [
                            {"option_id": option.option_id, "value": option.value}
                            for option in question.canonical_options
                        ],
                        "generator_correct_option_id": question.correct_option_id,
                    },
                    "math_authority": {
                        "status": (
                            "unavailable"
                            if outcome.outcome.startswith("verifier_error:")
                            else "responded"
                        ),
                        "decision": outcome.verifier_decision,
                        "valid_option_ids": list(outcome.authority_valid_option_ids),
                    },
                    "qualification_outcome": outcome.outcome,
                }
            )
    return {
        "format": _REVIEW_CAPTURE_FORMAT,
        "run_id": run_id,
        "candidate_model_alias": candidate_model_alias,
        "records": records,
    }


def _authority_status(authority: object) -> Literal["responded", "unavailable"]:
    if not isinstance(authority, dict):
        raise ValueError("private review record is missing authority data.")
    status = authority.get("status")
    if status in {"responded", "unavailable"}:
        return status
    if status is not None:
        raise ValueError("private review authority status is invalid.")
    return "responded" if authority.get("decision") is not None else "unavailable"


def _authority_payload_from_outcome(outcome: CandidateOutcome) -> dict[str, object]:
    return {
        "status": ("unavailable" if outcome.outcome.startswith("verifier_error:") else "responded"),
        "decision": outcome.verifier_decision,
        "valid_option_ids": list(outcome.authority_valid_option_ids),
    }


def _reverification_topic(topic: str) -> QualificationTopic:
    for candidate in _TOPICS:
        if candidate.topic == topic:
            return candidate
    raise ValueError("private review record has an unsupported Math topic.")


def _reverification_slot(*, topic: QualificationTopic, slot_id: str) -> PlannerSlot:
    for slot in _build_slots(
        topic=topic.topic,
        description=topic.description,
        difficulty=topic.difficulty,
        count=5,
    ):
        if slot.slot_id == slot_id:
            return slot
    raise ValueError("private review record has an unsupported slot ID.")


def _question_for_reverification(
    *,
    raw_question: object,
    topic: QualificationTopic,
) -> GeneratedQuestion:
    from features.practice_generation.schemas import GeneratedQuestion

    if not isinstance(raw_question, dict):
        raise ValueError("private review record is missing question data.")
    raw_options = raw_question.get("options")
    correct_option_id = raw_question.get("generator_correct_option_id")
    if not isinstance(raw_options, list) or correct_option_id not in {"0", "1", "2", "3"}:
        raise ValueError("private review record has an invalid question contract.")
    matching_option = next(
        (
            option.get("value")
            for option in raw_options
            if isinstance(option, dict) and option.get("option_id") == correct_option_id
        ),
        None,
    )
    if not isinstance(matching_option, str):
        raise ValueError("private review record has no generator-key option.")
    return GeneratedQuestion.model_validate(
        {
            "schema_version": raw_question.get("schema_version"),
            "generation_item_id": raw_question.get("generation_item_id"),
            "bucket_id": f"math-{topic.topic}",
            "slot_id": raw_question.get("slot_id"),
            "question": raw_question.get("stem"),
            "question_type": "mcq",
            "options": raw_options,
            "correct_option_id": correct_option_id,
            "correct_answer": matching_option,
            "subject": "math",
            "topic": topic.topic,
            "difficulty": topic.difficulty,
        }
    )


def _reverify_unavailable_capture(
    *,
    capture: dict[str, object],
    verifier: Any,
    run_id: str,
) -> tuple[dict[str, object], list[CandidateOutcome]]:
    if capture.get("format") != _REVIEW_CAPTURE_FORMAT:
        raise ValueError("private review capture format is not supported.")
    raw_records = capture.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("private review capture records must be a list.")

    rechecked_outcomes: list[CandidateOutcome] = []
    refreshed_records: list[dict[str, object]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("private review record is invalid.")
        record = dict(raw_record)
        authority = record.get("math_authority")
        if _authority_status(authority) == "responded":
            refreshed_records.append(record)
            continue
        arm = record.get("arm")
        topic_id = record.get("topic")
        difficulty = record.get("difficulty")
        model = record.get("model")
        raw_question = record.get("question")
        if (
            not isinstance(arm, str)
            or not isinstance(topic_id, str)
            or not isinstance(difficulty, str)
            or not isinstance(model, str)
        ):
            raise ValueError("private review record is missing re-verification metadata.")
        topic = _reverification_topic(topic_id)
        if difficulty != topic.difficulty:
            raise ValueError("private review record has an invalid difficulty.")
        try:
            question = _question_for_reverification(raw_question=raw_question, topic=topic)
            if question.slot_id is None:
                raise ValueError("private review record is missing a slot ID.")
            slot = _reverification_slot(topic=topic, slot_id=question.slot_id)
            request = _build_request(
                run_id=run_id,
                topic=topic.topic,
                difficulty=topic.difficulty,
                expected_count=1,
            )
            bucket = _build_bucket(
                topic=topic.topic,
                description=topic.description,
                difficulty=topic.difficulty,
                expected_count=1,
            )
            outcome = _outcome_for_question(
                run_id=run_id,
                verifier=verifier,
                request=request,
                bucket=bucket,
                slot=slot,
                question=question,
                arm=arm,
                model=model,
                finish_reason=None,
            )
        except ValueError:
            outcome = CandidateOutcome(
                run_id=run_id,
                arm=arm,
                topic=topic.topic,
                difficulty=topic.difficulty,
                outcome="verifier_error:InputValidationError",
                model=model,
            )
        record["math_authority"] = _authority_payload_from_outcome(outcome)
        record["qualification_outcome"] = outcome.outcome
        refreshed_records.append(record)
        rechecked_outcomes.append(outcome)
    return {
        "format": _REVIEW_CAPTURE_FORMAT,
        "records": refreshed_records,
    }, rechecked_outcomes


def _reverification_report(
    *,
    outcomes: list[CandidateOutcome],
    metrics: list[ExecutionMetric],
) -> dict[str, object]:
    return {
        "run_kind": "local_no_persistence_practice_math_authority_reverification",
        "student_credit_enforcement": "disabled",
        "production_route_changed": False,
        "authority": "normal_subject_qualified_route",
        "reviewed_unavailable_count": len(outcomes),
        "outcomes": dict(sorted(Counter(outcome.outcome for outcome in outcomes).items())),
        "execution_metrics": _summarize_metrics(metrics),
        "private_review_capture": "created",
    }


def _percentile_95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return ordered[index]


def _summarize_metrics(metrics: list[ExecutionMetric]) -> dict[str, object]:
    total_input = sum(metric.input_tokens or 0 for metric in metrics)
    total_output = sum(metric.output_tokens or 0 for metric in metrics)
    missing_usage = sum(
        metric.input_tokens is None or metric.output_tokens is None for metric in metrics
    )
    missing_cost = sum(metric.estimated_cost_usd is None for metric in metrics)
    latency_values = [metric.latency_ms for metric in metrics if metric.latency_ms is not None]
    return {
        "call_count": len(metrics),
        "generator_call_count": sum(metric.task_role == "generator" for metric in metrics),
        "verifier_call_count": sum(metric.task_role == "verifier" for metric in metrics),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "missing_usage_call_count": missing_usage,
        "estimated_cost_usd": (
            round(sum(metric.estimated_cost_usd or 0.0 for metric in metrics), 12)
            if missing_cost == 0
            else None
        ),
        "missing_cost_call_count": missing_cost,
        "latency_ms": {
            "mean": round(sum(latency_values) / len(latency_values), 2) if latency_values else None,
            "p95": _percentile_95(latency_values),
        },
        "observed_models": sorted({metric.observed_model for metric in metrics}),
        "configured_routes": sorted({metric.route_id for metric in metrics}),
        "fallback_used_call_count": sum(metric.fallback_used for metric in metrics),
    }


def _manual_adjudication_report(
    *,
    capture: dict[str, object],
    adjudication: dict[str, object],
) -> dict[str, object]:
    if capture.get("format") != _REVIEW_CAPTURE_FORMAT:
        raise ValueError("private review capture format is not supported.")
    if adjudication.get("format") != _ADJUDICATION_FORMAT:
        raise ValueError("manual adjudication format is not supported.")
    capture_records = capture.get("records")
    adjudication_records = adjudication.get("records")
    if not isinstance(capture_records, list) or not isinstance(adjudication_records, list):
        raise ValueError("private review and adjudication records must be lists.")

    records_by_id: dict[str, dict[str, object]] = {}
    for raw_record in capture_records:
        if not isinstance(raw_record, dict) or not isinstance(raw_record.get("review_id"), str):
            raise ValueError("private review record is invalid.")
        review_id = raw_record["review_id"]
        if review_id in records_by_id:
            raise ValueError("private review contains duplicate review IDs.")
        records_by_id[review_id] = raw_record

    classifications: dict[str, dict[str, object]] = {}
    for raw_entry in adjudication_records:
        if not isinstance(raw_entry, dict):
            raise ValueError("manual adjudication record is invalid.")
        review_id = raw_entry.get("review_id")
        bucket = raw_entry.get("primary_bucket")
        true_option_id = raw_entry.get("true_option_id")
        if not isinstance(review_id, str) or review_id not in records_by_id:
            raise ValueError("manual adjudication refers to an unknown review ID.")
        if review_id in classifications:
            raise ValueError("manual adjudication contains duplicate review IDs.")
        if not isinstance(bucket, str) or bucket not in _SEMANTIC_BUCKETS:
            raise ValueError("manual adjudication primary_bucket is invalid.")
        if bucket.startswith("VALID_"):
            if true_option_id not in {"0", "1", "2", "3"}:
                raise ValueError("valid manual adjudications require true_option_id.")
        elif true_option_id is not None:
            raise ValueError("invalid manual adjudications must not include true_option_id.")
        classifications[review_id] = {
            "primary_bucket": bucket,
            "true_option_id": true_option_id,
        }

    if set(classifications) != set(records_by_id):
        raise ValueError("manual adjudication must classify every private review record.")

    semantic_counts_by_arm: dict[str, Counter[str]] = {}
    authority_counts_by_arm: dict[str, Counter[str]] = {}
    authority_technical_counts_by_arm: dict[str, Counter[str]] = {}
    topic_counts_by_arm: dict[str, dict[str, Counter[str]]] = {}
    for review_id, record in records_by_id.items():
        arm = record.get("arm")
        topic = record.get("topic")
        question = record.get("question")
        authority = record.get("math_authority")
        if not isinstance(arm, str) or not isinstance(topic, str):
            raise ValueError("private review record is missing arm or topic.")
        if not isinstance(question, dict) or not isinstance(authority, dict):
            raise ValueError("private review record is missing question or authority.")
        generator_key = question.get("generator_correct_option_id")
        valid_ids = authority.get("valid_option_ids")
        decision = authority.get("decision")
        if not isinstance(valid_ids, list) or not all(
            isinstance(value, str) for value in valid_ids
        ):
            raise ValueError("private review authority data is invalid.")
        classification = classifications[review_id]
        bucket = str(classification["primary_bucket"])
        true_option_id = classification["true_option_id"]
        if bucket == "VALID_CORRECT_KEY" and generator_key != true_option_id:
            raise ValueError("VALID_CORRECT_KEY must match the generator key.")
        if bucket == "VALID_WRONG_KEY_ONLY" and generator_key == true_option_id:
            raise ValueError("VALID_WRONG_KEY_ONLY must differ from the generator key.")

        semantic_counts_by_arm.setdefault(arm, Counter())[bucket] += 1
        topic_counts_by_arm.setdefault(arm, {}).setdefault(topic, Counter())[bucket] += 1
        effective_authority_status = _authority_status(authority)
        if effective_authority_status == "unavailable":
            authority_technical_counts_by_arm.setdefault(arm, Counter())["unavailable"] += 1
            continue
        authority_accepted = decision == "ACCEPT" and len(valid_ids) == 1
        semantic_valid = bucket in {"VALID_CORRECT_KEY", "VALID_WRONG_KEY_ONLY"}
        exact_authority_answer = authority_accepted and valid_ids[0] == true_option_id
        if semantic_valid and exact_authority_answer:
            authority_class = "true_approvals"
        elif semantic_valid:
            authority_class = "false_rejections"
        elif authority_accepted:
            authority_class = "false_approvals"
        else:
            authority_class = "true_rejections"
        authority_counts_by_arm.setdefault(arm, Counter())[authority_class] += 1

    def arm_report(arm: str) -> dict[str, object]:
        semantic_counts = semantic_counts_by_arm.get(arm, Counter())
        total = sum(semantic_counts.values())
        semantic_valid = (
            semantic_counts["VALID_CORRECT_KEY"] + semantic_counts["VALID_WRONG_KEY_ONLY"]
        )
        authority_counts = authority_counts_by_arm.get(arm, Counter())
        authority_technical_counts = authority_technical_counts_by_arm.get(arm, Counter())
        topic_summary: dict[str, object] = {}
        for topic, counts in sorted(topic_counts_by_arm.get(arm, {}).items()):
            topic_total = sum(counts.values())
            topic_valid = counts["VALID_CORRECT_KEY"] + counts["VALID_WRONG_KEY_ONLY"]
            topic_summary[topic] = {
                "candidate_count": topic_total,
                "semantic_valid_count": topic_valid,
                "semantic_validity_rate": round(topic_valid / topic_total, 4)
                if topic_total
                else 0.0,
                "primary_buckets": dict(sorted(counts.items())),
            }
        return {
            "candidate_count": total,
            "semantic_valid_count": semantic_valid,
            "semantic_validity_rate": round(semantic_valid / total, 4) if total else 0.0,
            "initial_key_correct_count": semantic_counts["VALID_CORRECT_KEY"],
            "wrong_key_only_count": semantic_counts["VALID_WRONG_KEY_ONLY"],
            "primary_buckets": dict(sorted(semantic_counts.items())),
            "authority_confusion": {
                "true_approvals": authority_counts["true_approvals"],
                "true_rejections": authority_counts["true_rejections"],
                "false_approvals": authority_counts["false_approvals"],
                "false_rejections": authority_counts["false_rejections"],
            },
            "authority_technical": {
                "unavailable_count": authority_technical_counts["unavailable"],
            },
            "topic_summary": topic_summary,
        }

    arms = {arm: arm_report(arm) for arm in sorted(semantic_counts_by_arm)}
    candidate = arms.get("candidate")
    candidate_rate = (
        candidate.get("semantic_validity_rate") if isinstance(candidate, dict) else None
    )
    candidate_false_approvals = (
        candidate.get("authority_confusion", {}).get("false_approvals")
        if isinstance(candidate, dict)
        else None
    )
    candidate_unavailable = (
        candidate.get("authority_technical", {}).get("unavailable_count")
        if isinstance(candidate, dict)
        else None
    )
    candidate_qualification_decision = "NOT_ELIGIBLE_PENDING_SEMANTIC_QUALITY_GATE"
    if candidate_rate is not None and candidate_rate >= 0.98:
        candidate_qualification_decision = (
            "LIFECYCLE_QUALIFICATION_REQUIRED"
            if candidate_false_approvals == 0 and candidate_unavailable == 0
            else "NOT_ELIGIBLE_PENDING_AUTHORITY_COMPLETENESS"
        )
    elif candidate_rate is not None and candidate_rate >= 0.96:
        candidate_qualification_decision = (
            "EXPANDED_QUALIFICATION_REQUIRED"
            if candidate_false_approvals == 0 and candidate_unavailable == 0
            else "NOT_ELIGIBLE_PENDING_AUTHORITY_COMPLETENESS"
        )
    return {
        "run_kind": "local_no_llm_practice_math_manual_adjudication",
        "production_route_changed": False,
        "student_credit_enforcement": "disabled_during_source_capture",
        "independent_semantic_adjudication": "COMPLETED_FROM_PRIVATE_REVIEW",
        "promotion_decision": candidate_qualification_decision,
        "arms": arms,
    }


def main() -> int:
    # This check intentionally precedes every application import, credential lookup, and
    # provider construction. The script is otherwise a zero-cost no-op.
    args = _parse_args()
    if args.summarize_adjudication and args.reverify_unavailable:
        print("--summarize-adjudication and --reverify-unavailable cannot be combined.")
        return 2
    if args.authority_controls and (
        args.summarize_adjudication
        or args.reverify_unavailable
        or args.review_capture_path
        or args.adjudication_path
        or args.reverified_capture_path
    ):
        print("--authority-controls cannot be combined with private review modes.")
        return 2
    if args.summarize_adjudication:
        if os.environ.get(_PRIVATE_REVIEW_FLAG, "").strip().lower() != "true":
            print(
                f"{_PRIVATE_REVIEW_FLAG}=true is required to read private "
                "qualification review data."
            )
            return 2
        if not args.review_capture_path or not args.adjudication_path:
            print(
                "--summarize-adjudication requires --review-capture-path and --adjudication-path."
            )
            return 2
        try:
            review_paths = tuple(
                _validate_private_path(value.strip(), require_new=False)
                for value in args.review_capture_path.split(",")
                if value.strip()
            )
            if not review_paths:
                raise ValueError("At least one private review capture is required.")
            adjudication_path = _validate_private_path(
                args.adjudication_path,
                require_new=False,
            )
            report_path = (
                _validate_private_path(args.report_path, require_new=True)
                if args.report_path
                else None
            )
            report = _manual_adjudication_report(
                capture=_combined_review_capture(review_paths),
                adjudication=_read_private_json(adjudication_path),
            )
        except ValueError as exc:
            print(str(exc))
            return 2
        rendered = json.dumps(report, sort_keys=True)
        if report_path is not None:
            _write_private_json(report_path, report)
        print(rendered)
        return 0
    if args.reverify_unavailable:
        if os.environ.get(_PRIVATE_REVIEW_FLAG, "").strip().lower() != "true":
            print(
                f"{_PRIVATE_REVIEW_FLAG}=true is required to recheck private "
                "qualification review data."
            )
            return 2
        if not args.review_capture_path or not args.reverified_capture_path:
            print(
                "--reverify-unavailable requires --review-capture-path and "
                "--reverified-capture-path."
            )
            return 2
    if os.environ.get(_RUN_FLAG, "").strip().lower() != "true":
        print(f"{_RUN_FLAG}=true is required; no qualification was run.")
        return 0
    if os.environ.get("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "").strip().lower() != "false":
        print(
            "STUDENT_CREDIT_ENFORCEMENT_ENABLED=false is required for this "
            "no-persistence qualification harness."
        )
        return 2

    if (
        not args.reverify_unavailable
        and not args.authority_controls
        and not 1 <= args.samples_per_topic <= 5
    ):
        print("--samples-per-topic must be between 1 and 5.")
        return 2
    try:
        topics = (
            _selected_topics(args.topics)
            if not args.reverify_unavailable and not args.authority_controls
            else ()
        )
        arms = (
            _selected_arms(args.arms)
            if not args.reverify_unavailable and not args.authority_controls
            else ()
        )
    except ValueError as exc:
        print(str(exc))
        return 2
    try:
        report_path = (
            _validate_private_path(args.report_path, require_new=True) if args.report_path else None
        )
        if args.reverify_unavailable:
            review_paths = tuple(
                _validate_private_path(value.strip(), require_new=False)
                for value in args.review_capture_path.split(",")
                if value.strip()
            )
            if not review_paths:
                raise ValueError("At least one private review capture is required.")
            reverified_capture_path = _validate_private_path(
                args.reverified_capture_path,
                require_new=True,
            )
            review_capture_path = None
        else:
            review_paths = ()
            reverified_capture_path = None
            review_capture_path = (
                _validate_private_path(args.review_capture_path, require_new=True)
                if args.review_capture_path
                else None
            )
    except ValueError as exc:
        print(str(exc))
        return 2
    if review_capture_path is not None and (
        os.environ.get(_PRIVATE_REVIEW_FLAG, "").strip().lower() != "true"
    ):
        print(f"{_PRIVATE_REVIEW_FLAG}=true is required for --review-capture-path.")
        return 2

    app_root = str(Path(__file__).resolve().parents[1])
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    from config import get_settings
    from features.practice_generation.providers import (
        RoutedQuestionGenerator,
        RoutedQuestionVerifier,
    )
    from services.llm.orchestration.config_registry import LlmConfigRegistry
    from services.llm.orchestration.orchestrator import LlmOrchestrator
    from services.llm.pricing import load_pricing_config
    from services.llm.runtime_factory import build_model_executor

    settings = get_settings()
    if not settings.enable_real_llm:
        print("ENABLE_REAL_LLM=true is required; no qualification was run.")
        return 2

    registry = LlmConfigRegistry()
    base_executor = build_model_executor(settings)
    if args.authority_controls:
        authority_executor = _RecordingGeneratorOverride(
            base_executor,
            alias=None,
            model_configs=registry.model_map,
            pricing_config=load_pricing_config(),
        )
        authority_verifier = RoutedQuestionVerifier(
            LlmOrchestrator(model_executor=authority_executor)
        )
        started_at = time.monotonic()
        report = {
            **_authority_control_report(
                evaluations=_evaluate_authority_controls(
                    verifier=authority_verifier,
                    run_id=uuid.uuid4().hex[:12],
                ),
                metrics=authority_executor.metrics,
            ),
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        }
        if report_path is not None:
            _write_private_json(report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.reverify_unavailable:
        recheck_executor = _RecordingGeneratorOverride(
            base_executor,
            alias=None,
            model_configs=registry.model_map,
            pricing_config=load_pricing_config(),
        )
        recheck_orchestrator = LlmOrchestrator(model_executor=recheck_executor)
        recheck_verifier = RoutedQuestionVerifier(recheck_orchestrator)
        run_id = uuid.uuid4().hex[:12]
        started_at = time.monotonic()
        try:
            refreshed_capture, outcomes = _reverify_unavailable_capture(
                capture=_combined_review_capture(review_paths),
                verifier=recheck_verifier,
                run_id=run_id,
            )
        except ValueError as exc:
            print(str(exc))
            return 2
        report = {
            **_reverification_report(outcomes=outcomes, metrics=recheck_executor.metrics),
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        }
        _write_private_json(reverified_capture_path, refreshed_capture)
        if report_path is not None:
            _write_private_json(report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.candidate_model not in registry.model_map:
        print("--candidate-model must be an existing registry alias.")
        return 2
    current_executor = _RecordingGeneratorOverride(
        base_executor,
        alias=None,
        model_configs=registry.model_map,
        pricing_config=load_pricing_config(),
    )
    candidate_executor = _RecordingGeneratorOverride(
        base_executor,
        alias=args.candidate_model,
        model_configs=registry.model_map,
        pricing_config=load_pricing_config(),
    )
    current_orchestrator = LlmOrchestrator(model_executor=current_executor)
    candidate_orchestrator = LlmOrchestrator(model_executor=candidate_executor)
    current_generator = RoutedQuestionGenerator(current_orchestrator)
    candidate_generator = RoutedQuestionGenerator(candidate_orchestrator)
    # The verifier intentionally stays on its normally resolved, subject-qualified route.
    current_verifier = RoutedQuestionVerifier(current_orchestrator)
    candidate_verifier = RoutedQuestionVerifier(candidate_orchestrator)
    run_id = uuid.uuid4().hex[:12]
    started_at = time.monotonic()
    evaluations_by_arm: dict[str, list[CandidateEvaluation]] = {}
    metrics_by_arm: dict[str, list[ExecutionMetric]] = {}
    if "current" in arms:
        evaluations_by_arm["current"] = _evaluate_arm(
            arm="current",
            generator=current_generator,
            verifier=current_verifier,
            run_id=run_id,
            topics=topics,
            samples_per_topic=args.samples_per_topic,
        )
        metrics_by_arm["current"] = current_executor.metrics
    if "candidate" in arms:
        evaluations_by_arm["candidate"] = _evaluate_arm(
            arm="candidate",
            generator=candidate_generator,
            verifier=candidate_verifier,
            run_id=run_id,
            topics=topics,
            samples_per_topic=args.samples_per_topic,
        )
        metrics_by_arm["candidate"] = candidate_executor.metrics
    report = {
        "run_kind": "local_no_persistence_practice_math_qualification",
        "student_credit_enforcement": "disabled",
        "production_route_changed": False,
        "candidate_model_alias": args.candidate_model,
        "topics": [{"topic": item.topic, "difficulty": item.difficulty} for item in topics],
        "samples_per_topic": args.samples_per_topic,
        "authority": "normal_subject_qualified_route",
        "independent_semantic_adjudication": "PENDING_PRIVATE_REVIEW",
        "promotion_decision": "NOT_ELIGIBLE_PENDING_INDEPENDENT_ADJUDICATION",
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        "arms": {
            arm: _summarize(evaluations) for arm, evaluations in sorted(evaluations_by_arm.items())
        },
        "execution_metrics": {
            arm: _summarize_metrics(metrics) for arm, metrics in sorted(metrics_by_arm.items())
        },
        "private_review_capture": (
            "created" if review_capture_path is not None else "not_requested"
        ),
    }
    rendered = json.dumps(report, sort_keys=True)
    if review_capture_path is not None:
        _write_private_json(
            review_capture_path,
            _private_review_payload(
                run_id=run_id,
                candidate_model_alias=args.candidate_model,
                evaluations_by_arm=evaluations_by_arm,
            ),
        )
    if report_path is not None:
        _write_private_json(report_path, report)
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
