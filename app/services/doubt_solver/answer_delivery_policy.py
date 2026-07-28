"""Provider-neutral delivery policy for orchestrated answer streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import Settings, get_settings
from schemas.doubt_solver import CanonicalLanguage

DeliveryStrategy = Literal["live_stream", "verified_replay"]
DeliveryRiskLevel = Literal["low", "medium", "high"]

_DIFFICULTY_RANK = {
    "default": 0,
    "basic": 1,
    "intermediate": 2,
    "advanced": 3,
}


@dataclass(frozen=True)
class AnswerDeliverySignals:
    """Safe, pre-token signals used to select an answer delivery strategy."""

    classifier_confidence: float | None
    classifier_fallback: bool
    source_modality: Literal["text", "image"]
    image_confidence: float | None
    image_uncertain: bool
    difficulty: str
    current_fact_dependency: bool
    retrieval_used: bool
    retrieval_mode: str | None
    pattern_confidence: float | None
    pattern_candidate_conflict: bool
    provider_fallback: bool
    needs_review: bool
    language: CanonicalLanguage = "english"


@dataclass(frozen=True)
class AnswerDeliveryDecision:
    strategy: DeliveryStrategy
    risk_level: DeliveryRiskLevel
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AnswerDeliveryPolicy:
    mode: Literal["always_verified", "adaptive", "always_live"]
    min_classifier_confidence: float
    min_image_confidence: float
    min_pattern_confidence: float
    max_live_difficulty: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AnswerDeliveryPolicy:
        cfg = settings or get_settings()
        return cls(
            mode=cfg.answer_delivery_policy,  # type: ignore[arg-type]
            min_classifier_confidence=cfg.answer_live_stream_min_classifier_confidence,
            min_image_confidence=cfg.answer_live_stream_min_image_confidence,
            min_pattern_confidence=cfg.answer_live_stream_min_pattern_confidence,
            max_live_difficulty=cfg.answer_live_stream_max_difficulty,
        )

    def decide(self, signals: AnswerDeliverySignals) -> AnswerDeliveryDecision:
        if signals.language != "english":
            return AnswerDeliveryDecision(
                strategy="verified_replay",
                risk_level="high",
                reason_codes=("language_verification_required",),
            )
        if self.mode == "always_live" and signals.current_fact_dependency:
            return AnswerDeliveryDecision(
                strategy="verified_replay",
                risk_level="high",
                reason_codes=("current_fact_dependency",),
            )
        if self.mode == "always_verified":
            return AnswerDeliveryDecision(
                strategy="verified_replay",
                risk_level="high",
                reason_codes=("policy_always_verified",),
            )
        if self.mode == "always_live":
            return AnswerDeliveryDecision(
                strategy="live_stream",
                risk_level="low",
                reason_codes=("policy_always_live",),
            )

        reasons: list[str] = []
        if signals.classifier_confidence is None:
            reasons.append("classifier_confidence_missing")
        elif signals.classifier_confidence < self.min_classifier_confidence:
            reasons.append("classifier_confidence_below_threshold")
        if signals.classifier_fallback:
            reasons.append("classifier_fallback")
        if signals.source_modality == "image":
            if signals.image_confidence is None:
                reasons.append("image_confidence_missing")
            elif signals.image_confidence < self.min_image_confidence:
                reasons.append("image_confidence_below_threshold")
            if signals.image_uncertain:
                reasons.append("image_uncertain")
        if _DIFFICULTY_RANK.get(signals.difficulty, -1) > _DIFFICULTY_RANK[
            self.max_live_difficulty
        ]:
            reasons.append("difficulty_above_live_limit")
        if signals.current_fact_dependency:
            reasons.append("current_fact_dependency")
        if signals.retrieval_used:
            if signals.retrieval_mode != "runtime_ready":
                reasons.append("retrieval_not_runtime_ready")
            elif signals.pattern_confidence is None:
                reasons.append("pattern_confidence_missing")
            elif signals.pattern_confidence < self.min_pattern_confidence:
                reasons.append("pattern_confidence_below_threshold")
        if signals.pattern_candidate_conflict:
            reasons.append("pattern_candidate_conflict")
        if signals.provider_fallback:
            reasons.append("provider_fallback")
        if signals.needs_review:
            reasons.append("needs_review")

        if not reasons:
            return AnswerDeliveryDecision(
                strategy="live_stream",
                risk_level="low",
                reason_codes=("approved_low_risk",),
            )
        risk_level: DeliveryRiskLevel = "high" if any(
            code in {
                "classifier_fallback",
                "image_uncertain",
                "current_fact_dependency",
                "provider_fallback",
                "needs_review",
            }
            for code in reasons
        ) else "medium"
        return AnswerDeliveryDecision(
            strategy="verified_replay",
            risk_level=risk_level,
            reason_codes=tuple(reasons),
        )
