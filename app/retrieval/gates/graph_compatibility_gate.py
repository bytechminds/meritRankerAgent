"""Deterministic compatibility gate for student retrieval bundles."""

from __future__ import annotations

import re

from retrieval.models import GraphGateDecision, PatternRuntimeBundle

_APP_SUBJECT_TO_BUNDLE: dict[str, str] = {
    "math": "QUANT",
    "quant": "QUANT",
    "quantitative": "QUANT",
    "reasoning": "REASONING",
    "english": "ENGLISH",
    "general": "GK",
}

_FLOW_BY_INTENT: dict[str, frozenset[str]] = {
    "solve": frozenset({"COMPUTATIONAL_SOLVE_FLOW", "REASONING_TRACE_FLOW"}),
    "explain": frozenset({
        "ANSWER_EXPLANATION_FLOW",
        "FACT_CONCEPT_ANSWER_FLOW",
        "REASONING_TRACE_FLOW",
    }),
    "practice": frozenset({"COMPUTATIONAL_SOLVE_FLOW", "REASONING_TRACE_FLOW"}),
    "visualize": frozenset({"ANSWER_EXPLANATION_FLOW", "FACT_CONCEPT_ANSWER_FLOW"}),
}


class GraphCompatibilityGate:
    """Apply the mandatory student-safety admission rules to DynamoDB bundles."""

    def evaluate(
        self,
        *,
        query: str,
        subject: str,
        intent: str,
        bundle: PatternRuntimeBundle,
    ) -> GraphGateDecision:
        if bundle.pattern_status != "approved":
            return GraphGateDecision(allowed=False, reason="pattern_not_approved")
        if not bundle.can_use_for_retrieval:
            return GraphGateDecision(allowed=False, reason="retrieval_not_allowed")
        if bundle.student_visibility == "admin_only":
            return GraphGateDecision(allowed=False, reason="admin_only")
        if bundle.answer_data_status in {"ambiguous", "inconsistent"}:
            return GraphGateDecision(allowed=False, reason="unsafe_answer_data")
        if bundle.material_conflicts:
            return GraphGateDecision(allowed=False, reason="material_conflicts")
        if self._violates_not_same_when(query, bundle.not_same_when):
            return GraphGateDecision(allowed=False, reason="not_same_when_violation")

        expected_subject = _APP_SUBJECT_TO_BUNDLE.get(subject.strip().lower())
        if expected_subject and bundle.subject != expected_subject:
            return GraphGateDecision(allowed=False, reason="subject_mismatch")

        allowed_flows = _FLOW_BY_INTENT.get(intent.strip().lower())
        if allowed_flows and bundle.flow_type not in allowed_flows:
            return GraphGateDecision(allowed=False, reason="flow_type_mismatch")

        confidence = 0.85
        if expected_subject:
            confidence += 0.05
        if bundle.topic and bundle.topic.lower() in query.lower():
            confidence += 0.05
        return GraphGateDecision(allowed=True, reason="compatible", confidence=min(confidence, 1.0))

    @staticmethod
    def _violates_not_same_when(query: str, rules: list[str]) -> bool:
        normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        if not normalized_query:
            return False
        for rule in rules:
            normalized_rule = " ".join(re.findall(r"[a-z0-9]+", rule.lower()))
            if normalized_rule and normalized_rule in normalized_query:
                return True
        return False
