"""Render safe internal retrieval context for the existing generator prompt path."""

from __future__ import annotations

import json

from retrieval.models import StudentRetrievalContext

_MAX_RENDERED_BUNDLE_CHARS = 6000


def render_retrieval_context(context: StudentRetrievalContext) -> str:
    """Render the approved retrieval contract without raw vector metadata."""
    if context.mode == "fresh_solve":
        return ""

    lines = [
        "[RETRIEVAL_CONTEXT]",
        "provider: s3_vector",
        f"mode: {context.mode}",
        f"confidence: {context.confidence:.2f}",
        f"canUseForFinalAnswer: {str(context.can_use_for_final_answer).lower()}",
        f"selectedPatternId: {context.selected_pattern_id or ''}",
        f"subject: {context.subject or ''}",
        f"topic: {context.topic or ''}",
        f"flowType: {context.flow_type or ''}",
        "retrievalTrace: "
        + json.dumps(context.retrieval_trace.model_dump(by_alias=True), separators=(",", ":")),
        "warnings: " + json.dumps(context.warnings, ensure_ascii=True, separators=(",", ":")),
    ]
    if context.can_use_for_final_answer:
        lines.append(
            "Use the approved SolveFlow and answerGuide as the main strategy, adapt to the "
            "student's exact values, and independently verify the final answer. Never copy a "
            "stored final answer unless the values and options match exactly."
        )
    else:
        lines.append(
            "PatternGraph is a method hint only. Solve fresh and verify independently; do not "
            "treat this retrieval as final-answer authority."
        )

    bundle = {
        "patternGraphCompact": context.pattern_graph_compact,
        "solveFlow": context.solve_flow if context.can_use_for_final_answer else {},
        "answerGuide": context.answer_guide if context.can_use_for_final_answer else {},
        "notSameWhen": context.not_same_when,
        "materialConflicts": context.material_conflicts,
    }
    serialized = json.dumps(bundle, ensure_ascii=True, separators=(",", ":"))
    if len(serialized) > _MAX_RENDERED_BUNDLE_CHARS:
        serialized = serialized[:_MAX_RENDERED_BUNDLE_CHARS]
    lines.extend(["[RETRIEVAL_BUNDLE]", serialized])
    return "\n".join(lines)
