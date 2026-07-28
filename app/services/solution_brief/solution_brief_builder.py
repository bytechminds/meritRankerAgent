"""Deterministic SolutionBrief builder for generator context."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from services.context_retrieval.context_models import (
    ContextRetrievalRequest,
    RetrievedContextItem,
)
from services.solution_brief.metadata_helpers import (
    humanize_token,
    normalize_metadata_key,
    safe_list,
    safe_str,
    subject_label,
)
from services.solution_brief.models import SolutionBrief
from services.solution_brief.planner_policy import PlannerNeedPolicy
from tools.web_search.models import WebSearchItem

logger = logging.getLogger(__name__)

_KB_RISK_META_KEYS: tuple[str, ...] = (
    "traps",
    "constraints",
    "hiddenConditions",
    "hidden_conditions",
    "not_same_when",
    "notSameWhen",
)

_KB_APPROACH_META_KEYS: tuple[str, ...] = (
    "solving_style",
    "solvingStyle",
    "solution_approach",
)

_KB_INSTRUCTION_META_KEYS: tuple[str, ...] = (
    "answer_style",
    "answerStyle",
)


@dataclass(frozen=True)
class SolutionBriefBuildResult:
    """Builder output for context composition."""

    brief: SolutionBrief | None
    brief_text: str
    used: bool
    context_sources: str


class SolutionBriefBuilder:
    """Build compact SolutionBrief text from query, classification, and context."""

    def __init__(self, *, planner_need_policy: PlannerNeedPolicy | None = None) -> None:
        self._planner_need_policy = planner_need_policy or PlannerNeedPolicy()

    def build(
        self,
        request: ContextRetrievalRequest,
        *,
        kb_items: list[RetrievedContextItem] | None = None,
        web_items: list[WebSearchItem] | None = None,
    ) -> SolutionBriefBuildResult:
        kb_items = kb_items or []
        web_items = web_items or []
        sources = _context_sources(kb_items, web_items)

        decision = self._planner_need_policy.decide(
            request,
            kb_item_count=len(kb_items),
            web_item_count=len(web_items),
        )
        if not decision.use_solution_brief:
            logger.info(
                "solution_brief_builder  request_id=%s  used=false  "
                "context_sources=%s  subject=%s  topic=%s  given_count=0  "
                "context_count=0  core_concept_count=0  risk_count=0  brief_chars=0  "
                "decision_reason=%s",
                request.request_id,
                sources,
                request.subject,
                request.topic or "",
                decision.reason,
            )
            return SolutionBriefBuildResult(
                brief=None,
                brief_text="",
                used=False,
                context_sources=sources,
            )

        brief = self._build_brief(request, kb_items=kb_items, web_items=web_items)
        brief_text = self.format_brief(brief)
        logger.info(
            "solution_brief_builder  request_id=%s  used=true  "
            "context_sources=%s  subject=%s  topic=%s  given_count=%d  "
            "context_count=%d  core_concept_count=%d  risk_count=%d  brief_chars=%d",
            request.request_id,
            sources,
            brief.subject,
            brief.topic,
            len(brief.given),
            len(brief.context),
            len(brief.core_concepts),
            len(brief.risk_flags),
            len(brief_text),
        )
        return SolutionBriefBuildResult(
            brief=brief,
            brief_text=brief_text,
            used=True,
            context_sources=sources,
        )

    @staticmethod
    def compose_context_text(
        *,
        brief_text: str,
        web_section: str,
        max_chars: int,
    ) -> str:
        """Compose bounded context without truncating web source cards."""
        brief = brief_text.strip()
        web = web_section.strip()
        if not brief and not web:
            return ""
        if max_chars <= 0:
            return ""
        if not web:
            return _trim_to_line_boundary(brief, max_chars)
        if not brief:
            return _trim_to_line_boundary(web, max_chars)

        separator = "\n\n"
        if len(web) >= max_chars:
            # The web formatter receives this same cap and owns atomic source
            # card formatting. Do not slice it again after briefing.
            return web
        brief_budget = max_chars - len(web) - len(separator)
        if brief_budget <= 0:
            return web
        bounded_brief = _trim_to_line_boundary(brief, brief_budget)
        return f"{bounded_brief}{separator}{web}" if bounded_brief else web

    @staticmethod
    def format_brief(brief: SolutionBrief) -> str:
        lines: list[str] = ["[Solution Brief]"]
        if brief.subject:
            lines.append(f"Subject: {brief.subject}")
        if brief.topic:
            lines.append(f"Topic: {brief.topic}")
        if brief.given:
            lines.append("")
            lines.append("Given:")
            lines.extend(f"- {item}" for item in brief.given)
        if brief.find:
            lines.append("")
            lines.append(f"Find: {brief.find}")
        if brief.core_concepts:
            lines.append("")
            lines.append("Core concepts:")
            lines.extend(f"- {item}" for item in brief.core_concepts)
        if brief.context:
            lines.append("")
            lines.append("Context:")
            lines.extend(f"- {item}" for item in brief.context)
        if brief.solution_approach:
            lines.append("")
            lines.append("Solution approach:")
            lines.extend(f"- {item}" for item in brief.solution_approach)
        if brief.risk_flags:
            lines.append("")
            lines.append("Risk flags:")
            lines.extend(f"- {item}" for item in brief.risk_flags)
        if brief.generator_instructions:
            lines.append("")
            lines.append("Generator instructions:")
            lines.extend(f"- {item}" for item in brief.generator_instructions)
        return "\n".join(lines)

    def _build_brief(
        self,
        request: ContextRetrievalRequest,
        *,
        kb_items: list[RetrievedContextItem],
        web_items: list[WebSearchItem],
    ) -> SolutionBrief:
        subject = _subject_label(request.subject)
        topic = _resolve_topic(request, kb_items)
        given = _extract_given(request.query)
        find = _extract_find(request.query, request.intent)
        core_concepts: list[str] = []
        context: list[str] = []
        solution_approach: list[str] = []
        risk_flags: list[str] = []
        generator_instructions: list[str] = []

        for item in kb_items:
            self._merge_kb_item(
                item,
                request=request,
                core_concepts=core_concepts,
                context=context,
                solution_approach=solution_approach,
                risk_flags=risk_flags,
                generator_instructions=generator_instructions,
            )

        if web_items and not context:
            context.append("Fresh web sources selected for time-sensitive facts.")

        generator_instructions.extend(_default_generator_instructions(request))

        return SolutionBrief(
            subject=safe_str(subject, max_length=64),
            topic=safe_str(topic, max_length=128),
            given=_dedupe_cap(given),
            find=safe_str(find, max_length=512),
            context=_dedupe_cap(context),
            core_concepts=_dedupe_cap(core_concepts),
            solution_approach=_dedupe_cap(solution_approach),
            risk_flags=_dedupe_cap(risk_flags),
            generator_instructions=_dedupe_cap(generator_instructions),
        )

    @staticmethod
    def _merge_kb_item(
        item: RetrievedContextItem,
        *,
        request: ContextRetrievalRequest,
        core_concepts: list[str],
        context: list[str],
        solution_approach: list[str],
        risk_flags: list[str],
        generator_instructions: list[str],
    ) -> None:
        meta = item.metadata or {}
        tags_raw = normalize_metadata_key(
            meta,
            ("conceptTags", "concept_tags", "coreConcepts", "core_concepts"),
        )
        if tags_raw is not None:
            for tag in safe_list(tags_raw):
                clean = humanize_token(tag)
                if clean:
                    core_concepts.append(clean)

        hint = safe_str(item.text)
        if hint:
            if len(hint) > 220:
                hint = hint[:220].rstrip() + "..."
            context.append(hint)

        risk_text = safe_str(item.risk, max_length=128)
        if risk_text and risk_text.lower() not in {"none", ""}:
            risk_flags.append(risk_text)

        risk_meta = normalize_metadata_key(meta, _KB_RISK_META_KEYS)
        if risk_meta is not None:
            risk_flags.extend(safe_list(risk_meta))

        approach_meta = normalize_metadata_key(meta, _KB_APPROACH_META_KEYS)
        if approach_meta is not None:
            solution_approach.extend(safe_list(approach_meta))

        instruction_meta = normalize_metadata_key(meta, _KB_INSTRUCTION_META_KEYS)
        if instruction_meta is not None:
            generator_instructions.extend(safe_list(instruction_meta))

        topic_key = normalize_metadata_key(meta, ("patternTopicKey", "pattern_topic_key", "topic"))
        if topic_key and not request.topic:
            _ = topic_key  # topic resolved separately; never expose raw key in brief


def _context_sources(
    kb_items: list[RetrievedContextItem],
    web_items: list[WebSearchItem],
) -> str:
    has_kb = bool(kb_items)
    has_web = bool(web_items)
    if has_kb and has_web:
        return "both"
    if has_kb:
        return "kb"
    if has_web:
        return "web"
    return "none"


def _subject_label(subject: str) -> str:
    return subject_label(subject)


def _resolve_topic(
    request: ContextRetrievalRequest,
    kb_items: list[RetrievedContextItem],
) -> str:
    if request.topic:
        return humanize_token(request.topic.strip())
    for item in kb_items:
        meta = item.metadata or {}
        topic_key = normalize_metadata_key(
            meta,
            ("patternTopicKey", "pattern_topic_key", "topic"),
        )
        if topic_key:
            return humanize_token(safe_str(topic_key))
    if request.pattern_topic_candidate:
        return humanize_token(request.pattern_topic_candidate)
    return ""


def _extract_given(query: str) -> list[str]:
    numbers = re.findall(
        r"\b\d+(?:\.\d+)?(?:\s*(?:km/hr|km/h|m/s|%|rs|₹|years?|months?))?\b",
        query,
        flags=re.IGNORECASE,
    )
    given: list[str] = []
    if numbers:
        given.append(f"Numeric data mentioned: {', '.join(numbers[:5])}")
    if_parts = re.split(r"\sif\s", query, maxsplit=1, flags=re.IGNORECASE)
    if len(if_parts) > 1:
        clause = if_parts[1].strip()
        if clause and len(clause) <= 120:
            given.append(clause.rstrip("?.!"))
    return given[:5]


def _extract_find(query: str, intent: str) -> str:
    stripped = query.strip()
    if len(stripped) <= 180:
        return stripped
    if intent == "solve":
        return "Solve the student's quantitative or reasoning problem."
    return stripped[:180].rstrip() + "..."


def _default_generator_instructions(request: ContextRetrievalRequest) -> list[str]:
    instructions: list[str] = []
    if request.intent == "solve":
        instructions.append("Show clear solving steps suitable for exam preparation.")
    elif request.intent == "explain":
        instructions.append("Explain conceptually without exposing internal retrieval metadata.")
    if request.need_web_search or request.web_search_reason:
        instructions.append(
            "Use fresh web context only when relevant; mention uncertainty if sources conflict."
        )
    return instructions


def _dedupe_cap(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= 7:
            break
    return result


def _trim_to_line_boundary(text: str, max_chars: int) -> str:
    """Bound internal context at a complete line when practical."""
    if len(text) <= max_chars:
        return text
    bounded = text[:max_chars].rstrip()
    last_newline = bounded.rfind("\n")
    if last_newline > 0:
        return bounded[:last_newline].rstrip()
    return bounded
