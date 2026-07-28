"""
app/services/context_retrieval/context_retrieval_service.py
------------------------------------------------------------
Graph-facing context retrieval service.

The orchestrated graph calls only ContextRetrievalService.retrieve_context().
Bedrock KB details, filters, reranking, and formatting live here — not in graph nodes.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from config import get_settings
from observability import log_event
from retrieval.context.data_context_builder import render_retrieval_context
from retrieval.models import StudentRetrievalContext
from retrieval.retrieval_service import StudentRetrievalService
from services.context_retrieval.bedrock_kb_retriever import BedrockKnowledgeBaseRetriever
from services.context_retrieval.context_models import (
    ContextRetrievalDecision,
    ContextRetrievalRequest,
    ContextRetrievalResult,
    PatternHints,
    PatternHintStrength,
    RetrievedContextItem,
)
from services.context_retrieval.web_search_decision import (
    evaluate_web_search_decision,
    resolve_web_search_query,
    should_attempt_web_fallback,
    should_skip_kb_for_direct_web,
)
from services.solution_brief.metadata_helpers import (
    humanize_token,
    normalize_metadata_key,
    safe_list,
    safe_str,
    subject_label,
)
from services.solution_brief.solution_brief_builder import SolutionBriefBuilder
from tools.web_search.models import WebSearchItem, WebSearchRequest
from tools.web_search.web_search_tool import WebSearchTool, credentials_ready

logger = logging.getLogger(__name__)

_EXAM_KEYWORDS: tuple[str, ...] = (
    "sbi po",
    "ssc cgl",
    "cat ",
    " cat",
    "upsc",
    "ibps",
    "rrb",
)

_RETRIEVAL_KEYWORDS: tuple[str, ...] = (
    "pattern",
    "shortcut",
    "similar",
    "trap",
    "previous year",
    "pyq",
    "concept",
    "formula",
)

_SIMPLE_ARITHMETIC_RE = re.compile(
    r"^\s*(?:what\s+is\s+)?\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$",
    re.IGNORECASE,
)

_KB_INTENTS: frozenset[str] = frozenset({"explain", "practice", "visualize"})
_KB_DIFFICULTIES: frozenset[str] = frozenset({"intermediate", "advanced"})

_RERANK_CONFIDENCE_THRESHOLD = 0.85
_MAX_RETRIEVAL_LANES = 5

LANE_SUBJECT_TOPIC_FAMILY = "SUBJECT_TOPIC_FAMILY"
LANE_SUBJECT_TOPIC = "SUBJECT_TOPIC"
LANE_SUBJECT_ONLY = "SUBJECT_ONLY"
LANE_BROAD_SEMANTIC = "BROAD_SEMANTIC"
LANE_RELAXED_SUBJECT_ONLY = "RELAXED_SUBJECT_ONLY"

RELAXED_LANES: frozenset[str] = frozenset({LANE_BROAD_SEMANTIC, LANE_RELAXED_SUBJECT_ONLY})

_APP_TO_KB_SUBJECT: dict[str, str] = {
    "math": "QUANT",
    "quantitative": "QUANT",
    "quant": "QUANT",
    "reasoning": "REASONING",
    "english": "ENGLISH",
    "general": "GK",
    "gk": "GK",
}

_KB_SUBJECT_VALUES: frozenset[str] = frozenset({"QUANT", "REASONING", "ENGLISH", "GK"})

# (keywords, pattern_topic_key, optional_family_key) — longer phrases first within each group
_PATTERN_HINT_RULES: tuple[tuple[tuple[str, ...], str, str | None], ...] = (
    # Reasoning
    (
        ("coded inequality", "symbol inequality", "conclusions follow"),
        "CODED_INEQUALITY",
        None,
    ),
    (
        (
            "seating arrangement",
            "circular arrangement",
            "linear arrangement",
            "circular seating",
            "facing north",
            "facing south",
            "facing center",
        ),
        "SEATING_ARRANGEMENT",
        None,
    ),
    (("floor puzzle", "building numbered", "floors"), "FLOOR_PUZZLE", None),
    (
        (
            "direction sense",
            "turns left",
            "turns right",
            "north",
            "south",
            "east",
            "west",
        ),
        "DIRECTION_SENSE",
        None,
    ),
    (
        (
            "blood relation",
            "pointing to",
            "father",
            "mother",
            "sister",
            "brother",
        ),
        "BLOOD_RELATION",
        None,
    ),
    (
        ("syllogism", "statements and conclusions", "all some no"),
        "SYLLOGISM",
        None,
    ),
    (("number series", "missing number", "series"), "SERIES", None),
    (
        ("coding decoding", "code language", "coded as", "coding-decoding"),
        "CODING_DECODING",
        None,
    ),
    (("input output", "machine input", "input-output"), "INPUT_OUTPUT", None),
    (
        ("caselet", "table data", "data interpretation"),
        "CASELET_REASONING",
        None,
    ),
    (("puzzle", "arrangement"), "SEATING_ARRANGEMENT", None),
    # Quant
    (
        (
            "profit",
            "loss",
            "discount",
            "marked price",
            "selling price",
            "cost price",
        ),
        "PROFIT_LOSS_DISCOUNT",
        None,
    ),
    (("percentage", "percent"), "PERCENTAGE", None),
    (("mixture", "alligation", "concentration"), "MIXTURE_ALLIGATION", None),
    (("average",), "AVERAGE", None),
    (
        (
            "time speed distance",
            "time and distance",
            "relative speed",
            "cross each other",
            "train",
            "boat",
        ),
        "TIME_SPEED_DISTANCE",
        None,
    ),
    (("time and work", "work-time", "time work"), "TIME_WORK", None),
    (("ratio", "proportion"), "RATIO_PROPORTION", None),
    (("partnership",), "PARTNERSHIP", None),
    (
        ("simple interest", "compound interest", "interest"),
        "INTEREST",
        None,
    ),
    # English
    (("grammar", "sentence correction"), "GRAMMAR", None),
    (("synonym", "antonym"), "VOCABULARY", None),
    (("cloze test",), "CLOZE_TEST", None),
    (("reading comprehension",), "READING_COMPREHENSION", None),
    (("para jumble",), "PARA_JUMBLE", None),
)

_ADVANCED_REASONING_TOPICS: frozenset[str] = frozenset(
    {
        "SEATING_ARRANGEMENT",
        "FLOOR_PUZZLE",
        "CODED_INEQUALITY",
        "CASELET_REASONING",
    }
)

_INTERMEDIATE_REASONING_TOPICS: frozenset[str] = frozenset(
    {
        "DIRECTION_SENSE",
        "BLOOD_RELATION",
        "SERIES",
        "CODING_DECODING",
        "INPUT_OUTPUT",
        "SYLLOGISM",
    }
)

_SUBJECT_HINT_PREFIXES: dict[str, frozenset[str]] = {
    "reasoning": frozenset(
        {
            "CODED_INEQUALITY",
            "SEATING_ARRANGEMENT",
            "FLOOR_PUZZLE",
            "DIRECTION_SENSE",
            "BLOOD_RELATION",
            "SYLLOGISM",
            "SERIES",
            "CODING_DECODING",
            "INPUT_OUTPUT",
            "CASELET_REASONING",
        }
    ),
    "math": frozenset(
        {
            "PROFIT_LOSS_DISCOUNT",
            "PERCENTAGE",
            "MIXTURE_ALLIGATION",
            "AVERAGE",
            "TIME_SPEED_DISTANCE",
            "TIME_WORK",
            "RATIO_PROPORTION",
            "PARTNERSHIP",
            "INTEREST",
        }
    ),
    "english": frozenset(
        {
            "GRAMMAR",
            "VOCABULARY",
            "CLOZE_TEST",
            "READING_COMPREHENSION",
            "PARA_JUMBLE",
        }
    ),
}

_CANONICAL_TOPIC_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")

_KNOWN_TOPIC_TO_CANONICAL: dict[str, str] = {
    "time_speed_distance": "TIME_SPEED_DISTANCE",
    "time speed distance": "TIME_SPEED_DISTANCE",
    "tsd": "TIME_SPEED_DISTANCE",
    "age": "AGE",
    "profit_loss_discount": "PROFIT_LOSS_DISCOUNT",
    "profit and loss": "PROFIT_LOSS_DISCOUNT",
    "percentage": "PERCENTAGE",
    "percent": "PERCENTAGE",
    "ratio": "RATIO_PROPORTION",
    "mixture": "MIXTURE_ALLIGATION",
    "alligation": "MIXTURE_ALLIGATION",
    "coded_inequality": "CODED_INEQUALITY",
    "seating_arrangement": "SEATING_ARRANGEMENT",
    "direction_sense": "DIRECTION_SENSE",
    "blood_relation": "BLOOD_RELATION",
    "grammar": "GRAMMAR",
    "vocabulary": "VOCABULARY",
}

_context_retrieval_service: ContextRetrievalService | None = None


def map_app_subject_to_kb(subject: str) -> str | None:
    """Map app-level subject labels to KB metadata subject values."""
    normalized = subject.strip().lower()
    mapped = _APP_TO_KB_SUBJECT.get(normalized)
    if mapped:
        return mapped
    upper = subject.strip().upper()
    if upper in _KB_SUBJECT_VALUES:
        return upper
    return None


def infer_pattern_topic_key(query: str, topic: str | None) -> str | None:
    """Infer canonical patternTopicKey from classification topic or query keywords."""
    hints = resolve_retrieval_hints(query, "general", {"topic": topic})
    return hints.pattern_topic_key


def normalize_retrieval_tags(raw_tags: Any, *, max_tags: int | None = None) -> list[str]:
    """Normalize classifier/KB retrieval tags to lower_snake_case, deduped, capped."""
    settings = get_settings()
    limit = max_tags if max_tags is not None else settings.context_max_retrieval_tags
    if raw_tags is None:
        return []
    items: list[str] = []
    if isinstance(raw_tags, str):
        items = [t.strip() for t in raw_tags.replace(";", ",").split(",") if t.strip()]
    elif isinstance(raw_tags, (list, tuple, set)):
        for entry in raw_tags:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in items:
        clean = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
        if len(normalized) >= limit:
            break
    return normalized


def _map_topic_label_to_canonical(topic: str | None) -> str | None:
    if not topic:
        return None
    stripped = topic.strip()
    if _CANONICAL_TOPIC_RE.match(stripped):
        return stripped
    mapped = _KNOWN_TOPIC_TO_CANONICAL.get(stripped.lower())
    if mapped:
        return mapped
    normalized = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    return _KNOWN_TOPIC_TO_CANONICAL.get(normalized)


def _is_canonical_pattern_key(value: str | None) -> bool:
    return bool(value and _CANONICAL_TOPIC_RE.match(value.strip()))


def resolve_retrieval_hints(
    query: str,
    subject: str,
    classification: dict[str, Any] | None = None,
) -> PatternHints:
    """Resolve KB retrieval hints from classifier output with deterministic fallback."""
    classification = classification or {}
    settings = get_settings()
    threshold = settings.context_topic_hint_confidence_threshold

    topic_conf = classification.get("topic_confidence")
    try:
        topic_confidence = float(topic_conf) if topic_conf is not None else None
    except (TypeError, ValueError):
        topic_confidence = None

    retrieval_tags = normalize_retrieval_tags(classification.get("retrieval_tags"))
    pattern_candidate = classification.get("pattern_topic_candidate")
    family_candidate = classification.get("pattern_family_candidate")
    pattern_candidate_str = str(pattern_candidate).strip() if pattern_candidate else None
    family_candidate_str = str(family_candidate).strip() if family_candidate else None

    if (
        _is_canonical_pattern_key(pattern_candidate_str)
        and topic_confidence is not None
        and topic_confidence >= threshold
    ):
        return PatternHints(
            pattern_topic_key=pattern_candidate_str,
            pattern_family_key=family_candidate_str
            if _is_canonical_pattern_key(family_candidate_str)
            and topic_confidence >= threshold
            else None,
            topic_hint=pattern_candidate_str,
            matched_signals=[f"classifier_pattern_topic:{pattern_candidate_str}"],
            strength="strong",
            retrieval_tags=retrieval_tags,
            topic_confidence=topic_confidence,
            hint_source="classifier_pattern_topic",
        )

    mapped_topic = _map_topic_label_to_canonical(classification.get("topic"))
    if mapped_topic and topic_confidence is not None and topic_confidence >= threshold:
        return PatternHints(
            pattern_topic_key=mapped_topic,
            pattern_family_key=family_candidate_str
            if _is_canonical_pattern_key(family_candidate_str)
            and topic_confidence >= threshold
            else None,
            topic_hint=str(classification.get("topic") or mapped_topic),
            matched_signals=[f"classifier_topic_map:{mapped_topic}"],
            strength="strong",
            retrieval_tags=retrieval_tags,
            topic_confidence=topic_confidence,
            hint_source="classifier_topic_map",
        )

    derived = derive_pattern_hints(query, subject, classification)
    if derived.pattern_topic_key:
        return PatternHints(
            pattern_topic_key=derived.pattern_topic_key,
            pattern_family_key=derived.pattern_family_key,
            topic_hint=derived.topic_hint,
            matched_signals=derived.matched_signals,
            strength=derived.strength,
            retrieval_tags=retrieval_tags
            or derived.matched_signals[: settings.context_max_retrieval_tags],
            topic_confidence=topic_confidence,
            hint_source="deterministic_derive",
        )

    return PatternHints(
        topic_hint=str(classification.get("topic")) if classification.get("topic") else None,
        matched_signals=derived.matched_signals,
        strength="weak",
        retrieval_tags=retrieval_tags,
        topic_confidence=topic_confidence,
        hint_source="tags_only" if retrieval_tags else "none",
    )


def _hint_strength_for_match(keyword: str, topic_hits: int) -> PatternHintStrength:
    if topic_hits >= 2 or len(keyword) >= 14:
        return "strong"
    if len(keyword) >= 6:
        return "medium"
    return "weak"


def derive_pattern_hints(
    query: str,
    subject: str,
    classification: dict[str, Any] | None = None,
) -> PatternHints:
    """Derive deterministic topic/pattern hints from query text and subject."""
    classification = classification or {}
    topic = classification.get("topic")
    if topic and _CANONICAL_TOPIC_RE.match(str(topic).strip()):
        canonical = str(topic).strip()
        return PatternHints(
            pattern_topic_key=canonical,
            topic_hint=canonical,
            matched_signals=[f"classification_topic:{canonical}"],
            strength="strong",
        )

    query_lower = query.lower()
    subject_norm = subject.strip().lower()
    allowed_topics = _SUBJECT_HINT_PREFIXES.get(subject_norm)

    best_topic: str | None = None
    best_family: str | None = None
    best_keyword = ""
    matched_signals: list[str] = []
    topic_hit_counts: dict[str, int] = {}

    for keywords, topic_key, family_key in _PATTERN_HINT_RULES:
        if allowed_topics is not None and topic_key not in allowed_topics:
            continue
        for kw in keywords:
            if kw in query_lower:
                matched_signals.append(kw)
                topic_hit_counts[topic_key] = topic_hit_counts.get(topic_key, 0) + 1
                if len(kw) > len(best_keyword):
                    best_keyword = kw
                    best_topic = topic_key
                    best_family = family_key

    if not best_topic:
        return PatternHints(matched_signals=matched_signals, strength="weak")

    strength = _hint_strength_for_match(best_keyword, topic_hit_counts.get(best_topic, 1))
    if strength == "weak":
        return PatternHints(
            matched_signals=matched_signals,
            strength="weak",
        )

    return PatternHints(
        pattern_topic_key=best_topic,
        pattern_family_key=best_family,
        topic_hint=best_topic,
        matched_signals=matched_signals,
        strength=strength,
    )


@dataclass(frozen=True)
class RerankScoreBreakdown:
    """Safe rerank component breakdown for logging — no chunk text."""

    bedrock_score: float | None
    subject_match: bool
    topic_match: bool
    family_match: bool
    keyword_overlap_score: float
    concept_tag_overlap_score: float
    tag_overlap_score: float
    matched_tags_count: int
    matched_tags_sample: str
    approved_signal: bool
    risk: str
    rejection_reason: str
    final_confidence: float


def _parse_metadata_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _dedupe_short(items: list[str], *, limit: int = 5) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _format_kb_fallback_context(
    request: ContextRetrievalRequest,
    kb_items: list[RetrievedContextItem],
    *,
    max_chars: int,
    max_items: int = 2,
) -> str:
    """Compact KB fallback when SolutionBriefBuilder fails after selection."""
    display_subject = subject_label(request.subject)
    topic = humanize_token(safe_str(request.topic, max_length=128))
    if not topic:
        for item in kb_items[:max_items]:
            meta = item.metadata or {}
            topic_key = normalize_metadata_key(
                meta,
                ("patternTopicKey", "pattern_topic_key", "topic"),
            )
            if topic_key:
                topic = humanize_token(safe_str(topic_key))
                break
    if not topic and request.pattern_topic_candidate:
        topic = humanize_token(request.pattern_topic_candidate)

    core_concepts: list[str] = []
    for item in kb_items[:max_items]:
        meta = item.metadata or {}
        tags_raw = normalize_metadata_key(
            meta,
            ("conceptTags", "concept_tags", "coreConcepts", "core_concepts"),
        )
        if tags_raw is not None:
            core_concepts.extend(safe_list(tags_raw))

    lines: list[str] = ["[Relevant KB Context]"]
    if display_subject:
        lines.append(f"Subject: {display_subject}")
    if topic:
        lines.append(f"Topic: {topic}")
    deduped_concepts = _dedupe_short(core_concepts)
    if deduped_concepts:
        lines.append(f"Core concepts: {', '.join(deduped_concepts)}")
    lines.append("Context:")
    for item in kb_items[:max_items]:
        excerpt = safe_str(item.text)
        if not excerpt:
            continue
        if len(excerpt) > 220:
            excerpt = excerpt[:220].rstrip() + "..."
        lines.append(f"- {excerpt}")
    lines.extend(
        [
            "Instruction:",
            "Use this only if relevant. Do not copy blindly. "
            "Solve the student's question step by step.",
        ]
    )
    combined = "\n".join(lines)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip()
    return combined


class ContextRequestBuilder:
    """Build ContextRetrievalRequest from query + lean orchestrated classification."""

    @staticmethod
    def from_query_and_classification(
        *,
        request_id: str,
        query: str,
        classification: dict[str, Any],
        confidence: float | None = None,
    ) -> ContextRetrievalRequest:
        settings = get_settings()
        exam = ContextRequestBuilder._detect_exam(query)
        topic = classification.get("topic")
        topic_str = str(topic) if topic else None
        tags = normalize_retrieval_tags(classification.get("retrieval_tags"))
        topic_conf = classification.get("topic_confidence")
        topic_confidence = float(topic_conf) if topic_conf is not None else None
        pattern_topic_candidate = classification.get("pattern_topic_candidate")
        pattern_family_candidate = classification.get("pattern_family_candidate")
        need_web_search = bool(classification.get("need_web_search"))
        web_search_reason = classification.get("web_search_reason")
        web_search_query = classification.get("web_search_query")
        return ContextRetrievalRequest(
            request_id=request_id,
            query=query,
            subject=str(classification.get("subject") or "general"),
            intent=str(classification.get("intent") or "explain"),
            difficulty=str(classification.get("difficulty") or "default"),
            confidence=confidence,
            topic=topic_str,
            topic_confidence=topic_confidence,
            pattern_topic_candidate=str(pattern_topic_candidate)
            if pattern_topic_candidate
            else None,
            pattern_family_candidate=str(pattern_family_candidate)
            if pattern_family_candidate
            else None,
            retrieval_tags=tags,
            exam=exam,
            max_context_chars=settings.context_max_chars,
            max_results=settings.context_kb_top_k,
            retrieval_version=settings.context_retrieval_version,
            need_web_search=need_web_search,
            web_search_reason=str(web_search_reason) if web_search_reason else None,
            web_search_query=str(web_search_query) if web_search_query else None,
        )

    @staticmethod
    def _detect_exam(query: str) -> str | None:
        lower = query.lower()
        for keyword in _EXAM_KEYWORDS:
            if keyword in lower:
                return keyword.strip()
        return None


class ContextRetrievalService:
    """Retrieve, rerank, and format compact context_text for the generator."""

    def __init__(
        self,
        *,
        kb_retriever: BedrockKnowledgeBaseRetriever | None = None,
        web_search_tool: WebSearchTool | None = None,
        brief_builder: SolutionBriefBuilder | None = None,
        student_retrieval_service: StudentRetrievalService | None = None,
    ) -> None:
        self._kb_retriever = kb_retriever or BedrockKnowledgeBaseRetriever()
        self._web_search_tool = web_search_tool or WebSearchTool()
        self._brief_builder = brief_builder or SolutionBriefBuilder()
        self._student_retrieval_service = student_retrieval_service or StudentRetrievalService()

    def retrieve_context(
        self,
        request: ContextRetrievalRequest,
        *,
        on_before_web_search: Callable[[], None] | None = None,
        on_web_search_retry: Callable[[], None] | None = None,
        on_web_search_weak_context: Callable[[], None] | None = None,
    ) -> ContextRetrievalResult:
        settings = get_settings()
        web_decision = evaluate_web_search_decision(request, settings)
        logger.info(
            "web_search_decision  request_id=%s  need_web_search=%s  reason=%s  "
            "provider=%s  enabled=%s  will_call=%s",
            request.request_id,
            web_decision.need_web_search,
            web_decision.reason,
            web_decision.provider,
            web_decision.enabled,
            web_decision.will_call,
        )
        log_event(
            "web_search_decision",
            component="doubt_solver.web_search",
            stage="retrieve",
            status="required" if web_decision.direct_web else "skipped",
            details={
                "required": web_decision.direct_web,
                "reason": web_decision.reason,
                "provider": web_decision.provider,
                "provider_enabled": web_decision.enabled,
                "provider_configured": self._credentials_ready(settings),
                "will_call": web_decision.will_call,
                "search_term_present": bool(web_decision.query),
            },
        )

        if request.need_web_search:
            return self._retrieve_direct_web_context(
                request,
                web_decision=web_decision,
                on_before_web_search=on_before_web_search,
                on_web_search_retry=on_web_search_retry,
                on_web_search_weak_context=on_web_search_weak_context,
            )

        if settings.retrieval_provider == "s3_vector":
            return self._retrieve_s3_vector_context(request)

        kb_result = self._retrieve_kb_context(request)
        if kb_result.retrieval_used and kb_result.context_text:
            return kb_result

        if should_attempt_web_fallback(request, kb_selected=False, settings=settings):
            fallback_will_call = web_decision.enabled and self._credentials_ready(settings)
            if fallback_will_call and on_before_web_search is not None:
                on_before_web_search()
            web_result = self._run_web_search(
                request,
                settings=settings,
                on_web_search_retry=on_web_search_retry,
                on_web_search_weak_context=on_web_search_weak_context,
            )
            if web_result.context_text:
                return web_result
        return kb_result

    def _retrieve_s3_vector_context(
        self,
        request: ContextRetrievalRequest,
    ) -> ContextRetrievalResult:
        """Use S3 Vectors as the only student PatternGraph retrieval provider."""
        retrieval_context = self._student_retrieval_service.retrieve(request)
        context_text = render_retrieval_context(retrieval_context)
        retrieval_used = retrieval_context.mode != "fresh_solve"
        return ContextRetrievalResult(
            context_text=context_text,
            item_count=1 if retrieval_used else 0,
            retrieval_used=retrieval_used,
            reason=retrieval_context.retrieval_trace.fallback_reason or retrieval_context.mode,
            retrieval_context=retrieval_context,
        )

    def _retrieve_direct_web_context(
        self,
        request: ContextRetrievalRequest,
        *,
        web_decision,
        on_before_web_search: Callable[[], None] | None = None,
        on_web_search_retry: Callable[[], None] | None = None,
        on_web_search_weak_context: Callable[[], None] | None = None,
    ) -> ContextRetrievalResult:
        if web_decision.will_call and on_before_web_search is not None:
            on_before_web_search()

        if should_skip_kb_for_direct_web(request):
            return self._run_web_search(
                request,
                settings=get_settings(),
                on_web_search_retry=on_web_search_retry,
                on_web_search_weak_context=on_web_search_weak_context,
            )

        web_result = self._run_web_search(
            request,
            settings=get_settings(),
            on_web_search_retry=on_web_search_retry,
            on_web_search_weak_context=on_web_search_weak_context,
        )
        if web_result.context_text:
            return web_result
        return self._retrieve_kb_context(request)

    def _retrieve_kb_context(self, request: ContextRetrievalRequest) -> ContextRetrievalResult:
        kb_subject = map_app_subject_to_kb(request.subject)
        hints = resolve_retrieval_hints(
            request.query,
            request.subject,
            {
                "topic": request.topic,
                "topic_confidence": request.topic_confidence,
                "pattern_topic_candidate": request.pattern_topic_candidate,
                "pattern_family_candidate": request.pattern_family_candidate,
                "retrieval_tags": request.retrieval_tags,
            },
        )
        pattern_topic_key = hints.pattern_topic_key
        pattern_family_key = hints.pattern_family_key

        decision = self.decide_retrieval(request)
        logger.info(
            "context_retrieval_decision  request_id=%s  use_kb=%s  reason=%s  "
            "subject=%s  intent=%s  difficulty=%s  topic_hint=%s  "
            "pattern_topic_key=%s  hint_strength=%s  hint_source=%s  "
            "retrieval_tags_count=%d  kb_subject=%s",
            request.request_id,
            decision.use_kb,
            decision.reason,
            request.subject,
            request.intent,
            request.difficulty,
            hints.topic_hint or "",
            pattern_topic_key or "",
            hints.strength,
            hints.hint_source,
            len(hints.retrieval_tags),
            kb_subject or "",
        )

        if not decision.use_kb:
            logger.info(
                "context_retrieval_summary  request_id=%s  use_kb=false  reason=%s  "
                "context_chars=0",
                request.request_id,
                decision.reason,
            )
            return ContextRetrievalResult(
                context_text="",
                item_count=0,
                retrieval_used=False,
                reason=decision.reason,
            )

        retrieval_query = self._build_retrieval_query(
            request,
            kb_subject=kb_subject,
            pattern_topic_key=pattern_topic_key,
        )
        items, aws_count, winning_lane = self._retrieve_with_lanes(
            request,
            decision,
            kb_subject=kb_subject,
            pattern_topic_key=pattern_topic_key,
            pattern_family_key=pattern_family_key,
            retrieval_query=retrieval_query,
        )
        selected = self._rerank_and_select(
            request,
            decision,
            items,
            kb_subject=kb_subject,
            pattern_topic_key=pattern_topic_key,
            pattern_family_key=pattern_family_key,
            retrieval_tags=hints.retrieval_tags,
        )
        context_text = self._compose_generator_context(
            request,
            kb_items=selected,
            web_items=[],
            max_chars=decision.max_context_chars,
        )
        result_reason = self._resolve_result_reason(
            decision,
            items,
            selected,
            any_aws_results=aws_count > 0,
        )
        logger.info(
            "context_retrieval_format  request_id=%s  context_chars=%d  item_count=%d  "
            "result_reason=%s",
            request.request_id,
            len(context_text),
            len(selected),
            result_reason,
        )
        logger.info(
            "context_retrieval_summary  request_id=%s  use_kb=true  subject=%s  "
            "topic=%s  lane=%s  aws=%d  normalized=%d  selected=%d  reason=%s  "
            "context_chars=%d",
            request.request_id,
            request.subject,
            pattern_topic_key or "none",
            winning_lane or "none",
            aws_count,
            len(items),
            len(selected),
            result_reason,
            len(context_text),
        )
        return ContextRetrievalResult(
            context_text=context_text,
            item_count=len(selected),
            retrieval_used=bool(selected),
            reason=result_reason,
        )

    @staticmethod
    def _credentials_ready(settings) -> bool:
        return credentials_ready(settings)

    def _run_web_search(
        self,
        request: ContextRetrievalRequest,
        *,
        settings,
        on_web_search_retry: Callable[[], None] | None = None,
        on_web_search_weak_context: Callable[[], None] | None = None,
    ) -> ContextRetrievalResult:
        search_query = resolve_web_search_query(request)
        started_at = time.monotonic()
        try:
            web_result = self._web_search_tool.search(
                WebSearchRequest(
                    request_id=request.request_id,
                    query=request.query,
                    web_search_query=search_query,
                    subject=request.subject,
                    topic=request.topic,
                    retrieval_tags=request.retrieval_tags,
                    web_search_reason=request.web_search_reason,
                    timeout_seconds=settings.web_search_timeout_seconds,
                ),
                on_retry_sources=on_web_search_retry,
            )
        except Exception as exc:
            log_event(
                "web_search_execution",
                component="doubt_solver.web_search",
                stage="retrieve",
                status="failed",
                duration_ms=int((time.monotonic() - started_at) * 1000),
                error_code="WEB_SEARCH_PROVIDER_FAILED",
                details={
                    "provider": settings.web_search_provider,
                    "error_type": type(exc).__name__,
                },
                level=logging.WARNING,
            )
            raise
        execution_status = (
            "succeeded"
            if web_result.used
            else "weak"
            if web_result.weak_context
            else "skipped"
            if not web_result.executed
            else "empty"
        )
        log_event(
            "web_search_execution",
            component="doubt_solver.web_search",
            stage="retrieve",
            status=execution_status,
            duration_ms=web_result.duration_ms,
            details={
                "provider": web_result.provider,
                "attempts": web_result.attempt_count,
                "candidate_count": web_result.candidate_count,
                "selected_count": len(web_result.items),
                "official_count": web_result.official_count,
                "reputable_count": web_result.reputable_count,
                "context_characters": len(web_result.context_text),
            },
        )
        if web_result.weak_context or not web_result.items:
            if request.need_web_search and on_web_search_weak_context is not None:
                on_web_search_weak_context()
            safe_context = web_result.context_text or ""
            return ContextRetrievalResult(
                context_text=safe_context,
                item_count=0,
                retrieval_used=bool(safe_context),
                reason=web_result.reason,
                retrieval_context=StudentRetrievalContext.fresh_solve(
                    web_result.reason
                ),
                web_search_executed=web_result.executed,
                web_search_status=execution_status,
                web_search_provider=web_result.provider,
                web_context_chars=0,
                web_result_count=0,
                web_citation_count=0,
            )

        brief_result = self._brief_builder.build(
            request,
            kb_items=[],
            web_items=web_result.items,
        )
        context_text = self._brief_builder.compose_context_text(
            brief_text=brief_result.brief_text,
            web_section=web_result.context_text,
            max_chars=request.max_context_chars or settings.context_max_chars,
        )
        return ContextRetrievalResult(
            context_text=context_text,
            item_count=len(web_result.items),
            retrieval_used=bool(context_text),
            reason=web_result.reason,
            retrieval_context=StudentRetrievalContext.fresh_solve(
                web_result.reason
            ),
            web_search_executed=web_result.executed,
            web_search_status=execution_status,
            web_search_provider=web_result.provider,
            web_context_chars=len(web_result.context_text),
            web_result_count=len(web_result.items),
            web_citation_count=sum(bool(item.url) for item in web_result.items),
        )

    def _compose_generator_context(
        self,
        request: ContextRetrievalRequest,
        *,
        kb_items: list[RetrievedContextItem],
        web_items: list[WebSearchItem],
        max_chars: int,
    ) -> str:
        if not kb_items and not web_items and request.difficulty != "advanced":
            return ""

        selected_count = len(kb_items)
        try:
            brief_result = self._brief_builder.build(
                request,
                kb_items=kb_items,
                web_items=web_items,
            )
            context_text = self._brief_builder.compose_context_text(
                brief_text=brief_result.brief_text,
                web_section="",
                max_chars=max_chars,
            )
            if context_text:
                logger.info(
                    "context_retrieval_brief  request_id=%s  solution_brief_builder_used=true  "
                    "fallback_context_used=false  selected_count=%d  brief_chars=%d  "
                    "context_chars=%d",
                    request.request_id,
                    selected_count,
                    len(brief_result.brief_text),
                    len(context_text),
                )
                return context_text
            if not brief_result.used and kb_items:
                context_text = _format_kb_fallback_context(
                    request,
                    kb_items,
                    max_chars=max_chars,
                )
                logger.info(
                    "context_retrieval_brief  request_id=%s  solution_brief_builder_used=false  "
                    "fallback_context_used=true  selected_count=%d  context_chars=%d",
                    request.request_id,
                    selected_count,
                    len(context_text),
                )
                return context_text
            if not kb_items:
                return ""
            raise ValueError("solution_brief_empty_with_selected_kb")
        except Exception as exc:
            if not kb_items:
                logger.warning(
                    "context_retrieval_brief  request_id=%s  solution_brief_failed=true  "
                    "error_type=%s  error_message_short=%s  phase=solution_brief  "
                    "selected_count=0  context_chars_before_error=0",
                    request.request_id,
                    type(exc).__name__,
                    str(exc)[:120],
                )
                logger.debug(
                    "context_retrieval_brief traceback  request_id=%s",
                    request.request_id,
                    exc_info=True,
                )
                raise

            context_text = _format_kb_fallback_context(
                request,
                kb_items,
                max_chars=max_chars,
            )
            logger.warning(
                "context_retrieval_brief  request_id=%s  solution_brief_failed=true  "
                "fallback_context_used=true  error_type=%s  error_message_short=%s  "
                "phase=solution_brief  selected_count=%d  final_context_chars=%d",
                request.request_id,
                type(exc).__name__,
                str(exc)[:120],
                selected_count,
                len(context_text),
            )
            logger.debug(
                "context_retrieval_brief traceback  request_id=%s",
                request.request_id,
                exc_info=True,
            )
            return context_text

    def decide_retrieval(self, request: ContextRetrievalRequest) -> ContextRetrievalDecision:
        settings = get_settings()
        query_lower = request.query.lower()
        max_chars = request.max_context_chars or settings.context_max_chars
        top_k = request.max_results or settings.context_kb_top_k
        rerank_top_n = settings.context_rerank_top_n

        if not settings.enable_kb_retrieval:
            return ContextRetrievalDecision(
                use_kb=False,
                reason="kb_disabled",
                top_k=0,
                rerank_top_n=0,
                max_context_chars=max_chars,
            )

        if self._should_skip_kb(request, query_lower):
            return ContextRetrievalDecision(
                use_kb=False,
                reason="simple_query_skip",
                top_k=0,
                rerank_top_n=0,
                max_context_chars=max_chars,
            )

        use_kb = False
        reason = "default_kb"

        if request.intent in _KB_INTENTS:
            use_kb = True
            reason = f"intent_{request.intent}"
        elif request.difficulty in _KB_DIFFICULTIES:
            use_kb = True
            reason = f"difficulty_{request.difficulty}"
        elif request.subject == "reasoning" and request.difficulty != "basic":
            use_kb = True
            reason = "reasoning_subject"
        elif request.exam:
            use_kb = True
            reason = "exam_keyword"
        elif any(kw in query_lower for kw in _RETRIEVAL_KEYWORDS):
            use_kb = True
            reason = "retrieval_keyword"
        else:
            use_kb = True
            reason = "default_unsure_kb"
            top_k = min(top_k, 3)

        return ContextRetrievalDecision(
            use_kb=use_kb,
            reason=reason,
            filters={},
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            max_context_chars=max_chars,
        )

    def _build_retrieval_lanes(
        self,
        *,
        kb_subject: str | None,
        pattern_topic_key: str | None,
        pattern_family_key: str | None,
    ) -> list[tuple[str, dict[str, str]]]:
        settings = get_settings()
        production = self._production_safe_filters(settings)
        lanes: list[tuple[str, dict[str, str]]] = []

        if kb_subject and pattern_topic_key and pattern_family_key:
            lane_filters = {
                "subject": kb_subject,
                "patternTopicKey": pattern_topic_key,
                "patternFamilyKey": pattern_family_key,
                **production,
            }
            lanes.append((LANE_SUBJECT_TOPIC_FAMILY, lane_filters))

        if kb_subject and pattern_topic_key:
            lane_filters = {
                "subject": kb_subject,
                "patternTopicKey": pattern_topic_key,
                **production,
            }
            lanes.append((LANE_SUBJECT_TOPIC, lane_filters))

        if kb_subject:
            lane_filters = {"subject": kb_subject, **production}
            lanes.append((LANE_SUBJECT_ONLY, lane_filters))

        if kb_subject:
            lanes.append((LANE_RELAXED_SUBJECT_ONLY, {"subject": kb_subject}))

        broad_filters: dict[str, str] = {}
        if (
            settings.context_kb_schema_version
            and settings.context_kb_schema_version_mandatory
        ):
            broad_filters["schemaVersion"] = settings.context_kb_schema_version
        lanes.append((LANE_BROAD_SEMANTIC, broad_filters))

        return lanes[:_MAX_RETRIEVAL_LANES]

    @staticmethod
    def _production_safe_filters(settings: Any) -> dict[str, str]:
        filters: dict[str, str] = {}
        if settings.context_kb_schema_version:
            filters["schemaVersion"] = settings.context_kb_schema_version
        if settings.context_kb_taxonomy_approved_only:
            filters["taxonomyReviewRequired"] = "false"
        return filters

    def _retrieve_with_lanes(
        self,
        request: ContextRetrievalRequest,
        decision: ContextRetrievalDecision,
        *,
        kb_subject: str | None,
        pattern_topic_key: str | None,
        pattern_family_key: str | None,
        retrieval_query: str,
    ) -> tuple[list[RetrievedContextItem], int, str | None]:
        lanes = self._build_retrieval_lanes(
            kb_subject=kb_subject,
            pattern_topic_key=pattern_topic_key,
            pattern_family_key=pattern_family_key,
        )
        top_k = decision.top_k
        lanes_attempted = 0
        any_aws_results = False
        last_aws_count = 0

        for lane, filters in lanes:
            lanes_attempted += 1
            relaxed = lane in RELAXED_LANES
            relaxed_keys = sorted(
                key
                for key in ("schemaVersion", "taxonomyReviewRequired")
                if key not in filters
            )
            items, aws_count = self._kb_retriever.retrieve_lane(
                request=request,
                lane=lane,
                filters=filters,
                retrieval_query=retrieval_query,
                top_k=top_k,
                relaxed_filters=relaxed,
            )
            if aws_count > 0:
                any_aws_results = True
                last_aws_count = aws_count
            if items:
                if relaxed and relaxed_keys:
                    logger.info(
                        "context_retrieval_lane_relaxed  request_id=%s  lane=%s  "
                        "relaxed_filter_keys=%s",
                        request.request_id,
                        lane,
                        relaxed_keys,
                    )
                return items, aws_count, lane

        outcome = (
            "normalization_dropped_all_candidates"
            if any_aws_results
            else "no_kb_candidates"
        )
        logger.info(
            "context_retrieval_lanes_exhausted  request_id=%s  lanes_attempted=%d  "
            "normalized_count=0  any_aws_results=%s  outcome=%s",
            request.request_id,
            lanes_attempted,
            any_aws_results,
            outcome,
        )
        return [], last_aws_count, None

    @staticmethod
    def _resolve_result_reason(
        decision: ContextRetrievalDecision,
        items: list[RetrievedContextItem],
        selected: list[RetrievedContextItem],
        *,
        any_aws_results: bool = False,
    ) -> str:
        if not items:
            if any_aws_results:
                return "normalization_dropped_all_candidates"
            return "no_kb_candidates"
        if not selected:
            return "no_high_confidence_context"
        return "context_selected"

    @staticmethod
    def _build_retrieval_query(
        request: ContextRetrievalRequest,
        *,
        kb_subject: str | None,
        pattern_topic_key: str | None,
    ) -> str:
        subject_label = kb_subject or request.subject
        lines = [
            f"Question: {request.query.strip()}",
            f"Subject: {subject_label}",
        ]
        if pattern_topic_key:
            lines.append(f"Topic hint: {pattern_topic_key}")
        lines.append(f"Intent: {request.intent}")
        lines.append(f"Difficulty: {request.difficulty}")
        return "\n".join(lines)

    def _should_skip_kb(self, request: ContextRetrievalRequest, query_lower: str) -> bool:
        if request.intent != "solve":
            return False
        if request.subject not in {"math", "general"}:
            return False
        if request.difficulty not in {"basic", "default"}:
            return False
        if len(request.query.strip()) > 40:
            return False
        if request.exam or any(kw in query_lower for kw in _RETRIEVAL_KEYWORDS):
            return False
        if _SIMPLE_ARITHMETIC_RE.match(request.query):
            return True
        words = query_lower.split()
        return len(words) <= 6 and "profit" not in query_lower and "percent" not in query_lower

    def _rerank_and_select(
        self,
        request: ContextRetrievalRequest,
        decision: ContextRetrievalDecision,
        items: list[RetrievedContextItem],
        *,
        kb_subject: str | None,
        pattern_topic_key: str | None,
        pattern_family_key: str | None,
        retrieval_tags: list[str] | None = None,
    ) -> list[RetrievedContextItem]:
        if not items:
            logger.info(
                "context_retrieval_rerank  request_id=%s  candidate_count=0  "
                "selected_count=0  outcome=no_candidates_for_rerank",
                request.request_id,
            )
            return []

        settings = get_settings()
        query_tokens = set(re.findall(r"[a-z0-9]+", request.query.lower()))
        scored: list[tuple[float, float, RetrievedContextItem, RerankScoreBreakdown]] = []

        for item in items:
            is_relaxed_lane = (item.match_lane or "") in RELAXED_LANES
            rerank_score, rerank_confidence, why, risk, breakdown = self._score_candidate(
                request,
                item,
                kb_subject=kb_subject,
                pattern_topic_key=pattern_topic_key,
                pattern_family_key=pattern_family_key,
                query_tokens=query_tokens,
                retrieval_tags=retrieval_tags or [],
                is_relaxed_lane=is_relaxed_lane,
                taxonomy_approved_only=settings.context_kb_taxonomy_approved_only,
            )
            if rerank_confidence <= 0.0:
                continue
            enriched = item.model_copy(
                update={
                    "rerank_score": rerank_score,
                    "rerank_confidence": rerank_confidence,
                    "why_candidate": why,
                    "risk": risk or item.risk,
                }
            )
            scored.append((rerank_confidence, rerank_score, enriched, breakdown))

        scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)

        above_threshold = [
            (confidence, score, item, breakdown)
            for confidence, score, item, breakdown in scored
            if confidence >= _RERANK_CONFIDENCE_THRESHOLD
        ]
        selected_before_dedupe = len(above_threshold)

        seen: set[str] = set()
        selected: list[RetrievedContextItem] = []
        low_confidence = len(scored) - len(above_threshold)
        limit = max(decision.rerank_top_n, 1)

        for _confidence, _score, item, _breakdown in above_threshold:
            dedupe_key = item.source_id or item.text[:120]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            selected.append(item)
            if len(selected) >= limit:
                break

        logger.info(
            "context_rerank_selection  request_id=%s  scored_count=%d  "
            "selected_before_dedupe=%d  selected_after_dedupe=%d  "
            "low_confidence_count=%d",
            request.request_id,
            len(scored),
            selected_before_dedupe,
            len(selected),
            low_confidence,
        )

        self._log_rerank_breakdown(request.request_id, scored, selected)

        if selected:
            top_confidence = selected[0].rerank_confidence or 0.0
            pattern_ids = [
                str(item.metadata.get("patternId", ""))
                for item in selected
                if item.metadata.get("patternId")
            ]
            match_lanes = [item.match_lane or "" for item in selected]
            logger.info(
                "context_rerank_summary  request_id=%s  candidate_count=%d  "
                "selected_count=%d  top_confidence=%.2f  reason=context_selected",
                request.request_id,
                len(items),
                len(selected),
                top_confidence,
            )
            logger.info(
                "context_retrieval_rerank  request_id=%s  candidate_count=%d  "
                "selected_count=%d  top_confidence=%.2f  selected_pattern_ids=%s  "
                "top_match_lanes=%s  low_confidence_candidates=%d",
                request.request_id,
                len(items),
                len(selected),
                top_confidence,
                pattern_ids,
                match_lanes,
                low_confidence,
            )
            return selected

        top_confidence = scored[0][0] if scored else 0.0
        near_miss = 0.70 <= top_confidence < _RERANK_CONFIDENCE_THRESHOLD
        top_pattern_ids = [
            str(item.metadata.get("patternId", ""))
            for _conf, _score, item, _bd in scored[:3]
            if item.metadata.get("patternId")
        ]
        top_match_lanes = [item.match_lane or "" for _conf, _score, item, _bd in scored[:3]]
        if not scored:
            logger.info(
                "context_retrieval_rerank  request_id=%s  candidate_count=%d  "
                "selected_count=0  outcome=all_candidates_rejected",
                request.request_id,
                len(items),
            )
            logger.info(
                "context_rerank_summary  request_id=%s  candidate_count=%d  "
                "selected_count=0  top_confidence=0.00  reason=all_rejected",
                request.request_id,
                len(items),
            )
        else:
            logger.info(
                "context_retrieval_rerank  request_id=%s  candidate_count=%d  "
                "selected_count=0  top_confidence=%.2f  threshold=%.2f  "
                "near_miss=%s  reason=below_confidence_threshold  top_pattern_ids=%s  "
                "top_match_lanes=%s  low_confidence_candidates=%d",
                request.request_id,
                len(items),
                top_confidence,
                _RERANK_CONFIDENCE_THRESHOLD,
                str(near_miss).lower(),
                top_pattern_ids,
                top_match_lanes,
                low_confidence or len(scored),
            )
            logger.info(
                "context_rerank_summary  request_id=%s  candidate_count=%d  "
                "selected_count=0  top_confidence=%.2f  near_miss=%s  "
                "reason=no_high_confidence_context",
                request.request_id,
                len(items),
                top_confidence,
                str(near_miss).lower(),
            )
        return []

    @staticmethod
    def _log_rerank_breakdown(
        request_id: str,
        scored: list[tuple[float, float, RetrievedContextItem, RerankScoreBreakdown]],
        selected: list[RetrievedContextItem],
    ) -> None:
        selected_ids = {
            str(item.metadata.get("patternId", "")) or (item.source_id or "")
            for item in selected
        }
        for rank, (confidence, _score, item, breakdown) in enumerate(scored[:3], start=1):
            pattern_id = str(item.metadata.get("patternId", "")) or "unknown"
            below = confidence < _RERANK_CONFIDENCE_THRESHOLD
            rejection = breakdown.rejection_reason if below else "none"
            logger.info(
                "context_rerank_breakdown  request_id=%s  rank=%d  patternId=%s  "
                "match_lane=%s  final_confidence=%.2f  bedrock_score=%s  "
                "subject_match=%s  topic_match=%s  family_match=%s  "
                "keyword_overlap_score=%.2f  concept_tag_overlap_score=%.2f  "
                "tag_overlap_score=%.2f  matched_tags_count=%d  matched_tags_sample=%s  "
                "approved_signal=%s  risk=%s  rejection_reason=%s  selected=%s",
                request_id,
                rank,
                pattern_id,
                item.match_lane or "unknown",
                breakdown.final_confidence,
                f"{breakdown.bedrock_score:.4f}"
                if breakdown.bedrock_score is not None
                else "none",
                str(breakdown.subject_match).lower(),
                str(breakdown.topic_match).lower(),
                str(breakdown.family_match).lower(),
                breakdown.keyword_overlap_score,
                breakdown.concept_tag_overlap_score,
                breakdown.tag_overlap_score,
                breakdown.matched_tags_count,
                breakdown.matched_tags_sample or "none",
                str(breakdown.approved_signal).lower(),
                breakdown.risk or "none",
                rejection,
                str(pattern_id in selected_ids).lower(),
            )

    def _score_candidate(
        self,
        request: ContextRetrievalRequest,
        item: RetrievedContextItem,
        *,
        kb_subject: str | None,
        pattern_topic_key: str | None,
        pattern_family_key: str | None,
        query_tokens: set[str],
        retrieval_tags: list[str],
        is_relaxed_lane: bool,
        taxonomy_approved_only: bool,
    ) -> tuple[float, float, str, str, RerankScoreBreakdown]:
        empty_breakdown = RerankScoreBreakdown(
            bedrock_score=item.score,
            subject_match=False,
            topic_match=False,
            family_match=False,
            keyword_overlap_score=0.0,
            concept_tag_overlap_score=0.0,
            tag_overlap_score=0.0,
            matched_tags_count=0,
            matched_tags_sample="",
            approved_signal=False,
            risk="",
            rejection_reason="empty_text",
            final_confidence=0.0,
        )
        if not item.text.strip():
            return 0.0, 0.0, "rejected", "empty_text", empty_breakdown

        meta = item.metadata or {}
        reasons: list[str] = []
        risks: list[str] = []
        if item.risk:
            risks.append(item.risk)

        kb_meta_subject = str(meta.get("subject", "")).strip().upper()
        expected_subject = (kb_subject or "").upper()
        if expected_subject and kb_meta_subject and kb_meta_subject != expected_subject:
            bd = RerankScoreBreakdown(
                bedrock_score=item.score,
                subject_match=False,
                topic_match=False,
                family_match=False,
                keyword_overlap_score=0.0,
                concept_tag_overlap_score=0.0,
                tag_overlap_score=0.0,
                matched_tags_count=0,
                matched_tags_sample="",
                approved_signal=False,
                risk="subject_mismatch",
                rejection_reason="subject_mismatch",
                final_confidence=0.0,
            )
            return 0.0, 0.0, "rejected", "subject_mismatch", bd

        meta_topic = str(meta.get("patternTopicKey", "")).strip().upper()
        expected_topic = (pattern_topic_key or "").upper()
        if expected_topic and meta_topic and meta_topic != expected_topic:
            bd = RerankScoreBreakdown(
                bedrock_score=item.score,
                subject_match=expected_subject == kb_meta_subject,
                topic_match=False,
                family_match=False,
                keyword_overlap_score=0.0,
                concept_tag_overlap_score=0.0,
                tag_overlap_score=0.0,
                matched_tags_count=0,
                matched_tags_sample="",
                approved_signal=False,
                risk="topic_mismatch",
                rejection_reason="topic_mismatch",
                final_confidence=0.0,
            )
            return 0.0, 0.0, "rejected", "topic_mismatch", bd

        meta_family = str(meta.get("patternFamilyKey", "")).strip().upper()
        expected_family = (pattern_family_key or "").upper()
        # Family hints are soft signals — downrank mismatch, never hard-reject.
        family_mismatch_penalty = 0.0

        tax_raw = meta.get("taxonomyReviewRequired")
        approved_signal = False
        if tax_raw is not None and str(tax_raw).strip():
            tax_review = str(tax_raw).strip().lower()
            if tax_review == "true":
                if taxonomy_approved_only and not is_relaxed_lane:
                    bd = RerankScoreBreakdown(
                        bedrock_score=item.score,
                        subject_match=False,
                        topic_match=False,
                        family_match=False,
                        keyword_overlap_score=0.0,
                        concept_tag_overlap_score=0.0,
                        tag_overlap_score=0.0,
                        matched_tags_count=0,
                        matched_tags_sample="",
                        approved_signal=False,
                        risk="taxonomy_review_required",
                        rejection_reason="taxonomy_review_required",
                        final_confidence=0.0,
                    )
                    return 0.0, 0.0, "rejected", "taxonomy_review_required", bd
                risks.append("taxonomy_review_required_true")
            elif tax_review == "false":
                approved_signal = True
                reasons.append("taxonomy_approved")
        else:
            risks.append("missing_taxonomy_review_flag")

        base = item.score if item.score is not None else 0.45
        bonus = 0.0
        penalty = 0.0
        subject_match = False
        topic_match = False
        family_match = False
        keyword_overlap_score = 0.0
        concept_tag_overlap_score = 0.0
        tag_overlap_score = 0.0
        matched_tags_count = 0
        matched_tags_sample = ""

        if expected_subject and kb_meta_subject == expected_subject:
            bonus += 0.12
            subject_match = True
            reasons.append("subject_match")
        elif expected_subject and not kb_meta_subject:
            penalty += 0.06
            risks.append("missing_subject_metadata")

        if expected_topic and meta_topic == expected_topic:
            bonus += 0.15
            topic_match = True
            reasons.append("topic_match")
        elif expected_topic and not meta_topic:
            risks.append("missing_pattern_topic_key")

        if expected_family and meta_family == expected_family:
            bonus += 0.10
            family_match = True
            reasons.append("family_match")
        elif expected_family and meta_family and meta_family != expected_family:
            family_mismatch_penalty = 0.08
            penalty += family_mismatch_penalty
            risks.append("family_mismatch")
        elif expected_family and not meta_family:
            risks.append("missing_pattern_family_key")

        if "missing_pattern_id" in risks or not meta.get("patternId"):
            if "missing_pattern_id" not in risks:
                risks.append("missing_pattern_id")
            penalty += 0.03

        text_tokens = set(re.findall(r"[a-z0-9]+", item.text.lower()))
        overlap = len(query_tokens & text_tokens)
        if overlap:
            keyword_overlap_score = min(overlap * 0.03, 0.15)
            bonus += keyword_overlap_score
            reasons.append(f"keyword_overlap={overlap}")

        tags_raw = meta.get("conceptTags", "")
        meta_tag_tokens: set[str] = set()
        if isinstance(tags_raw, str) and tags_raw.strip():
            meta_tag_tokens = {
                re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")
                for t in tags_raw.split(",")
                if t.strip()
            }
            tag_overlap = len(query_tokens & meta_tag_tokens)
            if tag_overlap:
                concept_tag_overlap_score = min(tag_overlap * 0.04, 0.10)
                bonus += concept_tag_overlap_score
                reasons.append(f"concept_overlap={tag_overlap}")

        hint_tag_tokens = {t.lower() for t in retrieval_tags if t}
        if meta_topic:
            meta_tag_tokens.add(meta_topic.lower())
        if meta_family:
            meta_tag_tokens.add(meta_family.lower())
        if hint_tag_tokens and meta_tag_tokens:
            matched = sorted(hint_tag_tokens & meta_tag_tokens)
            matched_tags_count = len(matched)
            if matched_tags_count:
                tag_overlap_score = min(matched_tags_count * 0.05, 0.15)
                bonus += tag_overlap_score
                matched_tags_sample = ",".join(matched[:5])
                reasons.append(f"retrieval_tag_overlap={matched_tags_count}")

        if tax_raw is not None and str(tax_raw).strip().lower() == "false":
            bonus += 0.05

        parsed_conf = _parse_metadata_confidence(meta.get("confidence"))
        if parsed_conf is not None and parsed_conf >= 0.85:
            bonus += 0.05
            reasons.append("high_metadata_confidence")
        elif meta.get("confidence") in (None, ""):
            risks.append("missing_confidence")

        if tax_raw in (None, ""):
            penalty += 0.02

        if request.difficulty == "advanced":
            comp = str(meta.get("complexityLevel", ""))
            if comp in {"3", "4", "5"}:
                bonus += 0.03
                reasons.append("complexity_soft_match")

        if tax_raw is not None and str(tax_raw).strip().lower() == "true":
            penalty += 0.20

        rerank_score = base + bonus - penalty
        rerank_confidence = min(max(rerank_score, 0.0), 1.0)
        why = ", ".join(reasons) if reasons else "vector_score"
        risk = ", ".join(dict.fromkeys(risks)) if risks else ""
        rejection_reason = ""
        if rerank_confidence < _RERANK_CONFIDENCE_THRESHOLD:
            if not topic_match and expected_topic:
                rejection_reason = "missing_topic_match"
            elif keyword_overlap_score < 0.06:
                rejection_reason = "low_keyword_overlap"
            elif base < 0.50:
                rejection_reason = "low_bedrock_score"
            else:
                rejection_reason = "below_confidence_threshold"
        breakdown = RerankScoreBreakdown(
            bedrock_score=item.score,
            subject_match=subject_match,
            topic_match=topic_match,
            family_match=family_match,
            keyword_overlap_score=keyword_overlap_score,
            concept_tag_overlap_score=concept_tag_overlap_score,
            tag_overlap_score=tag_overlap_score,
            matched_tags_count=matched_tags_count,
            matched_tags_sample=matched_tags_sample,
            approved_signal=approved_signal,
            risk=risk,
            rejection_reason=rejection_reason,
            final_confidence=rerank_confidence,
        )
        return rerank_score, rerank_confidence, why, risk, breakdown

    def _format_context(
        self,
        request: ContextRetrievalRequest,
        decision: ContextRetrievalDecision,
        items: list[RetrievedContextItem],
    ) -> str:
        """Deprecated — use _compose_generator_context."""
        return self._compose_generator_context(
            request,
            kb_items=items,
            web_items=[],
            max_chars=decision.max_context_chars,
        )


def get_context_retrieval_service() -> ContextRetrievalService:
    """Return the module-level ContextRetrievalService singleton."""
    global _context_retrieval_service  # noqa: PLW0603
    if _context_retrieval_service is None:
        _context_retrieval_service = ContextRetrievalService()
    return _context_retrieval_service


def reset_context_retrieval_service() -> None:
    """Reset singleton — for tests only."""
    global _context_retrieval_service  # noqa: PLW0603
    _context_retrieval_service = None
