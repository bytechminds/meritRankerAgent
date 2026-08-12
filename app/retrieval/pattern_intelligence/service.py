"""Deterministic Pattern candidate discovery, hydration, gating, and projection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from retrieval.gates.score_policy import is_strong_runtime_match
from retrieval.pattern_intelligence.interfaces import (
    CanonicalPatternStore,
    LinkedPatternQuestionStore,
    LinkedQuestionReranker,
    PatternVectorCandidateFinder,
    PlayablePatternQuestionStore,
)
from retrieval.pattern_intelligence.models import (
    CanonicalPatternRecord,
    CanonicalPlayableQuestion,
    DoubtPatternContext,
    DoubtQuestionReference,
    DoubtSolveFlowStep,
    PatternGenerationContext,
    PatternMatchDecision,
    PatternMatchTier,
    PatternRetrievalPlan,
    PatternRuntimeMemo,
    PatternRuntimeMode,
    PatternRuntimeRequest,
    PatternRuntimeResult,
    VectorPatternCandidate,
)

_SUBJECT_ALIASES = {
    "math": "quant",
    "quantitative": "quant",
    "quant": "quant",
    "reasoning": "reasoning",
    "english": "english",
    "general": "gk",
    "gk": "gk",
}
_BLOCKED_GRAPH_KEYS = {
    "answer",
    "answerguide",
    "canonicalanswer",
    "correctanswer",
    "explanation",
    "rawcorrectanswer",
    "solution",
    "solutiontext",
}
_SAFE_GRAPH_KEYS = (
    "identity",
    "label",
    "name",
    "operation",
    "rule",
    "formula",
    "target",
)
_SAFE_SOLVE_FLOW_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,79}$", re.IGNORECASE)
_SAFE_SOLVE_FLOW_INSTRUCTION = re.compile(
    r"^[A-Za-z][A-Za-z ,;:()'/-]{0,159}[.!?]?$"
)
_SAFE_RAW_CONDITION = re.compile(r"^[A-Za-z][A-Za-z ,;:()'/-]{0,159}[.!?]?$"
)
_SAFE_STRUCTURED_GRAPH_TOKEN = re.compile(
    r"^[A-Za-z][A-Za-z0-9 _./:+*^()%-]{0,95}$"
)


class PatternCompatibilityGuard:
    """Fail-closed compatibility checks against the raw canonical Pattern contract."""

    def evaluate(
        self,
        *,
        request: PatternRuntimeRequest,
        record: CanonicalPatternRecord,
    ) -> PatternMatchDecision:
        if _normalized(record.status) != "active":
            return _ignore(record.pattern_id, "pattern_not_active")
        if not _has_required_graph_shape(record.pattern_graph):
            return _ignore(record.pattern_id, "pattern_graph_incomplete")
        if _subject_key(request.subject) != _subject_key(record.subject):
            return _ignore(record.pattern_id, "subject_mismatch")
        if not _matches_text(request.topic, record.topic):
            return _ignore(record.pattern_id, "topic_mismatch")
        if not _matches_text(request.category, record.category):
            return _ignore(record.pattern_id, "category_mismatch")
        if not _matches_complexity(request.difficulty, record.complexity_level):
            return _ignore(record.pattern_id, "difficulty_mismatch")
        if not _matches_exam_ids(request.exam_ids, record.exams):
            return _ignore(record.pattern_id, "exam_mismatch")
        if not _matches_text(request.question_type, record.question_type):
            return _ignore(record.pattern_id, "question_type_mismatch")
        if not _matches_text(request.pattern_family_id, record.pattern_family_id):
            return _ignore(record.pattern_id, "pattern_family_mismatch")
        if not _has_required_operation_match(
            request.required_operation_ids,
            record.pattern_graph,
        ):
            return _ignore(record.pattern_id, "required_operation_mismatch")
        if not _has_required_target_match(
            request.required_target_ids,
            record.pattern_graph,
        ):
            return _ignore(record.pattern_id, "required_target_mismatch")
        if _not_same_when_conflicts(request.excluded_conditions, record.pattern_graph):
            return _ignore(record.pattern_id, "slot_not_same_when_conflict")
        if _not_same_when_matches(request.query, record.pattern_graph):
            return _ignore(record.pattern_id, "not_same_when_matched")
        return PatternMatchDecision(
            patternId=record.pattern_id,
            tier=PatternMatchTier.GUIDANCE_SAFE,
            reason="authoritative_graph_compatible",
        )


class PatternRuntimeService:
    """Resolve one safe Pattern context without global caches or AWS dependencies."""

    def __init__(
        self,
        *,
        vector_finder: PatternVectorCandidateFinder,
        pattern_store: CanonicalPatternStore,
        linked_question_store: LinkedPatternQuestionStore | None = None,
        linked_question_reranker: LinkedQuestionReranker | None = None,
        playable_question_store: PlayablePatternQuestionStore | None = None,
        reuse_enabled: bool = False,
        compatibility_guard: PatternCompatibilityGuard | None = None,
    ) -> None:
        self._vector_finder = vector_finder
        self._pattern_store = pattern_store
        self._linked_question_store = linked_question_store
        self._linked_question_reranker = linked_question_reranker
        self._playable_question_store = playable_question_store
        self._reuse_enabled = reuse_enabled
        self._compatibility_guard = compatibility_guard or PatternCompatibilityGuard()

    def resolve_practice(
        self,
        request: PatternRuntimeRequest,
        *,
        memo: PatternRuntimeMemo | None = None,
    ) -> PatternRuntimeResult:
        """Return guidance-only context; current contracts never emit REUSE_SAFE."""
        return self._resolve(
            request=request.model_copy(update={"mode": PatternRuntimeMode.PRACTICE}),
            memo=memo or PatternRuntimeMemo(),
        )

    def resolve_doubt(
        self,
        request: PatternRuntimeRequest,
        *,
        memo: PatternRuntimeMemo | None = None,
    ) -> PatternRuntimeResult:
        """Return guidance plus optional bounded linked question references."""
        return self._resolve(
            request=request.model_copy(update={"mode": PatternRuntimeMode.DOUBT}),
            memo=memo or PatternRuntimeMemo(),
        )

    def _resolve(
        self,
        *,
        request: PatternRuntimeRequest,
        memo: PatternRuntimeMemo,
    ) -> PatternRuntimeResult:
        plan = build_retrieval_plan(request)
        warnings: list[str] = []
        candidate_ids = self._candidate_ids(
            request=request,
            plan=plan,
            memo=memo,
            warnings=warnings,
        )
        records = self._hydrate_patterns(candidate_ids=candidate_ids, memo=memo, warnings=warnings)
        decisions: list[PatternMatchDecision] = []
        selected: CanonicalPatternRecord | None = None
        for pattern_id in candidate_ids:
            record = records.get(pattern_id)
            if record is None:
                decisions.append(_ignore(pattern_id, "pattern_not_found"))
                continue
            if plan.strategy == "vector":
                version_hash = memo.candidate_versions.get(
                    _candidate_cache_key(request, plan), {}
                ).get(pattern_id)
                if not version_hash or not record.current_version_hash:
                    decisions.append(_ignore(pattern_id, "pattern_version_unavailable"))
                    continue
                if record.current_version_hash != version_hash:
                    decisions.append(_ignore(pattern_id, "pattern_version_mismatch"))
                    continue
            elif (
                request.server_pattern_version_hash
                and record.current_version_hash != request.server_pattern_version_hash
            ):
                decisions.append(_ignore(pattern_id, "pattern_version_mismatch"))
                continue
            decision = self._compatibility_guard.evaluate(request=request, record=record)
            decisions.append(decision)
            if decision.tier is PatternMatchTier.GUIDANCE_SAFE:
                selected = record
                break

        if selected is None:
            return PatternRuntimeResult(
                plan=plan,
                decisions=tuple(decisions),
                warnings=tuple(warnings),
            )

        generation_context = build_generation_context(selected)
        if request.mode is PatternRuntimeMode.PRACTICE:
            reuse_question = self._select_reuse_question(
                request=request,
                pattern=selected,
                memo=memo,
                warnings=warnings,
            )
            return PatternRuntimeResult(
                plan=plan,
                tier=(
                    PatternMatchTier.REUSE_SAFE
                    if reuse_question is not None
                    else PatternMatchTier.GUIDANCE_SAFE
                ),
                selectedPatternId=selected.pattern_id,
                selectedPatternVersionHash=selected.current_version_hash,
                generationContext=generation_context,
                reuseQuestion=reuse_question,
                decisions=tuple(decisions),
                warnings=tuple(warnings),
            )

        references, rerank_used, rerank_warning = self._linked_references(
            request=request,
            pattern_id=selected.pattern_id,
            memo=memo,
        )
        if rerank_warning:
            warnings.append(rerank_warning)
        return PatternRuntimeResult(
            plan=plan,
            tier=PatternMatchTier.GUIDANCE_SAFE,
            selectedPatternId=selected.pattern_id,
            selectedPatternVersionHash=selected.current_version_hash,
            doubtContext=DoubtPatternContext(
                **generation_context.model_dump(by_alias=False),
                questionReferences=tuple(references),
                solveFlowSteps=_approved_solve_flow_steps(selected),
            ),
            decisions=tuple(decisions),
            warnings=tuple(warnings),
            rerankUsed=rerank_used,
        )

    def _select_reuse_question(
        self,
        *,
        request: PatternRuntimeRequest,
        pattern: CanonicalPatternRecord,
        memo: PatternRuntimeMemo,
        warnings: list[str],
    ) -> CanonicalPlayableQuestion | None:
        if (
            not self._reuse_enabled
            or self._playable_question_store is None
            or not request.student_history_checked
        ):
            return None
        questions = self._load_playable_questions(
            pattern_id=pattern.pattern_id,
            memo=memo,
            warnings=warnings,
        )
        excluded = {
            value.strip()
            for value in (*request.excluded_question_ids, *request.seen_question_ids)
            if value.strip()
        }
        for question in questions:
            if question.question_id in excluded:
                continue
            if _playable_question_matches(request, pattern, question):
                return question
        return None

    def _candidate_ids(
        self,
        *,
        request: PatternRuntimeRequest,
        plan: PatternRetrievalPlan,
        memo: PatternRuntimeMemo,
        warnings: list[str],
    ) -> tuple[str, ...]:
        if plan.strategy == "exact":
            return plan.pattern_ids
        cache_key = _candidate_cache_key(request, plan)
        cached = memo.candidate_ids.get(cache_key)
        if cached is not None:
            return cached
        try:
            raw_candidates = self._vector_finder.find_candidates(
                query=request.query,
                subject=request.subject,
                limit=plan.candidate_limit,
            )
        except Exception:
            warnings.append("vector_discovery_unavailable")
            return ()
        candidates = _bounded_strong_candidates(raw_candidates, limit=plan.candidate_limit)
        pattern_ids = tuple(candidate.pattern_id for candidate in candidates)
        memo.candidate_versions[cache_key] = {
            candidate.pattern_id: candidate.version_hash
            for candidate in candidates
            if candidate.version_hash
        }
        if raw_candidates and not pattern_ids:
            warnings.append("vector_confidence_insufficient")
        memo.candidate_ids[cache_key] = pattern_ids
        return pattern_ids

    def _hydrate_patterns(
        self,
        *,
        candidate_ids: tuple[str, ...],
        memo: PatternRuntimeMemo,
        warnings: list[str],
    ) -> dict[str, CanonicalPatternRecord | None]:
        missing_ids = [
            pattern_id for pattern_id in candidate_ids if pattern_id not in memo.patterns
        ]
        if missing_ids:
            try:
                raw_records = self._pattern_store.batch_fetch(pattern_ids=missing_ids)
            except Exception:
                warnings.append("pattern_hydration_unavailable")
                return {pattern_id: memo.patterns.get(pattern_id) for pattern_id in candidate_ids}
            records_by_id: dict[str, CanonicalPatternRecord] = {}
            for raw_record in raw_records:
                record = CanonicalPatternRecord.from_raw(raw_record)
                if record is not None and record.pattern_id in missing_ids:
                    records_by_id[record.pattern_id] = record
            for pattern_id in missing_ids:
                memo.patterns[pattern_id] = records_by_id.get(pattern_id)
        return {pattern_id: memo.patterns.get(pattern_id) for pattern_id in candidate_ids}

    def _linked_references(
        self,
        *,
        request: PatternRuntimeRequest,
        pattern_id: str,
        memo: PatternRuntimeMemo,
    ) -> tuple[list[DoubtQuestionReference], bool, str | None]:
        if request.max_linked_questions == 0:
            return [], False, None
        questions = self._load_playable_questions(pattern_id=pattern_id, memo=memo)
        references = [
            DoubtQuestionReference(
                questionId=question.question_id,
                questionText=question.question[:4000],
                options=question.options,
            )
            for question in questions[: _linked_candidate_limit(request.max_linked_questions)]
        ]
        if not references or self._linked_question_reranker is None:
            return (
                _bounded_distinct_references(references, request.max_linked_questions),
                False,
                None,
            )
        try:
            reranked, used, warning = self._linked_question_reranker.rerank(
                query=request.query,
                candidates=references,
            )
        except Exception:
            return (
                _bounded_distinct_references(references, request.max_linked_questions),
                False,
                "linked_question_rerank_unavailable",
            )
        return _bounded_distinct_references(reranked, request.max_linked_questions), used, warning

    def _load_playable_questions(
        self,
        *,
        pattern_id: str,
        memo: PatternRuntimeMemo,
        warnings: list[str] | None = None,
    ) -> tuple[CanonicalPlayableQuestion, ...]:
        cached = memo.playable_questions.get(pattern_id)
        if cached is not None:
            return cached
        if self._playable_question_store is None:
            return ()
        try:
            raw_questions = self._playable_question_store.list_by_pattern_id(
                pattern_id=pattern_id,
                limit=5,
            )
        except Exception:
            if warnings is not None:
                warnings.append("pattern_reuse_lookup_unavailable")
            return ()
        questions_by_id: dict[str, CanonicalPlayableQuestion] = {}
        for raw_question in raw_questions:
            question = CanonicalPlayableQuestion.from_raw(raw_question)
            if (
                question is not None
                and question.pattern_id == pattern_id
            ):
                questions_by_id.setdefault(question.question_id, question)
        questions = tuple(questions_by_id[key] for key in sorted(questions_by_id))
        memo.playable_questions[pattern_id] = questions
        return questions


def build_retrieval_plan(request: PatternRuntimeRequest) -> PatternRetrievalPlan:
    """Construct a bounded exact-or-vector plan without score thresholds."""
    exact_ids = {
        pattern_id.strip()
        for pattern_id in (*request.server_pattern_ids, request.server_pattern_id or "")
        if pattern_id.strip()
    }
    if exact_ids:
        return PatternRetrievalPlan(
            mode=request.mode,
            strategy="exact",
            candidateLimit=len(exact_ids),
            patternIds=tuple(sorted(exact_ids)),
        )
    return PatternRetrievalPlan(
        mode=request.mode,
        strategy="vector",
        candidateLimit=request.candidate_limit,
    )


def build_generation_context(record: CanonicalPatternRecord) -> PatternGenerationContext:
    """Project allowed PatternGraph structure without answer or solution material."""
    graph = record.pattern_graph
    return PatternGenerationContext(
        patternId=record.pattern_id,
        target=tuple(_compact_values(graph.get("find"), max_items=3)),
        givens=tuple(_compact_values(graph.get("given"), max_items=4)),
        conditions=tuple(_compact_values(graph.get("condition"), max_items=4)),
        operationHints=tuple(
            _compact_values(
                [
                    graph.get("operations"),
                    graph.get("formulas"),
                    graph.get("formula"),
                    graph.get("relations"),
                    graph.get("rules"),
                    graph.get("constraints"),
                ],
                max_items=6,
            )
        ),
        notSameWhen=tuple(
            _compact_values(
                graph.get("not_same_when", graph.get("notSameWhen")),
                max_items=4,
                allow_raw_text=True,
            )
        ),
        trapCues=tuple(
            _compact_values(
                graph.get("hidden_traps", graph.get("hiddenTraps")),
                max_items=4,
            )
        ),
        complexityLevel=record.complexity_level,
        variationFocus=record.core_concept,
    )


def _candidate_cache_key(
    request: PatternRuntimeRequest,
    plan: PatternRetrievalPlan,
) -> tuple[str, str, int]:
    return (
        _normalized(request.query),
        _subject_key(request.subject),
        plan.candidate_limit,
    )


def _bounded_strong_candidates(
    values: Sequence[VectorPatternCandidate | Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[VectorPatternCandidate, ...]:
    candidates: list[VectorPatternCandidate] = []
    seen: set[str] = set()
    for value in values:
        candidate = VectorPatternCandidate.from_raw(value)
        if (
            candidate is not None
            and candidate.pattern_id not in seen
            and _is_strong_vector_candidate(candidate)
        ):
            candidates.append(candidate)
            seen.add(candidate.pattern_id)
        if len(candidates) == limit:
            break
    return tuple(candidates)


def _is_strong_vector_candidate(candidate: VectorPatternCandidate) -> bool:
    """Use the established runtime floor only as a negative candidate gate."""
    if candidate.score is None:
        return False
    from retrieval.models import RetrievedCandidate  # noqa: PLC0415

    return is_strong_runtime_match(
        [RetrievedCandidate(patternId=candidate.pattern_id, score=candidate.score)]
    )


def _linked_candidate_limit(output_limit: int) -> int:
    """Keep ColBERT's candidate pool small while leaving it a meaningful choice."""
    return min(max(output_limit * 3, 3), 5)


def _bounded_distinct_references(
    values: list[DoubtQuestionReference],
    limit: int,
) -> list[DoubtQuestionReference]:
    seen: set[str] = set()
    references: list[DoubtQuestionReference] = []
    for reference in values:
        if reference.question_id not in seen and all(
            _references_are_materially_distinct(reference, existing)
            for existing in references
        ):
            references.append(reference)
            seen.add(reference.question_id)
        if len(references) == limit:
            break
    return references


def _references_are_materially_distinct(
    candidate: DoubtQuestionReference,
    existing: DoubtQuestionReference,
) -> bool:
    """Avoid spending a second reference slot on near-identical question text."""
    candidate_tokens = set(_normalized(candidate.question_text).split())
    existing_tokens = set(_normalized(existing.question_text).split())
    if not candidate_tokens or not existing_tokens:
        return candidate.question_id != existing.question_id
    overlap = len(candidate_tokens.intersection(existing_tokens))
    union = len(candidate_tokens.union(existing_tokens))
    return union == 0 or overlap / union < 0.85


def _approved_solve_flow_steps(
    record: CanonicalPatternRecord,
) -> tuple[DoubtSolveFlowStep, ...]:
    """Mirror the upstream replay-safe admission gate for compact solve guidance."""
    if _normalized(record.solve_flow_status) != "approved":
        return ()
    if (
        _normalized(_safe_text(record.solve_flow_meta.get("replayDecision"))) != "pass"
        or _normalized(_safe_text(record.solve_flow_meta.get("replayEvidenceStatus")))
        != "consistent"
    ):
        return ()
    if _normalized(_safe_text(record.solve_flow.get("reviewStatus"))) != "approved":
        return ()
    raw_steps = record.solve_flow.get("steps")
    if not isinstance(raw_steps, list):
        return ()
    steps: list[DoubtSolveFlowStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            continue
        action = _safe_solve_flow_identifier(raw_step.get("action"))
        target = _safe_solve_flow_identifier(raw_step.get("target"))
        instruction = _safe_solve_flow_instruction(raw_step.get("instruction"))
        if not action or not target or not instruction:
            continue
        steps.append(
            DoubtSolveFlowStep(
                action=action,
                target=target,
                instruction=instruction,
            )
        )
        if len(steps) == 6:
            break
    return tuple(steps)


def _safe_solve_flow_identifier(value: object) -> str:
    """Allow only compact, source-independent SolveFlow identifiers."""
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    blocked_terms = (
        "answer",
        "option",
        "solution",
        "ignore",
        "system",
        "assistant",
        "developer",
        "instruction",
        "override",
    )
    if not _SAFE_SOLVE_FLOW_IDENTIFIER.fullmatch(normalized):
        return ""
    return (
        ""
        if any(term in normalized.casefold() for term in blocked_terms)
        else normalized
    )


def _safe_solve_flow_instruction(value: object) -> str:
    """Allow bounded method prose while rejecting answer values and prompt control text."""
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.replace("\x00", " ").split())[:160]
    if not normalized or not _SAFE_SOLVE_FLOW_INSTRUCTION.fullmatch(normalized):
        return ""
    blocked_markers = (
        "correct answer",
        "final answer",
        "answer is",
        "correct option",
        "solution is",
        "therefore answer",
        "ignore previous",
        "ignore prior",
        "system prompt",
        "developer message",
        "follow these instructions",
        "disregard",
        "override",
        "must follow",
    )
    return "" if any(marker in normalized.casefold() for marker in blocked_markers) else normalized


def _ignore(pattern_id: str, reason: str) -> PatternMatchDecision:
    return PatternMatchDecision(patternId=pattern_id, tier=PatternMatchTier.IGNORE, reason=reason)


def _has_required_graph_shape(graph: Mapping[str, Any]) -> bool:
    return all(
        _compact_values(graph.get(field), max_items=1)
        for field in ("given", "condition", "find")
    )


def _matches_text(expected: str | None, actual: str) -> bool:
    if not expected or not expected.strip():
        return True
    return bool(actual.strip()) and _normalized(expected) == _normalized(actual)


def _matches_complexity(expected: str | None, actual: str) -> bool:
    if not expected or not expected.strip():
        return True
    normalized_expected = _normalized(expected)
    try:
        level = int(actual.strip())
    except ValueError:
        return _matches_text(expected, actual)
    actual_band = "low" if level <= 3 else "medium" if level <= 7 else "high"
    return normalized_expected == actual_band


def _matches_exam_ids(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    requested = {_normalized(exam_id) for exam_id in expected if _normalized(exam_id)}
    if not requested:
        return True
    available = {_normalized(exam_id) for exam_id in actual if _normalized(exam_id)}
    return bool(requested.intersection(available))


def _playable_question_matches(
    request: PatternRuntimeRequest,
    pattern: CanonicalPatternRecord,
    question: CanonicalPlayableQuestion,
) -> bool:
    if (
        question.pattern_id != pattern.pattern_id
        or question.pattern_version_hash != pattern.current_version_hash
        or _normalized(question.pattern_link_evidence)
        not in {"verified generation", "verified ingestion"}
        or not question.explanation
        or _subject_key(request.subject) != _subject_key(question.subject)
        or not _matches_text(request.category, question.category)
        or not _matches_text(request.topic, question.topic)
        or not _matches_text(request.question_type, question.question_type)
        or not _matches_text(request.language, question.language)
        or not _matches_text(request.pattern_family_id, question.pattern_family_id)
        or not _matches_exam_ids(request.exam_ids, question.exam_ids)
    ):
        return False
    expected_difficulty = {
        "low": "easy",
        "medium": "medium",
        "high": "hard",
    }.get(_normalized(request.difficulty or ""), _normalized(request.difficulty or ""))
    if expected_difficulty and expected_difficulty != _normalized(question.difficulty):
        return False
    normalized_question = _normalized(question.question)
    return not any(
        normalized and normalized in normalized_question
        for condition in request.excluded_conditions
        if (normalized := _normalized(condition))
    )


def _has_required_operation_match(
    required_operation_ids: tuple[str, ...],
    graph: Mapping[str, Any],
) -> bool:
    """Require an exact structured operation identity when the planner supplies one."""
    required = {
        _normalized(value)
        for value in required_operation_ids
        if _normalized(value)
    }
    if not required:
        return True
    available = {
        _normalized(value)
        for value in _compact_values(
            [
                graph.get("operations"),
                graph.get("formulas"),
                graph.get("formula"),
                graph.get("relations"),
                graph.get("rules"),
                graph.get("constraints"),
            ],
            max_items=16,
        )
        if _normalized(value)
    }
    return bool(required.intersection(available))


def _has_required_target_match(
    required_target_ids: tuple[str, ...],
    graph: Mapping[str, Any],
) -> bool:
    required = {_normalized(value) for value in required_target_ids if _normalized(value)}
    if not required:
        return True
    available = {
        _normalized(value)
        for value in _compact_values(
            [
                graph.get("find"),
                graph.get("operations"),
                graph.get("relations"),
                graph.get("rules"),
            ],
            max_items=16,
        )
        if _normalized(value)
    }
    return bool(required.intersection(available))


def _not_same_when_matches(query: str, graph: Mapping[str, Any]) -> bool:
    normalized_query = _normalized(query)
    if not normalized_query:
        return False
    for condition in _compact_values(
        graph.get("not_same_when", graph.get("notSameWhen")),
        max_items=8,
        allow_raw_text=True,
    ):
        normalized_condition = _normalized(condition)
        if normalized_condition and normalized_condition in normalized_query:
            return True
    return False


def _not_same_when_conflicts(
    excluded_conditions: tuple[str, ...],
    graph: Mapping[str, Any],
) -> bool:
    """Reject a candidate when immutable slot exclusions overlap its graph guard."""
    requested = [_normalized(value) for value in excluded_conditions if _normalized(value)]
    if not requested:
        return False
    graph_conditions = [
        _normalized(value)
        for value in _compact_values(
            graph.get("not_same_when", graph.get("notSameWhen")),
            max_items=8,
            allow_raw_text=True,
        )
    ]
    for requested_condition in requested:
        for graph_condition in graph_conditions:
            if (
                requested_condition == graph_condition
                or requested_condition in graph_condition
                or graph_condition in requested_condition
            ):
                return True
    return False


def _compact_values(
    value: object,
    *,
    max_items: int,
    allow_raw_text: bool = False,
) -> list[str]:
    values: list[str] = []
    _collect_compact_values(
        value,
        values,
        allow_raw_text=allow_raw_text,
        structured_leaf=False,
    )
    unique: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = _normalized(item)
        if key and key not in seen:
            unique.append(item[:320])
            seen.add(key)
        if len(unique) == max_items:
            break
    return unique


def _collect_compact_values(
    value: object,
    values: list[str],
    *,
    allow_raw_text: bool,
    structured_leaf: bool,
) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text and (
            (structured_leaf and _is_safe_structured_graph_token(text))
            or (allow_raw_text and _is_safe_raw_graph_text(text))
        ):
            values.append(text)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_compact_values(
                item,
                values,
                allow_raw_text=allow_raw_text,
                structured_leaf=structured_leaf,
            )
        return
    if not isinstance(value, Mapping):
        return
    for key in _SAFE_GRAPH_KEYS:
        normalized_key = _normalized(key)
        if normalized_key in _BLOCKED_GRAPH_KEYS:
            continue
        nested = value.get(key)
        if nested is not None:
            _collect_compact_values(
                nested,
                values,
                allow_raw_text=allow_raw_text,
                structured_leaf=True,
            )


def _is_safe_structured_graph_token(value: str) -> bool:
    """Allow only compact identity/formula tokens from explicitly whitelisted keys."""
    normalized = _normalized(value)
    blocked_markers = (
        "correct answer",
        "correct option",
        "final answer",
        "answer is",
        "solution is",
        "ignore previous",
        "ignore prior",
        "system prompt",
        "developer message",
        "follow these instructions",
        "disregard",
        "override",
    )
    return bool(_SAFE_STRUCTURED_GRAPH_TOKEN.fullmatch(value)) and not any(
        marker in normalized for marker in blocked_markers
    )


def _is_safe_raw_graph_text(value: str) -> bool:
    """Allow only compact condition labels from the raw JSON escape hatch."""
    normalized = _normalized(value)
    blocked_markers = (
        "correct answer",
        "correct option",
        "final answer",
        "answer is",
        "solution is",
        "therefore answer",
        "ignore previous",
        "ignore prior",
        "system prompt",
        "developer message",
        "follow these instructions",
        "disregard",
        "override",
        "must follow",
        "answer",
        "option",
        "solution",
    )
    return bool(_SAFE_RAW_CONDITION.fullmatch(value.strip())) and not any(
        marker in normalized for marker in blocked_markers
    )


def _subject_key(value: str) -> str:
    normalized = _normalized(value)
    return _SUBJECT_ALIASES.get(normalized, normalized)


def _safe_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
