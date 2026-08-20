"""Question semantic reuse (Phase D).

Recovers trusted QuestionBank Questions whose educational intent matches a planner
slot even when the exact topic/category labels differ — the proven Path A blind
spot (``profit_and_loss`` vs ``percentage_discount``).

S3 Vectors is discovery only.  Nothing reaches a student until the authoritative
QuestionBank row has been re-read, its version hash recomputed in Python and matched
against the vector's metadata, its identity re-derived, and its trust and structural
compatibility re-checked.  Any doubt leaves the slot in deficit for Pattern and
generation to fill.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from features.practice_generation.events import emit_practice_event
from features.practice_generation.matching import (
    ReusableQuestion,
    question_bank_identity_is_intact,
    question_bank_version_hash_from_item,
    reusable_question_from_item,
)
from features.practice_generation.metadata_normalization import (
    normalize_difficulty,
    normalize_exam_ids,
    normalize_language,
    normalize_subject,
)
from features.practice_generation.schemas import PlannerSlot

_QB_DIFFICULTY = {"basic": "EASY", "intermediate": "MEDIUM", "advanced": "HARD"}
# QueryVectors is not a paginated API: it has no nextToken in either its input or
# output shape, so one logical semantic search is always exactly one API call and the
# only way to survive rejections is to ask for headroom up front.
_CANDIDATE_HEADROOM_MULTIPLIER = 2
_MAX_CANDIDATE_TOP_K = 200
# Deliberately high: "same template, different numbers" is the reuse we want, so only
# near-verbatim repeats are suppressed.
_NEAR_DUPLICATE_RATIO = 0.92


def resolve_candidate_top_k(configured_top_k: int, required_count: int) -> int:
    """Bounded TopK with headroom for stale, untrusted, and duplicate candidates."""
    return min(
        max(configured_top_k, required_count * _CANDIDATE_HEADROOM_MULTIPLIER),
        _MAX_CANDIDATE_TOP_K,
    )


def _stem_tokens(value: str) -> frozenset[str]:
    return frozenset(value.casefold().split())


def is_near_duplicate(candidate_stem: str, selected_stems: Sequence[frozenset[str]]) -> bool:
    """Deterministic near-verbatim suppression. No model call.

    Numeric variants of one template share most tokens but not all, so they stay
    below the ratio and remain reusable; a re-worded copy of the same sentence does
    not.
    """
    tokens = _stem_tokens(candidate_stem)
    if not tokens:
        return False
    for existing in selected_stems:
        union = tokens | existing
        if union and len(tokens & existing) / len(union) >= _NEAR_DUPLICATE_RATIO:
            return True
    return False


class QuestionCandidateFinder(Protocol):
    """Bounded questions-v1 discovery."""

    def find_candidates(
        self,
        *,
        demand_text: str,
        metadata_filter: dict[str, object],
        top_k: int,
    ) -> Sequence[tuple[str, float, str]]:
        """Return (qbId, similarity, vector versionHash) triples."""


@dataclass(frozen=True)
class SemanticDemandGroup:
    """One unique educational demand shared by every slot inside it."""

    group_id: str
    slots: tuple[PlannerSlot, ...]

    @property
    def representative(self) -> PlannerSlot:
        return self.slots[0]

    @property
    def required_count(self) -> int:
        return len(self.slots)


@dataclass
class SemanticReuseOutcome:
    selected_by_slot: dict[str, ReusableQuestion] = field(default_factory=dict)
    group_count: int = 0
    embedding_call_count: int = 0
    semantic_search_count: int = 0
    vector_api_call_count: int = 0
    near_duplicate_rejected_count: int = 0
    candidate_count: int = 0
    hydrated_count: int = 0
    version_parity_rejected_count: int = 0
    identity_rejected_count: int = 0
    trust_rejected_count: int = 0
    compatibility_rejected_count: int = 0
    duplicate_rejected_count: int = 0
    threshold_rejected_count: int = 0
    would_reuse_count: int = 0


def _slot_demand_key(slot: PlannerSlot, *, language: str) -> str:
    """Group slots that share one educational demand.

    Deliberately identical in spirit to the Pattern grouping contract: conservative,
    so two slots only share a query when they share the whole educational intent.
    """
    exam_ids = ",".join(sorted({value.strip() for value in slot.exam_ids if value.strip()}))
    return "|".join(
        (
            slot.subject_id,
            slot.topic_id,
            slot.category_id,
            slot.difficulty.value,
            slot.question_type.value,
            exam_ids,
            slot.target_skill,
            normalize_language(language) or "",
        )
    )


def group_semantic_demands(
    slots: Sequence[PlannerSlot],
    *,
    language: str,
) -> tuple[SemanticDemandGroup, ...]:
    """Collapse slots to unique demands so N slots cost one query, not N."""
    grouped: dict[str, list[PlannerSlot]] = defaultdict(list)
    for slot in slots:
        grouped[_slot_demand_key(slot, language=language)].append(slot)
    return tuple(
        SemanticDemandGroup(group_id=f"demand-{index + 1:03d}", slots=tuple(grouped[key]))
        for index, key in enumerate(sorted(grouped))
    )


def build_demand_text(group: SemanticDemandGroup) -> str:
    """Deterministic demand representation. No LLM, no answer-bearing content.

    Mirrors the indexed Question document so demand and inventory share a space.
    """
    slot = group.representative
    skill = slot.target_skill.replace("_", " ")
    intent = (slot.reasoning_target or skill).replace("_", " ")
    return "\n".join(
        (
            f"subject:{slot.subject_id.replace('_', ' ')}",
            f"topic:{slot.topic_id.replace('_', ' ')}",
            f"category:{slot.category_id.replace('_', ' ')}",
            f"question:{intent}",
        )
    )


def build_metadata_filter(group: SemanticDemandGroup, *, language: str) -> dict[str, object] | None:
    """Structural prefilters only.

    Topic and category are intentionally absent: making them exact filters would
    rebuild the Path A blind spot this phase exists to fix.
    """
    slot = group.representative
    subject = normalize_subject(slot.subject_id)
    difficulty = normalize_difficulty(slot.difficulty.value)
    canonical_language = normalize_language(language)
    exam_ids = normalize_exam_ids(slot.exam_ids)
    if subject is None or difficulty is None or canonical_language is None or exam_ids is None:
        return None
    clauses: list[dict[str, object]] = [
        {"subject": subject},
        {"difficulty": _QB_DIFFICULTY[difficulty]},
        {"language": canonical_language},
    ]
    if exam_ids:
        # The indexed `exam` metadata is an array; membership is expressed with $in.
        clauses.append({"exam": {"$in": list(exam_ids)}})
    return {"$and": clauses}


def _structurally_compatible(
    candidate: ReusableQuestion,
    slot: PlannerSlot,
    *,
    language: str,
) -> bool:
    """Semantic reuse compatibility: strict on structure, silent on topic/category."""
    expected_exams = normalize_exam_ids(slot.exam_ids)
    candidate_exams = normalize_exam_ids(candidate.exam_ids)
    if expected_exams is None or candidate_exams is None:
        return False
    if normalize_subject(candidate.subject) != normalize_subject(slot.subject_id):
        return False
    if normalize_difficulty(candidate.difficulty) != slot.difficulty.value:
        return False
    if normalize_language(candidate.language) != normalize_language(language):
        return False
    if candidate.question_type != slot.question_type.value:
        return False
    if expected_exams and not set(expected_exams).issubset(candidate_exams):
        return False
    return bool(candidate.solution)


class QuestionSemanticReuseResolver:
    """Discover, authoritatively validate, and assign reusable Questions."""

    def __init__(
        self,
        *,
        finder: QuestionCandidateFinder,
        hydrate: Any,
        threshold: float,
        top_k: int,
    ) -> None:
        self._finder = finder
        self._hydrate = hydrate
        self._threshold = threshold
        self._top_k = top_k

    def resolve(
        self,
        *,
        test_id: str,
        groups: Sequence[SemanticDemandGroup],
        language: str,
        excluded_question_ids: set[str],
    ) -> SemanticReuseOutcome:
        outcome = SemanticReuseOutcome(group_count=len(groups))
        selected_ids = set(excluded_question_ids)
        for group in groups:
            metadata_filter = build_metadata_filter(group, language=language)
            if metadata_filter is None:
                continue
            try:
                outcome.embedding_call_count += 1
                outcome.semantic_search_count += 1
                # One logical search == one API call, because QueryVectors does not
                # paginate. Counted separately so the invariant stays observable.
                outcome.vector_api_call_count += 1
                candidates = self._finder.find_candidates(
                    demand_text=build_demand_text(group),
                    metadata_filter=metadata_filter,
                    top_k=resolve_candidate_top_k(self._top_k, group.required_count),
                )
            except Exception:  # noqa: BLE001 - discovery is an optimization
                emit_practice_event(
                    "QUESTION_SEMANTIC_RETRIEVAL_FAILED",
                    test_id=test_id,
                    status="fallback",
                    details={"groupId": group.group_id, "reasonCode": "DISCOVERY_UNAVAILABLE"},
                    level=logging.WARNING,
                )
                continue
            self._resolve_group(
                test_id=test_id,
                group=group,
                language=language,
                candidates=candidates,
                selected_ids=selected_ids,
                outcome=outcome,
            )
        return outcome

    def _resolve_group(
        self,
        *,
        test_id: str,
        group: SemanticDemandGroup,
        language: str,
        candidates: Sequence[tuple[str, float, str]],
        selected_ids: set[str],
        outcome: SemanticReuseOutcome,
    ) -> None:
        ranked = sorted(candidates, key=lambda entry: entry[1], reverse=True)
        outcome.candidate_count += len(ranked)
        eligible: list[tuple[str, float, str]] = []
        for qb_id, score, vector_hash in ranked:
            if score < self._threshold:
                outcome.threshold_rejected_count += 1
                continue
            if not qb_id or qb_id in selected_ids:
                outcome.duplicate_rejected_count += 1
                continue
            eligible.append((qb_id, score, vector_hash))
        if not eligible:
            return
        rows = {
            str(row.get("qbId") or ""): row
            for row in self._hydrate([qb_id for qb_id, _score, _hash in eligible])
        }
        outcome.hydrated_count += len(rows)
        open_slots = [slot for slot in group.slots if slot.slot_id not in outcome.selected_by_slot]
        selected_stems: list[frozenset[str]] = [
            _stem_tokens(chosen.question) for chosen in outcome.selected_by_slot.values()
        ]
        top_score: float | None = None
        for qb_id, score, vector_hash in eligible:
            if not open_slots:
                break
            row = rows.get(qb_id)
            if row is None:
                outcome.version_parity_rejected_count += 1
                continue
            # Recomputed from the current row: a stored hash is never trusted.
            if question_bank_version_hash_from_item(row) != vector_hash:
                outcome.version_parity_rejected_count += 1
                continue
            if not question_bank_identity_is_intact(row):
                outcome.identity_rejected_count += 1
                continue
            candidate = reusable_question_from_item(row, requested_language=language)
            if candidate is None:
                outcome.trust_rejected_count += 1
                continue
            if is_near_duplicate(candidate.question, selected_stems):
                outcome.near_duplicate_rejected_count += 1
                continue
            slot = next(
                (
                    value
                    for value in open_slots
                    if _structurally_compatible(candidate, value, language=language)
                ),
                None,
            )
            if slot is None:
                outcome.compatibility_rejected_count += 1
                continue
            outcome.selected_by_slot[slot.slot_id] = candidate
            selected_stems.append(_stem_tokens(candidate.question))
            outcome.would_reuse_count += 1
            selected_ids.add(qb_id)
            open_slots.remove(slot)
            top_score = score if top_score is None else max(top_score, score)
        emit_practice_event(
            "QUESTION_SEMANTIC_CANDIDATE_DECISION",
            test_id=test_id,
            status="completed",
            details={
                "groupId": group.group_id,
                "requiredCount": group.required_count,
                "candidateCount": len(ranked),
                "eligibleCandidateCount": len(eligible),
                "selectedCount": group.required_count - len(open_slots),
                "topVectorScore": top_score,
            },
            level=logging.DEBUG,
        )


class S3QuestionCandidateFinder:
    """questions-v1 discovery over the shared Titan embedder and vector client."""

    def __init__(self, *, embedder: Any = None, vector_client: Any = None) -> None:
        from retrieval.embeddings.bedrock_titan_embedder import (  # noqa: PLC0415
            BedrockTitanEmbedder,
        )
        from retrieval.s3_vectors.client import S3VectorClient  # noqa: PLC0415

        self._embedder = embedder or BedrockTitanEmbedder()
        self._vector_client = vector_client or S3VectorClient()

    def find_candidates(
        self,
        *,
        demand_text: str,
        metadata_filter: dict[str, object],
        top_k: int,
    ) -> Sequence[tuple[str, float, str]]:
        query_vector = self._embedder.embed_query(demand_text)
        candidates = self._vector_client.query_question_candidates(
            query_vector=query_vector,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
        return tuple(
            (candidate.pattern_id, float(candidate.score), str(candidate.version_hash or ""))
            for candidate in candidates
            if candidate.pattern_id
        )


def build_question_semantic_resolver(
    *,
    mode: str,
    threshold: float,
    top_k: int,
    hydrate: Any,
) -> QuestionSemanticReuseResolver | None:
    """Construct the resolver only when Phase D is actually switched on."""
    if mode == "off":
        return None
    return QuestionSemanticReuseResolver(
        finder=S3QuestionCandidateFinder(),
        hydrate=hydrate,
        threshold=threshold,
        top_k=top_k,
    )
