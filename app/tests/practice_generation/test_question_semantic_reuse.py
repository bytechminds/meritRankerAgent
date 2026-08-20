"""Phase D: discovery, authoritative validation, and fail-closed behaviour."""

from __future__ import annotations

import json
from typing import Any

import pytest

from features.practice_generation.matching import (
    question_bank_version_hash_from_item,
)
from features.practice_generation.question_semantic_reuse import (
    QuestionSemanticReuseResolver,
    build_demand_text,
    build_metadata_filter,
    group_semantic_demands,
)
from features.practice_generation.repositories import build_question_bank_id
from features.practice_generation.schemas import PlannerSlot


def slot(index: int = 1, **overrides: Any) -> PlannerSlot:
    payload: dict[str, Any] = {
        "slot_id": f"slot-{index:03d}",
        "subject_id": "math",
        "topic_id": "percentage_discount",
        "category_id": "percentage_discount",
        "difficulty": "basic",
        "complexity": "low",
        "exam_ids": ["CAT"],
        "question_type": "mcq",
        "target_skill": "selling_price_from_discount",
        "variation_hint": "vary_values",
        "generator_route_hint": "math.generator.basic",
    }
    payload.update(overrides)
    return PlannerSlot(**payload)


def bank_row(qb_id_topic: str = "profit_and_loss", **overrides: Any) -> dict[str, Any]:
    """A stored row whose topic label differs from the requesting slot."""
    row: dict[str, Any] = {
        "question": "A shop gives a discount on a marked price. Find the selling price.",
        "answers": json.dumps({"options": ["1000", "1200", "1400", "1500"]}),
        "correctAnswer": "1200",
        "explanation": "Selling price is marked price minus discount.",
        "category": qb_id_topic,
        "difficulty": "EASY",
        "meta": {
            "status": "ACTIVE",
            "qualityStatus": "VERIFIED",
            "reusable": True,
            "visibility": "PLATFORM",
            "subject": "math",
            "topic": qb_id_topic,
            "questionType": "mcq",
            "language": "english",
            "examIds": ["CAT"],
            "confidence": 1,
        },
    }
    row.update(overrides)
    row["qbId"] = build_question_bank_id(
        subject="math",
        difficulty="basic",
        exam_ids=["CAT"],
        language="english",
        question_type="mcq",
        question=str(row["question"]),
        options=tuple(json.loads(str(row["answers"]))["options"]),
        correct_answer=str(row["correctAnswer"]),
    )
    row["versionHash"] = question_bank_version_hash_from_item(row)
    return row


class _Finder:
    """Single-page finder: the whole pool arrives in one API call."""

    def __init__(
        self,
        results: list[tuple[str, float, str]],
        *,
        fail: bool = False,
        api_call_count: int = 1,
    ) -> None:
        self.results = results
        self.fail = fail
        self.api_call_count = api_call_count
        self.calls: list[dict[str, Any]] = []

    def find_candidates(self, *, demand_text, metadata_filter, top_k):
        from features.practice_generation.question_semantic_reuse import CandidateDiscovery

        self.calls.append(
            {"demand_text": demand_text, "filter": metadata_filter, "top_k": top_k}
        )
        if self.fail:
            raise RuntimeError("questions-v1 unavailable")
        return CandidateDiscovery(
            candidates=tuple(self.results), api_call_count=self.api_call_count
        )


def resolver(rows: list[dict[str, Any]], finder: _Finder, *, threshold: float = 0.30):
    by_id = {row["qbId"]: row for row in rows}
    return QuestionSemanticReuseResolver(
        finder=finder,
        hydrate=lambda ids: [by_id[i] for i in ids if i in by_id],
        threshold=threshold,
        top_k=12,
    )


def run(rows, finder, slots, **kwargs):
    return resolver(rows, finder, **kwargs).resolve(
        test_id="test-1",
        groups=group_semantic_demands(slots, language="english"),
        language="english",
        excluded_question_ids=set(),
    )


# --- grouping and demand text -------------------------------------------------

def test_equivalent_slots_collapse_to_one_demand_group() -> None:
    groups = group_semantic_demands([slot(i) for i in range(1, 101)], language="english")

    assert len(groups) == 1
    assert groups[0].required_count == 100


def test_distinct_demands_stay_separate() -> None:
    slots = [
        slot(1),
        slot(2, topic_id="linear_equations", category_id="linear_equations"),
        slot(3, difficulty="advanced"),
    ]

    assert len(group_semantic_demands(slots, language="english")) == 3


def test_demand_text_is_deterministic_and_answer_free() -> None:
    group = group_semantic_demands([slot()], language="english")[0]
    text = build_demand_text(group)

    assert text == build_demand_text(group)
    for forbidden in ("answer", "solution", "correct", "1200"):
        assert forbidden not in text.lower()


# --- filters ------------------------------------------------------------------

def test_filter_uses_and_composition_and_exam_membership() -> None:
    group = group_semantic_demands([slot()], language="english")[0]
    built = build_metadata_filter(group, language="english")

    assert built == {
        "$and": [
            {"subject": "math"},
            {"difficulty": "EASY"},
            {"language": "english"},
            {"exam": {"$in": ["CAT"]}},
        ]
    }


def test_filter_never_constrains_topic_or_category() -> None:
    group = group_semantic_demands([slot()], language="english")[0]
    rendered = json.dumps(build_metadata_filter(group, language="english"))

    # The whole point of Phase D: topic/category stay semantic.
    assert "topic" not in rendered
    assert "category" not in rendered
    assert "percentage_discount" not in rendered


def test_filter_omits_exam_clause_when_the_slot_has_no_exam_scope() -> None:
    group = group_semantic_demands([slot(exam_ids=[])], language="english")[0]

    assert build_metadata_filter(group, language="english") == {
        "$and": [{"subject": "math"}, {"difficulty": "EASY"}, {"language": "english"}]
    }


# --- the incident: taxonomy-variant recovery ----------------------------------

def test_recovers_a_question_stored_under_a_different_topic_label() -> None:
    row = bank_row("profit_and_loss")
    finder = _Finder([(row["qbId"], 0.44, row["versionHash"])])

    outcome = run([row], finder, [slot()])

    assert outcome.would_reuse_count == 1
    assert outcome.selected_by_slot["slot-001"].question_id == row["qbId"]


def test_one_query_and_one_embedding_serve_a_hundred_slots() -> None:
    rows = [
        bank_row("profit_and_loss", question=f"Discount question variant {i} for selling price.")
        for i in range(100)
    ]
    finder = _Finder([(r["qbId"], 0.9 - i * 0.001, r["versionHash"]) for i, r in enumerate(rows)])

    outcome = run(rows, finder, [slot(i) for i in range(1, 101)])

    assert outcome.group_count == 1
    assert outcome.embedding_call_count == 1
    assert outcome.semantic_search_count == 1
    assert outcome.vector_api_call_count == 1
    assert outcome.would_reuse_count == 100
    assert len(set(c.question_id for c in outcome.selected_by_slot.values())) == 100


# --- authoritative validation: every gate fails closed ------------------------

def test_below_threshold_candidates_are_never_served() -> None:
    row = bank_row()
    finder = _Finder([(row["qbId"], 0.29, row["versionHash"])])

    outcome = run([row], finder, [slot()], threshold=0.30)

    assert outcome.would_reuse_count == 0
    assert outcome.threshold_rejected_count == 1


def test_stale_vector_hash_is_rejected() -> None:
    row = bank_row()
    finder = _Finder([(row["qbId"], 0.90, "stale-hash")])

    outcome = run([row], finder, [slot()])

    assert outcome.would_reuse_count == 0
    assert outcome.version_parity_rejected_count == 1


def test_content_edited_after_indexing_is_rejected_by_recomputation() -> None:
    row = bank_row()
    indexed_hash = row["versionHash"]
    row["explanation"] = "An administrator rewrote this explanation."
    finder = _Finder([(row["qbId"], 0.90, indexed_hash)])

    outcome = run([row], finder, [slot()])

    # The stored hash is stale but recomputation catches it: fail closed.
    assert outcome.would_reuse_count == 0
    assert outcome.version_parity_rejected_count == 1


def test_tampered_identity_is_rejected() -> None:
    row = bank_row()
    row["question"] = "A completely different question stem substituted in place."
    finder = _Finder([(row["qbId"], 0.90, question_bank_version_hash_from_item(row))])

    outcome = run([row], finder, [slot()])

    assert outcome.would_reuse_count == 0
    assert outcome.identity_rejected_count == 1


def test_missing_row_is_rejected() -> None:
    row = bank_row()
    finder = _Finder([(row["qbId"], 0.90, row["versionHash"])])

    outcome = run([], finder, [slot()])

    assert outcome.would_reuse_count == 0
    assert outcome.version_parity_rejected_count == 1


@pytest.mark.parametrize(
    "meta_override",
    [
        {"status": "INACTIVE"},
        {"qualityStatus": "PENDING"},
        {"reusable": False},
        {"needsReview": True},
        {"visibility": "PRIVATE"},
    ],
)
def test_untrusted_rows_are_rejected(meta_override: dict[str, Any]) -> None:
    row = bank_row()
    row["meta"] = {**row["meta"], **meta_override}
    row["versionHash"] = question_bank_version_hash_from_item(row)
    finder = _Finder([(row["qbId"], 0.90, row["versionHash"])])

    outcome = run([row], finder, [slot()])

    assert outcome.would_reuse_count == 0
    assert outcome.trust_rejected_count == 1


# --- hard negatives: structural compatibility stays strict --------------------

@pytest.mark.parametrize(
    ("row_kwargs", "meta_override"),
    [
        ({"difficulty": "HARD"}, {}),
        ({}, {"language": "hindi"}),
        ({}, {"subject": "physics"}),
        ({}, {"examIds": ["GMAT"]}),
    ],
)
def test_structurally_incompatible_candidates_are_rejected(
    row_kwargs: dict[str, Any], meta_override: dict[str, Any]
) -> None:
    row = bank_row(**row_kwargs)
    if meta_override:
        row["meta"] = {**row["meta"], **meta_override}
    row["versionHash"] = question_bank_version_hash_from_item(row)
    finder = _Finder([(row["qbId"], 0.95, row["versionHash"])])

    outcome = run([row], finder, [slot()])

    assert outcome.would_reuse_count == 0


# --- diversity and dedupe -----------------------------------------------------

def test_one_question_never_fills_two_slots() -> None:
    row = bank_row()
    finder = _Finder([(row["qbId"], 0.90, row["versionHash"])])

    outcome = run([row], finder, [slot(1), slot(2)])

    assert outcome.would_reuse_count == 1
    assert len(outcome.selected_by_slot) == 1


def test_already_used_questions_are_excluded() -> None:
    row = bank_row()
    finder = _Finder([(row["qbId"], 0.90, row["versionHash"])])
    outcome = resolver([row], finder).resolve(
        test_id="test-1",
        groups=group_semantic_demands([slot()], language="english"),
        language="english",
        excluded_question_ids={row["qbId"]},
    )

    assert outcome.would_reuse_count == 0
    assert outcome.duplicate_rejected_count == 1


# --- availability -------------------------------------------------------------

def test_discovery_failure_falls_through_without_raising() -> None:
    outcome = run([], _Finder([], fail=True), [slot()])

    assert outcome.would_reuse_count == 0
    assert outcome.selected_by_slot == {}


def test_partial_fill_leaves_the_remainder_in_deficit() -> None:
    rows = [bank_row("profit_and_loss", question=f"Discount variant {i} selling price.")
            for i in range(3)]
    finder = _Finder([(r["qbId"], 0.8, r["versionHash"]) for r in rows])

    outcome = run(rows, finder, [slot(i) for i in range(1, 11)])

    assert outcome.would_reuse_count == 3
    assert len(outcome.selected_by_slot) == 3


# --- orchestration modes ------------------------------------------------------

class _RecordingResolver:
    def __init__(self, selections: dict[str, Any]) -> None:
        self.selections = selections
        self.calls = 0

    def resolve(self, *, test_id, groups, language, excluded_question_ids):
        from features.practice_generation.question_semantic_reuse import SemanticReuseOutcome

        self.calls += 1
        return SemanticReuseOutcome(
            selected_by_slot=dict(self.selections),
            group_count=len(groups),
            would_reuse_count=len(self.selections),
        )


class _LinkRecordingQuestions:
    def __init__(self) -> None:
        self.linked: list[str] = []

    def link_reused(self, *, test_id, bucket_id, slot_id, question, **_kwargs):
        self.linked.append(slot_id)
        return True


def _orchestrator(mode: str, resolver_obj):
    from features.practice_generation.config import PracticeGenerationConfig
    from features.practice_generation.orchestration import PracticeGenerationOrchestrator

    orchestrator = PracticeGenerationOrchestrator.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = PracticeGenerationConfig(
        enabled=True,
        pattern_context_enabled=False,
        pattern_reuse_enabled=False,
        assessment_table="a",
        question_table="q",
        question_bank_table="qb",
        question_bank_category_index="ci",
        question_test_index="ti",
        aws_region="ap-south-1",
        question_semantic_reuse_mode=mode,
    )
    orchestrator._semantic_resolver = resolver_obj
    orchestrator._questions = _LinkRecordingQuestions()
    return orchestrator


def _slot_blueprint_and_request(count: int = 2):
    from features.practice_generation.planning import (
        deterministic_blueprint,
        resolve_practice_request,
    )

    request = resolve_practice_request(
        request_id="r1", user_id="u1", conversation_id="c1", turn_id="t1",
        query=f"Create {count} percentage discount questions",
        subject="math", topic="percentage_discount", difficulty="basic",
        language="english", exam_id=None, exam_stage=None,
    )
    return deterministic_blueprint(request), request


def _apply(mode: str, selections: dict[str, Any]):
    blueprint, request = _slot_blueprint_and_request()
    resolver_obj = _RecordingResolver(selections) if mode != "off" else _RecordingResolver({})
    orchestrator = _orchestrator(mode, resolver_obj)
    deficits = {s.slot_id for s in blueprint.slots}
    remaining = orchestrator._apply_question_semantic_reuse(
        test_id="test-1", request=request, blueprint=blueprint,
        deficit_slot_ids=deficits, filled_slot_ids=set(),
        already_linked_source_ids=set(), accepted_ids=[],
    )
    return orchestrator, resolver_obj, remaining, blueprint


def test_mode_off_never_queries_and_leaves_deficits_untouched() -> None:
    orchestrator, resolver_obj, remaining, blueprint = _apply("off", {})

    assert resolver_obj.calls == 0
    assert remaining == {s.slot_id for s in blueprint.slots}
    assert orchestrator._questions.linked == []


def test_shadow_mode_validates_but_serves_nothing() -> None:
    blueprint, _request = _slot_blueprint_and_request()
    first = blueprint.slots[0].slot_id
    candidate = bank_row()
    from features.practice_generation.matching import reusable_question_from_item

    hydrated = reusable_question_from_item(candidate, requested_language="english")
    orchestrator, resolver_obj, remaining, blueprint = _apply("shadow", {first: hydrated})

    assert resolver_obj.calls == 1
    # Full retrieval ran, but no deficit was reduced and nothing was linked.
    assert remaining == {s.slot_id for s in blueprint.slots}
    assert orchestrator._questions.linked == []


def test_on_mode_serves_and_reduces_the_deficit() -> None:
    blueprint, _request = _slot_blueprint_and_request()
    first = blueprint.slots[0].slot_id
    from features.practice_generation.matching import reusable_question_from_item

    hydrated = reusable_question_from_item(bank_row(), requested_language="english")
    orchestrator, resolver_obj, remaining, blueprint = _apply("on", {first: hydrated})

    assert resolver_obj.calls == 1
    assert first not in remaining
    assert orchestrator._questions.linked == [first]
    # Pattern and generation only ever see what semantic reuse could not fill.
    assert len(remaining) == len(blueprint.slots) - 1


# --- headroom, API-call accounting, and near-duplicate diversity ---------------

def test_installed_sdk_supports_query_vectors_pagination() -> None:
    """Capability, not a frozen SDK shape: the runtime must be able to send and
    receive a continuation token, otherwise deep candidate pools are unreachable."""
    import boto3

    model = boto3.client(
        "s3vectors", region_name="ap-south-1",
        aws_access_key_id="x", aws_secret_access_key="y",
    ).meta.service_model.operation_model("QueryVectors")
    assert "nextToken" in model.input_shape.members
    assert "nextToken" in model.output_shape.members


def test_top_k_carries_bounded_headroom_over_demand() -> None:
    from features.practice_generation.question_semantic_reuse import resolve_candidate_top_k

    # Headroom so stale/untrusted/duplicate candidates do not starve the demand.
    assert resolve_candidate_top_k(12, 1) == 12
    assert resolve_candidate_top_k(12, 100) == 200
    # And still bounded: never an unbounded fetch.
    assert resolve_candidate_top_k(12, 5000) == 200


def test_numeric_variants_are_not_treated_as_near_duplicates() -> None:
    from features.practice_generation.question_semantic_reuse import is_near_duplicate

    first = "A shop gives a 20 percent discount on an item marked 1500 find the selling price"
    second = "A shop gives a 15 percent discount on an item marked 2400 find the selling price"

    # This is exactly the reuse Phase D exists to enable.
    assert not is_near_duplicate(second, [frozenset(first.casefold().split())])


def test_near_verbatim_repeats_are_suppressed() -> None:
    from features.practice_generation.question_semantic_reuse import is_near_duplicate

    first = "A shop gives a 20 percent discount on an item marked 1500 find the selling price"
    near = "A shop gives a 20 percent discount on an item marked 1500 find the selling price now"

    assert is_near_duplicate(near, [frozenset(first.casefold().split())])


def test_hundred_slots_with_a_large_rejecting_pool_still_fill_exactly() -> None:
    """100 equivalent slots, >100 candidates, with stale and duplicate entries mixed
    in, and enough valid candidates deeper in the ranking to satisfy demand."""
    valid = [
        bank_row("profit_and_loss", question=f"Discount scenario {i} find the selling price.")
        for i in range(100)
    ]
    stale = [
        bank_row("profit_and_loss", question=f"Stale scenario {i} find the selling price.")
        for i in range(40)
    ]
    results: list[tuple[str, float, str]] = []
    # Stale entries rank first, so valid candidates only appear deeper in the pool.
    for i, row in enumerate(stale):
        results.append((row["qbId"], 0.99 - i * 0.0001, "stale-hash"))
    for i, row in enumerate(valid):
        results.append((row["qbId"], 0.80 - i * 0.0001, row["versionHash"]))
    # Duplicate deliveries of the first valid candidate.
    results.append((valid[0]["qbId"], 0.79, valid[0]["versionHash"]))
    finder = _Finder(results)

    outcome = run(valid + stale, finder, [slot(i) for i in range(1, 101)])

    assert outcome.group_count == 1
    assert outcome.embedding_call_count == 1
    assert outcome.semantic_search_count == 1
    assert outcome.vector_api_call_count == 1
    assert finder.calls[0]["top_k"] == 200, "headroom must exceed demand"
    assert outcome.version_parity_rejected_count == 40
    assert outcome.would_reuse_count == 100
    assert len(set(c.question_id for c in outcome.selected_by_slot.values())) == 100


# --- service paging: real finder over a fake paging vector client -------------

class _PagingVectorClient:
    """Serves candidates across pages, exactly as questions-v1 does."""

    def __init__(self, pages: list[list[tuple[str, float, str]]]) -> None:
        self.pages = pages
        self.requests: list[str | None] = []

    def query_question_page(self, *, query_vector, metadata_filter, top_k, next_token=None):
        from retrieval.models import RetrievedCandidate

        self.requests.append(next_token)
        index = 0 if next_token is None else int(next_token)
        page = self.pages[index]
        following = str(index + 1) if index + 1 < len(self.pages) else None
        return (
            [
                RetrievedCandidate(pattern_id=key, score=score, version_hash=vhash)
                for key, score, vhash in page
            ],
            following,
        )


class _StubEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.01] * 1024


def _paging_finder(pages):
    from features.practice_generation.question_semantic_reuse import S3QuestionCandidateFinder

    client = _PagingVectorClient(pages)
    finder = S3QuestionCandidateFinder(embedder=_StubEmbedder(), vector_client=client)
    return finder, client


def test_a_single_sufficient_page_makes_no_further_request() -> None:
    page_one = [(f"qb-v2-{i}", 0.9, "h") for i in range(12)]
    finder, client = _paging_finder([page_one, [("qb-v2-later", 0.5, "h")]])

    discovery = finder.find_candidates(demand_text="d", metadata_filter={}, top_k=12)

    assert discovery.api_call_count == 1
    assert client.requests == [None], "must not fetch a page it does not need"


def test_pagination_continues_until_the_demand_can_be_met() -> None:
    pages = [
        [(f"qb-v2-p1-{i}", 0.9, "h") for i in range(5)],
        [(f"qb-v2-p2-{i}", 0.8, "h") for i in range(5)],
        [(f"qb-v2-p3-{i}", 0.7, "h") for i in range(5)],
    ]
    finder, client = _paging_finder(pages)

    discovery = finder.find_candidates(demand_text="d", metadata_filter={}, top_k=12)

    assert discovery.api_call_count == 3
    assert len(discovery.candidates) == 15
    assert client.requests == [None, "1", "2"]


def test_pagination_stops_when_the_service_is_exhausted() -> None:
    finder, client = _paging_finder([[("qb-v2-only", 0.9, "h")]])

    discovery = finder.find_candidates(demand_text="d", metadata_filter={}, top_k=50)

    assert discovery.api_call_count == 1
    assert len(discovery.candidates) == 1


def test_pagination_is_bounded_even_with_an_endless_service() -> None:
    from features.practice_generation.question_semantic_reuse import _MAX_DISCOVERY_PAGES

    class _Endless(_PagingVectorClient):
        def query_question_page(self, *, query_vector, metadata_filter, top_k, next_token=None):
            from retrieval.models import RetrievedCandidate

            self.requests.append(next_token)
            token = str(len(self.requests))
            return (
                [RetrievedCandidate(pattern_id=f"qb-{token}", score=0.9, version_hash="h")],
                token,
            )

    from features.practice_generation.question_semantic_reuse import S3QuestionCandidateFinder

    client = _Endless([])
    finder = S3QuestionCandidateFinder(embedder=_StubEmbedder(), vector_client=client)

    discovery = finder.find_candidates(demand_text="d", metadata_filter={}, top_k=10_000)

    assert discovery.api_call_count == _MAX_DISCOVERY_PAGES


def test_hundred_slots_filled_from_candidates_spanning_multiple_pages() -> None:
    """First page is all stale; the valid candidates only arrive on later pages."""
    valid = [
        bank_row("profit_and_loss", question=f"Discount scenario {i} find the selling price.")
        for i in range(100)
    ]
    stale = [
        bank_row("profit_and_loss", question=f"Stale scenario {i} find the selling price.")
        for i in range(60)
    ]
    pages = [
        [(r["qbId"], 0.99, "stale-hash") for r in stale],
        [(r["qbId"], 0.80, r["versionHash"]) for r in valid[:50]],
        [(r["qbId"], 0.70, r["versionHash"]) for r in valid[50:]],
    ]
    finder, client = _paging_finder(pages)
    by_id = {row["qbId"]: row for row in valid + stale}
    from features.practice_generation.question_semantic_reuse import (
        QuestionSemanticReuseResolver,
    )

    outcome = QuestionSemanticReuseResolver(
        finder=finder,
        hydrate=lambda ids: [by_id[i] for i in ids if i in by_id],
        threshold=0.30,
        top_k=12,
    ).resolve(
        test_id="test-1",
        groups=group_semantic_demands([slot(i) for i in range(1, 101)], language="english"),
        language="english",
        excluded_question_ids=set(),
    )

    assert outcome.embedding_call_count == 1
    assert outcome.semantic_search_count == 1
    assert outcome.vector_api_call_count > 1, "deep pool must require paging"
    assert outcome.version_parity_rejected_count == 60
    assert outcome.would_reuse_count == 100
    assert len(set(c.question_id for c in outcome.selected_by_slot.values())) == 100
