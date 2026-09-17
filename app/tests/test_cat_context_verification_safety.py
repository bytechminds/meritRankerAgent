"""A dependent follow-up must not be certified standalone, nor skip verification.

Production request 14c780aa asked "What could be the size of a team that includes both F
and H?" one turn after the constraint set that defines F, H and the selection rules. The
gate certified it as `complete_self_contained_request`, so no history was read, the
classifier was forced to NEW_QUESTION, difficulty landed on `default`, and `default`
excluded the answer from correctness verification. A provably wrong answer was delivered
and persisted.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import combinations
from types import SimpleNamespace

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from schemas.conversation import RecentContextLoadResult, RecentConversationTurn
from schemas.doubt_solver import QueryClassification
from services.classification.contracts import ClassificationStageResult
from services.conversation.context_need_gate import ContextNeedGate
from services.conversation.conversation_understanding import ConversationUnderstandingService
from services.conversation.reference_resolution import assess_candidate_compatibility
from services.doubt_solver.answer_correctness import (
    requires_independent_correctness_verification,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)

_CONSTRAINTS = (
    "A team is to be selected from among ten persons — A, B, C, D, E, F, G, H, I and J — "
    "subject to the following conditions.\n\n"
    "Exactly two among E, J, I and C must be selected.\n"
    "If F is selected, then J cannot be selected.\n"
    "Exactly one among A and C must be selected.\n"
    "Unless A is selected, E cannot be selected.\n"
    "If and only if G is selected, D must not be selected.\n"
    "If D is not selected, then H must be selected.\n"
    "The size of a team is defined as the number of members in the team.\n\n"
    "Q1. Who among the following cannot be a member of a team of size 4?"
)
_FOLLOW_UP = "What could be the size of a team that includes both F and H?"
_WRONG_ANSWER = (
    "Since no additional constraints are provided, the size could be any size from 2 "
    "upwards.\n\n**Answer:** any number greater than or equal to 2"
)


def _possible_sizes_with_f_and_h() -> set[int]:
    """Deterministic oracle for the regression — never consulted by production code."""
    people = "ABCDEFGHIJ"
    sizes = set()
    for size in range(len(people) + 1):
        for team in combinations(people, size):
            members = set(team)
            if len({"E", "J", "I", "C"} & members) != 2:
                continue
            if "F" in members and "J" in members:
                continue
            if len({"A", "C"} & members) != 1:
                continue
            if "E" in members and "A" not in members:
                continue
            if ("G" in members) == ("D" in members):
                continue
            if "D" not in members and "H" not in members:
                continue
            if {"F", "H"} <= members:
                sizes.add(len(members))
    return sizes


def test_the_regression_oracle_is_five_six_or_seven() -> None:
    assert _possible_sizes_with_f_and_h() == {5, 6, 7}


class _Persistence:
    """Only the recent-context read the understanding service uses."""

    def __init__(self, *turns: RecentConversationTurn) -> None:
        self.turns = turns
        self.reads = 0

    def load_recent_context(self, _actor: str, _conversation: str, limit: int):
        self.reads += 1
        return RecentContextLoadResult(
            source="agentcore_memory" if self.turns else "none",
            memory_attempted=True,
            memory_status="succeeded" if self.turns else "empty",
            turns=self.turns[-limit:],
        )


def _turn(turn_id: str, query: str, answer: str) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_id=turn_id,
        original_query=query,
        final_answer=answer,
        created_at=datetime.now(UTC),
    )


# --- 1. The gate: dependency detection -------------------------------------------------


def test_the_dependent_follow_up_is_no_longer_certified_standalone() -> None:
    assessment = ContextNeedGate().evaluate(_FOLLOW_UP)

    assert assessment.decision != "CONTEXT_NOT_NEEDED"
    assert assessment.reason_codes[0] == "unresolved_local_reference"
    assert "unbound_symbolic_entity" in assessment.matched_signals


def test_the_explicit_follow_up_control_is_unchanged() -> None:
    assessment = ContextNeedGate().evaluate(
        "from last question statement asnwer Who could be the lowest seeded player "
        "facing the player seeded 1?"
    )

    assert assessment.decision == "CONTEXT_REQUIRED"
    assert assessment.matched_signals == ("previous_reference",)


@pytest.mark.parametrize(
    "query",
    [
        "What is 15% of 200?",
        "What is the capital of Rajasthan?",
        "Solve x + 5 = 12.",
        "A price of ₹570 is reduced by 10%. What is the new price?",
        "In triangle PQR, find angle P when Q = 60 and R = 70.",
        _CONSTRAINTS,
    ],
)
def test_standalone_questions_still_need_no_history(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    ("query", "signal"),
    [
        ("Which of them could be selected?", "unresolved_anaphor"),
        ("What would its speed be then?", "unresolved_anaphor"),
        ("Can F and H both be in a team of size 5?", "unbound_symbolic_entity"),
    ],
)
def test_implicit_symbolic_and_pronoun_follow_ups_are_recognized(
    query: str, signal: str
) -> None:
    """These would otherwise be certified standalone; the new signal withholds that."""
    assessment = ContextNeedGate().evaluate(query)

    assert assessment.decision == "UNCERTAIN"
    assert signal in assessment.matched_signals


@pytest.mark.parametrize(
    "query",
    [
        # Naming more of an earlier turn's cast is stronger evidence of dependency, not
        # weaker: the guard must not be defeated by adding one more label.
        "What is the maximum size of a team containing F, G and H?",
        "Who among A, B, C and D cannot be in a team of size 5?",
        "Can B, C and D all be selected together?",
    ],
)
def test_more_labels_do_not_defeat_the_guard(query: str) -> None:
    assessment = ContextNeedGate().evaluate(query)

    assert assessment.decision == "UNCERTAIN"
    assert "unbound_symbolic_entity" in assessment.matched_signals


@pytest.mark.parametrize(
    "query",
    [
        "Is it true that light travels faster than sound?",
        "Is it possible to divide by zero?",
        "Is it safe to mix bleach and ammonia?",
        "Is it correct to write 'different than'?",
        "Is it compulsory to study Hindi in CBSE?",
    ],
)
def test_an_expletive_it_is_not_treated_as_a_reference(query: str) -> None:
    """"It" here stands for the clause the same turn goes on to state."""
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize("query", ["Is P taller than Q?", "Which one is heavier, P or Q?"])
def test_short_symbolic_questions_keep_history_eligible(query: str) -> None:
    """Already uncertain before this change, by the gate's own standalone evidence rule."""
    assert ContextNeedGate().evaluate(query).decision == "UNCERTAIN"


@pytest.mark.parametrize(
    "query",
    [
        "What is vitamin C good for?",
        "What is vitamin B12 deficiency?",
        "What is a G20 summit?",
        "What does Theory X say about management?",
        "What is the role of the X chromosome?",
        "What is the difference between Class B and Class C IP addresses?",
        "In triangle PQR, find angle P when Q = 60 and R = 70.",
    ],
)
def test_letters_used_as_ordinary_content_are_not_treated_as_labels(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


# --- 2. Fetching context and using it stay separate decisions --------------------------


def test_a_dependent_question_with_no_history_stays_empty_handed() -> None:
    service = ConversationUnderstandingService(persistence=_Persistence())

    preparation = service.understand(
        actor_id="student", conversation_id="conv", query=_FOLLOW_UP
    )

    assert preparation.gate.decision == "UNCERTAIN"
    assert preparation.candidates == ()
    assert preparation.context_load.source == "none"


def test_unrelated_history_is_not_allowed_to_contaminate_the_answer() -> None:
    unrelated = _turn("older", "Find the HCF of 24 and 36.", "**Answer:** 12")
    service = ConversationUnderstandingService(persistence=_Persistence(unrelated))

    preparation = service.understand(
        actor_id="student", conversation_id="conv", query=_FOLLOW_UP
    )

    assert preparation.gate.decision == "UNCERTAIN"
    # The turn may be fetched; compatibility is what decides whether it may be used.
    compatibility = assess_candidate_compatibility(_FOLLOW_UP, preparation.candidates)
    assert all(not item.compatible for item in compatibility)


def test_the_relevant_prior_turn_is_loaded_for_the_dependent_follow_up() -> None:
    prior = _turn("prior", _CONSTRAINTS, "**Answer:** B cannot be a member.")
    persistence = _Persistence(prior)
    service = ConversationUnderstandingService(persistence=persistence)

    preparation = service.understand(
        actor_id="student", conversation_id="conv", query=_FOLLOW_UP
    )

    assert persistence.reads == 1
    assert [card.turn_id for card in preparation.candidates] == ["prior"]


# --- 3. `default` difficulty may no longer bypass verification -------------------------


@pytest.mark.parametrize("context_need", ["UNCERTAIN", "CONTEXT_REQUIRED"])
def test_an_uncertain_default_solve_is_verified(context_need: str) -> None:
    assert requires_independent_correctness_verification(
        subject="reasoning",
        difficulty="default",
        intent="solve",
        requested_action="ANSWER_CURRENT",
        context_need=context_need,
    )


@pytest.mark.parametrize(
    ("subject", "difficulty", "intent", "context_need"),
    [
        ("reasoning", "default", "solve", "CONTEXT_NOT_NEEDED"),
        ("math", "default", "solve", "CONTEXT_NOT_NEEDED"),
        ("math", "default", "solve", None),
        ("english", "default", "solve", "UNCERTAIN"),
        ("math", "default", "explain", "UNCERTAIN"),
    ],
)
def test_a_certified_standalone_default_keeps_the_low_cost_path(
    subject: str, difficulty: str, intent: str, context_need: str | None
) -> None:
    assert not requires_independent_correctness_verification(
        subject=subject,
        difficulty=difficulty,
        intent=intent,
        requested_action="ANSWER_CURRENT",
        context_need=context_need,
    )


@pytest.mark.parametrize("difficulty", ["intermediate", "advanced"])
def test_the_existing_difficulty_rule_is_untouched(difficulty: str) -> None:
    assert requires_independent_correctness_verification(
        subject="math", difficulty=difficulty, intent="solve", context_need="CONTEXT_NOT_NEEDED"
    )


# --- 4. End to end: the old causal chain cannot complete silently ----------------------


class _Orchestrator:
    def __init__(self) -> None:
        self.verifications: list[str] = []

    def generate(self, **kwargs: object):
        self.verifications.append(str(kwargs.get("query") or ""))
        return SimpleNamespace(
            content=(
                '{"status":"MISMATCH","independent_answer":"5, 6 or 7",'
                '"single_defensible_answer":true,"reason":"sizes are constrained"}'
            )
        )


class _Adapter:
    def __init__(self, orchestrator: _Orchestrator) -> None:
        from services.doubt_solver.answer_correctness import AnswerCorrectnessVerifier

        self.correctness_verifier = AnswerCorrectnessVerifier(orchestrator=orchestrator)  # type: ignore[arg-type]
        self.prompts: list[str] = []

    def generate(self, **kwargs: object) -> str:
        self.prompts.append(" ".join(str(value) for value in kwargs.values()))
        return _WRONG_ANSWER


@pytest.fixture
def _verified_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_RECOVERY_ENABLED", "false")
    cfg_module._settings = None
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    yield
    cfg_module._settings = None


def _classification_stage(**_: object) -> ClassificationStageResult:
    """What the classifier returns once it can see the candidate card.

    Difficulty stays `default` on purpose: this is the state the production request
    reached, and the point of the fix is that it no longer disables verification.
    """
    raw = QueryClassification(
        intent="solve_question",
        subject="reasoning",
        confidence=0.95,
        difficulty="default",
        classification_source="llm",
        relation="FOLLOW_UP",
        requested_action="ANSWER_WITH_CONTEXT",
        selected_turn_id="prior",
        requires_recent_conversation=True,
    )
    return ClassificationStageResult(
        raw=raw,
        classification={
            "subject": "reasoning",
            "intent": "solve",
            "difficulty": "default",
            "need_web_search": False,
            "classifier_confidence": 0.95,
            "classification_source": "llm",
            "requested_action": "ANSWER_WITH_CONTEXT",
        },
        modality="text",
        status="accepted",
        classifier_confidence=0.95,
        classifier_fallback=False,
    )


def test_the_14c780aa_chain_cannot_complete_silently(
    _verified_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        streaming_module, "orchestrated_classify_query_stage", _classification_stage
    )
    prior = _turn("prior", _CONSTRAINTS, "**Answer:** B cannot be a member.")
    understanding = ConversationUnderstandingService(persistence=_Persistence(prior))
    orchestrator = _Orchestrator()
    adapter = _Adapter(orchestrator)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="cat-regression",
                query=_FOLLOW_UP,
                language="english",
                actor_id="student",
                conversation_id="conv",
            ),
            adapter=adapter,  # type: ignore[arg-type]
            conversation_understanding=understanding,  # type: ignore[arg-type]
        )
    )

    # The generator was given the constraints the follow-up depends on.
    assert any("Exactly two among" in prompt for prompt in adapter.prompts)
    # `default` difficulty no longer skips the check, and the wrong answer is caught.
    assert len(orchestrator.verifications) == 1
    assert "any number greater than or equal to 2" in orchestrator.verifications[0]
    terminals = [index for index, e in enumerate(events) if e.type in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0] == len(events) - 1
    assert events[-1].type == "error"
    assert _WRONG_ANSWER not in "".join(
        str(getattr(event, "content", "") or "") for event in events
    )

