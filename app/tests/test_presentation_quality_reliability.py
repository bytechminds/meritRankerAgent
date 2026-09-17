"""Presentation repair must return the same solution in different formatting.

Two defects are covered. `math_single_dollar` used a broad `$...$` regex that read ordinary
currency prose ("the cost is $5 and the price is $12") as a malformed math span, so a
correct Quant answer was sent for an LLM rewrite and could fail closed before the
correctness verifier ever ran (production request 45e227c7). And the rewrite replaced the
draft whether or not it was accepted, so a failed reformat could discard a usable answer and
a reformat that changed the answer was indistinguishable from one that changed the layout.
"""

from __future__ import annotations

import pytest

from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_quality import (
    GENERATION_FAILURE_MESSAGE,
    AnswerQualityPolicy,
    is_presentation_only_failure,
    preserves_answer_surface,
    validate_answer_quality,
)
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor

_QUERY = "A shopkeeper buys and sells an item. Find the profit."


def _policy() -> AnswerQualityPolicy:
    return AnswerQualityPolicy.from_settings()


def _quality(content: str, *, difficulty: str = "advanced", subject: str = "math"):
    return validate_answer_quality(
        content,
        subject=subject,
        difficulty=difficulty,
        intent="solve",
        query=_QUERY,
        language="english",
        policy=_policy(),
    )


# --- 1. Currency must never read as malformed inline math ------------------------------

_CURRENCY_ANSWER = (
    "**Approach:** Compare cost price and selling price.\n\n"
    "Step 1: The cost price is $5 and the selling price is $12 per unit.\n"
    "Step 2: Profit per unit is $12 - $5 = $7.\n"
    "Step 3: Over 20 units the profit is $140.\n\n"
    "**Answer:** $140"
)


@pytest.mark.parametrize(
    "content",
    [
        _CURRENCY_ANSWER,
        "Item A costs $40.\n\nItem B costs $60.\n\n**Answer:** $100",
        "Costs are $10, $20 and $30 respectively.\n\n**Answer:** $60",
        "| Item | Price |\n|---|---|\n| Pen | $5 |\n| Book | $12 |\n\n**Answer:** $17",
        "The price is \\$5 and then \\$12.\n\n**Answer:** \\$17",
    ],
)
def test_currency_is_not_reported_as_single_dollar_math(content: str) -> None:
    assert "math_single_dollar" not in _quality(content).reason_codes


# --- 2. Genuine single-dollar LaTeX is still detected -----------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "Step 1: We are given $3x + 7 = 22$.\n\n**Answer:** 5",
        "Let $x+1$ be the expression.\n\n**Answer:** 5",
    ],
)
def test_real_single_dollar_math_is_still_detected(content: str) -> None:
    assert "math_single_dollar" in _quality(content).reason_codes


def test_currency_and_real_math_together_are_still_detected() -> None:
    content = "Profit is $200 while $x+1$ models the cost.\n\n**Answer:** 5"

    assert "math_single_dollar" in _quality(content).reason_codes


# --- 3. Display-block count is presentation-only, and stays so --------------------------


def test_many_display_blocks_remain_a_presentation_only_failure() -> None:
    steps = "\n\n".join(f"Step {i}:\n\n\\[ x_{i} = {i} \\]" for i in range(1, 9))
    result = _quality(f"{steps}\n\n**Answer:** 8")

    assert "too_many_display_math_blocks" in result.reason_codes
    assert is_presentation_only_failure(result)


# --- 4 & 7. A repair may not change the answer it is repairing --------------------------

_DRAFT = "Step 1: work with enough body text.\nStep 2: more work here.\n\n**Answer:** 42"


@pytest.mark.parametrize(
    ("candidate", "preserved"),
    [
        # Reformatting the body and the markup around the answer is what repair is for.
        ("Step 1: work with enough body text.\n\nStep 2: more work here.\n\n**Answer:** 42", True),
        (
            "Step 1: work with enough body text.\nStep 2: more work here.\n\n**Answer:** \\(42\\)",
            True,
        ),
        # Everything that changes what the student reads as the answer.
        ("Step 1: work with enough body text.\nStep 2: more work here.\n\n**Answer:** 7", False),
        (
            "Step 1: work with enough body text.\nStep 2: more work here.\n\n**Answer:** 42 km",
            False,
        ),
        ("Step 1: work, restated briefly with no answer heading at all here.", False),
    ],
)
def test_answer_surface_preservation(candidate: str, preserved: bool) -> None:
    assert preserves_answer_surface(_DRAFT, candidate) is preserved


@pytest.mark.parametrize(
    ("draft_answer", "candidate_answer"),
    [
        ("12 s", "12 m"),                       # unit carries the meaning
        ("12.5% profit", "12.5% loss"),         # same number, opposite outcome
        ("Cannot be determined from the data", "25 km"),  # conclusion flipped
        ("E", "B"),                             # option outside the A-D scalar range
        ("Rs 1200", "Rs 900"),                  # currency answer
        ("3/4", "7/8"),                         # ratio
    ],
)
def test_a_changed_answer_is_rejected_whatever_shape_it_takes(
    draft_answer: str, candidate_answer: str
) -> None:
    body = "A draft body long enough to clear the repair length floor comfortably here."
    draft = f"{body}\n\n**Answer:** {draft_answer}"
    candidate = f"{body}\n\n**Answer:** {candidate_answer}"

    assert preserves_answer_surface(draft, candidate) is False


def test_a_multi_part_answer_must_keep_every_part() -> None:
    body = "A draft body long enough to clear the repair length floor comfortably here."
    draft = f"{body}\n\n**Answer (a):** 12.5 km/h\n\n**Answer (b):** 2.5 km/h"

    assert preserves_answer_surface(draft, f"{body}\n\n**Answer (a):** 3 km/h") is False


def test_a_self_contradicting_draft_is_not_resolved_by_presentation_repair() -> None:
    """Choosing between two stated answers is a semantic decision, not a reformat."""
    draft = "Body text.\n\n**Answer:** 3\n\nThe derivation shows 2.5 clearly.\n\n**Answer:** 2.5"

    assert preserves_answer_surface(draft, "Body text.\n\n**Answer:** 3") is False


def test_a_repair_that_discards_most_of_the_draft_is_rejected() -> None:
    """The final answer surviving is not enough: the working has to survive too."""
    draft = "Step: " + "a long derivation sentence. " * 20 + "\n\n**Answer:** 4"

    assert preserves_answer_surface(draft, "Short claim.\n\n**Answer:** 4") is False


def test_percent_wording_is_not_treated_as_a_changed_answer() -> None:
    body = "A draft body long enough to clear the repair length floor comfortably here."

    assert preserves_answer_surface(
        f"{body}\n\n**Answer:** 25%", f"{body}\n\n**Answer:** 25 percent"
    )


# --- Orchestrator behaviour ------------------------------------------------------------

_REQUEST = RouteRequest(
    request_id="presentation-quality",
    subject="math",
    task_role="generator",
    difficulty="intermediate",
    intent="solve",
)


def _orchestrator(*contents: str) -> tuple[LlmOrchestrator, list[int]]:
    calls: list[int] = []

    class _Sequenced:
        last_stream_finish_reason = "stop"

        def execute(self, *, route_decision, messages):  # noqa: ANN001, ANN202
            calls.append(len(messages))
            content = contents[min(len(calls), len(contents)) - 1]
            return MockModelExecutor(content=content, finish_reason="stop").execute(
                route_decision=route_decision, messages=messages
            )

    return LlmOrchestrator(model_executor=_Sequenced()), calls


_CLEAN = (
    "**Approach:** Use the profit formula.\n\n"
    "Step 1: Cost price is 5 and selling price is 12.\n"
    "Step 2: Profit per unit is 7, so 20 units give 140.\n\n"
    "**Answer:** 140"
)


# --- 5. A failed rewrite must not discard a usable draft --------------------------------


def test_a_rewrite_that_changes_the_answer_is_rejected_and_the_draft_is_kept() -> None:
    draft = "Step 1: cost 5, price 12.\nStep 2: profit is 7 each.\n\n**Answer:** 140"
    changed = "Step 1: cost 5, price 12.\n\n**Answer:** 7"
    orchestrator, calls = _orchestrator(
        draft + "\n\\[ a \\]\n\\[ b \\]\n\\[ c \\]\n\\[ d \\]\n\\[ e \\]\n\\[ f \\]\n"
        "\\[ g \\]\n<ANSWER_DONE>",
        changed + "\n<ANSWER_DONE>",
    )

    result = orchestrator.generate(route_request=_REQUEST, query=_QUERY)

    assert len(calls) == 2  # draft plus exactly one rewrite; the budget is unchanged
    assert "**Answer:** 140" in result.final_answer.content
    assert "**Answer:** 7" not in result.final_answer.content


# --- 6. The 45e227c7 shape: currency no longer forces a rewrite -------------------------


def test_a_currency_answer_needs_no_rewrite_and_reaches_the_caller_clean() -> None:
    orchestrator, calls = _orchestrator(_CURRENCY_ANSWER + "\n<ANSWER_DONE>")

    result = orchestrator.generate(route_request=_REQUEST, query=_QUERY)

    assert len(calls) == 1  # no rewrite provider call on the healthy path
    assert result.final_answer.quality_status != "failed_quality_gate"
    assert "$140" in result.final_answer.content


# --- 8. Healthy answers are untouched ---------------------------------------------------


def test_a_healthy_answer_is_returned_unchanged_with_one_provider_call() -> None:
    orchestrator, calls = _orchestrator(_CLEAN + "\n<ANSWER_DONE>")

    result = orchestrator.generate(route_request=_REQUEST, query=_QUERY)

    assert len(calls) == 1
    assert result.final_answer.content == _CLEAN
    assert result.final_answer.quality_status != "failed_quality_gate"


def test_a_hard_failing_draft_is_still_replaced_after_a_failed_rewrite() -> None:
    """Keeping the draft is only for presentation defects; a malformed one still fails closed."""
    broken = _CLEAN + "\n\nCheck: \\( s + b = 15"
    orchestrator, calls = _orchestrator(broken + "\n<ANSWER_DONE>", broken + "\n<ANSWER_DONE>")

    result = orchestrator.generate(route_request=_REQUEST, query=_QUERY)

    assert len(calls) == 2
    assert result.final_answer.content == GENERATION_FAILURE_MESSAGE


# --- 9. Lifecycle on the repaired path --------------------------------------------------


class _Verifier:
    """Real verifier shape: records what it was asked to check."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def verify(self, **kwargs: object):  # noqa: ANN003
        from services.doubt_solver.answer_correctness import CorrectnessVerification

        self.checked.append(str(kwargs.get("candidate_answer") or ""))
        return CorrectnessVerification(
            status="match",
            single_defensible_answer=True,
            method="model",
            reason_code="NONE",
        )


class _CurrencyAdapter:
    def __init__(self, verifier: _Verifier) -> None:
        self.correctness_verifier = verifier
        self.generate_calls = 0

    def generate(self, **_: object) -> str:
        self.generate_calls += 1
        return _CURRENCY_ANSWER


def test_the_45e227c7_shape_reaches_the_verifier_with_one_generator_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A currency-heavy Quant answer used to fail closed before verification ever ran."""
    import config as cfg_module
    import services.doubt_solver.streaming_doubt_solver_service as streaming_module
    from services.doubt_solver.streaming_doubt_solver_service import (
        StreamDoubtSolverInput,
        stream_doubt_solver,
    )

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
    verifier = _Verifier()
    adapter = _CurrencyAdapter(verifier)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="presentation-quality-stream",
                query=_QUERY,
                language="english",
                classification={
                    "subject": "math",
                    "intent": "solve",
                    "difficulty": "advanced",
                    "need_web_search": False,
                    "classifier_confidence": 0.99,
                    "classification_source": "llm",
                },
                classifier_confidence=0.99,
            ),
            adapter=adapter,  # type: ignore[arg-type]
        )
    )
    cfg_module._settings = None

    assert adapter.generate_calls == 1  # no rewrite call for currency
    assert len(verifier.checked) == 1  # correctness verification actually ran
    assert "$140" in verifier.checked[0]
    terminals = [i for i, e in enumerate(events) if e.type in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0] == len(events) - 1
    assert events[-1].type == "complete"


# --- Padded LaTeX, escaping and scan cost -----------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "The speed is $ \\frac{d}{t} $ km/h.\n\n**Answer:** 5",
        "Result: $ \\sqrt{2} $ exactly.\n\n**Answer:** 5",
        "Write $ x^{2} $ here.\n\n**Answer:** 5",
        # A bare sub/superscript is the commoner model output and the same defect.
        "Take $ x^2 $ as the square.\n\n**Answer:** 5",
        "Let $ a_1 $ be the first term.\n\n**Answer:** 5",
    ],
)
def test_padded_latex_is_still_a_defect(content: str) -> None:
    """The renderer prints these literally, so the student sees raw LaTeX."""
    assert "math_single_dollar" in _quality(content).reason_codes


@pytest.mark.parametrize(
    "content",
    [
        "The total is $5 + $3 = $8 for the pair.\n\n**Answer:** $8",
        "Budget of $5,000 to $7,500 applies.\n\n**Answer:** $7,500",
        "Costs are $10, $20 and $30 in order.\n\n**Answer:** $60",
    ],
)
def test_currency_arithmetic_is_not_a_defect(content: str) -> None:
    assert "math_single_dollar" not in _quality(content).reason_codes


def test_escaping_markup_counts_as_preserving_the_answer() -> None:
    """The sanitizer escaping unsafe markup is a repair, not a changed answer."""
    draft = "Body text long enough to clear the repair floor here.\n\n**Answer:** ok <div>x</div>"
    escaped = (
        "Body text long enough to clear the repair floor here.\n\n"
        "**Answer:** ok &lt;div&gt;x&lt;/div&gt;"
    )

    assert preserves_answer_surface(draft, escaped)


def test_a_contradictory_draft_may_still_be_reduced_to_one_of_its_answers() -> None:
    """The existing gate asks the rewrite to resolve this shape; it may not invent a value."""
    draft = "**Answer:** 89\n\nThe HCF of 378, 513 and 621 is 27.\n\n**Answer:** 27"

    assert preserves_answer_surface(draft, "The HCF of 378, 513 and 621 is 27.\n\n**Answer:** 27")
    assert not preserves_answer_surface(draft, "Reworked at length here.\n\n**Answer:** 45")


def test_single_dollar_scan_stays_linear_on_currency_heavy_answers() -> None:
    """A long single-line currency answer used to cost seconds of quadratic scanning."""
    import time

    from services.doubt_solver.answer_quality import _has_single_dollar_math

    content = "$5 " * 15000
    started = time.perf_counter()
    flagged = _has_single_dollar_math(content)
    elapsed = time.perf_counter() - started

    assert flagged is False
    assert elapsed < 0.5


# --- Sign, multi-part answers and identifier-bearing currency prose ---------------------

_BODY = "A draft body long enough to clear the repair length floor comfortably in this test."


@pytest.mark.parametrize(
    ("draft_answer", "candidate_answer"),
    [
        ("-5", "5"),          # the smaller root of a quadratic
        ("5", "-5"),
        ("-12.5 m", "12.5 m"),  # signed displacement
    ],
)
def test_the_sign_of_an_answer_is_part_of_the_answer(
    draft_answer: str, candidate_answer: str
) -> None:
    draft = f"{_BODY}\n\n**Answer:** {draft_answer}"
    candidate = f"{_BODY}\n\n**Answer:** {candidate_answer}"

    assert preserves_answer_surface(draft, candidate) is False


def test_trailing_punctuation_is_still_treated_as_decoration() -> None:
    assert preserves_answer_surface(
        f"{_BODY}\n\n**Answer:** 42 —", f"{_BODY}\n\n**Answer:** 42"
    )


def test_a_multi_part_answer_is_not_a_contradiction_and_keeps_every_part() -> None:
    """`Answer (a)` and `Answer (b)` disagree by design, which is not a contradiction."""
    draft = f"{_BODY}\n\n**Answer (a):** 12.5 km/h\n\n**Answer (b):** 2.5 km/h"

    assert preserves_answer_surface(draft, draft)
    assert not preserves_answer_surface(draft, f"{_BODY}\n\n**Answer (a):** 12.5 km/h")


def test_a_repair_may_not_drop_precision_from_a_second_answer_surface() -> None:
    draft = f"{_BODY}\n\n**Answer:** approximately 2.5\n\n**Answer:** 2.53 km/h"

    assert not preserves_answer_surface(draft, f"{_BODY}\n\n**Answer:** approximately 2.5")


def test_a_contradictory_draft_may_not_gain_a_new_answer() -> None:
    draft = "**Answer:** 89\n\nThe HCF of 378, 513 and 621 is 27.\n\n**Answer:** 27"

    assert not preserves_answer_surface(draft, "Reworked derivation text here.\n\n**Answer:** 45")


@pytest.mark.parametrize(
    "content",
    [
        "Set item_1 to $5 and item_2 to $7 in the ledger.\n\n**Answer:** $12",
        "Costs $5, growth is 2^3 fold, final price $40.\n\n**Answer:** $40",
        "Price P_1 is $5 and P_2 is $7.\n\n**Answer:** $12",
        # Real inline math between two prices is the shape this fix exists to legalize.
        "The unit price is $5, and since \\(\\frac{3}{4}\\) sold, revenue is $9.\n\n**Answer:** $9",
        "Given \\(x_1\\) the price is $5 and \\(x_2\\) costs $9.\n\n**Answer:** $14",
        "The cost is $5 and _the rate_ is $9 per hour.\n\n**Answer:** $14",
        "Save $5 in C:\\Users\\data and $9 more.\n\n**Answer:** $14",
    ],
)
def test_identifiers_and_exponents_in_currency_prose_are_not_latex(content: str) -> None:
    assert "math_single_dollar" not in _quality(content).reason_codes
