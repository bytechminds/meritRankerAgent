"""Presentation-contract tests for subject- and intent-aware answer formatting.

Two layers are covered:

1. Prompt composition — the shared safe-presentation contract, the subject
   presentation guidance, and the intent presentation guidance are actually
   assembled into the generator system prompt, and differ across subjects.
2. Golden answer shapes — a representative answer written in each prescribed
   shape passes the deterministic answer-quality gate, so the prescribed
   presentation can never trigger a rewrite loop in production.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from schemas.llm_routing import RouteDecision
from services.doubt_solver.answer_quality import (
    AnswerQualityPolicy,
    validate_answer_quality,
)
from services.llm.orchestration.prompt_resolver import PromptResolver

_APP_DIR = Path(__file__).resolve().parents[1]
_PROMPT_ROOT = _APP_DIR / "prompts"

_SUBJECT_PROMPTS = {
    "math": "subjects/math_generator.md",
    "reasoning": "subjects/reasoning_generator.md",
    "english": "subjects/english_generator.md",
    "general": "subjects/general_generator.md",
}


def _system_prompt(subject: str, intent: str = "solve") -> str:
    route = RouteDecision(
        route_id=f"{subject}.generator.default",
        subject=subject,
        task_role="generator",
        difficulty="default",
        intent=intent,
        model="test_generator",
        prompt=_SUBJECT_PROMPTS[subject],
        intent_overlays={
            "solve": ["intents/solve.md"],
            "explain": ["intents/explain.md"],
            "visualize": ["intents/visualize.md"],
        },
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )
    resolver = PromptResolver(prompt_root=_PROMPT_ROOT)
    return resolver.resolve(route, query="Answer this question.")[0].content


@pytest.fixture
def policy() -> AnswerQualityPolicy:
    return AnswerQualityPolicy(
        validation_enabled=True,
        rewrite_enabled=True,
        max_rewrite_attempts=1,
        math_intermediate_max_chars=2200,
        max_visible_steps=8,
        max_display_math_blocks=6,
        max_math_line_chars=300,
        completion_marker="<ANSWER_DONE>",
    )


# ---------------------------------------------------------------------------
# Shared safe presentation contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subject", sorted(_SUBJECT_PROMPTS))
def test_shared_presentation_contract_is_composed_for_every_subject(subject: str) -> None:
    content = _system_prompt(subject)

    assert "## Presentation and structure" in content
    assert "Keep `**Answer:**` as the first line" in content
    assert "Leave a blank line between sections" in content
    assert "never more than eight numbered items" in content
    assert "Match length to the question" in content
    assert "Never emit an empty heading" in content


@pytest.mark.parametrize("subject", sorted(_SUBJECT_PROMPTS))
def test_answer_label_stays_bold_and_is_never_replaced_by_a_heading(subject: str) -> None:
    """`detect_final_answer` matches bold labels only — a `## Answer` heading
    would be read as a missing final answer and force a rewrite."""
    content = _system_prompt(subject)

    assert "Start with `**Answer:**` and give the direct answer first." in content
    assert "## Answer\n" not in content


@pytest.mark.parametrize("subject", sorted(_SUBJECT_PROMPTS))
def test_unsupported_markup_is_forbidden_for_every_subject(subject: str) -> None:
    content = _system_prompt(subject)

    assert "Never output Mermaid, Graphviz, PlantUML, SVG" in content
    assert "Avoid Markdown tables and fenced code blocks" in content
    assert "Never output raw HTML" in content


def test_visualize_overlay_no_longer_permits_diagram_markup() -> None:
    overlay = (_PROMPT_ROOT / "intents" / "visualize.md").read_text(encoding="utf-8")

    assert "Mermaid" in overlay  # present only as an explicit prohibition
    assert "use a table, numbered flow steps, or Mermaid diagram" not in overlay
    assert "Use Mermaid only when a flow or hierarchy genuinely adds clarity" not in overlay
    assert "Do not output Mermaid, Graphviz, PlantUML, SVG, HTML" in overlay


def test_streaming_safety_guidance_is_present() -> None:
    content = _system_prompt("general")

    assert "The answer is streamed to the student as it is written." in content


# ---------------------------------------------------------------------------
# Subject-aware guidance — present, and different per subject
# ---------------------------------------------------------------------------


def test_each_subject_contributes_its_own_answer_shape() -> None:
    shapes = {subject: _system_prompt(subject) for subject in _SUBJECT_PROMPTS}

    for content in shapes.values():
        assert "## Answer shape" in content

    assert "`**Formula / Method:**` in one line" in shapes["math"]
    assert "Give each deduction its own line." in shapes["reasoning"]
    assert "`Incorrect: <original>`" in shapes["english"]
    assert "## Subject presentation" in shapes["general"]


def test_subject_presentation_is_not_one_universal_template() -> None:
    math_content = _system_prompt("math")
    english_content = _system_prompt("english")
    general_content = _system_prompt("general")

    assert "## Subject presentation" not in math_content
    assert "Give each deduction its own line." not in english_content
    assert "`**Formula / Method:**` in one line" not in general_content


@pytest.mark.parametrize(
    ("family", "marker"),
    [
        ("physics", "Preserve units, signs, and dimensions."),
        ("chemistry", "reaction arrows"),
        ("biology", "`**Core Idea:**`"),
        ("history", "`**1857** — <event>` in date order"),
        ("geography", "Never fabricate a map or coordinates."),
        ("polity", "Never invent an article number, amendment, or legal provision"),
        ("economics", "never claim to draw one"),
        ("literature", "Never fabricate a quotation, line number, or attribution."),
        ("computer science", "do not emit fenced code blocks"),
        ("short factual", "Never expand a one-line factual answer into an essay."),
    ],
)
def test_general_route_carries_per_family_presentation_guidance(family: str, marker: str) -> None:
    """Physics, history, polity and the rest all classify as `general`, so their
    guidance must live in the general prompt — no new classifier field."""
    content = _system_prompt("general")

    assert marker in content, f"missing presentation guidance for {family}"


# ---------------------------------------------------------------------------
# Intent-aware guidance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        "**Define / what is**",
        "**Explain**",
        "**Why**",
        "**How / process**",
        "**Compare / difference between**",
        "**Why an option is right or wrong**",
    ],
)
def test_explain_intent_carries_question_shape_guidance(marker: str) -> None:
    content = _system_prompt("general", intent="explain")

    assert marker in content


def test_compare_uses_labelled_blocks_rather_than_a_table() -> None:
    content = _system_prompt("general", intent="explain")

    assert "so the two are easy to line up" in content
    assert "Do not use a table." in content


def test_solve_intent_keeps_scannable_working_without_forcing_a_template() -> None:
    content = _system_prompt("math", intent="solve")

    assert "one step or one equation per line" in content
    assert "do not force Given, Approach, Steps, or Final Answer headings." in content


# ---------------------------------------------------------------------------
# Golden answer shapes — must pass the deterministic quality gate
# ---------------------------------------------------------------------------

_GOLDEN_ANSWERS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "simple_arithmetic",
        "math",
        "basic",
        "solve",
        "**Answer:** 45\n\n**Solution:**\n\n15 × 3 = 45\n<ANSWER_DONE>",
    ),
    (
        "multi_step_quantitative",
        "math",
        "intermediate",
        "solve",
        "**Answer:** 89 seconds\n\n"
        "**Given**\n\n"
        "- Train length = 240 m\n"
        "- Platform length = 650 m\n"
        "- Pole-crossing time = 24 s\n\n"
        "**Formula / Method:** Speed = distance / time\n\n"
        "**Solution:**\n\n"
        "Speed = 240 / 24 = 10 m/s\n\n"
        "Distance = 240 + 650 = 890 m\n\n"
        "Time = 890 / 10 = 89 s\n\n"
        "**Final Answer:** 89 seconds\n<ANSWER_DONE>",
    ),
    (
        "conceptual_physics",
        "general",
        "intermediate",
        "explain",
        "**Answer:** Inertia is the resistance of a body to a change in its state of motion.\n\n"
        "**Concept / Rule:** A body stays at rest or in uniform motion unless a net external "
        "force acts on it.\n\n"
        "**Important Exam Facts:**\n\n"
        "- Mass is the measure of inertia.\n"
        "- Greater mass means greater inertia.\n"
        "<ANSWER_DONE>",
    ),
    (
        "chemistry_explanation",
        "general",
        "intermediate",
        "explain",
        "**Answer:** Rusting is the oxidation of iron in the presence of oxygen and moisture.\n\n"
        "**Concept / Rule:** Iron loses electrons and forms hydrated iron oxide.\n\n"
        "**Explanation:**\n\n"
        "Iron + Oxygen + Water → Hydrated iron(III) oxide\n\n"
        "**Important Exam Facts:**\n\n"
        "- Both oxygen and water are required.\n"
        "- Painting or galvanising prevents contact and stops rusting.\n"
        "<ANSWER_DONE>",
    ),
    (
        "biology_process",
        "general",
        "intermediate",
        "explain",
        "**Answer:** A reflex action is an involuntary response routed through the spinal cord.\n\n"
        "**Core Idea:** The signal bypasses the brain so the response is faster.\n\n"
        "**How It Works:**\n\n"
        "Stimulus\n"
        "→ Receptor\n"
        "→ Sensory neuron\n"
        "→ Spinal cord\n"
        "→ Motor neuron\n"
        "→ Response\n\n"
        "**Key Points:**\n\n"
        "- The pathway is called a reflex arc.\n"
        "- The brain is informed after the response occurs.\n"
        "<ANSWER_DONE>",
    ),
    (
        "history_chronology",
        "general",
        "intermediate",
        "explain",
        "**Answer:** The Indian National Congress was founded in 1885.\n\n"
        "**Context:** It became the main platform of the national movement.\n\n"
        "**Key Events:**\n\n"
        "**1885** — Indian National Congress founded\n"
        "**1905** — Partition of Bengal announced\n"
        "**1919** — Rowlatt Act passed\n\n"
        "**Important Exam Facts:**\n\n"
        "- A. O. Hume played a central organising role.\n"
        "<ANSWER_DONE>",
    ),
    (
        "geography_process",
        "general",
        "intermediate",
        "explain",
        "**Answer:** Relief rainfall occurs when moist air is forced up a mountain barrier.\n\n"
        "**Process:**\n\n"
        "Warm moist air\n"
        "→ rises over the slope\n"
        "→ cools\n"
        "→ condenses\n"
        "→ rainfall\n\n"
        "**Effects / Importance:**\n\n"
        "- The windward slope receives heavy rain.\n"
        "- The leeward side forms a rain shadow.\n"
        "<ANSWER_DONE>",
    ),
    (
        "polity_conceptual",
        "general",
        "intermediate",
        "explain",
        "**Answer:** The Preamble declares India a sovereign, socialist, secular, democratic "
        "republic.\n\n"
        "**Core Principle:** It states the source of authority and the objectives of the "
        "Constitution.\n\n"
        "**Important Exam Facts:**\n\n"
        "- It was adopted on 26 November 1949.\n"
        "- The words socialist and secular were added later by amendment.\n"
        "<ANSWER_DONE>",
    ),
    (
        "economics_explanation",
        "general",
        "intermediate",
        "explain",
        "**Answer:** Inflation is a sustained rise in the general price level.\n\n"
        "**Concept:** It reduces the purchasing power of money.\n\n"
        "**How It Works:**\n\n"
        "- Demand rises faster than supply, or input costs rise.\n"
        "- Producers raise prices to protect margins.\n\n"
        "**Effect / Implication:** Savers lose real value while borrowers repay in cheaper "
        "money.\n"
        "<ANSWER_DONE>",
    ),
    (
        "english_grammar",
        "english",
        "basic",
        "solve",
        "**Answer:** has\n\n"
        "**Rule / Reason:** A singular subject takes a singular verb.\n\n"
        "**Correction:**\n\n"
        "Incorrect: The list of items have been sent.\n"
        "Correct: The list of items has been sent.\n"
        "<ANSWER_DONE>",
    ),
    (
        "literature_explanation",
        "general",
        "intermediate",
        "explain",
        "**Answer:** The poem contrasts choice with regret.\n\n"
        "**Context / Theme:** The speaker looks back at a decision that shaped his life.\n\n"
        "**Explanation:**\n\n"
        "- The two roads represent competing life choices.\n"
        "- The closing sigh leaves the tone deliberately ambiguous.\n"
        "<ANSWER_DONE>",
    ),
    (
        "reasoning_problem",
        "reasoning",
        "intermediate",
        "solve",
        "**Answer:** South-East\n\n"
        "**Logic / Rule:** Track each turn from the starting direction.\n\n"
        "**Solution:**\n\n"
        "Start facing North\n"
        "→ right turn gives East\n"
        "→ right turn gives South\n\n"
        "The final displacement is South-East of the start point.\n"
        "<ANSWER_DONE>",
    ),
    (
        "computer_science_explanation",
        "general",
        "intermediate",
        "explain",
        "**Answer:** Binary search repeatedly halves a sorted range to locate a value.\n\n"
        "**Concept:** Each comparison discards half of the remaining elements.\n\n"
        "**How It Works:**\n\n"
        "- Compare the target with the middle element.\n"
        "- Discard the half that cannot contain the target.\n"
        "- Repeat until the range is empty.\n\n"
        "**Important Exam Facts:**\n\n"
        "- The list must be sorted first.\n"
        "<ANSWER_DONE>",
    ),
    (
        "very_short_factual",
        "general",
        "basic",
        "explain",
        "**Answer:** Dr. Rajendra Prasad\n\n"
        "**Important Exam Facts:**\n\n"
        "- He was the first President of India.\n"
        "- He held office from 1950 to 1962.\n"
        "<ANSWER_DONE>",
    ),
    (
        "long_explanatory_answer",
        "general",
        "advanced",
        "explain",
        "**Answer:** The monsoon is a seasonal reversal of wind driven by differential "
        "heating of land and sea.\n\n"
        "**Concept / Rule:** Land heats and cools faster than the ocean, so the pressure "
        "gradient reverses between summer and winter.\n\n"
        "**How It Works:**\n\n"
        "Intense summer heating of the landmass\n"
        "→ low pressure develops over the interior\n"
        "→ moist air flows in from the ocean\n"
        "→ widespread summer rainfall\n\n"
        "**Effects / Importance:**\n\n"
        "- Agricultural output depends closely on arrival and distribution.\n"
        "- Delayed onset affects sowing and reservoir levels.\n\n"
        "**Important Exam Facts:**\n\n"
        "- The retreating monsoon brings rain to the south-eastern coast.\n"
        "<ANSWER_DONE>",
    ),
)


@pytest.mark.parametrize(
    ("name", "subject", "difficulty", "intent", "content"),
    _GOLDEN_ANSWERS,
    ids=[case[0] for case in _GOLDEN_ANSWERS],
)
def test_golden_presentation_shapes_pass_the_quality_gate(
    name: str,
    subject: str,
    difficulty: str,
    intent: str,
    content: str,
    policy: AnswerQualityPolicy,
) -> None:
    result = validate_answer_quality(
        content,
        subject=subject,
        difficulty=difficulty,
        intent=intent,
        query="Answer this question.",
        policy=policy,
    )

    assert result.is_valid, f"{name} rejected: {result.reason_codes}"
    assert result.severity == "clean", f"{name}: {result.reason_codes}"


@pytest.mark.parametrize(
    ("name", "subject", "difficulty", "intent", "content"),
    _GOLDEN_ANSWERS,
    ids=[case[0] for case in _GOLDEN_ANSWERS],
)
def test_golden_shapes_start_with_a_detectable_direct_answer(
    name: str,
    subject: str,
    difficulty: str,
    intent: str,
    content: str,
) -> None:
    from services.doubt_solver.answer_quality import detect_final_answer

    assert content.startswith("**Answer:**"), name
    assert detect_final_answer(content), name


@pytest.mark.parametrize(
    ("name", "subject", "difficulty", "intent", "content"),
    _GOLDEN_ANSWERS,
    ids=[case[0] for case in _GOLDEN_ANSWERS],
)
def test_golden_shapes_use_only_the_safe_markdown_subset(
    name: str,
    subject: str,
    difficulty: str,
    intent: str,
    content: str,
) -> None:
    assert "```" not in content, f"{name} uses a fenced code block"
    assert "|" not in content, f"{name} uses a Markdown table"
    assert "<" not in content.replace("<ANSWER_DONE>", ""), f"{name} uses raw HTML"
    assert "$" not in content, f"{name} uses dollar math delimiters"
    assert "mermaid" not in content.lower(), f"{name} uses diagram markup"


def test_heading_style_answer_would_be_rejected_by_the_quality_gate(
    policy: AnswerQualityPolicy,
) -> None:
    """Regression guard for the reason the contract keeps `**Answer:**` bold."""
    heading_style = (
        "## Answer\n\n**89 seconds**\n\n## Final Answer\n\n**89 seconds**\n<ANSWER_DONE>"
    )

    result = validate_answer_quality(
        heading_style,
        subject="math",
        difficulty="default",
        intent="solve",
        query="Solve this.",
        policy=policy,
    )

    assert not result.is_valid
    assert "missing_final_answer" in result.reason_codes
