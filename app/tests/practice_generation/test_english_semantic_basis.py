"""Soft-semantic English fails closed without a trusted lexical basis.

`valid_option_ids` is necessary but not sufficient: qualification history includes a
genuine Gemini soft-semantic false accept, so availability must not rest on the
Authority alone. This gate is deterministic, classifies the question's FORM rather
than its topic, reads no model-authored label, and adds no model call.
"""

from __future__ import annotations

import pytest

from features.practice_generation.question_contract import (
    SemanticBasis,
    classify_semantic_basis,
)

# Real SSC/Banking lexical-equivalence formats. Several of these matched no known
# operator under the first denylist implementation and silently defaulted to
# RULE_BOUND, which is why classification is now safe-by-default.
LEXICAL_EQUIVALENCE_STEMS = [
    "Which word most nearly means 'obstinate'?",
    "Which is the best substitute for the word 'arduous'?",
    "Replace the underlined word with the most appropriate option.",
    "Choose the correct one-word substitution for 'a person who talks too much'.",
    "Select the alternative that expresses the same idea as the given phrase.",
    "Choose the word similar in meaning to 'lucid'.",
    "Choose the word opposite in meaning to 'benign'.",
    "Which word is a synonym of 'abundant'?",
    "Choose the antonym of 'vivid'.",
    "Select the word that means the same as 'cautious'.",
    "Which word is closest in meaning to 'elated'?",
    "Select the word that means the opposite of 'fragile'.",
    "Choose the word nearest in meaning to 'candid'.",
    "Pick the word that means the opposite of 'scarce'.",
    "Which option has the same meaning as 'reluctant'?",
]

RULE_BOUND_STEMS = [
    'Choose the option that is grammatically correct: '
    '"Each of the students ____ submitted the form."',
    "Which sentence is grammatically correct?",
    "Choose the grammatically correct sentence.",
    "Read the passage and answer the question. What did the team do third?",
    "Identify the part of the sentence that contains an error.",
    "Choose the sentence that uses the word \"discreet\" correctly.",
]


@pytest.mark.parametrize("stem", LEXICAL_EQUIVALENCE_STEMS)
def test_lexical_equivalence_stems_require_evidence(stem: str) -> None:
    assert classify_semantic_basis(stem) is SemanticBasis.EVIDENCE_REQUIRED


@pytest.mark.parametrize("stem", RULE_BOUND_STEMS)
def test_rule_bound_stems_stay_generatable(stem: str) -> None:
    """Grammar, error detection, passage ordering and in-context usage are decidable."""
    assert classify_semantic_basis(stem) is SemanticBasis.RULE_BOUND


def test_classification_ignores_topic_and_authored_labels() -> None:
    """Safety must never route on a free-form or model-authored label."""
    rule_bound = "Which sentence is grammatically correct?"
    assert classify_semantic_basis(rule_bound) is SemanticBasis.RULE_BOUND
    # The same stem stays RULE_BOUND regardless of any surrounding topic wording.
    assert classify_semantic_basis(rule_bound) is classify_semantic_basis(
        rule_bound.upper()
    )


@pytest.mark.parametrize(
    "stem",
    [
        "Some brand-new English format nobody has seen before.",
        "Pick the option that fits.",
        "",
        None,
    ],
)
def test_unrecognized_forms_fail_closed(stem: object) -> None:
    """The safety property: classification is an allowlist, not a denylist.

    An unseen English form must never silently become generatable. This is the guard
    that the first implementation failed — it defaulted unknown stems to RULE_BOUND.
    """
    assert classify_semantic_basis(stem) is SemanticBasis.EVIDENCE_REQUIRED
