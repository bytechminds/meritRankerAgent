"""Unit coverage for provider-neutral answer delivery decisions and replay chunks."""

from __future__ import annotations

from services.doubt_solver.answer_delivery_policy import (
    AnswerDeliveryPolicy,
    AnswerDeliverySignals,
)
from services.doubt_solver.markdown_replay import iter_markdown_replay_chunks


def _signals(**overrides: object) -> AnswerDeliverySignals:
    values: dict[str, object] = {
        "classifier_confidence": 0.99,
        "classifier_fallback": False,
        "source_modality": "text",
        "image_confidence": None,
        "image_uncertain": False,
        "difficulty": "basic",
        "current_fact_dependency": False,
        "retrieval_used": False,
        "retrieval_mode": "fresh_solve",
        "pattern_confidence": None,
        "pattern_candidate_conflict": False,
        "provider_fallback": False,
        "needs_review": False,
    }
    values.update(overrides)
    return AnswerDeliverySignals(**values)  # type: ignore[arg-type]


def _adaptive_policy() -> AnswerDeliveryPolicy:
    return AnswerDeliveryPolicy(
        mode="adaptive",
        min_classifier_confidence=0.93,
        min_image_confidence=0.90,
        min_pattern_confidence=0.90,
        max_live_difficulty="basic",
    )


def test_adaptive_policy_allows_only_explicit_low_risk_profile() -> None:
    decision = _adaptive_policy().decide(_signals())

    assert decision.strategy == "live_stream"
    assert decision.risk_level == "low"
    assert decision.reason_codes == ("approved_low_risk",)


def test_adaptive_policy_requires_verification_for_risk_signals() -> None:
    decision = _adaptive_policy().decide(
        _signals(
            classifier_confidence=0.50,
            classifier_fallback=True,
            current_fact_dependency=True,
        )
    )

    assert decision.strategy == "verified_replay"
    assert decision.risk_level == "high"
    assert "classifier_confidence_below_threshold" in decision.reason_codes
    assert "classifier_fallback" in decision.reason_codes
    assert "current_fact_dependency" in decision.reason_codes


def test_adaptive_policy_requires_verified_replay_for_uncertain_image() -> None:
    decision = _adaptive_policy().decide(
        _signals(
            source_modality="image",
            image_confidence=0.80,
            image_uncertain=True,
        )
    )

    assert decision.strategy == "verified_replay"
    assert "image_confidence_below_threshold" in decision.reason_codes
    assert "image_uncertain" in decision.reason_codes


def test_non_english_streams_are_verified_before_any_answer_chunk() -> None:
    for language in ("hindi", "hinglish"):
        decision = _adaptive_policy().decide(_signals(language=language))

        assert decision.strategy == "verified_replay"
        assert decision.reason_codes == ("language_verification_required",)


def test_always_live_does_not_bypass_non_english_language_verification() -> None:
    policy = AnswerDeliveryPolicy(
        mode="always_live",
        min_classifier_confidence=0.0,
        min_image_confidence=0.0,
        min_pattern_confidence=0.0,
        max_live_difficulty="advanced",
    )

    assert policy.decide(_signals(language="hindi")).strategy == "verified_replay"


def test_adaptive_policy_requires_runtime_ready_retrieval_for_live_delivery() -> None:
    decision = _adaptive_policy().decide(
        _signals(
            retrieval_used=True,
            retrieval_mode="pattern_assist",
            pattern_confidence=0.99,
        )
    )

    assert decision.strategy == "verified_replay"
    assert "retrieval_not_runtime_ready" in decision.reason_codes


def test_explicit_policy_modes_are_deterministic() -> None:
    always_verified = AnswerDeliveryPolicy(
        mode="always_verified",
        min_classifier_confidence=0.0,
        min_image_confidence=0.0,
        min_pattern_confidence=0.0,
        max_live_difficulty="advanced",
    )
    always_live = AnswerDeliveryPolicy(
        mode="always_live",
        min_classifier_confidence=1.0,
        min_image_confidence=1.0,
        min_pattern_confidence=1.0,
        max_live_difficulty="default",
    )

    assert always_verified.decide(_signals()).strategy == "verified_replay"
    assert always_live.decide(_signals(classifier_fallback=True)).strategy == "live_stream"
    assert (
        always_live.decide(_signals(current_fact_dependency=True)).strategy
        == "verified_replay"
    )


def test_markdown_replay_keeps_protected_blocks_whole() -> None:
    content = (
        "A short introduction.\n\n"
        "![diagram](https://example.test/a.png)\n\n"
        "\\[x = 2\\]\n\n"
        "| A | B |\n| - | - |\n| 1 | 2 |\n\n"
        "```mermaid\ngraph TD\nA --> B\n```\n"
    )

    chunks = list(iter_markdown_replay_chunks(content, max_chunk_chars=12))

    assert "".join(chunks) == content
    assert any("![diagram](https://example.test/a.png)" in chunk for chunk in chunks)
    assert any("\\[x = 2\\]" in chunk for chunk in chunks)
    assert any("| A | B |\n| - | - |\n| 1 | 2 |" in chunk for chunk in chunks)
    assert any("```mermaid\ngraph TD\nA --> B\n```" in chunk for chunk in chunks)
