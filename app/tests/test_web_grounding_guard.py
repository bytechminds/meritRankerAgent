"""Focused tests for mandatory current-information grounding."""

from services.context_retrieval.web_grounding import (
    required_web_answer_verified,
    required_web_context_verified,
    sanitize_required_web_answer,
)

_WEB_STATE = {
    "context_text": (
        "[Web Context]\n"
        "1. Title: Verified event\n"
        "   Content: A bounded factual event.\n"
        "   URL: https://example.gov/current-affairs\n"
    ),
    "retrieval_context": {
        "retrievalTrace": {"fallbackReason": "web_context_selected"}
    },
}


def test_required_web_search_is_blocked_after_provider_failure() -> None:
    assert (
        required_web_context_verified(
            {"need_web_search": True},
            {
                "context_text": "",
                "retrieval_context": {
                    "retrievalTrace": {"fallbackReason": "retrieval_error"}
                },
            },
        )
        is False
    )


def test_required_web_search_accepts_selected_cited_context() -> None:
    assert (
        required_web_context_verified(
            {"need_web_search": True},
            {
                "context_text": (
                    "[Web Context]\n"
                    "Source: https://example.gov/current-affairs"
                ),
                "retrieval_context": {
                    "retrievalTrace": {"fallbackReason": "web_context_selected"}
                },
            },
        )
        is True
    )


def test_static_request_does_not_require_web_context() -> None:
    assert required_web_context_verified({"need_web_search": False}, {}) is True


def test_current_affairs_practice_accepts_one_cited_question_per_source() -> None:
    answer = (
        "1. Which event was verified?\n"
        "Answer: The bounded event.\n"
        "Source: https://example.gov/current-affairs"
    )

    assert required_web_answer_verified(
        {"need_web_search": True, "intent": "practice"},
        _WEB_STATE,
        answer,
    )


def test_current_affairs_practice_rejects_more_questions_than_sources() -> None:
    answer = (
        "1. First unsupported question?\n"
        "2. Second unsupported question?\n"
        "Source: https://example.gov/current-affairs"
    )

    assert not required_web_answer_verified(
        {"need_web_search": True, "intent": "practice"},
        _WEB_STATE,
        answer,
    )


def test_required_web_answer_rejects_missing_or_unselected_citation() -> None:
    classification = {"need_web_search": True, "intent": "explain"}

    assert not required_web_answer_verified(
        classification,
        _WEB_STATE,
        "A plausible answer without a citation.",
    )
    assert not required_web_answer_verified(
        classification,
        _WEB_STATE,
        "Source: https://unselected.example/current-affairs",
    )


def test_practice_sanitizer_keeps_one_question_per_selected_source() -> None:
    answer = (
        "1. First supported question?\n"
        "Answer: Supported fact.\n"
        "Source: https://example.gov/current-affairs\n\n"
        "2. Repeated-source expansion?\n"
        "Answer: Unsupported expansion.\n"
        "Source: https://example.gov/current-affairs"
    )

    sanitized = sanitize_required_web_answer(
        {"need_web_search": True, "intent": "practice"},
        _WEB_STATE,
        answer,
    )

    assert sanitized is not None
    assert "First supported question" in sanitized
    assert "Repeated-source expansion" not in sanitized


def test_practice_sanitizer_rejects_blocks_without_selected_source() -> None:
    assert (
        sanitize_required_web_answer(
            {"need_web_search": True, "intent": "practice"},
            _WEB_STATE,
            "1. Unsupported?\nSource: https://unselected.example/event",
        )
        is None
    )
