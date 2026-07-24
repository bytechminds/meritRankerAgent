"""
app/tests/test_main_routing.py
--------------------------------
Tests for mode-based routing in app/main.py.

Exercises the invoke() function directly (without AgentCore HTTP layer).
No AWS credentials, network, or real LLM calls required.
Services are deterministic stubs.

[NOT VERIFIED] AgentCore HTTP runtime end-to-end (POST /invocations) is not
               tested here — only the Python invoke() boundary is exercised.
"""

from __future__ import annotations

import main


def _doubt_solver_payload(query: str | None = None, **values) -> dict:
    payload = {
        "mode": "doubt_solver",
        "user_id": "local-user",
        "conversation_id": "conversation-main-routing",
        "turn_id": "turn-main-routing",
        **values,
    }
    if query is not None:
        payload["query"] = query
    return payload


class TestDoubtSolverRouting:
    def test_doubt_solver_mode_returns_success(self):
        payload = _doubt_solver_payload("Explain what ratio means")
        result = main.invoke(payload)
        assert result["success"] is True
        assert result["mode"] == "doubt_solver"

    def test_doubt_solver_mode_returns_answer(self):
        payload = _doubt_solver_payload("Solve this: 2x = 10")
        result = main.invoke(payload)
        assert result["answer"]
        assert len(result["answer"]) > 0

    def test_doubt_solver_mode_returns_classification(self):
        payload = _doubt_solver_payload("Explain the concept of percentage")
        result = main.invoke(payload)
        assert "classification" in result
        assert result["classification"] is not None

    def test_doubt_solver_mode_returns_request_id(self):
        payload = _doubt_solver_payload("What is profit margin?")
        result = main.invoke(payload)
        assert "request_id" in result
        assert result["request_id"]

    def test_doubt_solver_empty_query_returns_error(self):
        payload = _doubt_solver_payload("")
        result = main.invoke(payload)
        assert result["success"] is False
        assert "Validation error" in result["answer"]

    def test_doubt_solver_missing_query_returns_error(self):
        payload = _doubt_solver_payload()
        result = main.invoke(payload)
        assert result["success"] is False

    def test_doubt_solver_invalid_language_returns_error(self):
        payload = _doubt_solver_payload("What is ratio?", language="fr")
        result = main.invoke(payload)
        assert result["success"] is False


class TestDemoRouting:
    def test_demo_mode_returns_success(self):
        payload = {"message": "hello", "mode": "demo"}
        result = main.invoke(payload)
        assert result["success"] is True
        assert result["mode"] == "demo"

    def test_default_mode_returns_success(self):
        payload = {"message": "hello"}
        result = main.invoke(payload)
        assert result["success"] is True

    def test_demo_route_unaffected_by_doubt_solver_changes(self):
        payload = {"message": "test message", "mode": "demo"}
        result = main.invoke(payload)
        assert result["success"] is True
        assert result["answer"]


# ---------------------------------------------------------------------------
# Part 4 field contract — answer_source and is_truncated present in response
# ---------------------------------------------------------------------------


class TestDoubtSolverPart4Fields:
    """Verify Part 4 fields are present and well-formed through the main route.

    These tests confirm that answer_source and is_truncated are propagated from
    the graph all the way through the DoubtSolverResponse serialisation.

    No mock patching is needed because ENABLE_REAL_LLM defaults to false —
    the mock answer generator always runs, producing answer_source="mock".
    """

    def test_response_contains_answer_source(self):
        payload = _doubt_solver_payload("What is a fraction?")
        result = main.invoke(payload)
        assert result["success"] is True
        assert "answer_source" in result

    def test_answer_source_is_valid_literal(self):
        payload = _doubt_solver_payload("Explain the concept of ratio")
        result = main.invoke(payload)
        assert result["success"] is True
        assert result["answer_source"] in {"mock", "llm", "fallback"}

    def test_answer_source_is_mock_when_llm_disabled(self):
        """With ENABLE_REAL_LLM=false (default), answer_source must be 'mock'."""
        import os  # noqa: PLC0415

        import config as cfg_module  # noqa: PLC0415

        original = os.environ.get("ENABLE_REAL_LLM")
        os.environ["ENABLE_REAL_LLM"] = "false"
        # Reset the module-level singleton so env var is re-read.
        cfg_module._settings = None
        try:
            payload = _doubt_solver_payload("Define denominator")
            result = main.invoke(payload)
            assert result["success"] is True
            assert result["answer_source"] == "mock"
        finally:
            if original is None:
                os.environ.pop("ENABLE_REAL_LLM", None)
            else:
                os.environ["ENABLE_REAL_LLM"] = original
            cfg_module._settings = None

    def test_response_contains_is_truncated(self):
        payload = _doubt_solver_payload("What is HCF?")
        result = main.invoke(payload)
        assert result["success"] is True
        assert "is_truncated" in result

    def test_is_truncated_is_bool(self):
        payload = _doubt_solver_payload("Explain percentage calculation")
        result = main.invoke(payload)
        assert result["success"] is True
        assert isinstance(result["is_truncated"], bool)

    def test_is_truncated_false_for_short_answer(self):
        """Mock answers are short — is_truncated must be False."""
        payload = _doubt_solver_payload("What is LCM?")
        result = main.invoke(payload)
        assert result["success"] is True
        assert result["is_truncated"] is False

    def test_needs_review_is_present(self):
        payload = _doubt_solver_payload("Solve for x: 3x + 5 = 14")
        result = main.invoke(payload)
        assert result["success"] is True
        assert "needs_review" in result
        assert isinstance(result["needs_review"], bool)

    def test_full_response_shape(self):
        """All required fields are present in a successful doubt_solver response."""
        required_fields = {
            "success", "request_id", "mode", "answer", "classification",
            "needs_review", "answer_source", "is_truncated",
        }
        payload = _doubt_solver_payload("What is a prime number?")
        result = main.invoke(payload)
        assert result["success"] is True
        missing = required_fields - result.keys()
        assert not missing, f"Missing fields in response: {missing}"

    def test_request_id_is_uuid_format(self):
        """request_id in response must be a non-empty UUID string."""
        import uuid  # noqa: PLC0415

        payload = _doubt_solver_payload("Explain decimals")
        result = main.invoke(payload)
        assert result["success"] is True
        # Validate it parses as a UUID (raises ValueError if not)
        parsed = uuid.UUID(result["request_id"])
        assert str(parsed) == result["request_id"]

    def test_classification_has_intent_and_confidence(self):
        """classification sub-object must include intent and confidence."""
        payload = _doubt_solver_payload("What is simple interest?")
        result = main.invoke(payload)
        assert result["success"] is True
        cls = result["classification"]
        assert "intent" in cls
        assert "confidence" in cls
        assert 0.0 <= cls["confidence"] <= 1.0
