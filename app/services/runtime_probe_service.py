"""
app/services/runtime_probe_service.py
--------------------------------------
Small helper utilities for local/manual runtime verification of the Doubt Solver
invoke path.  These functions are NOT called by the main application workflow —
they are used only by manual smoke scripts and developer tooling.

Exports
-------
build_doubt_solver_smoke_payload() -> dict
    Return a representative valid payload for a local HTTP smoke call.

validate_doubt_solver_response_shape(response: dict) -> tuple[bool, list[str]]
    Check that a raw response dict contains all required Doubt Solver fields.
    Returns (is_valid, list_of_missing_or_bad_fields).

Manual smoke workflow (no auto-invocation here):
    1. Start local runtime:   make dev
    2. Curl:                  make smoke-doubt-solver
    3. Validate response:     inspect returned JSON or call
                              validate_doubt_solver_response_shape(json.loads(...))

This file has no side effects on import.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

_SMOKE_QUERY = (
    "A shopkeeper marks goods 40% above cost price and gives a 20% discount. "
    "Find the profit or loss percentage. Show step-by-step working."
)

_REQUIRED_RESPONSE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "success": bool,
    "request_id": str,
    "mode": str,
    "answer": str,
    "classification": dict,
    "needs_review": bool,
    "answer_source": str,
    "is_truncated": bool,
}

_VALID_ANSWER_SOURCES = frozenset({"mock", "llm", "fallback"})


def build_doubt_solver_smoke_payload() -> dict:
    """Return a representative Doubt Solver payload for local HTTP smoke testing.

    The returned dict is safe to serialise with ``json.dumps`` and POST to
    ``http://localhost:8080/invocations`` while ``make dev`` is running.

    Example::

        import json, urllib.request
        payload = build_doubt_solver_smoke_payload()
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "http://localhost:8080/invocations",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        ok, issues = validate_doubt_solver_response_shape(result)
        assert ok, issues
    """
    return {
        "mode": "doubt_solver",
        "query": _SMOKE_QUERY,
        "user_id": "local-smoke",
        "conversation_id": "local-smoke-conversation",
        "turn_id": "local-smoke-turn",
        "language": "en",
    }


# ---------------------------------------------------------------------------
# Response shape validator
# ---------------------------------------------------------------------------


def validate_doubt_solver_response_shape(response: dict) -> tuple[bool, list[str]]:
    """Validate a raw Doubt Solver response dict against the expected contract.

    Checks:
    - All required fields are present.
    - Field types match the schema.
    - ``answer_source`` is a valid literal.
    - ``success`` is True (basic smoke check).
    - ``classification`` has ``intent`` and ``confidence`` sub-fields.

    Args:
        response: Raw dict from the HTTP response body.

    Returns:
        ``(True, [])`` when all checks pass.
        ``(False, [<issue>, ...])`` listing every problem found.
    """
    issues: list[str] = []

    for field, expected_type in _REQUIRED_RESPONSE_FIELDS.items():
        if field not in response:
            issues.append(f"missing field: {field!r}")
            continue
        if not isinstance(response[field], expected_type):
            issues.append(
                f"field {field!r}: expected {expected_type}, "
                f"got {type(response[field]).__name__}"
            )

    if "answer_source" in response and isinstance(response["answer_source"], str):
        if response["answer_source"] not in _VALID_ANSWER_SOURCES:
            issues.append(
                f"answer_source {response['answer_source']!r} not in "
                f"{sorted(_VALID_ANSWER_SOURCES)}"
            )

    if response.get("success") is False:
        issues.append("success is False — response indicates an error")

    cls = response.get("classification")
    if isinstance(cls, dict):
        for sub in ("intent", "confidence"):
            if sub not in cls:
                issues.append(f"classification missing sub-field: {sub!r}")
    elif "classification" in response:
        issues.append("classification is not a dict")

    return (len(issues) == 0), issues
