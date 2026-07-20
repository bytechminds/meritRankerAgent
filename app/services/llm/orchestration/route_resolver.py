"""
app/services/llm_orchestration/route_resolver.py
-------------------------------------------------
Deterministic LLM route resolver (Part 1).

Responsibilities:
- Normalize raw subject and difficulty strings.
- Look up the best matching route from the compiled registry.
- Apply a fixed, deterministic fallback chain.
- Resolve the route's YAML fallback symbols into typed FallbackAttempt objects.
- Return a RouteDecision with no credentials and no provider secrets.

Performance:
- No YAML parsing at request time.
- No file I/O at request time.
- No provider or LLM calls.
- All lookups are pure dict operations on the compiled registry maps.

Fixed lookup order (per request):
  1. (subject, task_role, difficulty)           → route_source="exact"
  2. (subject, task_role, "default")            → route_source="subject_default"
  3. ("general", task_role, "default")          → route_source="general_default"
  4. LlmRouteNotFoundError (safe_mock fallback
     is informational, not an automatic rescue
     for unsupported task roles)
"""

from __future__ import annotations

import logging

from schemas.llm_routing import (
    FallbackAttempt,
    ResolvedRouteEntry,
    RouteDecision,
    RouteRequest,
    TaskRole,
)
from services.llm.orchestration.config_registry import LlmConfigRegistry, get_registry
from services.llm.orchestration.errors import (
    LlmRouteNotFoundError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization tables
# ---------------------------------------------------------------------------

_SUBJECT_ALIASES: dict[str, str] = {
    "math": "math",
    "maths": "math",
    "mathematics": "math",
    "quantitative_aptitude": "math",
    "quant": "math",
    "quantitative": "math",
    "reasoning": "reasoning",
    "logical_reasoning": "reasoning",
    "verbal_reasoning": "reasoning",
    "critical_reasoning": "reasoning",
    "english": "english",
    "english_grammar": "english",
    "english_vocabulary": "english",
    "grammar": "english",
    "vocabulary": "english",
    "general": "general",
    "current_affairs": "current_affairs",
    "current affairs": "current_affairs",
    "practice": "practice",
}

_DIFFICULTY_ALIASES: dict[str, str] = {
    "basic": "basic",
    "easy": "basic",
    "beginner": "basic",
    "intermediate": "intermediate",
    "medium": "intermediate",
    "moderate": "intermediate",
    "advanced": "advanced",
    "hard": "advanced",
    "difficult": "advanced",
    "expert": "advanced",
    "default": "default",
}

# Fallback symbols that resolve to a route (vs. a direct model reference)
_ROUTE_FALLBACK_SYMBOLS: frozenset[str] = frozenset(
    {"basic", "intermediate", "advanced", "default", "general_default"}
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_subject(raw: str) -> str:
    """Normalize raw subject string to a known SubjectName or 'general'."""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _SUBJECT_ALIASES.get(cleaned, "general")


def normalize_difficulty(raw: str) -> str:
    """Normalize raw difficulty string to a known DifficultyLevel or 'default'."""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _DIFFICULTY_ALIASES.get(cleaned, "default")


# ---------------------------------------------------------------------------
# Fallback symbol resolution
# ---------------------------------------------------------------------------


def _resolve_fallback_symbol(
    symbol: str,
    subject: str,
    task_role: str,
) -> FallbackAttempt:
    """Translate a YAML fallback symbol into a typed FallbackAttempt (informational)."""
    if symbol == "safe_mock":
        return FallbackAttempt(
            kind="model",
            model="safe_mock",
            reason="safe_mock",
        )
    if symbol == "general_default":
        return FallbackAttempt(
            kind="route",
            subject="general",
            task_role=task_role,
            difficulty="default",
            reason="general_default",
        )
    # Difficulty-level symbol (basic / intermediate / advanced / default)
    return FallbackAttempt(
        kind="route",
        subject=subject,
        task_role=task_role,
        difficulty=symbol,
        reason=symbol,
    )


# ---------------------------------------------------------------------------
# Route resolver
# ---------------------------------------------------------------------------


def resolve_route(
    request: RouteRequest,
    registry: LlmConfigRegistry | None = None,
) -> RouteDecision:
    """Resolve a RouteDecision for the given RouteRequest.

    Uses the compiled registry (singleton by default) for all lookups.
    No I/O, no YAML parsing, no provider or LLM calls.

    Args:
        request:  The validated route request.
        registry: Optional registry override (for testing).

    Returns:
        A RouteDecision containing model alias, prompt path, and routing metadata.
        Contains no credentials or provider secret values.

    Raises:
        LlmRouteNotFoundError:   When no route can be found for the given
                                 task_role (even after subject/general fallback).
        LlmRouteResolutionError: For unexpected resolution errors.
    """
    reg = registry or get_registry()
    task_role: TaskRole = request.task_role

    subject = normalize_subject(request.subject)
    difficulty = normalize_difficulty(request.difficulty)

    logger.debug(
        "route_resolver.resolve  request_id=%s  subject_raw=%r  subject=%s  "
        "task_role=%s  difficulty_raw=%r  difficulty=%s",
        request.request_id,
        request.subject,
        subject,
        task_role,
        request.difficulty,
        difficulty,
    )

    # --- Step 1: exact match ---
    route = reg.get_route(subject, task_role, difficulty)
    if route is not None:
        return _build_decision(
            request=request,
            subject=subject,
            task_role=task_role,
            difficulty=difficulty,
            route=route,
            route_source="exact",
        )

    # --- Step 2: subject default ---
    if difficulty != "default":
        route = reg.get_route(subject, task_role, "default")
        if route is not None:
            logger.debug(
                "route_resolver  subject_default  subject=%s  task_role=%s",
                subject,
                task_role,
            )
            return _build_decision(
                request=request,
                subject=subject,
                task_role=task_role,
                difficulty="default",
                route=route,
                route_source="subject_default",
            )

    # --- Step 3: general default ---
    if subject != "general":
        route = reg.get_route("general", task_role, "default")
        if route is not None:
            logger.debug(
                "route_resolver  general_default  task_role=%s",
                task_role,
            )
            return _build_decision(
                request=request,
                subject="general",
                task_role=task_role,
                difficulty="default",
                route=route,
                route_source="general_default",
            )

    # --- No route found ---
    raise LlmRouteNotFoundError(
        f"No route found for task_role={task_role!r}, subject={subject!r}, "
        f"difficulty={difficulty!r}. "
        "The task_role may not have any configured routes. "
        "Use task_role='generator' or add routes to the LLM orchestration config."
    )


def _build_decision(
    *,
    request: RouteRequest,
    subject: str,
    task_role: TaskRole,
    difficulty: str,
    route: ResolvedRouteEntry,
    route_source: str,
) -> RouteDecision:
    """Build a RouteDecision from a resolved route entry."""
    assert isinstance(route, ResolvedRouteEntry)

    route_id = f"{subject}.{task_role}.{difficulty}"

    # Translate YAML fallback symbols to typed FallbackAttempt objects
    fallback_attempts = [
        _resolve_fallback_symbol(symbol, subject, task_role)
        for symbol in route.fallback
    ]

    decision = RouteDecision(
        route_id=route_id,
        subject=subject,
        task_role=task_role,
        difficulty=difficulty,
        intent=request.intent,
        exam=request.exam,
        exam_stage=request.exam_stage,
        model=route.model,
        prompt=route.prompt,
        overlays=list(route.overlays),
        intent_overlays=dict(route.intent_overlays),
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        provider_options=dict(route.provider_options),
        fallback_attempts=fallback_attempts,
        route_source=route_source,  # type: ignore[arg-type]
    )

    logger.info(
        "route_resolver.resolved  request_id=%s  route_id=%s  model=%s  "
        "route_source=%s  fallback_count=%d",
        request.request_id,
        route_id,
        route.model,
        route_source,
        len(fallback_attempts),
    )

    return decision
