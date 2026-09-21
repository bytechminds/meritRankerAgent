"""Deterministic validation of the model's Practice request interpretation.

Native structured outputs guarantee the response *shape*; they guarantee nothing
about its truth. Every business rule below is therefore enforced here, after
deserialization, and never delegated to the model:

- the question count stays the deterministically resolved one;
- explicit per-level counts must add up to it;
- every reported topic must be grounded in the student's own words;
- the topic list stays within the supported Practice bound.

A violation is a bounded, single-shot failure: the interpretation is discarded and
Practice continues on its existing deterministic request resolution. The model is
never called again to obtain a preferred answer.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Sequence
from typing import NamedTuple, Protocol

from pydantic import ValidationError

from observability.events import log_event
from practice_limits import MAX_REQUESTED_PRACTICE_QUESTIONS
from schemas.practice_request_intelligence import (
    CustomDifficulty,
    PracticeRequestIntelligence,
)

logger = logging.getLogger(__name__)

# One slot carries one topic, so a topic list can never usefully exceed the
# Practice topic contract bound.
MAX_INTERPRETED_TOPICS = 12

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)

# Intra-word characters that must not end a token. Everything else that is neither
# alphanumeric nor a combining mark separates tokens.
_TOKEN_JOINERS = frozenset("&'’-_/")


class QueryToken(NamedTuple):
    """One deterministic token and its exact bounds in the original query."""

    index: int
    text: str
    start: int
    end: int


def _is_token_char(character: str) -> bool:
    """Whether a character continues a token.

    Combining marks are included so a Devanagari cluster stays one token: the matras
    and virama in "प्रतिशत" are category Mn/Mc and are not alphanumeric on their own.
    """
    return (
        character.isalnum()
        or unicodedata.category(character)[0] == "M"
        or character in _TOKEN_JOINERS
    )


def tokenize_query(query: str) -> tuple[QueryToken, ...]:
    """Split a query into deterministic, language-neutral, Unicode-safe tokens.

    The same input always yields the same sequence. There is no dictionary, no
    language branch and no semantic splitting — tokens are only immutable coordinates
    the model can point at. Character offsets are deliberately not the contract:
    a model cannot be asked to count code points across combining marks.
    """
    tokens: list[QueryToken] = []
    start: int | None = None
    for position, character in enumerate(query):
        if _is_token_char(character):
            if start is None:
                start = position
            continue
        if start is not None:
            tokens.append(QueryToken(len(tokens), query[start:position], start, position))
            start = None
    if start is not None:
        tokens.append(QueryToken(len(tokens), query[start:], start, len(query)))
    return tuple(tokens)


def token_id(index: int) -> str:
    """Return the opaque, stable label the model selects a token by."""
    return f"T{index}"


def token_index(value: str) -> int | None:
    """Return the token index a label refers to, or None when it is not a label."""
    if not isinstance(value, str) or not value.startswith("T"):
        return None
    digits = value[1:]
    return int(digits) if digits.isdigit() else None


def reconstruct_source_text(
    query: str,
    tokens: tuple[QueryToken, ...],
    indexes: Sequence[int],
) -> str:
    """Return the exact original text covered by a validated token selection.

    The slice runs from the first token's start to the last token's end, so whatever
    separated them — commas, ampersands — is preserved verbatim. "Time, Speed &
    Distance" survives as one span; the model never retypes a character.
    """
    return query[tokens[indexes[0]].start : tokens[indexes[-1]].end]


class RequestIntelligenceError(Exception):
    """A structurally valid interpretation that violates a business invariant."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class PracticeRequestInterpreter(Protocol):
    """Returns the raw schema-constrained interpretation for one Practice request."""

    def interpret(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        language: str,
        exam_id: str | None,
        exam_stage: str | None,
    ) -> str: ...


def _comparable(value: str) -> str:
    """Fold text to a comparable token stream for grounding and duplicate checks.

    Case, Unicode form and punctuation are normalised away so a student's
    "ratio proporton" still grounds against the query as written, in any script.
    "&" folds to "and" exactly as the planner's own topic canonicalization does, so
    "Profit & Loss" and "Profit and Loss" are one constraint here too.
    """
    folded = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    return " ".join(_NON_WORD.sub(" ", folded).split())


def _resolve_selection(
    topic,
    *,
    token_count: int,
) -> list[int]:
    """Return the token indexes a topic selects, or raise if the selection is invalid.

    A selection must name real tokens, name each at most once, run in the query's own
    order, and be contiguous. Nothing is repaired and nothing is guessed.
    """
    indexes: list[int] = []
    for label in topic.token_ids:
        index = token_index(label)
        if index is None or not 0 <= index < token_count:
            raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNKNOWN")
        indexes.append(index)
    if not indexes:
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_TOKEN_EMPTY")
    if len(set(indexes)) != len(indexes):
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_TOKEN_DUPLICATED")
    if indexes != sorted(indexes):
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNORDERED")
    if indexes[-1] - indexes[0] + 1 != len(indexes):
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_TOKEN_NONCONTIGUOUS")
    return indexes


def _validate_topics(
    intelligence: PracticeRequestIntelligence,
    *,
    query: str,
    tokens: tuple[QueryToken, ...],
) -> PracticeRequestIntelligence:
    """Validate every topic selection and attach the exact source text it covers.

    Grounding is structural here: a selection either names real tokens of the
    student's query or it is rejected. There is nothing for the model to misspell and
    no arithmetic for it to get wrong.
    """
    topics = intelligence.topics
    if intelligence.interpretation_status == "BROAD" and topics:
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_BROAD_TOPICS_PRESENT")
    if len(topics) > MAX_INTERPRETED_TOPICS:
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_LIMIT_EXCEEDED")

    resolved: list = []
    seen_names: set[str] = set()
    seen_selections: set[tuple[int, ...]] = set()
    for topic in topics:
        indexes = _resolve_selection(topic, token_count=len(tokens))
        if not _comparable(topic.normalized_name):
            raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_NAME_EMPTY")
        selection = tuple(indexes)
        if selection in seen_selections:
            continue
        # An exact repeat is one constraint stated twice; a partial overlap is two
        # readings of the same words, which is contradictory rather than redundant.
        if any(not set(selection).isdisjoint(other) for other in seen_selections):
            raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_TOPIC_SPAN_OVERLAP")
        key = _comparable(topic.normalized_name)
        if key in seen_names:
            continue
        seen_selections.add(selection)
        seen_names.add(key)
        resolved.append(
            topic.model_copy(
                update={"source_text": reconstruct_source_text(query, tokens, indexes)}
            )
        )
    return intelligence.model_copy(update={"topics": resolved})


def interpreted_total_count(
    intelligence: PracticeRequestIntelligence,
) -> int | None:
    """Return the total the interpretation implies, or None when it implies none.

    An explicit per-level breakdown is itself a statement of the total, so a CUSTOM
    interpretation carrying no requestedCount still yields one deterministically —
    summed here, never by the model.
    """
    if intelligence.requested_count is not None:
        return intelligence.requested_count
    difficulty = intelligence.difficulty
    if isinstance(difficulty, CustomDifficulty):
        return difficulty.distribution.total()
    return None


def _validate_difficulty(
    intelligence: PracticeRequestIntelligence,
    *,
    explicit_count: int | None,
) -> None:
    """Validate what the grammar cannot express.

    Mode/payload consistency is structural in the v2 contract — a level exists only
    under SINGLE, a distribution only under CUSTOM — so only arithmetic and business
    bounds are checked here.
    """
    difficulty = intelligence.difficulty
    if not isinstance(difficulty, CustomDifficulty):
        return
    distribution = difficulty.distribution
    if min(distribution.basic, distribution.intermediate, distribution.advanced) < 0:
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_DIFFICULTY_INVALID")
    total = distribution.total()
    # The breakdown must agree with whichever total is actually asserted; when
    # neither the student nor the interpretation states one, the breakdown is
    # the total and only has to be a supportable Practice size.
    target = (
        intelligence.requested_count
        if intelligence.requested_count is not None
        else explicit_count
    )
    if target is not None and total != target:
        raise RequestIntelligenceError(
            "PRACTICE_INTELLIGENCE_DIFFICULTY_ARITHMETIC_INVALID"
        )
    if target is None and not 1 <= total <= MAX_REQUESTED_PRACTICE_QUESTIONS:
        raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_COUNT_OUT_OF_BOUNDS")


def parse_request_intelligence(
    raw: str,
    *,
    query: str,
    explicit_count: int | None,
) -> PracticeRequestIntelligence:
    """Deserialize and semantically validate one interpretation.

    Args:
        raw: The provider's schema-constrained JSON response.
        query: The original student query. It is tokenized deterministically here,
            and every topic span is resolved against those tokens — the model never
            supplies source text.
        explicit_count: The count the student demonstrably wrote, or None when the
            deterministic parser found none. Only a real count can contradict the
            interpretation — a practice-type default asserts nothing about intent
            and must never be compared against it.

    Raises:
        RequestIntelligenceError: The interpretation violates a business invariant.
        ValueError: The payload is not valid JSON or does not match the contract.
    """
    tokens = tokenize_query(query)
    intelligence = PracticeRequestIntelligence.model_validate(json.loads(raw))
    if intelligence.requested_count is not None:
        if not 1 <= intelligence.requested_count <= MAX_REQUESTED_PRACTICE_QUESTIONS:
            raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_COUNT_OUT_OF_BOUNDS")
        if explicit_count is not None and intelligence.requested_count != explicit_count:
            raise RequestIntelligenceError("PRACTICE_INTELLIGENCE_COUNT_MISMATCH")
    _validate_difficulty(intelligence, explicit_count=explicit_count)
    return _validate_topics(intelligence, query=query, tokens=tokens)


def interpret_practice_request(
    interpreter: PracticeRequestInterpreter | None,
    *,
    request_id: str,
    query: str,
    subject: str,
    language: str,
    exam_id: str | None,
    exam_stage: str | None,
    explicit_count: int | None,
) -> PracticeRequestIntelligence | None:
    """Interpret one free-text Practice request, or return None to fall back.

    Every failure — no interpreter configured, provider unavailable, malformed
    payload, failed semantic validation, or an interpretation the model itself
    marked AMBIGUOUS — resolves to None, which leaves the existing deterministic
    request resolution in charge. There is no semantic retry.
    """
    if interpreter is None:
        return None
    try:
        raw = interpreter.interpret(
            request_id=request_id,
            query=query,
            subject=subject,
            language=language,
            exam_id=exam_id,
            exam_stage=exam_stage,
        )
        intelligence = parse_request_intelligence(
            raw,
            query=query,
            explicit_count=explicit_count,
        )
    except RequestIntelligenceError as exc:
        _log_unusable(status="invalid", error_code=exc.reason_code)
        return None
    except (ValueError, TypeError, ValidationError) as exc:
        _log_unusable(
            status="invalid",
            error_code="PRACTICE_INTELLIGENCE_MALFORMED",
            error_type=type(exc).__name__,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — provider boundary owns its own retries
        _log_unusable(
            status="unavailable",
            error_code="PRACTICE_INTELLIGENCE_UNAVAILABLE",
            error_type=type(exc).__name__,
        )
        return None
    if intelligence.interpretation_status == "AMBIGUOUS":
        _log_unusable(status="ambiguous", error_code="PRACTICE_INTELLIGENCE_AMBIGUOUS")
        return None
    log_event(
        "practice_request_intelligence_resolved",
        component="practice.request_intelligence",
        stage="interpret",
        status="resolved",
        details={
            "interpretationStatus": intelligence.interpretation_status,
            "topicCount": len(intelligence.topics),
            "difficultyMode": intelligence.difficulty.mode,
        },
    )
    return intelligence


def _log_unusable(
    *,
    status: str,
    error_code: str,
    error_type: str | None = None,
) -> None:
    log_event(
        "practice_request_intelligence_unusable",
        component="practice.request_intelligence",
        stage="interpret",
        status=status,
        error_code=error_code,
        details={"errorType": error_type} if error_type else None,
        level=logging.WARNING,
    )
