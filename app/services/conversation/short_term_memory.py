"""AgentCore short-term-memory adapter for completed turn pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError
from pydantic import ValidationError

from schemas.conversation import CompletedConversationTurn, RecentConversationTurn


class ShortTermMemoryError(RuntimeError):
    """AgentCore Memory operation failed."""

    def __init__(self, message: str, *, reason: str = "memory_timeout") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ShortTermMemoryReadResult:
    turns: tuple[RecentConversationTurn, ...]
    failure_reason: str | None = None
    attempted: bool = True
    event_count: int = 0


def _memory_failure_reason(error: ClientError) -> str:
    code = str(error.response.get("Error", {}).get("Code") or "")
    if code in {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}:
        return "memory_permission_denied"
    if code in {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}:
        return "memory_resource_missing"
    return "memory_timeout"


class AgentCoreShortTermMemory:
    def __init__(self, *, memory_id: str, client: Any) -> None:
        self._memory_id = memory_id
        self._client = client

    def save_completed_turn(self, turn: CompletedConversationTurn) -> None:
        try:
            self._client.create_event(
                memoryId=self._memory_id,
                actorId=turn.actor_id,
                sessionId=turn.conversation_id,
                eventTimestamp=turn.created_at,
                clientToken=turn.turn_id,
                payload=[
                    {"conversational": {"role": "USER", "content": {"text": turn.original_query}}},
                    {
                        "conversational": {
                            "role": "ASSISTANT",
                            "content": {"text": turn.final_answer},
                        }
                    },
                ],
                metadata={
                    "turn_id": {"stringValue": turn.turn_id},
                    "language": {"stringValue": turn.language},
                    **(
                        {"exam_id": {"stringValue": turn.exam_id}}
                        if turn.exam_id
                        else {}
                    ),
                },
            )
        except ClientError as exc:
            raise ShortTermMemoryError(
                "AgentCore Memory write failed.", reason=_memory_failure_reason(exc)
            ) from exc
        except (ParamValidationError, ValueError) as exc:
            raise ShortTermMemoryError(
                "AgentCore Memory write failed.", reason="memory_resource_missing"
            ) from exc
        except BotoCoreError as exc:
            raise ShortTermMemoryError(
                "AgentCore Memory write failed.", reason="memory_timeout"
            ) from exc

    def list_recent_completed_turns(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> list[RecentConversationTurn]:
        result = self.read_recent_completed_turns(actor_id, conversation_id, limit)
        if result.failure_reason in {
            "memory_timeout",
            "memory_permission_denied",
            "memory_resource_missing",
        }:
            raise ShortTermMemoryError(
                "AgentCore Memory read failed.", reason=result.failure_reason
            )
        return list(result.turns)

    def read_recent_completed_turns(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> ShortTermMemoryReadResult:
        target = min(max(limit, 1), 5)
        events: list[dict[str, Any]] = []
        next_token: str | None = None
        pages = 0
        try:
            while pages < 5:
                kwargs: dict[str, Any] = {
                    "memoryId": self._memory_id,
                    "actorId": actor_id,
                    "sessionId": conversation_id,
                    "includePayloads": True,
                    "maxResults": 50,
                }
                if next_token:
                    kwargs["nextToken"] = next_token
                response = self._client.list_events(**kwargs)
                events.extend(response.get("events", []))
                next_token = response.get("nextToken")
                pages += 1
                if not next_token:
                    break
        except ClientError as exc:
            return ShortTermMemoryReadResult(
                turns=(), failure_reason=_memory_failure_reason(exc)
            )
        except (ParamValidationError, ValueError):
            return ShortTermMemoryReadResult(
                turns=(), failure_reason="memory_resource_missing"
            )
        except BotoCoreError:
            return ShortTermMemoryReadResult(
                turns=(), failure_reason="memory_timeout"
            )

        if not events:
            return ShortTermMemoryReadResult(turns=(), failure_reason="memory_no_events")

        normalized: dict[str, RecentConversationTurn] = {}
        saw_actor_mismatch = False
        saw_conversation_mismatch = False
        saw_malformed = False
        saw_incomplete_pair = False
        for event in events:
            if not isinstance(event, dict):
                saw_malformed = True
                continue
            if event.get("actorId") != actor_id:
                saw_actor_mismatch = True
                continue
            if event.get("sessionId") != conversation_id:
                saw_conversation_mismatch = True
                continue
            payload = event.get("payload") or []
            if not isinstance(payload, list):
                saw_malformed = True
                continue
            messages = [
                entry.get("conversational")
                for entry in payload
                if isinstance(entry, dict)
                and isinstance(entry.get("conversational"), dict)
            ]
            roles = [item.get("role") for item in messages]
            if len(messages) != 2 or roles != ["USER", "ASSISTANT"]:
                saw_incomplete_pair = True
                continue
            user_content = messages[0].get("content") or {}
            answer_content = messages[1].get("content") or {}
            if not isinstance(user_content, dict) or not isinstance(answer_content, dict):
                saw_malformed = True
                continue
            user_text = str(user_content.get("text") or "").strip()
            answer_text = str(answer_content.get("text") or "").strip()
            if not user_text or not answer_text:
                saw_incomplete_pair = True
                continue
            metadata = event.get("metadata") or {}
            if not isinstance(metadata, dict):
                saw_malformed = True
                continue
            turn_meta = metadata.get("turn_id") or {}
            if not isinstance(turn_meta, dict):
                saw_malformed = True
                continue
            turn_id = str(turn_meta.get("stringValue") or event.get("eventId") or "")
            if not turn_id:
                saw_malformed = True
                continue
            timestamp = event.get("eventTimestamp") or datetime.now(UTC)
            try:
                candidate = RecentConversationTurn(
                    turn_id=turn_id,
                    original_query=user_text,
                    final_answer=answer_text,
                    created_at=timestamp,
                )
            except ValidationError:
                saw_malformed = True
                continue
            existing = normalized.get(turn_id)
            if existing is None or candidate.created_at > existing.created_at:
                normalized[turn_id] = candidate
        turns = tuple(sorted(normalized.values(), key=lambda turn: turn.created_at)[-target:])
        if turns:
            return ShortTermMemoryReadResult(turns=turns, event_count=len(events))
        if saw_actor_mismatch:
            reason = "memory_actor_mismatch"
        elif saw_conversation_mismatch:
            reason = "memory_conversation_mismatch"
        elif saw_malformed:
            reason = "memory_malformed_events"
        elif saw_incomplete_pair:
            reason = "memory_no_completed_turns"
        else:
            reason = "memory_malformed_events"
        return ShortTermMemoryReadResult(
            turns=(), failure_reason=reason, event_count=len(events)
        )


class UnavailableShortTermMemory:
    """No-network adapter used when AgentCore Memory is not configured."""

    def save_completed_turn(self, turn: CompletedConversationTurn) -> None:
        del turn
        raise ShortTermMemoryError(
            "AgentCore Memory is not configured.", reason="memory_not_configured"
        )

    def list_recent_completed_turns(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> list[RecentConversationTurn]:
        del actor_id, conversation_id, limit
        return []

    def read_recent_completed_turns(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> ShortTermMemoryReadResult:
        del actor_id, conversation_id, limit
        return ShortTermMemoryReadResult(
            turns=(),
            failure_reason="memory_not_configured",
            attempted=False,
        )
