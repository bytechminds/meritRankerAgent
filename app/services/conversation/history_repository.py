"""DynamoDB repository for authoritative completed conversation turns."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import BotoCoreError, ClientError

from schemas.conversation import CompletedConversationTurn, RecentConversationTurn

_serializer = TypeSerializer()
_deserializer = TypeDeserializer()


class ConversationHistoryError(RuntimeError):
    """Conversation history operation failed."""

    def __init__(self, message: str, *, reason: str = "history_query_failed") -> None:
        super().__init__(message)
        self.reason = reason


class ConversationTurnConflictError(ConversationHistoryError):
    """A turn ID was reused for different immutable request identity."""


class ConversationHistoryIsolationError(ConversationHistoryError):
    """A history query returned a record outside the requested isolation scope."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class ConversationHistoryRepository:
    def __init__(self, *, table_name: str, client: Any) -> None:
        self._table_name = table_name
        self._client = client

    def get_completed_turn(
        self, actor_id: str, conversation_id: str, turn_id: str
    ) -> CompletedConversationTurn | None:
        try:
            response = self._client.get_item(
                TableName=self._table_name,
                Key={"id": {"S": turn_id}},
                ConsistentRead=True,
            )
        except ClientError as exc:
            raise ConversationHistoryError("Conversation history read failed.") from exc
        except BotoCoreError as exc:
            raise ConversationHistoryError(
                "Conversation history read failed.",
                reason="history_query_failed",
            ) from exc
        item = response.get("Item")
        if not item:
            return None
        turn = self._parse_completed_turn(item)
        if turn.actor_id != actor_id or turn.conversation_id != conversation_id:
            raise ConversationTurnConflictError("Turn identity does not match the stored turn.")
        return turn

    def save_completed_turn(self, turn: CompletedConversationTurn) -> bool:
        item = {
            "id": turn.turn_id,
            "userId": turn.actor_id,
            "conversationId": turn.conversation_id,
            "turnId": turn.turn_id,
            "originalQuery": turn.original_query,
            "finalAnswer": turn.final_answer,
            "examId": turn.exam_id,
            "examStage": turn.exam_stage,
            "language": turn.language,
            "subject": turn.subject,
            "topic": turn.topic,
            "qualityStatus": turn.quality_status,
            "wasRegenerated": turn.was_regenerated,
            "createdAt": turn.created_at.isoformat(),
            "__typename": "ConversationHistory",
        }
        encoded = {
            key: _serializer.serialize(value)
            for key, value in item.items()
            if value is not None
        }
        try:
            self._client.put_item(
                TableName=self._table_name,
                Item=encoded,
                ConditionExpression="attribute_not_exists(#id)",
                ExpressionAttributeNames={"#id": "id"},
            )
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                existing = self.get_completed_turn(
                    turn.actor_id, turn.conversation_id, turn.turn_id
                )
                if existing and existing.original_query == turn.original_query:
                    return False
                raise ConversationTurnConflictError(
                    "Turn ID already exists with different content."
                ) from exc
            raise ConversationHistoryError("Conversation history write failed.") from exc
        except BotoCoreError as exc:
            raise ConversationHistoryError(
                "Conversation history write failed.",
                reason="history_query_failed",
            ) from exc

    def list_recent_completed_turns(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> list[RecentConversationTurn]:
        target = min(max(limit, 1), 3)
        items: list[RecentConversationTurn] = []
        saw_incomplete_turn = False
        start_key: dict[str, Any] | None = None
        pages = 0
        while len(items) < target and pages < 5:
            kwargs: dict[str, Any] = {
                "TableName": self._table_name,
                "IndexName": "ConversationHistoryByConversation",
                "KeyConditionExpression": "#conversation = :conversation",
                "FilterExpression": "#user = :actor",
                "ExpressionAttributeNames": {
                    "#conversation": "conversationId",
                    "#user": "userId",
                },
                "ExpressionAttributeValues": {
                    ":conversation": {"S": conversation_id},
                    ":actor": {"S": actor_id},
                },
                "ScanIndexForward": False,
                "Limit": 20,
            }
            if start_key:
                kwargs["ExclusiveStartKey"] = start_key
            try:
                response = self._client.query(**kwargs)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code") or "")
                reason = (
                    "history_permission_denied"
                    if code
                    in {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}
                    else "history_query_failed"
                )
                raise ConversationHistoryError(
                    "Recent conversation query failed.", reason=reason
                ) from exc
            except BotoCoreError as exc:
                raise ConversationHistoryError(
                    "Recent conversation query failed.",
                    reason="history_query_failed",
                ) from exc
            for raw in response.get("Items", []):
                data = {key: _deserializer.deserialize(value) for key, value in raw.items()}
                if data.get("userId") != actor_id:
                    raise ConversationHistoryIsolationError(
                        "Recent conversation actor does not match.",
                        reason="history_actor_mismatch",
                    )
                if data.get("conversationId") != conversation_id:
                    raise ConversationHistoryIsolationError(
                        "Recent conversation ID does not match.",
                        reason="history_conversation_mismatch",
                    )
                if not data.get("originalQuery") or not data.get("finalAnswer"):
                    saw_incomplete_turn = True
                    continue
                items.append(
                    RecentConversationTurn(
                        turn_id=str(data.get("turnId") or data.get("id")),
                        original_query=str(data["originalQuery"]),
                        final_answer=str(data["finalAnswer"]),
                        created_at=datetime.fromisoformat(str(data["createdAt"])),
                    )
                )
                if len(items) >= target:
                    break
            start_key = response.get("LastEvaluatedKey")
            pages += 1
            if not start_key:
                break
        if not items and saw_incomplete_turn:
            raise ConversationHistoryError(
                "Recent conversation has no completed turns.",
                reason="history_no_completed_turns",
            )
        return sorted(items, key=lambda turn: turn.created_at)[-target:]

    @staticmethod
    def _parse_completed_turn(item: dict[str, Any]) -> CompletedConversationTurn:
        data = {key: _deserializer.deserialize(value) for key, value in item.items()}
        try:
            return CompletedConversationTurn(
                actor_id=data["userId"],
                conversation_id=data["conversationId"],
                turn_id=data.get("turnId") or data["id"],
                original_query=data["originalQuery"],
                final_answer=data["finalAnswer"],
                exam_id=data.get("examId"),
                exam_stage=data.get("examStage"),
                language=data["language"],
                subject=data.get("subject") or "unknown",
                topic=data.get("topic"),
                quality_status=data["qualityStatus"],
                was_regenerated=bool(data.get("wasRegenerated", False)),
                created_at=datetime.fromisoformat(data["createdAt"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationHistoryError("Stored conversation turn is invalid.") from exc
