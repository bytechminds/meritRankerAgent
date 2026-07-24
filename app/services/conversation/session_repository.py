"""DynamoDB repository for lightweight conversation session metadata."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from schemas.conversation import CompletedConversationTurn

_serializer = TypeSerializer()
_deserializer = TypeDeserializer()
_TITLE_LIMIT = 60
_FALLBACK_TITLE = "New conversation"


class ConversationSessionError(RuntimeError):
    """Conversation session metadata operation failed."""


class ConversationSessionConflictError(ConversationSessionError):
    """A conversation ID is already owned by a different actor."""


def build_conversation_title(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query).strip()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("C")
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return _FALLBACK_TITLE
    if len(normalized) <= _TITLE_LIMIT:
        return normalized
    return f"{normalized[: _TITLE_LIMIT - 1].rstrip()}…"


class ConversationSessionRepository:
    def __init__(self, *, table_name: str, client: Any) -> None:
        self._table_name = table_name
        self._client = client

    def create_session_if_absent(self, turn: CompletedConversationTurn) -> bool:
        timestamp = turn.created_at.isoformat()
        item = {
            "id": turn.conversation_id,
            "userId": turn.actor_id,
            "title": build_conversation_title(turn.original_query),
            "lastActivityAt": timestamp,
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "__typename": "ConversationSession",
        }
        try:
            self._client.put_item(
                TableName=self._table_name,
                Item={key: _serializer.serialize(value) for key, value in item.items()},
                ConditionExpression="attribute_not_exists(#id)",
                ExpressionAttributeNames={"#id": "id"},
            )
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            if code == "ConditionalCheckFailedException":
                return False
            raise ConversationSessionError("Conversation session create failed.") from exc

    def update_last_activity(self, turn: CompletedConversationTurn) -> bool:
        timestamp = turn.created_at.isoformat()
        try:
            self._client.update_item(
                TableName=self._table_name,
                Key={"id": {"S": turn.conversation_id}},
                UpdateExpression="SET #last_activity = :last_activity",
                ConditionExpression=(
                    "#user_id = :user_id AND "
                    "(attribute_not_exists(#last_activity) OR #last_activity < :last_activity)"
                ),
                ExpressionAttributeNames={
                    "#user_id": "userId",
                    "#last_activity": "lastActivityAt",
                },
                ExpressionAttributeValues={
                    ":user_id": {"S": turn.actor_id},
                    ":last_activity": {"S": timestamp},
                },
            )
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            if code != "ConditionalCheckFailedException":
                raise ConversationSessionError("Conversation session update failed.") from exc
            owner = self._get_owner(turn.conversation_id)
            if owner is not None and owner != turn.actor_id:
                raise ConversationSessionConflictError(
                    "Conversation ID belongs to a different actor."
                ) from exc
            if owner is None:
                raise ConversationSessionError(
                    "Conversation session disappeared during update."
                ) from exc
            return False

    def upsert_from_completed_turn(self, turn: CompletedConversationTurn) -> bool:
        if self.create_session_if_absent(turn):
            return True
        return self.update_last_activity(turn)

    def _get_owner(self, conversation_id: str) -> str | None:
        try:
            response = self._client.get_item(
                TableName=self._table_name,
                Key={"id": {"S": conversation_id}},
                ProjectionExpression="#user_id",
                ExpressionAttributeNames={"#user_id": "userId"},
                ConsistentRead=True,
            )
        except ClientError as exc:
            raise ConversationSessionError("Conversation session read failed.") from exc
        raw_owner = response.get("Item", {}).get("userId")
        if not raw_owner:
            return None
        return str(_deserializer.deserialize(raw_owner))
