"""Process-scoped construction for conversation persistence integrations."""

from __future__ import annotations

from services.aws_client_factory import (
    get_bedrock_agentcore_client,
    get_conversation_dynamodb_client,
)
from services.conversation.history_repository import ConversationHistoryRepository
from services.conversation.persistence import ConversationPersistenceService
from services.conversation.runtime_config import load_conversation_runtime_config
from services.conversation.session_repository import ConversationSessionRepository
from services.conversation.short_term_memory import (
    AgentCoreShortTermMemory,
    UnavailableShortTermMemory,
)


def build_conversation_persistence_service() -> ConversationPersistenceService:
    config = load_conversation_runtime_config()
    dynamodb_client = get_conversation_dynamodb_client(config.region_name)
    short_term_memory = (
        AgentCoreShortTermMemory(
            memory_id=config.memory_id,
            client=get_bedrock_agentcore_client(config.region_name),
        )
        if config.memory_id
        else UnavailableShortTermMemory()
    )
    return ConversationPersistenceService(
        history_repository=ConversationHistoryRepository(
            table_name=config.table_name,
            client=dynamodb_client,
        ),
        session_repository=(
            ConversationSessionRepository(
                table_name=config.conversation_session_table_name,
                client=dynamodb_client,
            )
            if config.conversation_session_table_name
            else None
        ),
        short_term_memory=short_term_memory,
    )
