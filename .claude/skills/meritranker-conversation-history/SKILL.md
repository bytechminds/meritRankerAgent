---
name: meritranker-conversation-history
description: Use for any change to conversation/session persistence, AgentCore Memory-first context selection, DynamoDB fallback, follow-up/reference resolution, or context-need gating feeding Doubt Solver. Covers app/services/conversation/.
---

# MeritRanker Conversation History

Completed-turn persistence, session metadata, and the read contract that
selects context for Doubt Solver.

## Current status

Dev infrastructure, persistence, Memory-first selection, exact-conversation
DynamoDB fallback, and independent-query isolation were previously live
verified. 2026-08-16 post-answer finalization reliability change registered
`conversation_persistence_started` as an observability event and added a narrow
post-answer failure diagnostic (no exception text/question/answer/JWT/secret
serialized). Full detail: `skills/features/conversation-history.md`.

## Owning code (verified against current tree)

`app/services/conversation/`: `persistence.py` (currently modified in the
working tree), `history_repository.py`, `session_repository.py`,
`context_need_gate.py`, `conversation_understanding.py`,
`follow_up_query_resolver.py`, `reference_resolution.py`,
`recent_context.py`/`recent_context_service.py`,
`selected_context_builder.py`, `candidate_builder.py`, `memory_hygiene.py`,
`short_term_memory.py`, `bootstrap.py`, `runtime_config.py`.

## Invariants

- AgentCore Memory and DynamoDB are context sources only —
  `ContextNeedGate` decides whether a read is skipped, required, or uncertain;
  it does not decide the final answer content.
- Required/uncertain requests load Memory first, then use the
  exact-conversation DynamoDB fallback on controlled empty/failure — never
  silently skip to a different retrieval order.
- Only one validated selected turn reaches generation.
- Root-lineage persistence, frontend reference IDs, summaries, and semantic
  recent-turn retrieval are explicitly out of scope — do not add them without
  an approved design change.
- Classifier projection is capped (3,200 formatted chars: 450/300 latest
  question/answer, 350/240 older) — do not silently raise this without
  checking downstream prompt budgets.
- Never persist the resolved-reference string — it stays in-memory generation
  context only.
- Post-answer failures must never replace an already-accepted/streamed answer
  with an error.

## Tests

`app/tests/test_conversation_history_memory.py`,
`test_conditional_conversation_context.py`,
`test_conversation_understanding.py`,
`test_contextual_reference_resolution.py`.

## Regression impact to check

Doubt Solver streaming finalize path (persistence runs after answer is
streamed), AI usage metering (operation lifecycle), language resolution for
persisted turns.
