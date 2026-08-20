---
name: meritranker-doubt-solver
description: Use for any change to the Doubt Solver feature — student question understanding, classification-to-generation flow, streaming answers, Pattern-aware guidance in doubt answers, or conversation-context selection feeding the doubt graph. Covers app/graphs/doubt_solver_graph.py, app/services/doubt_solver/, app/schemas/doubt_solver.py.
---

# MeritRanker Doubt Solver

Student-facing tutoring flow: understand the doubt, classify it, gather context,
generate a streamed explanation.

## Current status (see file for full detail, verify before relying on it)

Dev deployed — bounded AgentCore Memory-first follow-up understanding and DynamoDB
fallback are active; the deployed generator remains mock; Sandbox role attachment
and multi-account deployment are `[NOT VERIFIED]`. Full detail:
`skills/features/doubt-solver.md` (last updated 2026-08-10) plus the V1 planning
set under `skills/features/`: `doubt-solver-v1-product-brief.md`,
`doubt-solver-v1-ba-requirements.md`, `doubt-solver-v1-implementation-plan.md`,
`doubt-solver-v1-ai-architecture-plan.md`, `doubt-solver-v1-part10-release-gate.md`,
`doubt-solver-v1-env-audit-release-gate.md`.

## Owning code (verified against current tree)

- `app/graphs/doubt_solver_graph.py` — LangGraph topology (currently modified in
  the working tree — check `git diff` before assuming this doc is current).
- `app/services/doubt_solver/streaming_doubt_solver_service.py` — streaming
  entrypoint; also the AI-usage-metering finalize point (see
  `meritranker-ai-usage-metering`).
- `app/schemas/doubt_solver.py` — request/response/state schemas (also modified
  in the working tree).
- `app/services/conversation/` — history/session/context selection feeding the
  graph; see `meritranker-conversation-history` before touching context selection.
- `app/services/context_retrieval/` — Bedrock KB retrieval, web-grounding
  decision (`web_search_decision.py`, `web_grounding.py`).
- `app/retrieval/pattern_intelligence/` — canonical Pattern guidance path (see
  `meritranker-pattern-intelligence`); disabled by default in production.

## Invariants

- Public request/response schemas in `app/schemas/doubt_solver.py` are a contract
  — do not silently change shape (`CLAUDE.md` regression rule).
- Pattern-aware context (`DoubtPatternContext`) stays internal until the prompt
  boundary and never exposes raw answers beyond the bounded, answer-redacted
  linked-reference limit.
- Answer verification ownership is unchanged by Pattern-context changes — do not
  let generation logic bypass the existing verifier.
- No secrets, JWTs, or full prompts in logs — route through
  `app/observability/`.

## Tests

`app/tests/test_doubt_solver_graph.py`, `test_doubt_solver_schemas.py`,
`test_orchestrated_doubt_solver_graph_flow.py`,
`test_orchestrated_doubt_solver_graph_state.py`,
`test_integration_doubt_solver.py`, `test_stream_lifecycle.py`,
`test_pattern_intelligence_doubt_integration.py`,
`test_conditional_conversation_context.py`.

## Regression impact to check

Conversation history/context selection, AI usage metering (every doubt call is
metered), classification pipeline (feeds the graph's entry), web search source
policy, language resolution.
