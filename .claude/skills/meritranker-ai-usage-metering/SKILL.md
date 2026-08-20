---
name: meritranker-ai-usage-metering
description: Use for any change touching LLM token/cost usage tracking, shadow billing, credit estimation, or usage observability summaries. Covers app/services/llm/billing.py, app/schemas/billing.py, app/observability/llm_usage.py — shared by both Doubt Solver and Practice generation.
---

# MeritRanker AI Usage Metering

Measures provider-reported LLM usage per logical operation, calculates a
Decimal-safe shadow cost/credit estimate, and emits safe terminal observability.
**It does not read, reserve, or debit a student balance** — shadow-only by design.

## Current status

In Progress — implementation present; full lint/pytest validation
`[NOT VERIFIED]`. Full detail: `skills/features/ai-usage-metering.md`.
Planning/review history: `ai-usage-metering-*.md` under `skills/features/`.

## Entrypoints (verified)

| File | Function |
|---|---|
| `app/observability/lifecycle.py` | `observe_invocation()` — starts/finalizes a Doubt operation |
| `app/services/doubt_solver/streaming_doubt_solver_service.py` | `stream_doubt_solver()` |
| `app/features/practice_generation/agentcore_async.py` | `AgentCorePracticeAsyncLauncher` |
| `app/services/llm/billing.py` | `record_usage_for_current_operation()`, `calculate_operation_billing()`, `emit_operation_billing_summary()` |
| `app/observability/llm_usage.py` | `record_llm_call()` — shared capture sink, forwards to billing |
| `app/schemas/billing.py` | `BillingConfig`, `OperationBillingSummary` |
| `app/schemas/llm_usage.py` | `LLMUsageRecord` |

No LangGraph node or graph-state changes — metering lives outside graph topology.

## Invariants

- Shadow-only: never reads/writes a student's real balance.
- `emit_operation_billing_summary()` is fail-open — a billing failure must never
  block or fail the underlying Doubt/Practice request.
- Decimal arithmetic only for cost — no float rounding for money.
- Content-free terminal summary — no prompts, questions, or answers in the
  billing log.
- Every real provider call must be counted once, not duplicated across
  fallback/retry.

## Tests

`app/tests/test_ai_usage_billing.py`, `test_llm_usage_observability.py`,
`test_observability.py`.

## Regression impact to check

Doubt Solver streaming finalize path, Practice generator/verifier/planner calls,
any new provider adapter (must normalize usage counters — see
`app/services/llm_providers/`).
