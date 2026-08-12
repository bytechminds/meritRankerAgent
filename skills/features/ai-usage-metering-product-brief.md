# Product Brief: Centralized AI Usage Metering V1

## Goal

Measure provider-reported AI usage for one logical student operation, calculate its configured
shadow cost and MeritRanker credits, and emit one safe terminal summary. No student balance is
read or changed.

## Scope

- Capture provider-reported input and output tokens through the shared usage sink.
- Calculate LLM cost from configured USD-per-million rates, add one feature allowance, apply the
  commercial factor, and round credits up.
- Finalize Doubt Solver by request ID and Practice by its durable test ID.
- Keep missing usage or rates observable and non-blocking while shadow mode is enabled.

## Non-goals

- Wallet, ledger, reservation, Redis, DynamoDB billing persistence, payment, or deployment work.
- Prompt, model-routing, fallback, retry, token-budget, verification, PatternGraph, or Practice
  algorithm changes.

## Success Criteria

1. One terminal `AI_USAGE_SUMMARY` is emitted for each metered operation.
2. All central primary, fallback, capacity, retry, and provider-failure usage is included when
   provider counters are available.
3. No debit can occur in V1.
4. Existing student responses and Practice terminal rules are unchanged.
