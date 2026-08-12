# Feature: Centralized AI Usage Metering V1

## Purpose

Measures provider-reported LLM usage per logical operation, calculates a Decimal-safe shadow cost
and credit estimate, and emits safe terminal observability. It does not read, reserve, or debit a
student balance.

## Current Status

In Progress — implementation is present; full lint and pytest validation are [NOT VERIFIED].

## User Flow

1. A student starts a Doubt Solver request or Practice generation.
2. The existing provider execution boundary reports actual token counters to the central usage sink.
3. A terminal, content-free shadow summary is emitted; Practice uses its durable test ID.

## Entrypoints

| File | Function | Description |
|---|---|---|
| `app/observability/lifecycle.py` | `observe_invocation()` | Starts/finalizes a Doubt operation. |
| `app/services/doubt_solver/streaming_doubt_solver_service.py` | `stream_doubt_solver()` | Transfers/finalizes streaming operations. |
| `app/features/practice_generation/agentcore_async.py` | `AgentCorePracticeAsyncLauncher` | Hands an operation to the retained Practice task. |

## Graphs

No graph topology or graph state changes. Metering is outside LangGraph nodes.

## Schemas

| File | Model | Purpose |
|---|---|---|
| `app/schemas/billing.py` | `BillingConfig` | Immutable, startup-validated shadow configuration. |
| `app/schemas/billing.py` | `OperationBillingSummary` | Content-free terminal accounting result. |
| `app/schemas/llm_usage.py` | `LLMUsageRecord` | Adds resolved `model_alias` to an existing safe provider-attempt record. |

## Services

| File | Function | Description |
|---|---|---|
| `app/services/llm/billing.py` | `record_usage_for_current_operation()` | Appends safe central records to one locked operation accumulator. |
| `app/services/llm/billing.py` | `calculate_operation_billing()` | Uses Decimal rates, a fixed feature cost, factor, and ceiling conversion. |
| `app/services/llm/billing.py` | `emit_operation_billing_summary()` | Emits one fail-open terminal shadow summary and missing-profile diagnostic. |
| `app/observability/llm_usage.py` | `record_llm_call()` | Existing shared capture sink; forwards each safe record to billing. |

## Tools

None.

## Prompts

None.

## Tests

| File | Covers |
|---|---|
| `app/tests/test_ai_usage_billing.py` | Decimal input/output math, ceiling credits, unknown rates, missing usage, nested/thread context, no debit, one summary. |
| `app/tests/test_llm_usage_observability.py` | Existing provider usage extraction and exact reviewed pricing protections. |

## Config / Env

| Variable / File | Default | Description |
|---|---|---|
| `app/config/llm/model_pricing.yaml` | `metering_enabled: true` | Cached versioned rates, aliases, feature fixed cost, and credit conversion settings. |
| `credit_debit_enabled` | `false` | V1 hard guard; configuration validation rejects `true`. |

## Billing Formula and Configuration

`input_cost_per_million_tokens` and `output_cost_per_million_tokens` are USD per 1,000,000
provider-reported input and output tokens. For each call, V1 calculates
`(input / 1_000_000 × input rate) + (output / 1_000_000 × output rate)`; cached and reasoning
counts remain observability fields and are not separately charged. One configured `infra` fixed
cost is added per terminal feature operation. Credits are
`ceil((total_llm_cost + infra_cost) × pricing_factor / usd_per_credit)`.

To add or change a model price, first verify the exact resolved provider/deployment price, then
add one `models` entry and mark its matching alias profile `available`; unverified profiles must
remain `unknown`. Change the commercial multiplier under `credits.pricing_factor` and a feature
allowance under `infra.<feature>.fixed_cost`. `metering_enabled: false` is a no-op;
`credit_debit_enabled` must remain `false` in V1, so the calculated value is never debited.

## Known Limitations

- [NOT VERIFIED] Azure, DeepSeek, GPT-5, GPT-4o-mini, and Gemini 2.5 profiles have no
  project-reviewed provider rate. Their shadow totals remain incomplete rather than guessed.
- [DEFER] No wallet, ledger, persistence, retry queue, or reconciliation exists by design.
- [NOT VERIFIED] Full test suite and linter could not run because the local `uv` cache is outside
  the writable sandbox and the environment has no standalone `pytest` module.

## Latest Changes

- `2026-08-12` — Added central shadow metering, Decimal billing, safe operation summaries, and
  Practice context handoff without changing student response contracts.

## Next Steps

- [ ] [PROD BLOCKER] Run `make check` in a provisioned developer/CI environment.
- [ ] [PROD BLOCKER] Run a controlled real-provider dry run and reconcile safe summaries to an
  invoice/export before enabling any debit capability.
- [ ] [DEFER] Review each unpriced deployment before adding a rate.
