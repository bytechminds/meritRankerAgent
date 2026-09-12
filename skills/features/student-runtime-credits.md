# Feature: Student Runtime Credits

## Purpose

Charges a student's authoritative wallet for one accepted Doubt Solver answer or one delivered
Practice assessment, based on the actual provider cost of that operation. There is no per-feature
price: the only student pricing multiplier is the configured gross margin.

## Current Status

Implemented and reviewed. **Not enabled.** Real deductions are blocked until model pricing
coverage reaches 100% (see *Pricing coverage requirement*).

## Business formula

```
student_credits_to_debit =
    CEIL( total_llm_cost_usd * CREDITS_PER_USD / (1 - TARGET_GROSS_MARGIN) )
```

`total_llm_cost_usd` is the sum of the operation's priced provider calls, taken from the existing
`OperationBillingSummary`. It deliberately **excludes** the `infra.<feature>.fixed_cost` allowance,
so a zero-LLM operation can never produce a charge.

`TARGET_GROSS_MARGIN` is a margin, not a markup: 0.40 divides by 0.60. Ceiling rounding is applied
once, at the end, and always rounds in MeritRanker's favour.

At `CREDITS_PER_USD=50`, `TARGET_GROSS_MARGIN=0.40`: $1.00 of provider cost → 84 credits.

The legacy `calculated_credits` / `usd_per_credit` / `pricing_factor` values in
`app/config/llm/model_pricing.yaml` are **shadow telemetry only** and are never read by the wallet
path. Do not wire them into student debit.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STUDENT_CREDIT_ENFORCEMENT_ENABLED` | `false` | Master switch. False ⇒ no runtime is built, no wallet read, no settlement. |
| `STUDENT_CREDIT_DRY_RUN` | `true` | Observational. Calculates and logs; never writes; never refuses a student. |
| `CREDITS_PER_USD` | `50` | Decimal, must be > 0. |
| `TARGET_GROSS_MARGIN` | `0.40` | Decimal, `0 <= m < 1`. |
| `STUDENT_CREDIT_ROUNDING_MODE` | `CEIL` | Only `CEIL` is accepted; any other value fails at startup. |
| `DYNAMODB_USER_CREDITS_TABLE` | — | Required when enforcing. Injected by the CDK from SSM. |
| `DYNAMODB_CREDIT_LEDGER_TABLE` | — | Required when enforcing. Injected by the CDK from SSM. |

Policy is validated only when enforcement is enabled, so an unconfigured deployment starts
unchanged. Invalid values raise `ConfigurationError` at startup.

## Modes

**Enforcement off** — complete plug-out. The streaming call omits the `student_credits` keyword
entirely, so the call is identical to the pre-credit contract. A credit-store outage cannot affect
Doubt Solver.

**Dry run** (`ENABLED=true`, `DRY_RUN=true`) — reads the wallet, calculates, emits telemetry, and
mutates nothing. It never refuses a student, including when the credit store is unavailable; that
case is recorded as `student_credit_admission_status=unavailable` and generation continues.

**Enforcing** (`ENABLED=true`, `DRY_RUN=false`) — a missing wallet, a non-positive balance, or an
unreadable wallet blocks generation before any provider call. The unreadable-wallet case fails
closed by design: spend that settlement cannot recover must not start.

## Wallet and ledger ownership

`UserCredits` and `CreditLedger` are owned by `ai-tutor-backend` (Amplify). The runtime is a
consumer: it reads the balance and performs one conditional transaction. It **never** creates a
wallet row and never writes any other attribute. Purchase channels (Razorpay, Google Play, admin
grants) only ever CREDIT; the runtime only ever DEBITs.

Table identity reaches the runtime through the existing SSM contract under
`/meritranker/agent-runtime/v1/credits/…`, published by the backend and consumed by the AgentCore
CDK. IAM is `dynamodb:GetItem` + `dynamodb:TransactWriteItems` on those two table ARNs only.

## Idempotency key

```
student-credit:doubt:<user_id>:<turn_id>
student-credit:practice:<user_id>:<test_id>
```

`test_id` is derived from the Practice request idempotency key, so a resumed or replayed Practice
settles against the same reference.

`user_id` is the verified Cognito `sub`; `turn_id` is the ConversationHistory primary key and is
reused by the client on retry. The key is the `CreditLedger` partition key, written under
`attribute_not_exists(ledgerId)`. A repeated settlement returns `already_settled` with the recorded
amount rather than debiting again.

## Settlement transaction

One `TransactWriteItems`:

1. `Put` the ledger row, `ConditionExpression: attribute_not_exists(ledgerId)`.
2. `Update` the wallet, `ConditionExpression: attribute_exists(userId) AND creditsBalance >= :credits`,
   `UpdateExpression: ... ADD creditsBalance :negative, version :one`.

A conditional atomic `ADD` is used rather than a balance/version compare-and-set so that an
unrelated concurrent CREDIT (a purchase landing mid-generation) does not cancel a legitimate debit.
Because the post-state is not knowable at write time, the ledger's optional `balanceAfter`
attribute is deliberately **not** written — `amount` and `direction` are the audit values.

## Practice: the chargeable path

Practice charges only the calls that produced the delivered questions. Provider usage telemetry is
untouched — every call, including a failed one with unknown usage, stays in the operation's record
ledger and in `AI_USAGE_SUMMARY`. A second, narrower set is maintained alongside it on the same
`OperationUsageAccumulator` and priced by the same `calculate_operation_billing`, so student
charging can never drift from provider pricing.

Admitted to the chargeable set:

| Work | Chargeable |
|---|---|
| Classifier call of the launching request | Yes — the request hands its accumulator to the Practice operation, so it is already present. |
| Planner attempt that produced the accepted blueprint | Yes. A rejected attempt, and every attempt when the deterministic fallback produced the plan, are not. |
| Authoring call of a wave that yielded at least one accepted question | Yes, once per wave. |
| Authoring call of a wave whose questions were all rejected or that failed | No. |
| Verifier call that approved a question | Yes. |
| Verifier call that rejected a question, or that failed at the provider | No. |

Per-call attribution is exact and thread-safe: `capture_llm_usage()` binds a child accumulator for
the duration of one call, so concurrent slot groups and verifications never share billing state.
Every captured record is forwarded to the operation accumulator afterwards.

Settlement runs inside the existing background Practice worker, after final manifest validation and
before `READY` is published — the student's original request carries no extra billing latency. A
Practice that never reaches `READY` is never settled, whatever internal work it accumulated.

In enforcing mode, a settlement that cannot be confirmed (insufficient balance, or the credit store
being unreachable) fails the Practice with `PRACTICE_CREDIT_SETTLEMENT_FAILED` rather than
publishing a financially successful state on an unconfirmed debit. Dry run always calculates and
always continues to `READY`. An unpriced call on the accepted path blocks the debit and leaves the
questions delivered — never a guessed charge.

## Failure behaviour

| Outcome | Debit |
|---|---|
| Accepted fresh answer | Calculated credits |
| Auth failure, invalid request, generation failure, quality-gate failure, clarification | 0 |
| Practice launch turn (`practice_generation_started`) | 0 — the Practice operation settles separately, once, at `READY` |
| Replay / reconnect / zero LLM calls | 0 (guarded by `llm_call_count > 0`) |
| Zero provider cost | 0 |
| Unpriced model in the operation | 0, and `student_credit_pricing_incomplete` at ERROR — never guessed, never silently zero |
| Insufficient balance at settlement | 0, typed `INSUFFICIENT_CREDITS`, wallet untouched |
| Credit store unavailable at settlement | 0. The delivered answer is never retracted. |

## Pricing coverage requirement

`tests/test_student_credit_pricing_coverage.py` asserts that every billable model an active
production route can reach — after `model_registry_env` overlays, including provider-failure
fallback chains — resolves to a reviewed rate. **This test is currently red and must be green
before `STUDENT_CREDIT_DRY_RUN` is set to `false`.** An unpriced model must never become a
zero-cost charge.

Coverage is a property of a deployment, not of the YAML file: deployment names come from
environment variables, so the check must run against the target environment.

## Redis

Not used and not required. DynamoDB is the sole authority for both balance and settlement. A future
Redis layer may cache *displayed* balance only; it must never become authoritative for a debit.

## Known v1 limitations

- **Concurrency.** Two simultaneous operations on a thin balance may both generate. The first
  settles; the second is refused with `INSUFFICIENT_CREDITS` after its answer has already streamed.
  Wallet correctness is preserved; provider spend is wasted. No reservations by design.
- **Streamed content is not retracted.** Progressive chunks may already be visible when settlement
  fails. Only the terminal event reflects the failure.
- **Client disconnect before the terminal event.** Generation completes but settlement never runs,
  so the operation is not charged.
- **Persistence precedes settlement.** If settlement then refuses, the turn is durable and a retry
  replays it free. Under-charge only; never a double charge.
- **Legacy doubt path is not integrated.** Settlement is wired into the orchestrated paths only;
  `ENABLE_ORCHESTRATED_DOUBT_SOLVER=true` is therefore a financial precondition.
- **Unmetered spend.** Bedrock Titan embeddings and web search never reach `record_llm_call` and are
  not billed to the student.

## Tests

| File | Covers |
|---|---|
| `app/tests/test_student_credits.py` | Calculation, settlement, idempotency, concurrency, dry run, repository contract. |
| `app/tests/test_student_credit_review.py` | Adversarial: triple settle, concurrent CREDIT during DEBIT, cross-wallet attempts, dry-run outage, formula reconciliation against exact rational arithmetic. |
| `app/tests/test_student_credit_stream_integration.py` | Terminal ordering, plug-out, typed refusal, outage tolerance. |
| `app/tests/test_student_credit_config.py` | Startup validation and fail-fast. |
| `app/tests/test_student_credit_pricing_coverage.py` | The release gate. |
| `app/tests/test_practice_credit_charging.py` | Practice chargeable-path selection, settlement, replay, boundaries. |
