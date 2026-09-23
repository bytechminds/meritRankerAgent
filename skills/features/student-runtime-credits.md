# Feature: Student Runtime Credits

## Purpose

Student credits use a temporary, server-owned maximum-charge authorization before
chargeable Doubt or Practice work. The final debit remains the existing measured
provider-cost calculation:

```text
calculate_credits(actual measured provider cost)
```

Authorization is not fixed-price billing. Any unused authorization is returned;
if measured cost exceeds it, the student debit is capped at the authorization and
the platform records the overage.

## Configuration

`app/config/student_credit_policy.yaml` is the single versioned authority for
student business pricing and authorization. Its strict typed loader derives:

```text
credits_per_usd = usd_to_inr_business_rate / credit_value_inr
```

The current, locked `student-credit-v1` values are `1 credit = ₹1`, 100 credits
per USD, a 0.60 target gross margin, CEIL rounding, a 5-credit Doubt
authorization, and Practice authorization of `max(5, effectiveCount * 5)`
(5Q → 25, 10Q → 50, 20Q → 100, 50Q → 250). The hold is an authorization, not a
predicted bill: settlement charges measured accepted-path usage and refunds the rest. Missing or invalid
policy fails startup when enforcement is enabled; it is not read when
enforcement is disabled.

Only these operational controls remain environment-owned:

| Variable | Default | Meaning |
|---|---:|---|
| `STUDENT_CREDIT_ENFORCEMENT_ENABLED` | `false` | Enables the runtime boundary. |
| `STUDENT_CREDIT_DRY_RUN` | `true` | Logs would-authorize/would-settle; no mutations or refusals. |

The legacy `credits:` block in `app/config/llm/model_pricing.yaml` remains
shadow billing telemetry and is not read by the wallet path.

The AgentCore CDK entrypoint forwards the enforcement and dry-run flags to the
runtime; enabling enforcement also injects the published credit table names and
the corresponding least-privilege DynamoDB permission.

Practice authorization is:

```text
max(PRACTICE_MIN_AUTHORIZATION_CREDITS,
    effectiveCount * PRACTICE_AUTHORIZATION_CREDITS_PER_QUESTION)
```

`requestedCount` must never be used. The existing 50-question cap and all
Practice planning behavior remain outside this feature.

## Runtime lifecycle

```text
completed replay
  -> no credit work

new request
  -> preflight wallet read (below the configured minimum stops before
     image/classifier work; the minimum is the lowest configured authorization)
  -> existing classification / Practice request normalization
  -> atomic authorization
  -> existing execution
  -> measured calculate_credits()
  -> atomic settle (release unused amount), or atomic release on failure
```

Wallets at or above the admission minimum but below the operation-specific
authorization can incur existing classification/routing work, but cannot start
a Doubt generator/verifier or a Practice worker/planner/generator/verifier.

## DynamoDB authority

`UserCredits` and `CreditLedger` remain the only storage. No GraphQL schema
change is required: the ledger stores the private, schemaless
`authorizationState` attribute:

```text
AUTHORIZED -> SETTLED
AUTHORIZED -> RELEASED
```

The ledger's `amount` is the temporary authorized amount only while state is
`AUTHORIZED`. On `SETTLED`, it is overwritten with the actual captured debit.
On `RELEASED`, it becomes zero. Existing JSON `metadata` records authorization,
actual calculation, capture, and released amount.

### Atomic authorization

One `TransactWriteItems` transaction:

1. `Put CreditLedger`, `attribute_not_exists(ledgerId)`.
2. `Update UserCredits`, `attribute_exists(userId) AND creditsBalance >= :requiredCredits`,
   decrementing the authorization amount.

The deterministic IDs are unchanged:

```text
student-credit:doubt:<user_id>:<turn_id>
student-credit:practice:<user_id>:<test_id>
```

Duplicate authorization reads the ledger without a second wallet decrement,
then fails before paid work. Completed-turn replay remains owned by the existing
replay path.

### Atomic settlement and release

Both transitions condition the ledger on the same user, `AUTHORIZED` state, and
the original authorization amount. Settlement sets the durable debit to:

```text
captured = min(actualCalculatedCredits, authorizationCredits)
released = authorizationCredits - captured
```

It refunds `released` in the same transaction. When `released` is 0 the wallet
update must not declare the balance attribute at all: DynamoDB rejects any unused
expression name or value with `ValidationException`, which previously failed every
settlement whose actual cost reached its authorization. The test double enforces
the same rule. Failure/cancellation uses the
same conditional transaction to set `RELEASED` and refund the entire hold.
Terminal duplicate settle/release reads are idempotent and do not change the
wallet.

## Overage safety

When `actualCalculatedCredits > authorizationCredits`, the request succeeds and
the student is charged only `authorizationCredits`. The runtime emits the
high-severity, non-student-facing `STUDENT_CREDIT_AUTHORIZATION_EXCEEDED` event.
It does not retry generation, request more credits, or fail an answer/READY
assessment because of the overage.

## Path ownership

- `app/main.py` preserves replay-before-preflight and passes the one runtime to
  the existing Doubt graph and Practice launcher.
- Streaming Doubt authorizes after existing classification/Practice request
  resolution and before expensive execution. Its failure finalizer releases the
  Doubt hold.
- Non-stream Doubt uses the existing graph's generate and Practice-launch
  boundaries. It returns canonical `INSUFFICIENT_CREDITS` before paid execution.
- The existing Practice background launcher releases the deterministic Practice
  hold for cancellation, launch failure, and every non-READY terminal state. The
  owner is read from the assessment row's `userId`; the persisted
  `practiceRequest` does not carry it.
- The existing Practice orchestrator remains the sole caller that settles a
  READY assessment's measured charge.

The planner, generator, verifier, recovery policy, 50Q cap, model routes,
educator workflows, and public GraphQL schema are not modified.

## UX and telemetry

Both refusal gates (admission minimum and operation hold) return the same stable
fields, on the SSE `error` event metadata and on the non-stream response:

```text
code: INSUFFICIENT_CREDITS   retryable: false   action: ADD_CREDITS
requiredCredits: <int, when known>   availableCredits: <int, when known>
```

The streamed label is "Not enough credits to continue. Add credits and try
again." Store, IAM, validation, uncertain-transaction and pricing-incomplete
failures keep their own codes and never carry `action: ADD_CREDITS`.

The existing `useCreditErrorHandler` and `InsufficientCreditsModal` handle
canonical `INSUFFICIENT_CREDITS` in the Doubt/Practice request surface. No new
modal or purchase route exists. The hook labels detected Practice requests as
`AI_MOCK_TEST`; other Doubt requests use `AI_CHAT`.

Safe request-summary fields include preflight, authorization status and amount,
actual calculated credits, captured/released credits, settlement status,
overage flag, and idempotent replay flag. The only stored/logged user reference
is the existing non-reversible hash; logged `reference_id` values hash their
embedded user segment. An `unavailable` settlement also logs `exception_type`,
`aws_error_code`, `transaction_cancelled` and `cancellation_reason_codes`.

## Validation

`app/tests/test_student_credits.py` proves authorization atomicity,
same-reference execution exclusion, concurrency, effective-count policy,
normal/zero/overage settlement, exact-once release, and dry run. Adjacent Doubt
stream, Practice worker, configuration, telemetry, and adversarial review tests
validate their respective boundaries.

Dev-only live qualification remains required before setting
`STUDENT_CREDIT_DRY_RUN=false`. Production stays unchanged.
