# BA Requirements: Centralized AI Usage Metering V1

## Acceptance Criteria

| ID | Requirement |
|---|---|
| AC-01 | Provider-reported input and output tokens are accounted separately; configured limits and reasoning telemetry are never substituted for actual output usage. |
| AC-02 | Cost uses Decimal arithmetic: `(input / 1M * input rate) + (output / 1M * output rate)`. |
| AC-03 | A feature fixed allowance is added exactly once, then the pricing factor and ceiling credit conversion are applied. |
| AC-04 | `metering_enabled=false` is a no-op. `credit_debit_enabled=false` produces no wallet or ledger action and reports `credits_debited=0`. |
| AC-05 | Missing usage or a missing rate emits one safe diagnostic and produces an incomplete, non-debit shadow summary without changing the student operation. |
| AC-06 | Primary, failed, fallback, and Practice capacity attempts with provider usage are all included. |
| AC-07 | Concurrent Practice groups share only their operation accumulator; separate requests/test IDs remain isolated. |
| AC-08 | Doubt finalizes at its existing terminal lifecycle; Practice finalizes exactly once at READY or FAILED using its durable test ID. |
| AC-09 | Billing configuration is startup-validated, cached, immutable, versioned, and contains no secrets. |

## Explicit Non-Requirements

No wallet debit, credit reservation, ledger, Redis, new table, per-call persistence, queue,
background billing worker, provider/routing change, prompt change, or production deployment.
