# Implementation Report: Centralized AI Usage Metering V1

## Changed Files

| File | Action | Description |
|---|---|---|
| `app/schemas/billing.py` | New | Strict Decimal-safe billing contracts. |
| `app/services/llm/billing.py` | New | Cached configuration, locked operation aggregation, calculation, and safe emission. |
| `app/config/llm/model_pricing.yaml` | Modified | V1 flags, fixed costs, conversion settings, alias audit profiles; no guessed rate. |
| `app/observability/*` | Modified | Central record append, operation context, safe events, Doubt lifecycle finalization. |
| `app/main.py` | Modified | Startup configuration validation and streaming/replay transfer. |
| `app/features/practice_generation/*` | Modified | Durable handoff/finalization and copied context for parallel existing workers. |
| `app/tests/test_ai_usage_billing.py` | New | Focused shadow-mode behavior tests. |
| `skills/features/ai-usage-metering.md` | New | Feature context updated. |

## Behaviour Implemented

- [x] AC-01/06: `record_llm_call()` forwards existing provider-reported usage for primary,
  fallback, retry, failure, and capacity paths without deriving usage from limits.
- [x] AC-02/03: Decimal calculation handles cached input, configured fixed feature cost, factor,
  and ceiling conversion.
- [x] AC-04/05: V1 rejects debit enablement; unknown usage/rates make a fail-open incomplete
  summary and emit safe diagnostics.
- [x] AC-07/08: Existing Practice worker context is copied; one locked accumulator transfers to
  the durable test and finalizes once at its terminal lifecycle.
- [x] AC-09: YAML is cached, immutable, versioned, non-secret, and validated during startup.

## Impact Analysis

| Component | Impact |
|---|---|
| Public request/response schemas | No change. |
| LangGraph topology/state | No change. |
| Provider routing, prompts, retries | No behavior change; existing record sink observes results. |
| Practice task lifecycle | Preserved; only operation context/handoff is added. |

## Tests Added / Updated

| Test file | Test name | Covers |
|---|---|---|
| `app/tests/test_ai_usage_billing.py` | `test_billing_uses_decimal_rates_and_ceiling_credit_conversion` | Decimal formula and credit ceiling. |
| same | `test_unknown_provider_rate_is_incomplete_and_never_guessed` | Missing profile fail-open. |
| same | `test_central_llm_hook_emits_one_shadow_summary_without_debit` | Central hook, exactly once, no debit. |

## Commands Run

```text
python3 -m py_compile <changed Python files>
agentcore validate
git diff --check
PYTHONPATH=app app/.venv/bin/python -c '<billing Decimal/config smoke>'
PYTHONPATH=app app/.venv/bin/python -c '<central usage-hook smoke>'
cd app && uv run ruff check ...   # blocked by local dependency-cache sandbox
PYTHONPATH=app python3 -m pytest ...   # pytest not installed in system Python
```

## Test / Lint Result

| Check | Result |
|---|---|
| `py_compile` | PASS |
| `git diff --check` | PASS |
| Billing config/calculation smoke | PASS |
| Central usage-hook smoke | PASS |
| `ruff check` | [NOT VERIFIED] — `uv` cache access unavailable. |
| `pytest` | [NOT VERIFIED] — standalone Python lacks `pytest`; `uv` execution unavailable. |
| `agentcore validate` | PASS |

## Implementation Notes

The new code is bounded, request/task-local, and contains no external billing calls. Existing
provider adapters retain execution authority; only their pre-existing usage sink is observed.

## Known Limitations

- [PROD BLOCKER] Do not ship without CI/developer `make check` and a controlled real-provider
  reconciliation run.
- [DEFER] Credit debit, ledger, wallet, and reconciliation are intentionally absent.

## Documentation Status

| Doc file | Status |
|---|---|
| `skills/features/ai-usage-metering.md` | Updated. |
| `skills/core/architecture-principles.md` | Updated with the boundary decision. |

## Unverified Items

| Item | Label | Notes |
|---|---|---|
| Full test and lint run | [NOT VERIFIED] | Environment limitation recorded above. |
| Invoice reconciliation | [NOT VERIFIED] | Requires a controlled real-provider run. |

## Follow-up Needed

- [ ] Run CI `make check` and controlled dry run before release.
