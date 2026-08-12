# Architecture Review: Centralized AI Usage Metering V1

## Recommendation

**PASS WITH NOTES**

## Evidence Reviewed

| Area | Result |
|---|---|
| Service boundary | `record_llm_call()` remains the only capture join point; graph nodes remain untouched. |
| State ownership | Locked accumulator is request/task-local through `ExecutionContext`; no database, cache, or global user state. |
| Practice durability | Handoff key is the existing deterministic test ID; terminal task lifecycle finalizes it. |
| AI workflow | Provider results and routing decisions are observed only; prompts, model selection, retries, and validations are unchanged. |

## Findings

- [NOT VERIFIED] In-memory task-local aggregation cannot survive process termination before the
  existing Practice task finishes. This is acceptable for V1 shadow observability only; it cannot
  support financial debit or invoice-grade reconciliation.
- [PROD BLOCKER] Full automated test/lint evidence is still required before release.

## Final Decision Reason

The design follows the established service and observability boundaries, avoids a second provider
path, and keeps academic graph contracts unchanged. Production approval remains contingent on the
recorded validation and controlled runtime evidence.
