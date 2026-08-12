# Security Review: Centralized AI Usage Metering V1

## Recommendation

**PASS WITH WARNINGS**

## Evidence Reviewed

| Artefact | Reviewed? | Notes |
|---|---|---|
| Python schemas/services | Yes | Strict Pydantic config and content-free summary. |
| Logging calls | Yes | Summary includes IDs, counts, costs, flags, and versions only. |
| YAML configuration | Yes | Contains no secrets or endpoints. |
| Dependency scan | [NOT VERIFIED] | No automated CVE scan run. |

## Findings

| # | Finding | File / location | Label | Severity |
|---|---|---|---|---|
| 1 | Operation IDs are server-created UUIDs or deterministic test IDs; prompts/answers are not emitted. | observability/billing paths | None identified | Low |
| 2 | Existing request-identity boundary remains outside this feature. | runtime/proxy | [AUTH TODO] | Medium |

## Blocking Findings

None identified for the shadow-only implementation.

## Production Auth Notes

| Auth concern | Status | Label |
|---|---|---|
| Inbound identity verification | Existing proxy boundary; not changed here | [NOT VERIFIED] |
| Client identity handling | Not added by this feature | N/A |
| Wallet authorization | No wallet path exists | N/A |

## Privacy Notes

| Data field | Sensitivity | Logging status | Persistence status |
|---|---|---|---|
| Prompt/answer/reasoning text | Sensitive | Excluded by event key sanitization and summary design | Not persisted by billing. |
| Request/test ID | Operational metadata | Safe summary correlation | In-memory lifecycle only. |

## Dependency / Security Scan Status

| Check | Status | Notes |
|---|---|---|
| No hardcoded secrets in added code/config | Pass by inspection | No secret values or endpoint fields added. |
| No eval/exec/subprocess with user input | Pass by inspection | None added. |
| CVE scan | [NOT VERIFIED] | Not run. |

## Final Decision Reason

> The implementation is shadow-only, adds no external financial call or secret, and deliberately
> excludes student content from billing events. Existing runtime auth and dependency scanning still
> require their normal release evidence.
