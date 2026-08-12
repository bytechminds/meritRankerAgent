# Implementation Plan: Pattern Intelligence Runtime

> Role: Solution Architect

## Goal

Deliver a small, feature-flagged canonical Pattern runtime that can safely guide practice deficits and doubt solving without changing public schemas or bypassing established verifier/reuse authority.

## Non-Goals

- New infrastructure, agentcore configuration, queues, caches, or model routes.
- Replacing the legacy flattened `StudentRetrievalService`.
- Implementing Pattern-driven reusable practice without an authoritative relation.

## Current Context

| Component | Current state |
|---|---|
| Legacy retrieval | `app/retrieval/retrieval_service.py` requires flattened fields absent from canonical Pattern |
| Practice | `app/features/practice_generation/orchestration.py` has post-blueprint schema-v2 slot seam |
| Doubt | `app/services/context_retrieval/context_retrieval_service.py` is the orchestrated integration seam |
| Feature context | `skills/features/pattern-intelligence-runtime.md` will be added |

## Architecture Decision

**Decision:**
> Add `app/retrieval/pattern_intelligence/` as a canonical adapter. It uses a deterministic plan, direct S3 Vector candidate discovery, DynamoDB Pattern batch hydration, `questionsByPattern` only for bounded doubt references, compact context builders, and request-local memoization. The legacy retrieval bundle remains untouched for backwards compatibility.

**Alternatives considered:**
> Adapting canonical Pattern into `PatternRuntimeBundle` would fabricate legacy fields; using PatternQuestion as practice reuse would cross an unsafe data boundary. Both are rejected.

## Files to Add / Change

| File | Action | Description |
|---|---|---|
| `app/retrieval/pattern_intelligence/*` | New | Models, stores, guard, runtime, context/budget helpers |
| `app/config.py` | Modify | Add disabled-by-default runtime and bounded access settings |
| `app/retrieval/s3_vectors/client.py` | Modify | Add candidate-only canonical Pattern query without stale filters |
| `app/services/dynamodb_service.py` | Modify | Opt-in bounded BatchGet unprocessed-key retries |
| `app/features/practice_generation/pattern_context.py` | Modify | Post-blueprint Pattern guidance provider |
| `app/features/practice_generation/orchestration.py` | Modify | Resolve/persist compact selection metadata and pass guidance to generator |
| `app/features/practice_generation/providers.py` | Modify | Prompt-safe guidance + measured input budget |
| `app/services/context_retrieval/context_retrieval_service.py` | Modify | Feature-flagged doubt runtime selector |
| `app/graphs/doubt_solver_graph.py` | Modify | Route legacy S3 path through the shared selector |
| `app/tests/test_pattern_intelligence_runtime.py` | New | Runtime/guard/store-independent tests |
| `app/tests/practice_generation/test_pattern_context.py` | New | Practice integration tests |
| Existing focused tests | Modify | Config/Dynamo/prompt/graph regression coverage |
| `skills/features/*` | Add / Update | Current behavior and role reports |

## Data Flow

```text
Practice request → existing planner → slot retrieval groups → Pattern plan
→ candidate discovery → authoritative Pattern batch hydration → guard
→ existing QuestionBank reuse / deficit generation with compact guidance → verifier → manifest

Doubt request → existing classifier → Pattern plan → candidate discovery
→ Pattern hydration → optional questionsByPattern query → bounded ColBERT references
→ compact context → existing solver/prompt boundary
```

## Service Boundaries

| Integration | Service file | Status |
|---|---|---|
| S3 Vector candidates | `retrieval/pattern_intelligence` via S3 client | Feature-flagged [NOT VERIFIED] live |
| Canonical Pattern / PatternQuestion | `retrieval/pattern_intelligence` stores | Feature-flagged [NOT VERIFIED] runtime config |
| Existing generation / solver | Existing provider and adapter services | Active |

## Schema / Contract Changes

| Schema | Field | Change type | Backward-compatible? |
|---|---|---|---|
| Public practice/doubt request/response | None | None | Yes |
| Internal Pattern runtime models | New | Add | Yes |
| Existing generator internals | Optional compact context argument | Add | Yes |

## Config / Env Changes

| Variable | Purpose | Default | Required for deploy? |
|---|---|---|---|
| `PATTERN_INTELLIGENCE_ENABLED` | Enables canonical runtime | `false` | Yes for activation |
| `DYNAMODB_PATTERN_QUESTION_TABLE` | Canonical PatternQuestion table | empty | Yes for references |
| `DYNAMODB_PATTERN_QUESTION_BY_PATTERN_INDEX` | Existing `questionsByPattern` GSI | empty | Yes for references |
| bounded Pattern candidate/reference/prompt settings | Runtime limits | conservative | No until activation |

## Testing Plan

| Scenario | File | Test name | Type |
|---|---|---|---|
| Plan/guard/no answer leakage | `app/tests/test_pattern_intelligence_runtime.py` | targeted unit tests | Unit |
| Batch retry | `app/tests/test_dynamodb_service.py` | unprocessed key retry | Unit |
| Practice guidance and fallback | `app/tests/practice_generation/test_pattern_context.py` | targeted tests | Unit |
| Context-service and legacy graph routing | existing doubt tests | feature-flag tests | Graph/service |

## Security Notes

| Concern | Assessment | Label |
|---|---|---|
| Public schemas | Unchanged; no client Pattern ID accepted | Pass |
| Retrieved records | Pydantic-adapted and compact-projected | [AI RISK] controlled |
| Answer leakage | Correct answer, option selection, raw solution excluded | Must verify in tests |
| Logging | IDs/counts/reason codes only | Pass |

## Performance / Cost Notes

| Concern | Assessment | Label |
|---|---|---|
| Pattern access | Bounded group query + one BatchGet | [NOT MEASURED] |
| Linked references | One GSI query per selected doubt Pattern, capped | [NOT MEASURED] |
| ColBERT | Linked set only; no whole-bank rerank | [COLD START RISK] |
| Prompt | Explicit measured budget, no full graph | [MEASURE] |

## Risks

| Risk | Likelihood | Impact | Label | Mitigation |
|---|---|---|---|---|
| Live S3 contract differs | High | High | [NOT VERIFIED] | Feature flag and fail-closed fallback |
| No practice mapping | Confirmed | High | [BLOCKER] | Keep Pattern `REUSE_SAFE` disabled |
| Canonical graph variation | Medium | Medium | [AI RISK] | Exact structural validation and compact projection |

## Rollback / Disable Plan

> Set `PATTERN_INTELLIGENCE_ENABLED=false`; existing legacy retrieval, practice reuse, generation, and doubt behavior remain active.

## Acceptance Criteria

- [x] BA functional requirements have focused local regression tests.
- [ ] `make check` and `agentcore validate` pass — [NOT VERIFIED] execution approval rejected.
- [x] Feature context docs are updated.
- [x] No public schema or deployment/infrastructure change was made.

## Handoff to Engineer

**Approved by:** Self-reviewed Solution Architect  
**Date:** 2026-08-10

**Start with:** typed canonical runtime models and pure guard tests.

**Known unknowns / [NOT VERIFIED] items the engineer should flag:**

- Live S3 Vector ingestion/metadata/IAM and Pattern table configuration.
- Exact Pattern-to-QuestionBank mapping; do not infer it.

**Out of bounds — do not implement:**

- Redis, queues, managed KB, Prompt Management, AppConfig, scans, new LLM calls, or production deployment.

## Implementation Handoff Status — 2026-08-10

Implemented the approved local scope: a separate canonical runtime, disabled-by-default concrete
adapters, bounded no-scan access, strict redacted projections, practice deficit guidance, typed
doubt guidance, prompt budgets, and fallback behavior. The implementation intentionally added no
new deployment resource, model route, public request/response field, cache, or reusable-question
assumption. Release activation remains blocked by the missing direct S3 Vector contract and the
missing Pattern-to-playable-QuestionBank mapping; current full validation is [NOT VERIFIED] because
the environment rejected `uv` execution after the final edits.
