# BA Requirements: Practice Generation Recovery

## Requirement Summary

Practice generation must preserve valid work, bound repair cost, validate the final manifest, and
recover only proven bookkeeping defects without weakening academic acceptance.

## Actors / Users

| Actor | Role | Interaction |
|---|---|---|
| Student | Primary | Requests a practice assessment |
| Practice coordinator | System | Plans, verifies, persists, recovers, and publishes |

## Functional Requirements

| ID | Requirement | Priority | Notes |
|---|---|---|---|
| FR-01 | The system MUST bind each slot to its exact compatibility bucket. | Must | Six-field signature |
| FR-02 | The system MUST preserve accepted sibling slots. | Must | Immutable accepted map |
| FR-03 | The system MUST allow only initial, one repair, and one replacement. | Must | No extra loop |
| FR-04 | The final validator MUST identify failed slots and recoverability. | Must | Typed result |
| FR-05 | The system MUST perform at most one metadata-only manifest recovery. | Must | Full revalidation follows |
| FR-06 | The system MUST fail closed on academic or authoritative corruption. | Must | No weak READY |

## Input / Output Specification

No public request or response field changed. Internal final validation adds failed slot IDs,
expected/actual counts, and recoverability.

## User Scenarios

**Happy path:** Three verified unique questions reach READY.

**Repair path:** One rejected candidate receives slot-scoped feedback; accepted siblings remain.

**Failure path:** A second manifest mismatch or conditional-write conflict becomes typed FAILED.

## Edge Cases

| Edge Case | Expected Behaviour |
|---|---|
| Duplicate ID/text | Terminal typed failure |
| Missing/unverified Question | Terminal typed failure |
| Wrong verified bucket metadata | One conditional repair, then revalidate |
| Provider failure | Existing provider fallback path; no content repair |

## Acceptance Criteria

| ID | Criterion | Linked FR |
|---|---|---|
| AC-01 | Two category signatures map to 1/2 bucket counts. | FR-01 |
| AC-02 | Repair/replacement touches only pending slots. | FR-02, FR-03 |
| AC-03 | Final diagnostics name slots and counts. | FR-04 |
| AC-04 | Wrong bucket metadata reaches READY after one recovery. | FR-05 |
| AC-05 | Invalid authoritative records remain non-playable. | FR-06 |

## Requirement-to-Test Mapping

| Requirement | Test file | Test name | Status |
|---|---|---|---|
| FR-01/04 | `app/tests/practice_generation/test_practice_core.py` | `test_slot_bucket_binding_uses_the_full_compatibility_signature` | Exists |
| FR-02/03 | `app/tests/practice_generation/test_planner_first_generation.py` | repair/regenerate wave tests | Exists |
| FR-05 | `app/tests/practice_generation/test_practice_orchestration.py` | `test_finalization_repairs_wrong_bucket_metadata_once_and_revalidates` | Exists |
| FR-06 | existing final-gate regressions | invalid/duplicate/ownership tests | Exists |

## Non-Goals

Routing, models, budgets, Pattern, billing, AppSync schema, and deployment.

## Data Sensitivity Notes

Question and answer content is sensitive educational content and is excluded from telemetry.

## Dependency Failure Expectations

Provider failures use existing bounded fallback; DynamoDB conditional conflicts fail explicitly;
telemetry remains fail-open.

## Open Questions

Literal “as below” context-gate behavior is separate and `[NOT VERIFIED]` for historical replay.

## Assumptions

None; requirements follow persisted evidence and current contracts.

## Blockers

Aggregate worktree isolation is a `[PROD BLOCKER]` for promotion.
