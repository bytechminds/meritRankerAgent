# BA Requirements: Pattern Intelligence Runtime

> Role: Business Analyst

## Requirement Summary

The system shall use PatternGraph only as a bounded, authoritative method hint for existing practice and doubt flows. It must not treat a vector hit or raw PatternQuestion as student-safe answer or reuse authority.

## Actors / Users

| Actor | Role | Interaction |
|---|---|---|
| Student | Primary user | Requests practice or asks a doubt through existing contracts |
| Practice runtime | Automated | Plans slots, reuses verified QuestionBank items, generates deficits |
| Doubt runtime | Automated | Classifies, retrieves optional pattern context, solves current question |

## Functional Requirements

| ID | Requirement | Priority | Notes |
|---|---|---|---|
| FR-01 | The system SHALL create a deterministic retrieval plan before external Pattern access. | Must | No model routing. |
| FR-02 | The system SHALL treat S3 Vector results as candidate IDs only and hydrate authoritative Pattern records in bounded batches. | Must | No scan or per-candidate GetItem. |
| FR-03 | The system SHALL classify every candidate as `REUSE_SAFE`, `GUIDANCE_SAFE`, or `IGNORE` using authoritative compatibility checks. | Must | Vector score is not authority. |
| FR-04 | The system SHALL pass only compact, answer-redacted PatternGraph guidance to new practice generation. | Must | Existing verifier remains final authority. |
| FR-05 | The system SHALL not reuse PatternQuestion as a playable practice question. | Must | No verified playable mapping exists. |
| FR-06 | The system SHALL use bounded linked PatternQuestion references for doubt only, with ColBERT limited to that set. | Must | No whole-bank rerank. |
| FR-07 | The system SHALL fall back to the existing practice/doubt path on retrieval, hydration, linked-reference, or reranker failure. | Must | No new single point of failure. |
| FR-08 | The system SHALL measure and bound Pattern-aware prompt context before a generation call. | Must | Keep static and dynamic budget components separate. |
| FR-09 | The system SHALL not expose answers, option selections, raw solutions, or private graph data through client-facing state. | Must | Server-side only. |

## Input / Output Specification

### Input

| Field | Type | Required | Constraints | Notes |
|---|---|---|---|---|
| Existing practice request | Existing Pydantic model | Yes | Unchanged | No client Pattern ID is accepted |
| Existing doubt request/classification | Existing Pydantic model | Yes | Unchanged | Subject/topic hints are candidate signals only |
| Authoritative Pattern / PatternQuestion | DynamoDB record | Conditional | Validated adapter | Never trusted as raw prompt payload |

### Output

| Field | Type | Always present | Notes |
|---|---|---|---|
| Existing practice/doubt response | Existing model | Yes | Public contracts unchanged |
| Internal Pattern selection | Typed internal model | No | Excluded from student response |
| Compact prompt context | Typed internal model/rendering | No | Answer-redacted and bounded |

## User Scenarios

**Scenario 1: Compatible practice PatternGraph**
> The planner creates slots; a compatible active PatternGraph adds method constraints to deficit generation; the existing verifier accepts or rejects the generated question.

**Scenario 2: Pattern data unavailable**
> The runtime records a safe fallback and the existing reuse/generation or solver behavior proceeds.

**Scenario 3: Raw linked PatternQuestion**
> The record may be a bounded doubt reference only after strict lifecycle checks; it is never promoted into a playable practice item.

## Edge Cases

| Edge Case | Expected Behaviour |
|---|---|
| Exact Pattern ID is absent | Use bounded candidate discovery only when enabled/configured |
| Inactive/malformed/subject-mismatched Pattern | `IGNORE`; no prompt context |
| Batch hydration partial result | Ignore missing items; bounded retry for unprocessed keys |
| No linked question or ColBERT unavailable | PatternGraph-only doubt guidance or current path |
| Prompt budget pressure | Remove secondary reference/context before altering the student question or slot contract |

## Acceptance Criteria

| ID | Criterion | Linked FR |
|---|---|---|
| AC-01 | A feature-enabled request builds deterministic plan flags before data access. | FR-01 |
| AC-02 | Duplicate candidate IDs produce one bounded batch hydration, and malformed records are ignored. | FR-02 |
| AC-03 | Incompatible or unapproved Patterns cannot enter either prompt context or reuse. | FR-03 |
| AC-04 | A practice deficit receives compact guidance while existing QuestionBank reuse and verifier behavior remain unchanged. | FR-04, FR-05 |
| AC-05 | Doubt ColBERT only receives bounded linked references without answer/solution fields. | FR-06, FR-09 |
| AC-06 | Every external failure produces existing fallback behavior. | FR-07 |
| AC-07 | Prompt budget data is measured before structured generation. | FR-08 |

## Requirement-to-Test Mapping

| Requirement | Test File | Test Name | Status |
|---|---|---|---|
| FR-01–FR-03 | `app/tests/test_pattern_intelligence_runtime.py` | plan, score/version, metadata, graph, operation, and exclusion guards | Added — current execution [NOT VERIFIED] |
| FR-04–FR-05 | `app/tests/practice_generation/test_pattern_context.py` | compatible slot guidance and fail-closed type compatibility | Added — current execution [NOT VERIFIED] |
| FR-06 | `app/tests/test_pattern_intelligence_runtime.py` | bounded references, answer redaction, rerank fallback | Added — current execution [NOT VERIFIED] |
| FR-07 | `app/tests/test_pattern_intelligence_doubt_integration.py` | canonical IGNORE → legacy S3 fallback | Added — current execution [NOT VERIFIED] |
| FR-08 | `app/tests/test_pattern_intelligence_doubt_integration.py`, `app/tests/practice_generation/test_practice_prompts.py` | reference-first doubt budget and structured practice budget | Added — current execution [NOT VERIFIED] |

## Non-Goals

- Public request/response schema changes.
- New Pattern ingestion or a PatternQuestion-to-Question materialization system.
- Production deployment or unverified direct S3 Vector activation.

## Data Sensitivity Notes

| Data Field | Sensitivity | Handling Required |
|---|---|---|
| Student query / practice request | Potentially private | Do not log full payload or prompt |
| PatternQuestion answer/solution | Academic answer authority | Never include in reference context |
| PatternGraph/SolveFlow | Internal instructional data | Project compact safe projection only |

## Dependency Failure Expectations

| Service | Failure Mode | Expected System Behaviour |
|---|---|---|
| Embedding / S3 Vector | Unavailable or malformed | Existing flow without Pattern context |
| Pattern DynamoDB | Partial/error | Ignore affected candidates; existing flow continues |
| PatternQuestion GSI | Unavailable | PatternGraph-only doubt context or existing flow |
| ColBERT | Unavailable/timeouts | Deterministic linked-reference order or no reference |

## Open Questions

| # | Question | Owner | Status |
|---|---|---|---|
| 1 | Which Direct S3 Vector index and metadata contract is deployed? | Backend/runtime owner | Open [NOT VERIFIED] |
| 2 | Which verified QuestionBank record maps to a Pattern? | Backend data owner | Open [BLOCKER] |

## Assumptions

| # | Assumption | Label |
|---|---|---|
| 1 | `Pattern.status=active` represents the available canonical lifecycle state. | Confirmed by sibling schema |
| 2 | `questionsByPattern` returns raw provenance, not playable QuestionBank records. | Confirmed by sibling schema |

## Blockers

| # | Blocker | Label | Resolution Path |
|---|---|---|---|
| 1 | Pattern-driven playable reuse has no exact access path. | [PROD BLOCKER] for `REUSE_SAFE` | Add a sparse Pattern mapping GSI on existing QuestionBank, then validate/IAM-configure it. |
| 2 | Direct S3 Vector contract is not deployed/verified. | [PROD BLOCKER] for live activation | Publish index, ingestion, table settings, and IAM evidence. |

## Implementation Status — 2026-08-10

- FR-01 through FR-09 have local code paths and focused regression tests. `REUSE_SAFE` remains
  deliberately non-emitting, satisfying FR-05 rather than fabricating a QuestionBank reuse path.
- Pattern-aware practice is strict about source metadata. When the canonical record cannot prove
  category, question type, family, operation identity, or exact complexity parity, it returns
  `IGNORE` and existing deficit generation continues.
- Current lint/test/release execution is [NOT VERIFIED]: the environment rejected the required
  `uv` command due to its usage-limit approval gate after the latest edits. This does not alter the
  two production blockers above.
