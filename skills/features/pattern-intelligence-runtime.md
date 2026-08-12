# Pattern Intelligence Runtime

## Current Status

**Local implementation and validation complete; Dev activation remains security-blocked.**

`PATTERN_INTELLIGENCE_ENABLED`, `PATTERN_INTELLIGENCE_REUSE_ENABLED`, and
`PRACTICE_PATTERN_CONTEXT_ENABLED` default to `false`. Production is unchanged. Guidance and reuse
must remain off in Dev until the QuestionBank mutation boundary prevents authenticated owners from
forging authoritative Pattern linkage.

## Runtime Contract

```text
S3 Vector candidate discovery
→ vector-score floor (negative eligibility only)
→ authoritative Pattern BatchGet
→ exact vector/DynamoDB version-hash parity
→ deterministic Pattern/slot compatibility
→ IGNORE | GUIDANCE_SAFE | REUSE_SAFE
```

DynamoDB Pattern records remain PatternGraph/SolveFlow authority. QuestionBank remains playable
Question authority. Raw PatternQuestion records are never treated as playable and are no longer
used for Doubt references.

The compatibility guard fails closed on status, graph shape, canonical subject/topic/category,
numeric-complexity band, exam IDs, question type, Pattern family, required target/operation
identities, slot exclusions, and `not_same_when`. Canonical identifiers normalize case plus
space/hyphen/underscore separators at one runtime boundary. Vector candidates without a score,
without a version hash, below the established retrieval floor, or stale against
`kbCurrentVersionHash` are ignored.

## Practice

The planner builds immutable slots before any Pattern or QuestionBank lookup. Existing indexed
QuestionBank reuse runs first. Deficit slots are conservatively grouped, resolved through the
Pattern runtime, and receive either:

- `REUSE_SAFE`: one exact verified playable QuestionBank row with explicit Pattern ID/version and
  `VERIFIED_GENERATION` or `VERIFIED_INGESTION` evidence, full slot compatibility, checked student
  history, and no activity/seen/exclusion conflict.
- `GUIDANCE_SAFE`: compact answer-redacted Pattern target/givens/conditions/operations/traps and
  `not_same_when` supplied to the existing adaptive generator.
- `IGNORE`: unchanged generation fallback.

The existing independent verifier remains the only authority for generated questions. After it
accepts a Pattern-guided question, the runtime writes a deterministic QuestionBank row with exact
Pattern ID/version evidence and records its `sourceQuestionBankId` on the assessment Question so
subsequent PracticeAttempt history can exclude it. For the Performance producer contract, a
verified generated assessment Question receives the trusted runtime selection's exact canonical
`patternId` and `patternVersionHash` as direct fields. Generic reused QuestionBank content remains
provenance only until the QuestionBank write boundary is deployed and proven authoritative; missing
or untrusted linkage is not inferred. History uses the existing user attempt GSI,
bounded assessment BatchGet, and bounded Question BatchGet; there is no scan or per-question N+1.

A `REUSE_SAFE` runtime selection is also copied as the exact direct Pattern ID/version pair to its
assessment-owned reused Question. Generic QuestionBank reuse never reads or promotes the source
row's raw Pattern tuple, even when that row claims verified evidence. The reuse flag builds the
same server-side Pattern context provider as guidance, but all flags remain disabled until the
existing QuestionBank mutation boundary is hardened.

Prompt input is measured before model execution. Optional reference material is removed before
core Pattern semantics; a singleton whose core guidance cannot fit returns to a controlled path
instead of silently sending an oversized prompt.

## Doubt

The shared context service and legacy graph use the same feature-gated runtime. Any unavailable,
weak, stale, malformed, or incompatible Pattern returns to legacy S3 retrieval.

`DoubtPatternContext` contains only the compact Pattern projection, at most six replay-safe approved
SolveFlow steps, and at most two verified playable QuestionBank references. The QuestionBank
Pattern GSI supplies a bounded five-record pool; ColBERT may rerank only that pool. Correct answers
and explanations are validated server-side but never projected into the Doubt prompt. Normal,
streaming, and legacy answer paths apply the same reference-first token budget.

## Producer and Data Contracts

The upstream backend now has a direct `PutVectors` producer using deterministic `patternId` keys,
Titan v2 1024-dimensional normalized embeddings, contract version `pattern-runtime-v1`, and
metadata containing Pattern ID, canonical subject/topic, active status, canonical version hash,
and vector-document version. The structural document excludes source values, answers, solutions,
and explanations.

The bounded, paginated admin backfill accepts only active Patterns whose recomputed canonical hash
matches `kbCurrentVersionHash`. It derives category from canonical Pattern topic and question type
only from the source PatternQuestion response mode. Missing/ambiguous evidence fails that record in
isolation. Writes are idempotent.

QuestionBank adds optional `patternId`, `patternVersionHash`, and `patternLinkEvidence` plus sparse
`getByPatternId`. Linkage is written only from verified ingestion/generation evidence; no semantic
historical mapping is invented.

## Data Access Bounds

| Access | Bound |
|---|---:|
| S3 Vector query | one per distinct retrieval group |
| Pattern hydration | one BatchGet, max 24 IDs, one bounded UnprocessedKeys retry |
| QuestionBank Pattern candidates | one exact GSI query, max 5 |
| Student history | one attempt GSI query (max 50), then two bounded BatchGets |
| ColBERT | zero or one execution over 3–5 linked playable references |

## Configuration and Rollback

The master, reuse, and Practice guidance flags remain independently false by default. Runtime table,
index, and vector ARN values come from managed SSM publication and AgentCore environment wiring.
Setting the three feature flags false restores legacy behavior without data migration.

## Validation — 2026-08-10

- Focused Tutor tests: **197 passed**.
- Full `make check`: Ruff passed; **2846 passed, 1 skipped**.
- AgentCore CDK TypeScript build: passed.
- `agentcore validate`: valid.
- Tutor and backend `git diff --check`: passed.
- Backend focused producer/schema/persistence/publication tests: **38 passed**.
- Backend global TypeScript: `GLOBAL_BASELINE_UNHEALTHY`; the touched KB-sync handler has no scoped
  diagnostic. Existing createPattern test/type diagnostics remain outside this change.

## Remaining Blockers

- **[SECURITY BLOCKER]** QuestionBank currently permits owner mutation. Because clients could forge
  `patternId`/version/evidence, `PATTERN_INTELLIGENCE_REUSE_ENABLED` must remain false and playable
  Doubt references must not be enabled live until mutations are restricted to trusted admin/IAM
  writers. This authorization change requires explicit owner approval.
- **[NOT VERIFIED]** Managed Dev deploy, vector backfill counts, real authenticated Practice/Doubt
  E2E, player lifecycle, IAM deployed-policy evidence, quality/cost/latency/diversity metrics, and
  provider prompt-cache evidence.
- **[DEFER]** Assessment-wide Pattern diversity, cross-wave memo reuse, and bounded parallel group
  retrieval require measured Dev evidence before changing the current bounded serial design.

## Latest Change

- `2026-08-10` — Closed the local producer, version-parity, authoritative playable mapping,
  REUSE_SAFE, canonical compatibility, QuestionBank-only Doubt references, history exclusion,
  prompt-budget, and executable validation gaps. Release remains held at local verification because
  the existing QuestionBank owner-mutation boundary is not authoritative enough for live reuse.
