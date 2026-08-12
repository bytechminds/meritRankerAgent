# Security Review: Pattern Intelligence Runtime

> Date: 2026-08-10

## Decision

**SECURITY_BLOCKED**

Local controls pass: vectors exclude answer/source values; stale/malformed/weak candidates fail
closed; PatternGraph and SolveFlow projections are allowlisted; correct answers/explanations never
enter Doubt references; no Pattern identity comes from the client; logs contain only safe IDs,
counts, reason codes, budgets, and timing; data access is indexed and bounded; producer and consumer
IAM definitions use exact resources/actions.

The unresolved blocker is the existing QuestionBank model authorization. Authenticated owners can
mutate rows, including the new authoritative Pattern linkage fields. Therefore a client could forge
`patternId`, `patternVersionHash`, and `patternLinkEvidence`. The reuse flag remains false and no
live security approval is possible until QuestionBank mutations are limited to trusted admin/IAM
writers while required authenticated reads are preserved.

Managed Dev IAM, authenticated E2E, and dependency/CVE evidence remain NOT VERIFIED because the
security hold prevents deployment. Production is unchanged.
