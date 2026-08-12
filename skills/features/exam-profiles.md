# Exam Profiles

## Status

In Progress — local model, startup cache, and request integration are implemented. Dev Admin and
AgentCore E2E are not yet verified.

## Identity

- `examProfileId` is the immutable canonical `EXAM_ID#STAGE` identity.
- `examId` is the broad stable exam identity.
- `examName` is display metadata only.
- `stage` is profile metadata and joins `examId` deterministically.

## Runtime contract

The Amplify-owned ExamProfile table is published through the existing SSM root:
`/meritranker/agent-runtime/v1/exam-profile/{table-name,table-arn}`.

At AgentCore startup, `DynamoExamProfileSource` reads active records, validates each profile,
builds an immutable `MappingProxyType` snapshot, and atomically swaps it in. Request resolution
is only an O(1) snapshot dictionary lookup. A load failure keeps the previous snapshot; a first
load failure leaves the legacy YAML `ExamResponseProfileResolver` active.

`exam_profiles_load_*` emits load health at info level. Cache-hit resolution telemetry is debug
level to retain the zero-latency request path; fallback resolution is info level for diagnosis.

There is deliberately no Redis, AppConfig, Prompt Management, generic configuration framework,
periodic request-path refresh, or per-request DynamoDB read. Profile edits reach a running
runtime after restart in this initial safe version.

## Context projection

Normal generator context contains identity, description, matching canonical-subject sections,
level, styles, and exclusions. It excludes cutoff, timestamps, and marks/timing/counts. The
planner receives the same compact context; only a `FULL_MOCK` gets deterministic format totals
and section marks/timing/counts. The existing prompt templates remain unchanged.

The section projection includes the immutable section `name` in addition to its
ID and subject. This keeps a future server-owned real-exam snapshot compatible
with the backend's exact section mapping; it does not enable Real Exam launch.

## Compatibility

Resolution precedence is cached exact `examProfileId`, cached exact `(examId, stage)`, then the
existing legacy resolver. No fuzzy display-name or LLM identity mapping is permitted.

## Latest Changes

- Added the minimal Amplify model/resource parameters, admin form, startup cache, compact
  projection, and planner/prompt request propagation.
- Added local cache, model contract, resource-publication, and form-mapping tests.
- Local fixture measurements on 2026-08-07: 10/25/50/100 profiles serialized to
  4,624/11,599/23,224/46,477 bytes; startup snapshot construction took
  0.111/0.145/0.256/0.546 ms; cached lookup p95 was 2.167/2.125/2.166/2.208 µs.
  The same test process recorded RSS deltas of 81,920/196,608/294,912/376,832 bytes.
  These are local in-memory fixture measurements, not a deployed DynamoDB E2E result.
