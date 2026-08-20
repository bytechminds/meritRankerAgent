---
name: meritranker-exam-profiles
description: Use for any change to exam/stage profile identity, the AgentCore startup DynamoDB cache and snapshot, or profile-aware Practice planner/generator context. Covers app/schemas/exam_response_profiles.py and the DynamoExamProfileSource cache.
---

# MeritRanker Exam Profiles

Admin-managed exam/stage profile cache and the compact canonical context it
projects into the Practice planner/generator.

## Current status

In Progress — local model, startup cache, and request integration implemented;
Dev Admin and AgentCore E2E `[NOT VERIFIED]`. Full detail:
`skills/features/exam-profiles.md`.

## Identity model (do not redefine without approval)

- `examProfileId` — immutable canonical `EXAM_ID#STAGE` identity.
- `examId` — broad stable exam identity.
- `examName` — display metadata only, never used for matching.
- `stage` — profile metadata, joins `examId` deterministically.

## Runtime contract

The Amplify-owned ExamProfile table is published through
`/meritranker/agent-runtime/v1/exam-profile/{table-name,table-arn}` (SSM). At
AgentCore startup, `DynamoExamProfileSource` reads active records, validates
each, builds an immutable `MappingProxyType` snapshot, and atomically swaps it
in. Request resolution is an O(1) snapshot lookup — no per-request DynamoDB
read, no Redis/AppConfig/Prompt Management framework. A load failure keeps the
previous snapshot; a *first* load failure falls back to the legacy YAML
`ExamResponseProfileResolver`. Profile edits reach a running runtime only after
restart in this version — that is intentional, not a bug.

## Owning code

`app/schemas/exam_response_profiles.py`. Practice integration:
`app/features/practice_generation/planning.py`,
`app/features/practice_generation/providers.py` (`RoutedPlannerProvider`).

## Invariants

- Normal generator context: identity, description, matching canonical-subject
  sections, level, styles, exclusions — never cutoff, timestamps, or
  marks/timing/counts.
- Only `FULL_MOCK` receives deterministic format totals and section
  marks/timing/counts; quick practice always follows the user-requested count.
- `exam_profiles_load_*` load-health logs at info; cache-hit resolution stays
  debug-level to preserve the zero-latency request path; fallback resolution
  logs at info for diagnosis.

## Tests

`app/tests/test_exam_profile_cache.py`, `test_exam_response_profiles.py`.

## Regression impact to check

Practice planner/generator prompt context, existing prompt templates (must
remain unchanged by profile projection changes).
