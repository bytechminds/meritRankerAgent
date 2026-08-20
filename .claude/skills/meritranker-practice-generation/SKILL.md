---
name: meritranker-practice-generation
description: Use for any change to Practice/quiz/mock-test generation — planner, generator, verifier, matching/reuse, freshness-sensitive (current-affairs) web-evidence gating, exam-profile-aware planning, or async AgentCore task lifecycle for Practice. Covers app/features/practice_generation/ and app/prompts/practice_generation/.
---

# MeritRanker Practice Generation

Creates one-question practice, quizzes, topic/sectional tests, and full mocks.

## Current status

In progress: tracked background execution, IAM-signed AppSync progress
publication, and local deterministic validation are implemented; Sandbox live
generation is `[NOT VERIFIED]`. Freshness-sensitive Practice (current-affairs
web evidence) fail-closed path was live-verified locally on 2026-08-14; a
successful READY assessment with sufficient live evidence remains
`[NOT VERIFIED]`. Full detail: `skills/features/practice-generation.md`. Related
process history: `practice-generation-recovery-*.md`,
`practice-planner-reliability-*.md` (all under `skills/features/`).

## Architecture

```
classifier → deterministic Practice-creation gate
  → AgentCore tracked task → PracticeGenerationGraph
    → planner (exam-profile-aware for FULL_MOCK only)
    → QuestionBank reuse / Pattern-guided generation / adaptive generator
    → independent verifier
  → direct QuestionBank DynamoDB access
  → IAM-signed updatePracticeGenerationProgress AppSync mutation
```

## Owning code (verified against current tree)

`app/features/practice_generation/`: `agentcore_async.py` (thin AgentCore
integration boundary — does not own planning/generation/verification),
`generation.py`, `planning.py`, `matching.py`, `orchestration.py`,
`providers.py`, `question_contract.py`, `metadata_normalization.py`,
`repositories.py`, `schemas.py`, `config.py`, `pattern_context.py`,
`progress.py`/`progress_contract.py`, `appsync_progress_client.py`. Prompts:
`app/prompts/practice_generation/question_generator_v2.md`,
`question_regenerator.md`, `question_repair.md`, `question_verifier_v2.md`.

Note: most of these files are currently modified in the working tree — read the
actual diff before assuming `skills/features/practice-generation.md` reflects
the latest state, and update that file once the change lands (see
`meritranker-tutor-documentation-sync`).

## Invariants

- Static academic practice never triggers web search and stores no web evidence;
  only freshness-sensitive requests do.
- Fresh Practice launches only when selected evidence count ≥ accepted question
  count; insufficient evidence stops before planner/generator/verifier work
  (`PRACTICE_FRESH_EVIDENCE_COUNT_EXCEEDS_LIMIT` fails closed, never silently
  reduces the request).
- A verifier-approved fresh-fact question must cite one of the selected evidence
  URLs or it is deterministically returned for regeneration (`UNSUPPORTED_FACT`).
- Only `FULL_MOCK` receives exam-profile format counts/marks/timing; quick
  practice always follows the user-requested count.
- No raw snippets/URLs in logs — counts, windows, and status only.
- One Python application, one AgentCore runtime — no separate compute/transport
  path for Practice.

## Tests

`app/tests/practice_generation/` (`test_practice_core.py`,
`test_practice_freshness.py`, `test_practice_routing.py`),
`app/tests/test_web_search_source_policy.py`, `test_web_search_tool.py`,
`test_exam_profile_cache.py`.

## Regression impact to check

Exam profiles (planner context), Pattern Intelligence (guided generation path),
AI usage metering (every generator/verifier call is metered), web search source
policy, QuestionBank reuse/persistence, language resolution for generated
content.
