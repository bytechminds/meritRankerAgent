# Implementation Plan: Practice Generation Recovery

## Goal

Recover verified practice work through the smallest bounded changes to existing generation and
finalization boundaries.

## Non-Goals

No new agent/module, public schema, route, model, token, Pattern, billing, or infrastructure change.

## Current Context

`generation.py`, `providers.py`, `orchestration.py`, and `repositories.py` already own the required
boundaries; `skills/features/practice-generation.md` is the feature authority.

## Architecture Decision

Use the blueprint's canonical compatibility signature, enrich the existing final result, carry
slot-scoped repair context through the existing provider, and conditionally update only verified
bucket metadata. Alternatives involving another agent, broad regeneration, or extra LLM judging
were rejected as higher cost and weaker ownership.

## Files to Add / Change

Existing generation/provider/orchestration/repository/event/prompt files, focused tests, and the
feature context only. No runtime or schema file is added.

## Data Flow

`slot → generate → validate → independently verify → persist → final validate → optional one
metadata repair → full final validate → READY/FAILED`.

## Service Boundaries

LLMs remain behind `LlmOrchestrator`; DynamoDB remains behind `QuestionRepository`; AppSync progress
remains behind `AppSyncAssessmentProgressRepository`.

## Schema / Contract Changes

No public contract change. `FinalValidation` is internal and backward-compatible through defaults.

## Config / Env Changes

No new variables.

## Testing Plan

Exact category split, real repair payload, bounded waves, conditional write, integration recovery,
32-case deterministic matrix, full test suite, and local real-LLM replay.

## Security Notes

Telemetry contains only IDs/counts/reasons. Candidate content is sent only to the existing trusted
repair model and is never logged.

## Performance / Cost Notes

Recovery adds zero LLM calls. The existing maximum content attempts remains three per unresolved
slot. Local replay improved 12 calls to 10 and output tokens from 21,663 to 10,008.

## Risks

Second recovery failure is terminal; aggregate dirty-worktree scope must be isolated before release.

## Rollback / Disable Plan

Revert the scoped functions and prompt line; no data migration or config rollback is required.

## Acceptance Criteria

All BA criteria, `make check`, `agentcore validate`, docs, and local replay must pass.

## Handoff to Engineer

Approved by self-review on 2026-08-12. Out of bounds are all route/model/token/Pattern/billing and
public contract changes.
