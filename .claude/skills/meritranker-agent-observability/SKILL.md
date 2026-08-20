---
name: meritranker-agent-observability
description: Use for any change to structured events, readable request logging, tracing, or web-search/grounding/runtime-model diagnostics. Covers app/observability/ (events.py, lifecycle.py, logging.py, readable_log.py, summary.py, tracing.py, context.py) — the shared logging system every feature must reuse, never duplicate.
---

# MeritRanker Agent Observability

Structured events, request summaries, local inspection, and tracing shared by
every feature. **Do not create a second logging framework** — extend this one.

## Current status

Implemented locally. AgentCore runtime export, CloudWatch log delivery, custom
span visibility, and production volume/cost are `[NOT VERIFIED]` until
deployment. Full detail: `skills/features/agent-observability.md` (also
currently modified in the working tree alongside `app/main.py` — read the diff
before assuming the doc is current).

## Owning code

`app/observability/`: `events.py` (registered event catalogue — new event
types must be registered here, per the 2026-08-16 conversation-history example
of `conversation_persistence_started`), `lifecycle.py` (`observe_invocation()`),
`logging.py`, `readable_log.py`, `summary.py`, `tracing.py`, `context.py`,
`llm_usage.py` (see `meritranker-ai-usage-metering`).

## Invariants

- Never log secrets, JWTs, credentials, chain-of-thought, raw provider
  payloads, full prompts/questions/answers, or large web pages.
- Emitting an unregistered event type must fail loudly in a way that's caught
  by tests (observability-whitelist `ValueError`), not silently dropped —
  register new events in `events.py` before emitting them.
- `GROUNDING=grounded` is only emitted after required-web output actually cites
  a selected source URL and passes the bounded current-affairs cardinality
  check — provider success or context delivery alone is not "grounded".
- Production logs stay compact (milestones, warnings, all errors, safe
  correlation IDs); detailed lifecycle diagnostics are DEBUG/local only.
- Every unexpected error must expose component, stage, exception type/reason,
  and request correlation — without leaking payload content.

## Tests

`app/tests/test_observability.py`, `test_readable_request_log.py`,
`test_stream_lifecycle.py`, `test_llm_usage_observability.py`.

## Regression impact to check

Any feature that emits a new event type (Doubt Solver, Practice, Conversation
History) must register it here first. Changing log verbosity affects
production cost/searchability — treat as a Performance-Cost Reviewer concern
(see `meritranker-tutor-role-review`).
