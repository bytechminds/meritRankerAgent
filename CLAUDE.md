# MeritRanker Engineering Contract

## Product Goal

MeritRanker is an AI-assisted competitive-exam learning system.

The system must provide:
- highly accurate educational content;
- reliable generation and verification;
- deterministic business rules where possible;
- safe use of external/current information;
- strong Practice and Doubt Solver reliability;
- low unnecessary AI and infrastructure cost;
- bounded latency;
- scalable architecture without premature complexity.

Accuracy and correctness are non-negotiable.

## Engineering Priorities

1. Correctness and educational accuracy.
2. Reliability, stability, security, and durable state correctness.
3. Cost efficiency.
4. Latency and throughput.
5. Maintainability and scalability.

Cost or latency optimizations must never weaken correctness or verification.

## Core Development Rule

For any bug or feature:

DO NOT immediately edit code.

Follow:

1. Understand requirement.
2. Inspect actual execution path.
3. Reproduce or collect evidence.
4. Identify first incorrect boundary/root cause.
5. Identify existing ownership and reusable abstractions.
6. Design the smallest safe change.
7. Analyze impact and regressions.
8. Obtain architecture review when required.
9. Implement only approved scope.
10. Run focused tests.
11. Review the diff independently.
12. Run broader regression tests.
13. Run real/live verification when required.
14. Report PASS or NOT_READY honestly.

## Evidence Before Fixes

Never fix from a symptom alone.

Before modifying a defect, identify where practical:

- exact request/input;
- last successful stage;
- first failing stage;
- originating exception/reason;
- owning component;
- whether a recent change is causal;
- whether the affected code is shared.

Do not globally suppress an exception to hide its origin.

Fix the first invalid assumption at its owning layer.

## Change-Surface Rule

Prefer:
- 1–3 focused production modules;
- existing abstractions;
- existing configuration;
- existing persistence and observability.

Large patches require explicit architectural justification.

Do not:
- reorganize directories during a bug fix;
- rename unrelated APIs;
- rewrite working components;
- fix unrelated lint/type errors;
- modify unrelated dirty-worktree changes;
- introduce a new abstraction, helper, constant, dependency, or pattern beyond
  what the request strictly requires, however small.

If satisfying the request appears to require touching anything outside its
literal scope, stop and ask the user first — do not proceed and disclose it
afterward. `skills/roles/scope-guard.md` is the gate for this and runs on every
change regardless of size.

## Architecture

Keep layers clearly separated:

UI/API
→ request/authentication
→ domain policy
→ orchestration
→ retrieval/providers
→ generation/verification
→ persistence
→ observability

Do not leak:
- provider-specific parameters into classifiers/domain logic;
- database details into LLM orchestration;
- UI behavior into shared model execution;
- Practice-specific semantics into generic provider adapters.

Dependencies should point toward stable abstractions.

## AI / LLM Rules

Use LLMs only for work requiring probabilistic reasoning/generation.

Prefer deterministic code for:
- validation;
- counts;
- dates;
- identity;
- ownership;
- permissions;
- compatibility;
- filtering;
- state transitions;
- arithmetic when deterministic;
- routing rules that are unambiguous.

Do not add another LLM call when deterministic logic can safely solve the problem.

Generator output is never authoritative merely because a model produced it.

Use verification appropriate to the feature.

For current/fresh factual content, external evidence is authoritative over model memory.

## Provider Design

Provider implementations remain behind shared interfaces.

Do not couple product logic to one provider.

Adding another provider should normally require:
- adapter/configuration;
- routing configuration;
- tests;

not rewriting business logic.

Fallback must be bounded, observable, and cost-aware.

## Cost Rules

Count all real provider calls.

Avoid:
- duplicate LLM work;
- unnecessary fallback;
- per-item retrieval when grouping is possible;
- repeated DB reads inside loops;
- large unused prompt context;
- fetching data already available in the request/context.

Reuse already-paid verified work where safe.

Never weaken quality verification to save money.

## Reliability

Business correctness must not depend exclusively on process-local memory.

Use durable state for durable business facts.

RAM may be used for short-lived working state and optimization only.

Retries must be:
- bounded;
- typed;
- idempotent where required;
- prevented after cancellation or stale execution;
- safe against duplicate persistence.

## Performance

Measure before adding infrastructure.

Do not introduce Redis/cache/new services solely because latency is suspected.

First inspect:
- call counts;
- DB operation counts;
- provider latency;
- retrieval latency;
- serialization;
- repeated work.

Optimize the measured bottleneck.

## Logging / Observability

Use the shared structured logger.

DEBUG/local:
- detailed lifecycle diagnostics;
- safe IDs;
- stages;
- reason codes;
- duration;
- provider/route/model;
- counts.

Production:
- compact milestones;
- warnings;
- all errors/exceptions;
- safe request correlation.

Never log:
- secrets;
- JWTs;
- credentials;
- chain-of-thought;
- raw provider secrets;
- unnecessary raw prompts/questions/answers;
- large web pages.

Every unexpected error should expose enough safe metadata to locate:
- component;
- stage;
- exception type/reason;
- request correlation.

Do not create a second logging framework.

## Testing

Every changed behavior requires focused tests.

For shared code also test unaffected callers.

Where appropriate test:
- happy path;
- invalid input;
- boundary values;
- failure path;
- retry/fallback;
- concurrency/idempotency;
- cancellation;
- recovery/resume;
- language;
- security/ownership;
- cost/call counts.

Prefer deterministic/mocked tests before expensive real-provider tests.

Real-provider/live validation is required when local tests cannot prove the behavior.

Never infer live success from source tests.

## Regression Protection

Before completion ask:

- What previously working behavior could this change affect?
- Is this function shared?
- Does it alter persistence?
- Does it alter routing/models/tokens?
- Does it alter billing?
- Does it alter security/auth?
- Does it alter Practice or Doubt Solver?
- Does it alter Pattern/retrieval?
- Does it alter language?
- Does it alter cancellation/resume?
- Does it change API/schema/UI contracts?

Test the relevant areas.

## Language

Resolve user-facing language once at the operation boundary.

Explicit request/UI language wins over stored preference.

Persist or propagate resolved language through long-running operations.

Student-facing directly reused content must match the resolved language.

Internal evidence, Pattern data, or web sources may use another language if needed for source quality.

Generated student-facing content must use the requested language and be verified for language compliance.

## Current / Fresh Information

For current, recent, latest, or otherwise time-sensitive factual content:

- require fresh external evidence;
- resolve temporal intent deterministically;
- use runtime date;
- preserve explicit historical ranges;
- search with bounded purpose-aware queries;
- filter stale/unsupported evidence;
- ground generation and verification in evidence;
- fail closed when trusted evidence is insufficient.

Never silently substitute stale model memory.

## Security

Server-side identity and authorization are authoritative.

Never trust browser-provided ownership fields when authenticated identity is available.

Do not expose answer keys or private authoritative data unintentionally.

Do not weaken security to simplify tests.

## Documentation

Update feature documentation when:
- architecture changes;
- durable contracts change;
- new failure semantics are introduced;
- operational behavior changes materially.

Do not document trivial implementation details.

## Completion Contract

A task is complete only when:

- root cause/requirement is understood;
- implementation is integrated;
- focused tests pass;
- impact review completed;
- applicable regression tests pass;
- diff is clean;
- live validation completed when required;
- remaining limitations are stated.

Use NOT_READY for anything not actually proven.

Never deploy Production unless explicitly requested.