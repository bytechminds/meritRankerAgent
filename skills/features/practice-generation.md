# Practice Generation

## Purpose and status

PracticeGenerationGraph creates one-question practice, quizzes, topic/sectional tests, and full
mocks using the existing MeritRanker AgentCore Python application. The feature is in progress:
tracked background execution, IAM-signed AppSync progress publication, and local deterministic
validation are implemented; Sandbox live generation remains `[NOT VERIFIED]` until the controlled
live gate completes.

The existing classifier recognizes practice intent and a deterministic gate requires an explicit
creation signal. Eligible requests create or recover the deterministic private assessment,
register one AgentCore tracked task, persist the conversation linkage, start background execution,
and end the request stream with the typed `practice_generation_started` event containing
`responseType=practice_generation` and `practiceTestId`. Practice advice and strategy questions
continue through DoubtSolverGraph. The controlled `PRACTICE_AGENTCORE_ASYNC_NOT_CONFIGURED` result
remains only as a defensive injected-graph fallback when no launcher is supplied.

## Architecture

```text
Frontend
→ existing AgentCore runtime
→ existing classifier
   ├── DoubtSolverGraph
   └── AgentCore tracked task → PracticeGenerationGraph
→ direct Question/QuestionBank DynamoDB access
→ IAM-signed updatePracticeGenerationProgress AppSync mutation
→ existing progress Lambda and MockTestQuiz table
→ existing AppSync subscription and frontend
```

## Freshness-sensitive Practice evidence (2026-08-14)

After the existing classifier and deterministic Practice gate, requests for current affairs,
recent events, current office-holders, dated awards/appointments, or other freshness-sensitive
facts run one bounded existing web-search retrieval before the asynchronous assessment is created.
The request carries `requires_fresh_evidence`, a canonical freshness reason, and a compact selected
`FreshEvidenceBundle` through `PracticeGenerationRequest` and the existing
`MockTestQuiz.meta.practiceRequest` recovery payload. Static academic practice never triggers this
path and stores no web evidence.

The source-policy resolver supplies either the user-explicit date range or the existing current
window ending on the runtime date. The provider-neutral web tool retains only valid, de-duplicated,
date-compatible URLs with compact snippets; supplied publication dates outside the window are
discarded and undated metadata is allowed only when it is otherwise usable. The maximum retained
evidence set is 20 items. Requests needing more independently grounded facts stop with the typed
`PRACTICE_FRESH_EVIDENCE_COUNT_EXCEEDS_LIMIT` result rather than silently reducing the request.

Fresh Practice launches only when the selected evidence count equals or exceeds the accepted
question count. Provider failures, weak/empty context, malformed recovery metadata, or insufficient
evidence stop before planner, generator, or verifier model work. No later phase repeats web search.
The planner receives only freshness metadata; each generator/verifier call receives the exact
evidence subset for its slot IDs. A verifier-approved fresh-fact question must cite one of those
selected URLs, otherwise it is deterministically returned for regeneration as `UNSUPPORTED_FACT`.
Logs retain counts, windows, and status only; raw snippets and URLs are not logged.

Local current-code E2E on 2026-08-14 verified the fail-closed path for both an undated
current-affairs request and a `2024` historical current-affairs request: live bounded web
retrieval returned no source that passed the existing quality gate, so no assessment task,
generator, or verifier call was started and the caller received the controlled startup failure.
This proves the no-stale-memory guard under unavailable evidence. A successful READY assessment
with sufficient live evidence remains **[NOT VERIFIED]**; no deployment was performed.

## Exam Profile Context (2026-08-07)

`PracticeGenerationRequest` carries an optional canonical `exam_profile_id` alongside the existing
`exam_id`/`exam_stage` compatibility fields. `RoutedPlannerProvider` resolves it exclusively from
the process-local ExamProfile snapshot and adds only the compact canonical-subject section context
to its existing structured planner payload. A quick practice still follows the user-requested
count; only a `FULL_MOCK` can receive profile format counts, marks, and timing. Generation and
verification architecture remains unchanged.

There is one Python application and one AgentCore runtime. No separate practice compute,
transport, image, deployment path, or credential path exists.

`agentcore_async.py` is the thin integration boundary. It initializes the existing assessment,
registers/completes the SDK task, submits bounded in-process background work, drives persisted graph
commands to READY/FAILED, and consumes terminal exceptions. It does not own planning, generation,
verification, reuse, question persistence, or finalization rules.

## Retained implementation

- `graph.py`: routes neutral `plan_and_fill`, legacy `generate_group`, schema-v2
  `generate_wave`, and `finalize` graph commands.
- `orchestration.py`: blueprint/reuse/deficit/group generation/retry/finalization orchestration.
- `planning.py`: explicit creation gate, request resolver, subject-aware planner-family selection,
  schema-v2 slot validation, bounded repair, request-topic coverage validation, and deterministic
  slot fallback for allowed assessment types. Explicit schema-v1 persisted blueprints remain
  readable.
- `metadata_normalization.py`: centralized fail-closed subject/topic/category/difficulty/exam/
  language/activity/PatternFamily normalization for exact reuse metadata.
- `matching.py`: backend v1 reuse-key construction, schema-v2 exact slot assignment, eligibility,
  hydration, confidence checks, and one-to-one duplicate prevention. Category—not subject—owns the
  reuse partition-key segment; optional PatternFamily identity is never inferred from topic.
- `generation.py`: bounded groups, partial sibling retention, structural validation, conditional
  verification, deterministic question IDs, and final-set validation.
- `providers.py`: shared AgentCore model registry and provider executor adapters, including the
  quant/reasoning, English, and factual planner families using existing model aliases.
- `repositories.py`: MockTestQuiz, QuestionBank, and Question access; initial assessment
  creation/recovery, direct conditional Question writes, strongly consistent
  manifest hydration, and bounded GSI pagination.
- `appsync_progress_client.py`: exact deployed GraphQL mutation, AWS SigV4 signing, HTTP timeout,
  bounded retry, GraphQL error parsing, and typed safe response parsing.
- `progress.py`: full-meta merge, authoritative Question-backed manifest recalculation,
  monotonicity checks, and the single ongoing MockTestQuiz update path through AppSync.
- `resource_contract.py`: success-only cached SSM table/index contract loader.
- `resource_validation.py`: fail-closed table/index status, ARN, key-schema, and projection checks.
- `events.py`: typed, content-safe practice observability.
- `pattern_context.py`: stable provider boundary with a disabled no-op implementation.
- `app/prompts/practice_generation/`: shared contract, three planner-family prompts, generator,
  and verifier prompts.

Generation groups remain an internal batching and durable-progress concept. New writes use generic
PENDING/RUNNING/COMPLETED/FAILED state and retry/deficit metadata; they do not use delivery counts,
queue identifiers, dispatch markers, or leases. Legacy schemaless metadata may be ignored without
a destructive migration.

## DynamoDB contracts

`MockTestQuiz` retains request summary, blueprint, phase, playable flag, progress, counts,
generation groups, deficits, and the authoritative `readyQuestionIds`/`readyCount` manifest.
`QuestionBank` exact reuse uses the sparse `getByReuseBucket` GSI with the canonical
category/topic/question-type/language partition key and a `v1#<difficulty>#` sort-key prefix.
Repository pagination remains bounded and query-only. Unknown aliases, incompatible category,
exam or PatternFamily metadata, and low or malformed declared candidate confidence fail closed to
fresh generation; no scan, `contains`, fuzzy, or semantic fallback is introduced by this slice.
The older bounded category query remains available only for schema-v1 compatibility.
`Question` writes remain assessment-owned, direct, conditional, and deterministic. After each
accepted generation group is persisted, the progress coordinator rehydrates the current manifest
from Questions and sends one complete parent-state mutation. Existing valid manifest siblings are
preserved across GSI lag by strongly hydrating previously published IDs and current group IDs.

Direct MockTestQuiz access is retained only for initial create/recover/read and playable reads.
Ongoing progress, manifest, group state, READY, FAILED, and playable transitions use the existing
AppSync mutation. The deployed resolver remains responsible for ownership, transition, metadata,
and conditional validation and replaces the full `meta` object. If the primary terminal FAILED
publication cannot be sent, the existing conditional assessment failure write is used only to make
the stale state reconcilable; it cannot overwrite READY and does not create a second progress path.

Schema-v2 generated and reused Questions persist a private versioned answer contract with stable
`INDEX_V1` option IDs, the canonical correct option/value, verified explanation, `VERIFIED`
status, and answer version. Deterministic answer-position balancing updates that complete contract
atomically; it never downgrades it to the legacy answer shape.

READY requires exact accepted/ready/reused/generated/verified/bucket counters, zero failures,
terminal groups, a unique exact manifest, strong base-record ownership checks, valid answers and
solutions, exact planner-slot coverage, independent schema-v2 verification evidence, and a final
deterministic answer-position distribution.
After academic verification, but before positions and `READY`, the repository deterministically
reorders each question's four options from the stable assessment/question IDs. It preserves the
stored correct-answer reference, validates the final distribution (no more than
`ceil(questionCount / 4)` correct answers per option and no three adjacent equal positions), then
re-runs the full playable-question/final-manifest contract. Invalid distribution or revalidation
fails closed with `ANSWER_POSITION_DISTRIBUTION_INVALID`; the Question testId GSI is never final
authority.

## Resource contract and configuration

The root remains `/meritranker/agent-runtime/v1/practice`. Enabled graph configuration requires
only resource/reuse contract versions, table names and ARNs, and the three GSI names. Missing or
incompatible database resources fail closed; failed SSM loads are not cached. An enabled practice
runtime also requires `APPSYNC_GRAPHQL_ENDPOINT`; a disabled practice feature does not.

### Startup resource-resolution invariant

**Stable SSM-backed resource metadata resolves before the application is ready to accept
invocations. Runtime request paths consume the cached values and must never perform SSM resource
discovery.** Every reader executes at `main.py` module import — strictly before `@app.entrypoint`
becomes reachable — and each caches an immutable process-scoped snapshot:

| Reader | Feature | Required when |
|---|---|---|
| `features/practice_generation/resource_contract.py` | Practice core | `PRACTICE_GENERATION_ENABLED` |
| `features/practice_generation/pattern_resource_contract.py` | Pattern Intelligence | `PATTERN_INTELLIGENCE_ENABLED` (attempt-history identifiers only when `PATTERN_INTELLIGENCE_REUSE_ENABLED`) |
| `services/conversation/runtime_config.py` | history / session / memory | non-`test` `APP_ENV` |
| `services/doubt_solver/exam_profile_cache.py` | exam profiles (Doubt + Practice planner) | non-`test` `APP_ENV` |

There is no `ApplicationResourceSnapshot` aggregate and none is needed: composing these four
already-startup-resolved, already-batched, already-process-cached contracts would duplicate them
without changing behaviour. Resource identifiers are deployment-stable, so a restart or redeploy is
the only refresh boundary — there is deliberately no TTL cache, background polling, or per-request
`DescribeTable`/`GetIndex` verification.

Precedence is `explicit environment value → SSM → explicit configuration failure`. Deployed
runtimes receive CDK-injected environment variables and therefore make zero SSM calls for those
fields; only genuinely missing identifiers are read.

Fail-fast is part of the contract: for an explicitly **enabled** feature, a missing parameter,
`AccessDenied`, or an unparseable resource identifier raises `PracticeConfigurationError` (or
`ConversationConfigurationError`) out of module import, so the application never becomes
importable. An enabled feature must never silently degrade to a no-op provider because its
resources could not be resolved.

**One intentional exception:** `ExamProfileRuntime.load()` is **fail-open** — it catches every
exception and returns `False`, preserving the last-known-good snapshot, because exam-response
profiles are advisory during migration. This is the only reader that does not block startup and is
therefore the only divergence from the fail-fast contract above. Whether exam profiles should
become mandatory is an open **advisory-vs-mandatory policy decision** and is deliberately out of
scope for the resource-lifecycle work; it is recorded here so the exception is explicit rather
than accidental.

The invariant is enforced by `app/tests/test_startup_resource_contract.py`, which asserts a
*bounded* number of batched startup calls plus a *zero* post-startup delta for first request,
second request, Practice configuration, Pattern retrieval provider construction, Doubt Solver
exam-profile resolution, and conversation/session/memory resolution — including under concurrent
cold bootstrap. The exact startup call count is intentionally **not** pinned, so legitimately
adding or re-batching a parameter does not produce a false failure.

### Readiness lifecycle

Two distinct startup events, and no event claims readiness while mandatory initialization can
still fail:

- `runtime_started` (`status=started`) — emitted early, immediately after runtime identity is
  configured, so identity diagnostics survive a later startup failure. It deliberately does not
  claim readiness.
- `runtime_ready` (`status=ready`) — the authoritative readiness boundary, emitted only after
  conversation persistence, resource contracts, DynamoDB resource validation, Practice/Pattern
  runtime construction, and graph compilation have all succeeded. Details carry aggregate feature
  flags only (`practiceEnabled`, `patternContextEnabled`, `patternReuseEnabled`,
  `orchestratedDoubtSolverEnabled`) — never table names, ARNs, or index names.

Because every mandatory step raises out of module import, a failure anywhere above makes the
`runtime_ready` line unreachable and the application never importable. Prior to this split,
`runtime_started` carried `status=ready` while Practice/Pattern bootstrap, DynamoDB validation,
and graph construction all still followed — a false-ready window that is now closed and regression-
tested.

Local `agentcore dev` runs the sequence **twice** (`Started reloader process` then
`Started server process`): this is normal uvicorn `StatReload` behaviour, two separate OS
processes each performing a legitimate independent bootstrap. It is not deduplicated, and any
startup benchmark should account for it.

Supported settings:

- `PRACTICE_GENERATION_ENABLED=false`
- `APPSYNC_GRAPHQL_ENDPOINT=` (required when enabled)
- `PRACTICE_PATTERN_CONTEXT_ENABLED=false`
- `PRACTICE_RESOURCE_PARAMETER_ROOT=/meritranker/agent-runtime/v1/practice`
- `PRACTICE_GENERATION_GROUP_SIZE=3`
- `PRACTICE_GENERATION_GROUP_MAX=5`
- `PRACTICE_VERIFICATION_MAX_CONCURRENCY=3` (1–5) — bounded fan-out for the existing
  per-question verifier calls inside one generation group. Transport concurrency only: each
  question still receives its own verifier call on the same route, model, prompt, and schema,
  and outcomes are merged by `slot_id` in the original deterministic order, so completion order
  can never determine identity or event ordering. Default 3 was chosen by measurement — at the
  5-slot group maximum, bound 3 reaches the same wall time as bound 4, so 3 is the smallest
  value achieving the best result. Worst-case in-flight verifier calls is
  `PRACTICE_VERIFICATION_MAX_CONCURRENCY × 2` because up to two generation groups run
  concurrently.
- `PRACTICE_ITEM_RETRY_LIMIT=1`
- `PRACTICE_PLANNER_REPAIR_LIMIT=1`
- `PRACTICE_RECOVERY_STALE_SECONDS=120`
- `PRACTICE_GENERATION_MAX_WALL_TIME_SECONDS=900`
- bounded QuestionBank query, metadata-size, and finalization-attempt settings documented in
  `app/.env.local.example`.

Practice model calls use the same `LlmOrchestrator`, `PromptResolver`, model executor, provider
configuration, provider adapters, fallback handling, and environment credential resolver as
DoubtSolverGraph. Structured calls skip only student-answer completion, continuation, rewrite,
and answer-quality finalization so planner/generator/verifier JSON remains unchanged. Provider
credentials are resolved lazily only after an active route selects a provider.

The local AgentCore CDK definition grants exact access to the 11 legacy practice SSM parameters, the three
existing table ARNs, and only the named Question/QuestionBank indexes. MockTestQuiz direct access is
limited to `GetItem`/`PutItem`/conditional `UpdateItem`; Question access uses direct `PutItem`,
bounded query/batch reads, and position updates. When the existing deployment-time non-secret
practice flag is enabled, it grants only
`updatePracticeGenerationProgress` on the configured AppSync API—never an API wildcard. The
Pattern and PracticeAttempt SSM parameters, environment variables, and IAM grants are synthesized
only when their matching Pattern flags are enabled. With both flags off, the deployment contract is
the legacy Practice contract and contains no Pattern resource dependency. The compatible
Pattern-off AgentCore Dev revision was deployed on 2026-08-10 and its exact legacy index Query
permissions were verified. Live creation requests were misrouted by the deployed classifier before
Practice launch, so a READY assessment remains `[NOT VERIFIED]`.
No secret, queue, credential resolver, or wildcard resource was added.

## Tests

The focused suite covers routing and the controlled safe gate; graph imports and neutral command
routing; planning, reuse, generation, verification, replacement, persistence, exact manifests,
resource contract/validation, shared credentials, duplicate graph operations, restart behavior,
GSI lag, counter/ownership failures, deterministic option rebalancing, and repeated
5/50/100-question in-memory runs.

The 2026-08-10 P0 regression suite additionally proves that a Pattern-disabled five-question
assessment reaches `READY` through the exact legacy path, vector discovery failure returns empty
optional guidance, a missing Pattern QuestionBank index downgrades `REUSE_SAFE` to
`GUIDANCE_SAFE`, and a real legacy QuestionBank query failure remains fatal. Legacy repository
query failures emit content-safe structured diagnostics with operation, logical table/index, AWS
exception type/code, Pattern flag state, and the explicit fallback decision.

### Schema-v2 checkpoint compatibility repair — 2026-08-10

The Dev failure `practice-d5e83d833e7a09c28b70033890a2f022` was traced to the schema-v2
post-match checkpoint, not QuestionBank reuse or question persistence. The initial and manifest
`Question.testId` queries and exact `QuestionBank.getByReuseBucket` read succeeded; the existing
AppSync progress handler then rejected the unapproved `patternRuntime` metadata key with
`PRACTICE_PROGRESS_UNKNOWN_META_FIELD`. Pattern-disabled Practice now does not publish
Pattern-only metadata. Schema-v2 emits explicit reuse-query start/completion events, the generic
Question event identifies `operation=list_linked`, and the async terminal boundary records only
safe request correlation, repository operation, cause class, AWS code when present, and fallback
decision. With both Pattern flags off, schema-v2 does not invoke even the Pattern provider and
does not emit a Pattern retrieval event. It now publishes `EXISTING_MATCH_COMPLETED` and one
`GENERATION_GROUP_CREATED` event per persisted group before generation begins. The deterministic
five-slot CAT PRE Profit and Loss fallback regression asserts the deployed progress metadata
allowlist, canonical fallback keys, and the generation checkpoint. Dev READY/player evidence
remains pending the repaired revision deployment and authenticated lifecycle run.

The shared-credential composition test instantiates the routed practice planner, generator, and
verifier with one centralized runtime model executor. The normal AgentCore runtime uses sanitized
CodeZip staging rather than `app/` as a Docker build context, so a source `app/.dockerignore` is
not required.

Key files:

- `app/tests/practice_generation/test_practice_core.py`
- `app/tests/practice_generation/test_practice_matching_v2.py`
- `app/tests/practice_generation/test_practice_orchestration.py`
- `app/tests/practice_generation/test_practice_repository.py`
- `app/tests/practice_generation/test_practice_resource_contract.py`
- `app/tests/practice_generation/test_practice_resource_validation.py`
- `app/tests/practice_generation/test_practice_routing.py`
- `app/tests/practice_generation/test_practice_credentials.py`
- `app/tests/practice_generation/test_practice_prompts.py`
- `app/tests/practice_generation/test_practice_freshness.py`
- `app/tests/practice_generation/test_practice_agentcore_async.py`
- `app/tests/practice_generation/test_appsync_progress_client.py`
- `app/tests/practice_generation/test_appsync_progress_repository.py`

Tracked-task coverage uses the installed `bedrock-agentcore` application abstraction and verifies
HealthyBusy/Healthy transitions, exactly-once completion, terminal persistence ordering,
registration/start failure, and duplicate active invocation suppression. Routing coverage verifies
the existing stream/non-stream response contract and Hindi/Hinglish language preservation.

## Performance notes

The acknowledgement critical path is classifier → deterministic request resolution → one
conditional assessment write/read → AgentCore task registration → existing conversation
persistence → typed stream acknowledgement commit → background task start. Planning, QuestionBank reads,
generation, verification, and finalization run in the tracked background task. Background
execution is bounded to four in-process tasks per runtime instance. Within one schema-v2 wave, at
most two immutable provider/verifier workers run concurrently; workers never mutate repositories
and the coordinator commits their results sequentially.

## Security and known limitations

- Logs exclude prompts, questions, solutions, provider bodies, credentials, and raw user IDs.
- No cache, Redis, Step Functions, new database, or replacement async infrastructure was added.
- Fresh current-affairs creation is supported only with sufficient temporally eligible evidence;
  the controlled fail-closed path is the expected outcome when provider evidence is insufficient.
- Canonical Pattern context is implemented behind both disabled-by-default Pattern flags. It runs
  only after QuestionBank reuse for deficit slots, requires exact source compatibility, and falls
  back to ordinary generation when unavailable or unsafe. Live activation remains `[BLOCKED]` by
  the unverified direct S3 Vector contract and missing Pattern-to-playable-QuestionBank mapping.
- `[NOT VERIFIED]` Sandbox 1/5/50/100 live generation and start/save/resume/submit/review evidence
  is still required before a complete or production-ready claim.
- `[BLOCKED]` The deployed runtime has no established safe provider-secret injection channel:
  CodeZip excludes local `.env` files, AgentCore CLI 0.14 has no parameter override, and copying
  raw secret values into CDK would expose them in synthesized/CloudFormation artifacts. Existing
  environment credential names and resolver behavior are preserved; live Dev validation remains
  blocked until an approved deployment-time secret delivery mechanism exists.
- `[AUTH TODO]` `resolve_actor_id` remains the repository's compatibility identity seam and returns
  the validated request actor. A server-verified runtime principal is still required before direct
  production exposure; no new auth contract was invented in this change.
- A stale `GENERATING` activity can be recovered once through the existing parent metadata using
  an `expectedUpdatedAt` compare-and-set claim. Accepted slot-linked Questions are retained and
  only remaining slots return to `PENDING`. A second stale interruption fails the parent; no lease,
  queue, table, Lambda, or unbounded retry loop was added.
- Existing frontend Practice APIs and attempt flow remain owned by `AI-tutor-backend`.
- AppSync retries are limited to two and only cover transport timeout/connection, 429, 5xx, and
  temporary credential refresh. GraphQL authorization, owner, transition, metadata, schema, and
  conditional conflicts fail without blind replay. A transport retry recalculates the manifest
  from persisted Questions first.

## Latest changes

- 2026-08-12: Planner validation recovery now retains content-safe diagnostics for each bounded
  attempt: attempt/phase, schema name, error count, field paths, error types, reason code, slot
  count, and validation duration. The sole repair receives those metadata categories; no raw model
  output, prompt, answers, or student data is persisted or logged. A multi-topic deterministic
  fallback canonicalizes supported separators and round-robins every requested topic across the
  accepted slots, then passes the existing schema-v2 and system-bucket validation. If that fallback
  cannot construct a valid blueprint, it becomes the controlled
  `PRACTICE_PLANNER_FALLBACK_INVALID` terminal state instead of leaking a generic async
  `ValidationError`; no generator receives an invalid blueprint. The normal valid-planner path
  remains one call, malformed output remains initial call plus at most one repair, and fallback
  makes no LLM call. Repair payloads now identify `planner_phase=repair` and only allowlisted
  feedback, and all three planner prompts direct the repair action. The central event registry
  includes `planner_fallback_completed` and `planner_fallback_failed`, so safe fallback telemetry
  cannot itself terminate the async task. The exact local Dev request for SSC_GD/PRE, Time and
  Work plus Number System completed on 2026-08-12 as
  `practice-b06846a73f72427f563a55a1cdbfd6b6`: two invalid planner responses, a valid
  deterministic fallback, five generated questions, and terminal `READY`. Focused planner and
  observability coverage passed (175 tests); browser lifecycle, production deployment, and
  reliability/load evidence remain `[NOT VERIFIED]`.

- 2026-08-10: Added feature-gated canonical PatternGraph guidance for schema-v2 deficit slots.
  Candidate IDs are authoritatively BatchGet-hydrated, strict metadata/operation/exclusion gates
  precede `GUIDANCE_SAFE`, prompts enforce method preservation plus anti-copy variation, and input
  size is measured with adaptive splitting and explicit baseline fallback. Existing QuestionBank
  reuse and independent verification remain authoritative; `REUSE_SAFE` is intentionally disabled.
  Current post-edit lint/test execution and live activation are `[NOT VERIFIED]` / `[BLOCKED]`.

- 2026-08-07: Verification-provider execution failures are now separated from
  academic verification decisions. A normalized eligible `ProviderExecutionError` from the
  independent verifier ends the current content wave without entering repair/regeneration,
  preserves all already persisted VERIFIED slots, and schedules the existing single bounded
  provider replacement for only unresolved slots. Verifier rejections continue through the
  existing semantic repair policy and are never provider fallback. The same `openai_o4_mini`
  verifier route retains `reasoning_effort: medium` with a bounded 5,000 completion-token cap;
  the previous 3,000-token cap was exhausted by real verifier reasoning output. No model,
  provider, credential, prompt, queue, table, or verification contract was changed.

- 2026-08-07: The exact local ten-question mixed Quant request was rerun after the fallback and
  verifier-boundary changes. The task acknowledged and detached correctly; a DeepSeek
  `empty_answer` selected the configured O3 fallback, and verified slots were retained. The
  assessment correctly ended `FAILED` rather than sticking in `GENERATING` because a final
  permitted `math.generator.basic` regeneration returned `STRUCTURED_PARSE_INVALID` for one
  unresolved slot. This is a separate semantic-output defect, not a fallback bypass or verifier
  transport failure. The current local ten-question READY gate is therefore `[NOT VERIFIED]`;
  no prompt or retry-policy expansion was made without systematic evidence.

- 2026-08-07: Terminal generation failures now preserve a specific safe business code when the
  bounded structured-output attempts are exhausted: `STRUCTURED_PARSE_INVALID` becomes
  `PRACTICE_GENERATOR_OUTPUT_INVALID`, rather than the generic deficit-exhaustion code. The
  parser now emits that stable reason for malformed JSON and a non-array `questions` field, so the
  underlying safe reason remains available in correlated runtime events.

- 2026-08-07: Fixed the shared `RegistryBackedModelExecutor` response-error boundary for
  practice generators. A normalized `LlmProviderResponseError` now follows the existing
  model-registry fallback chain whenever its failure kind is already in
  `FALLBACK_ELIGIBLE_FAILURE_KINDS`; an empty provider response therefore reaches the configured
  fallback instead of terminating before fallback selection. Safety, credential/configuration,
  and other non-eligible failures remain terminal. The executor emits safe primary/fallback
  attempt, selection, success, failure, and exhaustion events, correlated with the restored
  request context and the server-issued activity/batch/slot identifiers. Prompts, generated
  content, answers, raw provider payloads, credentials, and tokens are not logged.
  The route resolver's four fallback symbols are informational route alternatives; the actual
  `math_advanced_generator` provider chain is its two registry aliases, `openai_o3` then
  `openai_gpt_5_4`, and is the bounded chain executed by this shared executor.

- 2026-08-07: A provider fallback-chain exhaustion preserves all already persisted, verified
  slots. The existing bounded `PRACTICE_ITEM_RETRY_LIMIT` (0 or 1) now authorizes one persisted
  `PROVIDER_REPLACEMENT` pass for the exact unresolved slots only; it reuses the normal generator
  prompt (`replacementWave=0`) and cannot enter content repair. A further eligible provider
  exhaustion transitions the parent with `PRACTICE_REPLACEMENT_EXHAUSTED`. No full-assessment
  restart, new retry framework, queue, table, provider, prompt, or infrastructure was added.

- 2026-08-07: Planner validation now retains the first safe reason code and observed slot count
  through its single repair. `PLANNER_SLOT_COUNT_MISMATCH`, `PLANNER_DUPLICATE_SLOT`,
  `PLANNER_INVALID_DIFFICULTY`, `PLANNER_MISSING_CATEGORY`, `PLANNER_INVALID_TOTAL`, and
  `PLANNER_SCHEMA_INVALID` are emitted without storing raw planner output. A deterministic
  fallback for an explicitly mixed request with no explicit difficulty cycles generic
  Basic/Intermediate/Advanced slots; valid planner slot difficulty and its generator-route hint
  remain authoritative. The focused fallback, partial-success, planner, and routing tests pass;
  the local live ten-question gate remains `[NOT VERIFIED]` until the runtime test below is run.

- 2026-08-06: Completed the schema-v2 planner-first execution path. Exact normalized QuestionBank
  reuse fills planner slots one-to-one, deficit is the remaining immutable slot set, and compatible
  slots form adaptive homogeneous groups capped at Basic 5, Intermediate 4, Advanced 2, and
  complex Advanced Reasoning 1. Up to two immutable generation contexts execute concurrently;
  only the coordinator conditionally persists verified results. Deterministic validation precedes
  a question/slot-bound independent verifier. `REPAIRABLE` receives one repair wave,
  `REGENERATE` skips directly to replacement, and a final replacement is the last content attempt;
  provider failures never enter content repair. Finalization requires exact slot coverage and
  `INDEPENDENT_MODEL_V2` evidence. A one-shot stale recovery uses the existing AppSync progress
  mutation with `expectedUpdatedAt` CAS, retains accepted slots, and fails on a second interruption.
  The private answer-v2 contract remains authoritative through option rebalancing. Its `INDEX_V1`
  option IDs and `correctOptionId` are numeric, zero-based indexes matching the deployed player
  contract. Live Dev model
  execution is still `[NOT VERIFIED]`.

- 2026-08-06: Schema-v2 groups that share a canonical bucket now run serially. Concurrent
  generation retains the existing maximum of two groups only for distinct buckets, because groups
  sharing a bucket need the first coordinator commit in their exclusion set before the next prompt.
  This prevents cross-group duplicate questions from reaching the final manifest gate.

- 2026-08-06: Schema-v2 persistence uses `answerExplanation` as the single authoritative
  explanation for both the private answer contract and the player-facing Question explanation.
  A generated auxiliary solution cannot diverge from the verified answer explanation at the
  final contract gate.

- 2026-08-06: New planner results use canonical schema-v2 `PlannerSlot` records with one stable
  slot per accepted question, explicit subject/topic/category/exam/type/skill/variation metadata,
  and deterministic route hints. Math and Reasoning use the quant/reasoning planner, English uses
  the English planner, and remaining supported subjects use the factual planner; all routes retain
  the existing model aliases and credential path. Planner-provided bucket metadata is ignored and
  compatibility `DemandBucket` records are deterministically derived from validated slots so the
  later matching/generation orders can migrate independently. Explicit schema-v1 persisted
  blueprints remain supported. This slice does not change matching, generation, orchestration,
  repositories, AgentCore task execution, or backend contracts.

- 2026-08-06: Practice parent-state metadata no longer stores or publishes raw request text.
  Legacy `practiceRequest.querySummary` values are stripped before every AppSync progress mutation,
  while recovery retains typed practice constraints and uses the existing safe request fallback.
  Verifier-supplied reasons are reduced to `VERIFIER_APPROVED` or `VERIFIER_REJECTED` at the
  observability boundary. Exhausted replacement now uses the existing shared FAILED publisher so
  standard failed and terminal events remain reconstructable. These changes do not alter prompts,
  providers, graph topology, retries, persistence schemas, credentials, or AppSync mutation shape.

- 2026-08-04: Finalization now deterministically rebalances four-option answer positions after
  academic verification and before position persistence/`READY`. The final manifest maintains its
  exact Question IDs and stored correct-answer references, is revalidated after the permutation,
  and fails safely as `ANSWER_POSITION_DISTRIBUTION_INVALID` if it cannot meet the bounded
  distribution. The update emits only safe question-count/position-count observability and does
  not alter providers, prompts, credentials, persistence models, asynchronous execution, AppSync,
  or progress subscriptions. The focused distribution/orchestration/repository suite passes;
  authenticated Dev player verification remains `[NOT VERIFIED]`.

- 2026-08-04: An advanced-practice generator response became the typed safe failure
  `PRACTICE_GENERATOR_OUTPUT_TOKEN_EXHAUSTED` when its provider completion limit was reached.
  The current policy above supersedes the earlier direct-fallback handling with one strictly larger,
  product-capped same-model recovery and capacity-aware fallback; it never accepts truncated output
  or repeats an unchanged primary budget. Safety refusals and other non-eligible failures remain
  terminal. The existing persisted `responseType=practice_generation` and stable `practiceTestId`
  conversation linkage remain the restoration contract. Historical authenticated local-browser
  READY evidence did not include the current token policy; current lifecycle evidence remains
  `[NOT VERIFIED]` in this repository.

- 2026-08-03: The current player contract is now explicitly MCQ-only. Planner and generator prompts
  advertise only `mcq`; generated and persisted Questions must contain exactly four distinct,
  non-empty options, one matching correct option, required solution text, matching language, and
  `questionType` metadata. Reuse, persistence, final-manifest validation, and the frontend mapping
  fail closed when that contract is violated. A group transitions to `RUNNING` through the existing
  conditional `claim_group` write before its provider call, so an unclaimed retry cannot be lost by
  an in-process restart. No question type is coerced into a player shape.

- 2026-08-03: Latest authenticated local-browser evidence reached `READY`, `playable=true`, and an
  exact five-question manifest through the AppSync subscription without continuous polling. The
  subsequent live `checkQuizAnswer` operation failed with Dev Lambda code
  `QUESTION_NOT_IN_ATTEMPT` even though all five manifest Question IDs exist. Player submission,
  review, and end-to-end readiness therefore remain `[NOT VERIFIED]`; this entry supersedes any
  earlier claim of a complete browser player flow.

- 2026-08-03: Fixed the confirmed 0% practice stall at the existing AppSync
  progress-serialization boundary. DynamoDB deserializes numeric metadata as
  `Decimal`; the first post-planning progress mutation attempted to JSON-encode
  that metadata directly and raised `TypeError` before any group state or
  progress was published. The task's existing failure path then encountered the
  same serialization error and left the assessment in `GENERATING`/`QUEUED`.
  The retained client now emits JSON-safe numeric values, and a terminal
  publication failure falls back to the existing conditional assessment failure
  write so reconciliation observes `FAILED` rather than an indefinite stale
  state. Focused tests cover both paths. A real local authenticated browser
  request progressed to `READY`, persisted five questions, and exposed one
  playable test route; no provider, credential, schema, or infrastructure
  change was made.

- 2026-08-03: A tracked background invocation that returns any nonterminal outcome now fails
  closed with `PRACTICE_GENERATION_STALLED` before task completion. This prevents the known
  `RUNNING`/`DELEGATED` branch from leaving its parent assessment indefinitely `GENERATING`.
  This does not add durable cross-process task recovery; process-restart recovery remains
  `[NOT VERIFIED]` until a permitted durable ownership mechanism is available and live-tested.

- 2026-08-03: Deferred live-SSE practice task submission until the unchanged
  `practice_generation_started` frame has been handed to the ASGI transport. Initialization, task
  registration, and conversation linkage still happen first. If the terminal frame is abandoned,
  the registered task is failed instead of remaining pending; a post-commit submit failure uses the
  launcher's existing FAILED progress path and never emits a second terminal event. Direct service,
  non-stream, replay, provider, AppSync, DynamoDB, and frontend contracts are unchanged.
  Terminal delivery resolution is atomic across the worker/async queue boundary. Duplicate
  idempotent deliveries are tracked by their existing request IDs: one committed delivery protects
  the shared task, while only abandonment of the final unresolved delivery can fail a pending task.

- 2026-08-03: Added `mini mock` to the existing generic practice-artifact gate. A live local
  request had already been classified by the real shared classifier as reasoning/practice, but
  the launch gate rejected the artifact and incorrectly continued through normal answer
  generation. Duration-based mini-mock creation now uses the existing practice launcher; advice
  requests, classifier confidence, model routing, planning, generation, and verification are
  unchanged.

- 2026-08-03: Corrected local integration configuration for the practice UI. The Python runtime
  now enables practice locally, maps the already-present Google Gemini key to the existing
  `GEMINI_API_KEY` name, loads the exact practice SSM contract, and validates the retained tables
  and indexes at startup. The authenticated frontend server uses the existing Dev Runtime ARN and
  AWS profile. Deployed Runtime version 11 still predates this practice implementation; no runtime
  deployment was performed.

- 2026-08-03: Routed planner, generator, and verifier calls through the same existing
  `LlmOrchestrator` and `PromptResolver` instance used by doubt solving, removed eager optional
  provider credential preflight, and made practice enablement/endpoint/root explicit non-secret
  CDK inputs with practice disabled by default. No provider, resolver, provider SSM key, backend
  contract, or AWS resource was introduced. Deployment and live validation were not performed.

- 2026-08-02: Connected the existing PracticeGenerationGraph to the deployed
  `updatePracticeGenerationProgress` mutation using AgentCore role credentials and SigV4; split
  Question persistence from parent mutation; added authoritative monotonic manifest publication,
  bounded retry/error handling, explicit runtime configuration, two-phase tracked-task launch, and
  the terminal `practice_generation_started` SSE event. Backend/AppSync/frontend contracts were not
  changed. Dev deployment and live 1/5/50/100 validation remain `[NOT VERIFIED]` until the release
  gate completes.

## Migration history

- 2026-08-06: The initial planner-first hardening established exact difficulty reuse and adaptive
  batch caps. It was subsequently completed by the coordinator-only two-worker wave and one-shot
  parent-metadata recovery documented above; live runtime recovery remains `[NOT VERIFIED]`.

On 2026-08-01 the existing runtime gained the AgentCore tracked-task launcher, immediate existing
practice completion envelope, shared model executor, exact practice-table IAM, and safe lifecycle
events. No backend schema, API, frontend, queue, Lambda, ECR, Dockerfile, or new credential path was
added.

On 2026-08-01 the undeployed separate worker design was removed: its SQS publisher/consumer,
queue/DLQ contract, Lambda entrypoint, lease semantics, worker-only Secrets Manager resolver,
Dockerfile, immutable image tooling, and worker deployment documentation. Reusable graph,
generation, persistence, prompts, shared provider adapters, and tests were retained.
# Pattern Intelligence release-blocker closure (2026-08-10)

Pattern resolution runs only after the planner and ordinary indexed QuestionBank reuse. Deficit
slots can now consume `GUIDANCE_SAFE`, and `REUSE_SAFE` exists behind
`PATTERN_INTELLIGENCE_REUSE_ENABLED=false`. Reuse requires an explicitly Pattern-linked, verified,
platform-playable QuestionBank row plus checked cross-activity history. Newly Pattern-guided
questions acquire deterministic QuestionBank linkage only after the existing independent verifier
accepts them; the resulting QuestionBank ID is recorded on the assessment Question for later seen
exclusion. No scan or request-path N+1 was added.

Local full validation is green (2850 passed, 1 skipped; Ruff passed). The upstream QuestionBank
owner-mutation authorization is corrected and tested locally, but its Dev deployment is blocked by
unrelated dirty-worktree changes. Pattern flags remain false; the deployed Pattern-off runtime has
the restored legacy Practice IAM contract, but READY/player E2E is still not proven because live
requests were classified outside the Practice route.

## Performance producer contract (2026-08-12)

New AI Tutor assessments now persist the server-owned top-level
`assessmentMode: PRACTICE` and, when the trusted request has one, the canonical
top-level `examProfileId`. The browser does not supply this mode and no Tutor
request path currently creates `REAL_EXAM`; `FULL_MOCK` and `SECTIONAL_TEST`
remain practice-condition activities until a separately trusted launch path can
freeze and enforce the complete exam snapshot.

Verified generated questions persist the trusted runtime selection's canonical
`patternId` and `patternVersionHash` directly on the playable assessment-owned
`Question`. Generic QuestionBank reuse retains only provenance; it does not
promote QuestionBank linkage into direct Pattern identity until that source's
write boundary is deployed and proven authoritative. Missing or untrusted
linkage is left unset; no topic-based Pattern inference is performed. Focused
repository/orchestration tests cover assessment mode/profile and the direct
generated-versus-reused boundary.

The AppSync progress transport continues to project only the approved `meta`
contract; top-level performance fields are not copied into that strict mutation.
Authenticated Dev creation/player and any Real Exam path remain `[NOT VERIFIED]`.

## Strict AppSync progress-meta transport correction (2026-08-10)

The schema-v2 stall after QuestionBank matching was conclusively caused by the durable top-level
`examProfileId` assessment field being merged into the strict
`updatePracticeGenerationProgress` payload. It is not in the existing backend resolver allowlist;
the handler therefore correctly rejected the parent update with
`PRACTICE_PROGRESS_UNKNOWN_META_FIELD` before group creation or generation.

`PracticeProgressMeta` is now the single typed Python producer contract and projects only the
backend's exact top-level allowlist. New assessments no longer write the redundant top-level
`examProfileId`; the approved durable recovery copy is the existing `practiceRequest.examProfileId`.
The projection filters that top-level field from legacy assessments without removing the recovery
copy. The client retains its independent key check. Any other unapproved producer field fails
publication; terminal direct-DynamoDB fallback is limited to unavailable credentials or transport,
never an invalid progress contract. Key-only diagnostics contain only stage,
`actual_keys`, `unknown_keys`, and `contractVersion`; no metadata values, query text, answers, or
provider output are logged.

The strict backend handler remains unchanged in acceptance behavior and now emits the same safe
key-only diagnostic for a rejected field. Focused regressions cover schema-v1/schema-v2,
Pattern-off/Pattern-on, READY/FAILED projection parity, the exact five-question Reasoning/CAT
deterministic fallback through `GENERATION_GROUP_STARTED`, backend-handler rejection diagnostics,
and the no-direct-fallback invariant. Dev runtime
`practice-v2-progress-contract-20260810.5` was deployed and its container confirmed that the
projection removes top-level `examProfileId`; authenticated fresh READY/player evidence remains
`[NOT VERIFIED]` because the earlier failed activity is terminal and was not recreated under a
student-owned frontend session.

## Flexible Practice completion-cap and concise-output policy (2026-08-11)

`PracticeGenerationCapacityPolicy` is the single authority for structured Practice output ceilings.
It runs after route/model resolution and is shared by schema-v1 and schema-v2 grouping, the
structured orchestrator, and the model executor. It selects `initial <= escalation <= product hard
<= model hard` only from subject family, difficulty, complexity, slot count, and whether the
selected model uses reasoning. It contains no topic rules. Model catalog metadata supplies the
configured safe model cap (8,000 maximum in this runtime); the policy’s product cap is always at
or below that value.

Basic non-reasoning one-slot work starts at 900 tokens, can recover once at 1,500, and has a 2,200
product cap. High-complexity Advanced Math/Reasoning uses one slot per call with a 4,000 initial
cap and one 5,600-token recovery cap. Existing model `reasoning_effort` configuration is unchanged:
Advanced Math remains `high`, Advanced Reasoning remains its configured `medium`; there is no
unproven reasoning-effort reduction. The capacity values are ceilings, not expected usage or cost.
Cost/usage remains provider-reported input/completion/reasoning usage.

`generator_output_policy.md` is applied once as a generator-only shared overlay to initial,
repair, and regeneration prompts. Its two bullets rank correctness, complete conditions/schema,
and exam fidelity ahead of conciseness and require the shortest complete, information-dense output;
it never asks for minimum tokens, fixed word limits, or literal two-line answers. The verifier,
PatternGraph constraints, independent verification, and playable-question contract are unchanged.

OpenAI/Azure, OpenAI-compatible/DeepSeek, Gemini, and mock adapters now retain the raw provider
finish reason while mapping it to one internal outcome: `completed`, `output_token_exhausted`,
`empty_response`, `content_filtered`, `provider_failure`, or `unknown`. A structured Practice
`OUTPUT_TOKEN_EXHAUSTED` response is never parsed, verified, persisted, or delivered. The executor
performs at most one same-model recovery at a strictly larger cap; if still exhausted, it selects
only a fallback that can send the escalation budget, otherwise fails in the existing bounded
provider path. It does not repeat an identical budget or add an independent retry loop.

Safe per-call telemetry records route/model/provider, subject/difficulty/complexity/slot count,
configured initial cap, actual cap sent, product cap, provider-reported usage, raw/normalized finish
reason, fallback and escalation status. It never records prompts, questions, answers, or reasoning
content. Contract validity and verifier outcome remain generation-stage evidence rather than being
claimed by provider-call telemetry.

Focused offline validation covers basic versus high-complexity capacity selection, the provider
parameter sent on primary/escalation/fallback, no identical-cap retry, insufficient-cap fallback
skipping, finish-reason normalization, prompt priority, Pattern prompt regression, and existing
academic contracts. Comparative live quality/cost acceptance data and authenticated Dev Practice
lifecycle evidence remain `[NOT VERIFIED]`; no Production deployment is authorized by this slice.

## Practice generation recovery and final-manifest hardening (2026-08-12)

The failed three-question Reasoning replay was caused after academic acceptance, not by the
planner or verifier. Its schema-v2 slots shared subject, topic, difficulty, and question type but
had two different category/route compatibility signatures. The slot-to-bucket lookup ignored the
category and route fields and consequently persisted every approved question as
`slot-bucket-001`; the final gate correctly rejected the 1/2 planned distribution as
`BLUEPRINT_DISTRIBUTION_MISMATCH` after 12 model calls. Slot-to-bucket binding now reconstructs the
same ordered six-field signature used by `PracticeBlueprint` bucket derivation and verifies the
derived bucket's required count before generation.

The existing content retry boundary remains exactly three waves per unresolved slot: initial,
one repair, and one fresh replacement. Accepted sibling slots remain immutable and are never sent
again. A rejected, schema-valid candidate and up to four stable verifier reason codes are now sent
only in that slot's `repair_context`; malformed output has no candidate and is recreated from the
immutable slot contract. Before a fresh replacement, rejected wording is added to the bounded
exclusion set. Provider failures retain their separate provider-fallback path and never enter
content repair.

Final validation now returns `ready`, `reason_code`, `failed_slot_ids`, expected and actual counts,
and `recoverable`. The only final recovery implemented is the proven bookkeeping defect: on the
first `BLUEPRINT_DISTRIBUTION_MISMATCH`, verified Question metadata is conditionally corrected by
question ID, assessment ownership, immutable slot ID, and verified status. The coordinator then
recalculates authoritative parent metadata and runs the complete final validator once more. A
conflict, incomplete update, second validation failure, missing/duplicate Question, invalid
answer, verification failure, ownership mismatch, or other authoritative corruption remains a
typed terminal failure. Recovery performs no LLM call and cannot weaken academic validation.

Safe lifecycle events cover question repair/replacement and manifest recovery start, completion,
and failure. They contain only IDs, attempt numbers, counts, reason codes, and recoverability; raw
questions, options, answers, solutions, prompts, and provider payloads remain excluded. No public
schema, graph operation, route, model, token ceiling, Pattern behavior, billing/credit behavior,
AppSync acceptance contract, infrastructure, or production deployment changed.

Offline evidence includes the exact two-category distribution regression, conditional repository
write contract, a first-pass recovery plus second-pass READY integration, slot-scoped real provider
payload inspection, bounded repair/replacement regressions, and a deterministic 32-case matrix
across Math/Reasoning, basic/advanced, single/multi-topic, and 3/5/10/20 question manifests.
The semantically equivalent self-contained local replay created
`practice-e1f748264ff1d978b02d59b695b6cc0e` and reached `READY`, `live=true`, with three unique
schema-v2 slots, mandatory independent verification, and exact bucket count. Slot 003 required and
completed the sole repair wave; accepted siblings were preserved. It used 10 model calls, 8,476
input tokens, and 10,008 output tokens, compared with 12 calls, 10,113 input tokens, and 21,663
output tokens in the reported failed run. The literal wording remained blocked by the independent
context gate when replayed without its historical source turn, so literal-query parity is
`[NOT VERIFIED]`. No Production deployment was performed or authorized.

## Cancel, crash recovery, and same-test resume (2026-08-13)

Practice generation now has a durable execution lease on the existing `MockTestQuiz` item. Each
attempt claims a fresh `activeExecutionId`, lease expiry, start time, attempt number, and resume
reason by conditional AppSync update. Only that execution may renew the lease, publish progress,
delete invalid resume rows, or persist a Question. Question persistence uses one DynamoDB
transaction that checks the parent execution fence before the idempotent put. The runtime role has
only the additional `TransactWriteItems` and `DeleteItem` actions on the existing assessment and
Question tables; no table, queue, workflow, checkpoint store, or runtime was added.

Authenticated cancel is cooperative and operation-scoped. The Next.js server derives the Cognito
owner and invokes `practice_control`; the Python runtime persists `cancelRequested` before signaling
the matching in-memory execution. Costly planner, generator, repair, replacement, and verifier
boundaries recheck cancellation and renew the coarse lease. A question already independently
approved when cancellation arrives is persisted and checkpointed before the worker stops. The
shared AgentCore session is never terminated because it may also own unrelated conversation work.
The terminal assessment is `FAILED` with phase `CANCELLED` and `USER_CANCELLED`; repeated cancel is
idempotent and READY assessments are not cancellable.

Resume always reuses the same assessment/test ID and immutable persisted blueprint. It strongly
reads existing schema-v2 Questions through the canonical playable-question validator, preserves
valid verified slots, deletes only exact invalid rows under the new execution fence, reconstructs
authoritative counts, and resets only incomplete or invalid groups. An active unexpired lease
blocks concurrent resume; an expired process can be replaced without any in-memory state. The old
process is fenced from all subsequent Question/progress writes. Resume remains unavailable for
non-resumable validation and contract failures, and the unchanged final validator is still the only
path to READY.

Safe events cover claim, cancel request/observation, resume reconstruction, stale fencing, and
execution release. They include identifiers, counters, states, and reason codes only. Focused
offline tests cover repeated cancel, cancellation boundaries, paid-approved persistence, active
lease contention, zero-RAM reconstruction, invalid-slot-only regeneration, stale-executor write
rejection, sequential registry cleanup, and unchanged final validation behavior. Authenticated
Sandbox cancel/process-crash/resume with real provider calls, measured latency/cost, and Production
deployment remain `[NOT VERIFIED]`.

## Strict cancellation cost guard (2026-08-13)

Practice now supplies one optional execution-scoped callback to the existing shared structured
model executor. The callback reuses `require_expensive_work_allowed`: it checks the process-local
cancel signal first, then conditionally renews the existing durable execution lease. The shared
executor has no Practice import, test ID, execution ID, or persistence knowledge. With no callback,
its invocation and retry/fallback behavior are unchanged.

The callback runs immediately before the one allowed same-model capacity escalation and before
each configured provider/model fallback. A `PRACTICE_CANCEL_OBSERVED` or
`PRACTICE_EXECUTION_FENCED` signal propagates as operation control flow and cannot be normalized as
a provider failure or trigger another fallback. An already-fired synchronous provider request may
finish and remain billable. No new provider call begins after its result reaches either guarded
retry boundary.

Existing graph boundaries remain authoritative and were not duplicated: generation waves,
verifier calls, repair/replacement waves, and subsequent generation groups already invoke the same
execution-control check. If cancellation arrives during an in-flight verifier, an approved result
is still authoritatively persisted and checkpointed before stopping; an unverified generator
result is not persisted merely for cancellation. Resume continues to reuse only verified,
persisted Questions under the existing fresh execution claim and final validator.

Focused offline regressions prove cancelled and superseded retry blocking, active-guard
escalation/fallback compatibility, explicit `attempt_guard=None` compatibility, typed signal
propagation through structured orchestration, no verifier after a generator completes under
cancellation, and approved in-flight verifier persistence before the next slot is blocked. The
guard adds no work to a successful primary call. On an active retry path it can add one existing
conditional lease-renewal write immediately before each new expensive attempt; a local cancel is
rejected without that write. No new dependency, module, state store, infrastructure, provider
adapter, route, model, token cap, prompt, Pattern behavior, billing rule, player behavior, or
public schema was added. Authenticated Dev/local real-provider UI cancellation evidence remains
`[NOT VERIFIED]`; no Production deployment is authorized by this change.

An authenticated local UI run created
`practice-d13016fb850913418f4a654a32d14449` at `2026-08-13T17:49:38.661672Z`.
Cancel was pressed at `17:50:03.679Z`; the UI showed `Stopping generation` on the next sampled
state and the durable item reached `CANCELLED` / `USER_CANCELLED` / `CANCEL_COMPLETED` at
`17:50:22.080834Z`. A keyed Question-index query returned zero persisted Questions, and the UI
offered same-test Resume. The local background worker's provider-attempt event stream was not
retained in the readable request log, so provider completion time, exact post-cancel attempt
counts, and token savings could not be independently reconstructed. Therefore the strict
real-provider event-level acceptance gate remains `[NOT VERIFIED]` and release remains
`NOT_READY` despite the successful authenticated UI lifecycle check.

## Question-language integrity (2026-08-15)

Practice delivery now resolves one canonical `english`, `hindi`, or `hinglish` value. A validated
request field remains the normal source; an unambiguous current query command such as `in Hindi`
or `हिंदी में` overrides it only for Practice generation. The resolved value and source are stored
inside the existing `practiceRequest` metadata, included in the idempotency key, passed unchanged
to planner/generator/verifier routing, and emitted as the safe `practice_language_resolved` event.

QuestionBank reuse requires explicit matching language metadata. A missing legacy language is
`UNKNOWN`, not English, and is excluded from an explicit-language reuse path without a runtime LLM
backfill. New assessment and Question persistence map canonical values to the existing backend
codes `en`, `hi`, and `hinglish`; newly persisted private activities validate that top-level value
before delivery. Historical private resume metadata lacking `languageSource` remains compatible
with its prior validated Question metadata, so this hardening does not mutate old attempts.

Generator, repair, replacement, and verifier prompts state that language applies to question,
options, answer explanation, and solution. Before verifier acceptance a bounded script signal
rejects clearly incompatible English/Hindi/Hinglish output; the existing bounded repair/replacement
flow handles the deficit and never publishes a partial set. Hindi delivery retains an English-subject
exception for textual grammar content. Current-affairs evidence source language remains independent
from requested delivery language. No PatternGraph identity, retrieval index, cancellation fence,
same-test resume, ExamProfile, Student Performance, outbox, provider, queue, or Production
deployment changed. Focused offline tests pass; authenticated three-language browser/provider E2E
is `[NOT VERIFIED]`.

## Exact count and fresh-evidence hardening (2026-08-16)

`MAX_PRACTICE_QUESTIONS` in `app/practice_limits.py` is the canonical
Practice count authority. Explicit requests accept only `1..100`; `0`, negative values, and
values above 100 return the typed `PRACTICE_REQUEST_COUNT_OUT_OF_RANGE` failure rather than
silently clamping. `requested_count` and `accepted_count` must be equal for every accepted
Practice request. The count travels unchanged into the blueprint, expected slots, persisted
assessment request, and final manifest. `PRACTICE_GENERATION_GROUP_MAX=5` remains an internal
bounded generation cap, never a total-assessment cap; the route capacity policy may use smaller
groups (for example, five groups of four for a 20-slot intermediate set) without dropping slots.

Fresh Practice now has an explicit deterministic temporal mode in the existing web-search policy:
`CURRENT`, `LATEST`, `RECENT`, `EXPLICIT_YEAR`, `EXPLICIT_MONTH`, or `EXPLICIT_DATE_RANGE`.
Explicit year/month/range intent takes precedence over generic current wording; latest/current
windows end on the runtime date. The existing source-policy and query-builder layers add temporal
coverage, competitive-exam context, and source-pack topic semantics to the provider-neutral request;
the classifier still supplies semantic web demand only. Current-affairs packs already use Tavily's
`news` topic. The Tavily adapter uses the documented Bearer authorization header and only the
supported existing controls: topic, `start_date`, `end_date`, `time_range`, domain filters,
`max_results`, and `search_depth`.

For a freshness-required request, candidates missing a parseable publication date or outside the
resolved date window are rejected **before** reranking. URL de-duplication then occurs while
constructing the compact evidence bundle. Generator, repair/replacement, and verifier receive the
same explicit `evidence_by_slot` subset; verifier approval still requires a cited selected URL and its prompt
rejects stale or unsupported facts. Insufficient qualified evidence fails before Practice launch;
no model-memory fallback is allowed. The web path records only safe counts, temporal mode/window,
date range summaries, and stale-rejection counts, never page bodies or prompts.

The normal provider limit remains 20 results for one Tavily call. A 100-slot fresh request performs
one bounded retrieval and fails closed if it cannot provide 100 independent eligible evidence items;
it never fans out to 100 searches. Deterministic tests can supply a complete 100-item evidence bundle
to verify count and slot contracts without paid provider calls. A successful real-provider 100-slot
fresh assessment is `[NOT VERIFIED]`; no source-category expansion or additional search service was
introduced. Static math, reasoning, and grammar requests retain zero web-search calls.

Focused offline evidence: count, freshness, Tavily adapter, query-builder, reranker, orchestration,
async, cancellation/resume, Pattern context, billing, and routing suites passed on 2026-08-16.
A live Tavily current-affairs call returned no dated, policy-eligible evidence and the fresh gate
failed closed; successful real-provider fresh generation remains `[NOT VERIFIED]`. The Azure smoke
is blocked by its endpoint/API-mode mismatch and the native OpenAI fallback by `RateLimitError`;
therefore end-to-end real LLM generation is `[NOT VERIFIED]`. No Production deployment is authorized
by this change. PatternGraph, cancellation/resume fencing, billing, provider/model routing, token
budgets, player/scoring contracts, and backend data schemas were not modified.

## Hyphenated-count parsing fix (2026-08-16)

Production incident: a request phrased as `"Create a 20-question ... Quick Practice ..."`
(count directly hyphen-attached to the unit word, e.g. `20-question`, not `20 question`)
resolved to 5 questions instead of 20. Root cause: `_COUNT_PATTERN` in
`app/features/practice_generation/planning.py` required literal whitespace between the
digits and the trailing unit keyword (`question`, `quiz`, `mock`, etc.). A hyphenated
compound never matched, so `resolve_requested_count` fell through the word-number
dictionary (no match — the count was numeric, not spelled out) straight to
`_DEFAULT_COUNTS[practice_type]`; since the query said "Quick Practice" and matched no
other type keyword, `practice_type` resolved to `QUICK_PRACTICE`, whose default is 5. This
predates the "Exact count and fresh-evidence hardening" work above — the `max(1, ...)` →
`int(...)` clamp change there did not touch this defect, since it only fires once a count is
already extracted. The existing test at `test_practice_core.py` covering
`"Create a 5-question Quick Practice ..."` had masked the bug for years because 5 (the
QUICK_PRACTICE default) happened to equal the intended count.

Fix: `_COUNT_PATTERN` now matches the digit-hyphen-unit compound directly
(`\d{1,3}-question\b`, no intervening words) as an explicit alternative, alongside the
original whitespace-separated form; the "search up to 3 words ahead" lookahead itself
still requires whitespace, so unrelated hyphenated compounds (`"10-minute mini mock"`)
are not misread as a count. `sum(bucket.required_count) == accepted_count` was already
enforced at the blueprint layer (`schemas.py`), and `GenerationGroup.required_count` was
already a separate, independently bounded (≤5) generation-batch concept — total-count vs.
batch-size separation required no change.

The reported Practice-ID mismatch between the conversation turn and the displayed activity
summary (`getPracticeActivitySummary`) could not be investigated from this repository:
`app/main.py` threads a single `practice_test_id` synchronously from launch through to both
the persisted conversation turn and the response payload, so this backend cannot itself
produce two different IDs for one request. The summary query, card-to-conversation
association, and any UI/query-variable selection are owned by the separate AI-tutor-backend
(Next.js/Amplify) repository and remain `[NOT VERIFIED]` here.

Regression tests added to `test_practice_core.py`: hyphenated counts at 1/5/20/50/100, the
exact reported multi-topic query, and the unspecified-count default. Focused
`practice_generation` suite (345 tests) and `ruff check` on the changed files pass. Full
`make check` was run; 5 unrelated pre-existing failures in `test_azure_first_fallback.py`,
`test_difficulty_classification.py`, `test_intent_overlay.py`, and
`test_orchestrated_streaming.py` (stale `OrchestratedDoubtSolverState` field-set assertions
missing `fresh_evidence`) predate this change and belong to the in-progress Doubt Solver
freshness work already in the working tree — not touched, per scope.

## Request Intelligence for free-text Practice requests (2026-08-28)

### Why

The classifier reports one broad `topic` per request, and `resolve_practice_request`
derives count, mixed/explicit difficulty and delivery language from regexes. Nothing in
that path could read a request such as `"ratio proporton aur percentge ke 50 questions,
10 easy 30 medium 10 hard"`: every slot went to the single classifier topic and the
explicit per-level counts were invisible. An earlier attempt resolved topics by matching
the query against `ExamProfileSection.topics`; it was abandoned, and its
`topic_resolution.py` module, the `ExamProfileSection.topics` field, and the ExamProfile
vocabulary lookup in `doubt_solver_graph.py` were reverted with this change so no second
dormant multi-topic architecture remains. The proven parts of that work —
`PracticeGenerationRequest.topics`, multi-topic `_requested_topic_ids`, feasibility-aware
slot-topic coverage, and the verifier `errorClass` diagnostic — were kept.

### Shape

```text
free-text request → existing classifier (unchanged)
→ resolve_practice_request  (deterministic count / type / language)
→ Request Intelligence      (interpretation only)
→ deterministic validation  (grounding, bounds, arithmetic)
→ existing deterministic planner → reuse / deficit / generation / verification
```

The model **interprets**; deterministic code **validates, distributes and executes**. The
model never allocates questions across topics, never chooses a mixed split, and never
overrides the resolved count: 50 questions over 11 interpreted topics is still divided
5/5/5/5/5/5/4/4/4/4/4 by `deterministic_blueprint`.

### Contract and route

`app/schemas/practice_request_intelligence.py` owns the shared contract and the single
static JSON Schema `practice_request_intelligence_v2` (interpretation status, requested
count, grounded topics, and one mutually exclusive difficulty object). It is shared
because both the provider adapter and the Practice feature read it. The schema is static
and versioned because Bedrock compiles and caches a structured-output grammar per schema;
it uses only Bedrock-supported constructs — no `minimum`/`maxItems`/`maxLength` — and all
business bounds are enforced afterwards in Python.

Route `general.request_intelligence.default` → model alias
`practice_request_intelligence` → provider `bedrock`, profile `bedrock_apsouth1`, model
`zai.glm-4.7-flash`, standard on-demand tier, `fallback: []` and no `fallback_models`.

`app/services/llm/providers/bedrock_provider.py` is a normal `ProviderAdapter` in the
existing factory. It calls Converse with
`outputConfig.textFormat = {type: json_schema, structure.jsonSchema}` — there is no
"return JSON" instruction, no markdown fence stripping and no JSON-repair retry anywhere
on this path. Authentication is the runtime's existing AWS IAM identity: no API key, no
new secret, no SSM or Secrets Manager entry. `ProviderProfile.region_env` /
`ProviderCredentials.region` were added so the region (`BEDROCK_LLM_REGION`, optional,
defaults to the SDK region) travels through the existing credential seam; the region is
not a secret and appears in `safe_metadata()`.

### Deterministic validation and failure policy

`app/features/practice_generation/request_intelligence.py` rejects any interpretation
that: disagrees with the deterministically resolved count, exceeds the supported count
bounds, reports a topic whose `sourceText` is not present in the student's own query,
returns more than `MAX_INTERPRETED_TOPICS` (12) topics, carries topics on a `BROAD`
status, or states per-level counts that do not sum to the requested count. Duplicate
topics collapse using the same `&`/punctuation folding the planner's own topic
canonicalization applies.

Every failure mode — no interpreter configured, provider unavailable, malformed payload,
failed semantic validation, or an `AMBIGUOUS` interpretation — resolves to `None` and
Practice continues on exactly the pre-existing deterministic resolution. **There is no
semantic retry**: the model is never called a second time to obtain a preferred answer,
and infrastructure retry policy is unchanged. Both outcomes emit
`practice_request_intelligence_resolved` / `practice_request_intelligence_unusable` with
a reason code and no user text; token usage, latency and cost flow through the existing
`ProviderAdapterExecutor` telemetry.

### Invocation and bypass

A caller that already holds trusted structured constraints passes `topics=` to
`resolve_practice_request` and the interpreter is not consulted at all — no model call.
Free-text Practice creation (both the non-streaming graph node and the streaming service)
passes `request_interpreter=`; the interpreter is injected once at the composition root in
`app/main.py`. Non-Practice routes never reach the seam.

### Durability

`topics` and `difficultyDistribution` are now persisted in `practiceRequest` meta and
rehydrated in `orchestration._request`. Without this, a resumed multi-topic Practice would
rehydrate a request whose topic set no longer matched its persisted blueprint and fail
`apply_system_bucket_policy` coverage validation.

### Stated limitation

`difficulty_distribution` is honoured by `deterministic_blueprint`, which serves
QUICK_PRACTICE (the default path) and every request of 2 or fewer questions. The LLM
blueprint planner used for QUIZ/TOPIC_TEST/SECTIONAL_TEST/FULL_MOCK is unchanged in this
round and does not receive the explicit per-level counts; interpreted **topics** do
constrain it, through the existing `apply_system_bucket_policy` coverage check.

A consequence to weigh in the next round: the planner payload in `providers.py` still
carries only `request.topic`, so on those practice types an interpreted multi-topic
request produces a blueprint the coverage check rejects, costing the one bounded repair
attempt before the deterministic fallback produces the correct multi-topic plan. The
outcome is correct either way; the wasted planner call is not. Deliberately not changed
here — it needs a planner payload and prompt decision, which is outside this round.

### Verification

58 new tests in `tests/practice_generation/test_request_intelligence.py` and
`tests/test_bedrock_provider.py` (classifier freeze, structured bypass with zero model
calls, free-text seam, broad/single/multi-topic, mixed and custom difficulty, invalid
arithmetic, ungrounded and duplicate topics, ambiguous handling, EN/HI/Hinglish fixtures,
Converse request shape, response deserialization, failure-kind mapping). `ruff check`
clean, `agentcore validate` Valid, full suite 3621 passed with the same 7 pre-existing
failures recorded above plus the two timing-sensitive `test_exam_profile_cache` stress
cases (both pass in isolation). Mocked fixtures prove the contract and the wiring only —
**the real model's semantic quality is `[NOT VERIFIED]`** and requires live qualification.

## Request Intelligence v2 — count authority and difficulty exclusivity (2026-08-28)

Two contract defects found during live GLM qualification are closed here. The model
choice is unaffected: `zai.glm-4.7-flash` remains `MODEL_QUALIFICATION_FAILED` on
semantic grounds (Hindi topic accuracy 8.3%), and this work is what a *replacement*
model will plug into.

### Difficulty is mutually exclusive by construction

The v1 contract carried three flat fields (`difficultyMode`, `singleDifficulty`,
`difficultyDistribution`) and let deterministic code reject meaningless combinations
afterwards. In live qualification the model emitted a non-null level in **50/50** calls,
28 of them alongside `difficultyMode: UNSPECIFIED`, so 78% of interpretations were
discarded and Practice silently fell back. Expressing the field as a null-typed union
(`{"type": ["string","null"], "enum": [...]}`) did **not** change that — proven by a
5-call live check. The flat shape itself was the problem.

`practice_request_intelligence_v2` replaces those three fields with one `difficulty`
object built from a closed `anyOf` branch per mode, each pinned by `const` and each
`additionalProperties: false`:

```text
{mode: UNSPECIFIED}
{mode: SINGLE,  level: BASIC|INTERMEDIATE|ADVANCED}   level required
{mode: MIXED}
{mode: CUSTOM,  distribution: {basic, intermediate, advanced}}   distribution required
```

SINGLE-without-level, MIXED-with-level, UNSPECIFIED-with-level, CUSTOM-without-
distribution and CUSTOM-with-level are now unrepresentable in the grammar rather than
rejected after the fact. The Pydantic side mirrors it exactly — a discriminated union
whose branches set `extra="forbid"` — so the deserializer refuses anything the schema
could not have produced. Deterministic code still owns everything the grammar cannot
express: count bounds, non-negative counts, and custom-distribution arithmetic.

The version moved because the wire contract genuinely changed shape. AWS documents
grammar caching for identical schemas; there is no evidence the schema *name* alone owns
cache identity, and the rename is not a workaround for one.

### A default count is not evidence of intent

`resolve_requested_count` previously collapsed "the student wrote no count" into the
practice-type default, and the validator then compared that default against the model's
reading — rejecting 15 of 50 live cases where the model was right and the parser was
blind (Devanagari units, typo'd units, a total implied only by a distribution).

`explicit_requested_count()` now returns `None` when nothing was written; the default is
layered on top in `resolve_requested_count`, whose behaviour is unchanged for every
existing caller. **No regex was added or modified, and no language-specific parsing
exists.** Precedence in `_resolve_practice_count`:

1. a trusted structured count supplied by the caller;
2. the interpretation's count (or a CUSTOM distribution's deterministic sum);
3. a count the parser demonstrably found;
4. the practice-type default.

Only a real explicit count can contradict the interpretation — a mismatch drops the whole
interpretation rather than silently preferring either value.

### Known ordering defect — reported, not fixed

`DOES_PRE_INTELLIGENCE_EVIDENCE_WORK_DEPEND_ON_REQUESTED_COUNT = YES`. For
freshness-sensitive Practice, `_orchestrated_collect_context_node` sizes web-evidence
retrieval via `required_fresh_evidence_count(query)` **before** interpretation runs, so a
request whose count the parser cannot see retrieves evidence for the default — measured:
`"मुझे हाल की current affairs पर 30 सवाल दो"` retrieves 10 facts for a 30-question ask,
while the English equivalent correctly retrieves 30. `_resolve_practice_count` therefore
pins the count to the deterministic value on that path, so generation can never outrun
the evidence actually gathered. That is a safety guard, **not** intended semantics: the
real correction is to interpret before sizing retrieval, which is a graph-ordering change
across two call sites and deliberately out of scope for this round.
