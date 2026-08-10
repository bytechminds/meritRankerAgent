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
  schema-v2 slot validation, bounded repair, and deterministic slot fallback for allowed
  assessment types. Explicit schema-v1 persisted blueprints remain readable.
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

Supported settings:

- `PRACTICE_GENERATION_ENABLED=false`
- `APPSYNC_GRAPHQL_ENDPOINT=` (required when enabled)
- `PRACTICE_PATTERN_CONTEXT_ENABLED=false`
- `PRACTICE_RESOURCE_PARAMETER_ROOT=/meritranker/agent-runtime/v1/practice`
- `PRACTICE_GENERATION_GROUP_SIZE=3`
- `PRACTICE_GENERATION_GROUP_MAX=5`
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

The local AgentCore CDK definition grants exact access to the 11 practice SSM parameters, the three
existing table ARNs, and only the named Question/QuestionBank indexes. MockTestQuiz direct access is
limited to `GetItem`/`PutItem`/conditional `UpdateItem`; Question access uses direct `PutItem`,
bounded query/batch reads, and position updates. When the existing deployment-time non-secret
practice flag is enabled, it grants only
`updatePracticeGenerationProgress` on the configured AppSync API—never an API wildcard. The
currently deployed Dev runtime remains drifted from this source and was not changed by this task.
No secret, queue, credential resolver, or wildcard resource was added.

## Tests

The focused suite covers routing and the controlled safe gate; graph imports and neutral command
routing; planning, reuse, generation, verification, replacement, persistence, exact manifests,
resource contract/validation, shared credentials, duplicate graph operations, restart behavior,
GSI lag, counter/ownership failures, deterministic option rebalancing, and repeated
5/50/100-question in-memory runs.

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
- Current-affairs set creation remains unsupported.
- Pattern context remains disabled and fails closed when force-enabled.
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

- 2026-08-04: An advanced-practice generator response becomes the typed safe failure
  `PRACTICE_GENERATOR_OUTPUT_TOKEN_EXHAUSTED` when `finish_reason=length` consumed the configured
  completion and reasoning budget. Fallback eligibility is decided by the shared provider failure
  classification; safety refusals and other non-eligible failures remain terminal. The actual
  executing model alias is retained for
  generated-question provenance. Exhausting that chain marks the parent activity `FAILED` without
  entering validation repair, replacement, or an unchanged primary retry. Advanced Reasoning now
  requests one question per provider call; other subject group sizes are unchanged. The existing
  persisted `responseType=practice_generation` and stable `practiceTestId` conversation linkage
  remain the restoration contract. An authenticated local-browser run reached `READY`, restored its
  persisted card after reload, and opened the five-question instruction route. External frontend
  batch-reconciliation request counts remain `[NOT VERIFIED]` in this repository.

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
