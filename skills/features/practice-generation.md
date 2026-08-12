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
