# Conversation History

## Status

Dev infrastructure, persistence, Memory-first selection, exact-conversation DynamoDB fallback, and
independent-query isolation were previously live verified. The 2026-07-25 conditional-context
implementation preserves those storage boundaries and makes reads conditional.

## Latest Changes — Post-answer finalization reliability (2026-08-16)

- `conversation_persistence_started` is a registered structured observability event. The existing
  persistence coordinator can therefore enter its worker phase without an observability-whitelist
  `ValueError` after a verified answer has been streamed.
- A narrow post-answer failure diagnostic records only request-scoped lifecycle stage, exception
  type, source-frame coordinates, answer-emitted state, and persistence boundary flags. It never
  serializes the exception text, question, answer, prompt, JWT, or secret; local DEBUG adds only
  sanitized frame coordinates.
- The exact real-model blood-relation regression completed with one terminal `complete` frame after
  History and Session succeeded. Local AgentCore Memory remained `failed_configuration`, which is
  reported as partial persistence without replacing the accepted answer.

## Conditional-context boundary (2026-07-25)

AgentCore Memory and DynamoDB remain context sources only. `ContextNeedGate` decides only whether a
read is skipped, required, or uncertain. Required/uncertain requests load Memory first and use the
exact-conversation DynamoDB fallback on controlled empty/failure. Deterministic hygiene and compact
card building do not select the final turn.

The existing query classifier owns both academic classification and the minimal
relation/action/selected-ID decision. Only one validated selected turn reaches generation.
Root-lineage persistence, frontend reference IDs, summaries, semantic recent-turn retrieval, and
new storage fields remain excluded.

External pronouns and indirect references are now validated against the same bounded clean candidate
cards after Memory/DynamoDB loading. This does not change retrieval order, read limits, fallback
behavior, persistence eligibility, History/Session/Memory schemas, or storage writes. The optional
resolved-reference string exists only in the internal generation context and is never persisted.
The classifier projection is capped at 3,200 formatted characters: latest question/answer clues use
450/300 characters and older clues use 350/240. Read counts, clean-turn eligibility, grounded
compatibility, and final selected-turn validation are unchanged.

Deterministic hygiene rejects correction/re-solve meta-query pairs from future candidates even when
the stored answer is substantive. Their History/Memory write contract is unchanged; the filter
prevents a later “solve it from scratch” request from treating “Your answer is wrong” as the
original academic problem.

Generic correction/re-solve wording beginning with `previous`, `last`, or `the previous/last`
selects only the newest already-hygienic substantive candidate. For that narrow case, generic
reference compatibility cannot reject the newest turn before the correction rule runs; selecting
an older turn remains invalid. This changes no read count, Memory hygiene, persistence, or storage
contract.

## Current behavior

Persistable finalized turns are stored in the Amplify-generated `ConversationHistory` table and in
AgentCore short-term memory. DynamoDB recent-context fallback queries the
`ConversationHistoryByConversation` GSI by `conversationId`, applies actor isolation, requests newest
items first, and returns bounded chronological context.

After history persistence succeeds, `ConversationSessionRepository.upsert_from_completed_turn()`
creates or updates the corresponding lightweight session in the generated `ConversationSession`
table. First create writes the conversation ID, actor ID, deterministic query title,
`lastActivityAt`, Amplify timestamp attributes, and typename. Later updates only advance
`lastActivityAt`; title is immutable and activity cannot move backward.

Title generation normalizes whitespace, removes control characters, preserves Unicode, caps output
at 60 code points with an ellipsis, and uses `New conversation` only when no usable query remains.
No model call is made.

Each persistence decision returns a typed `ConversationPersistenceResult` with independent history,
session, and memory statuses. Accepted statuses are explicitly limited to `checked` and
`passed_quality_gate`; empty, language-noncompliant, failed-quality, cancelled, non-finalized, and
clarification-only outputs are skipped. One safe structured summary is logged per decision.

The shared observability layer also emits typed persistence start, completion, skip, or partial-failure
events and updates the request execution summary with history, session, and memory statuses.
Recent-context loading emits source, usable-turn count, latency, fallback, and controlled failure
reason only. Persistence worker threads explicitly copy request context. Questions, answers,
formatted recent context, memory event payloads, and DynamoDB item bodies are never telemetry.

## Configuration

The existing process-cached SSM loader reads history and session table names and ARNs in one
`GetParameters` request. The runtime bootstrap builds available repositories with the shared cached
DynamoDB client. AWS Region is validated before SSM access. The Memory ID is independent and
optional: when absent, history and session remain active, no AgentCore client is constructed, and
Memory reads/writes return controlled `not_configured`/`failed_configuration` outcomes. No second
config loader, per-request SSM read, or graph-level DynamoDB call exists.

The AgentCore CDK stack creates one target-scoped short-term Memory with 30-day raw-event retention
and no long-term strategies. Its AWS-compatible physical name is
`meritranker_short_term_memory_<target>`; the service schema does not permit the requested hyphenated
form. The generated Memory ID is injected into the Runtime with
`MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID`, creating an implicit CloudFormation dependency.
Staging and production receive independent resources when their deployment targets are configured;
only Dev is currently configured and deployed.

The installed AgentCore alpha CodeZip packager copies source files without honoring `.gitignore`.
The CDK app therefore stages a deployment-only source tree under `agentcore/.cache`, excluding all
`.env` variants, tests, caches, bytecode, and logs. Contract tests cover this exclusion. The
runtime package includes AWS Distro for OpenTelemetry because the enabled instrumentation entrypoint
requires `opentelemetry-instrument`.

## Reliability

History and memory persistence are independent and may execute concurrently. History must succeed
or be an idempotent replay before session persistence is attempted. Each component retries once only
for recognized transient AWS/network failures. A remaining partial failure is reported explicitly
without replacing an accepted answer. Memory success with history failure emits a degraded-consistency
metric. Equal timestamp retries are no-ops. Cross-actor conversation ID collisions raise a repository
conflict. Streaming emits public terminal success only after persistence coordination returns.
Every `complete` and `practice_generation_started` terminal event now includes safe
`request_id`, `conversation_id`, `turn_id`, and `persisted` metadata. `persisted` is true only when
the exact history write is `succeeded` or `idempotent_replay`; failed, skipped, unavailable, and
clarification persistence never claim confirmation.
The persistence deadline bounds request coordination, not an already-running synchronous AWS SDK
call. A timed-out worker is marked `failed_transient`, may finish later, and is never treated as
confirmed persistence for the terminal result.

Recent context uses a typed load result with source `agentcore_memory`, `dynamodb_fallback`, or
`none`, plus `memory_attempted`, `memory_status`, `memory_event_count`, `dynamodb_attempted`,
`dynamodb_status`, `dynamodb_item_count`, bounded turns, usable count, latency, and a safe failure
reason. Separate `memory_failure_reason` and `dynamodb_failure_reason` fields preserve both store
outcomes when fallback also fails. Memory distinguishes not configured, resource missing,
permission denied, timeout, no events, malformed events, actor/session mismatch, and no completed
turns. DynamoDB distinguishes not configured, permission denied, query failure, no items,
actor/conversation mismatch, and no completed turns. DynamoDB fallback remains an
exact-conversation GSI query and validates actor and conversation identity before returning data.

Clear standalone requests perform no Memory or DynamoDB history read. Contextual/uncertain requests
normally load at most three completed pairs. An explicit correction/re-solve request may
exceptionally load up to five pairs so the original substantive question remains selectable through
a short correction chain; no larger read is allowed. AgentCore Memory remains primary; controlled
empty/unavailable/malformed results use the exact-conversation DynamoDB fallback.

AgentCore Memory and DynamoDB transport exceptions are normalized into typed, content-free
failures. Memory transport failure proceeds to DynamoDB; DynamoDB transport failure returns a
controlled unavailable-context result rather than escaping the boundary.

The readable local log records gate decision, Memory/DynamoDB attempts, statuses, counts, returned
turn IDs, candidate counts/size, relation/action/selected ID, selected-context policy/size, and
generation. Content previews are local-only and bounded; production forcibly disables them.

## IAM

History access is scoped to its exact table and `ConversationHistoryByConversation` index with
`GetItem`, `PutItem`, and `Query`. Session access is scoped to its exact table with `GetItem`,
`PutItem`, and `UpdateItem`. SSM access is scoped to the four exact history/session parameters.
There is no `dynamodb:*`, wildcard resource, session `Query`, or SQS path.
The deployed Dev role was also inspected directly and contains exactly
`bedrock-agentcore:CreateEvent` and `bedrock-agentcore:ListEvents` on the deployed Memory ARN.

## Unchanged boundaries

Public request and JSON schemas, retrieval, prompts, language policy, and AgentCore Memory resource
design are unchanged. The terminal SSE metadata contract now additionally proves request,
conversation, turn, and confirmed-history identity without exposing question, answer, prompt,
context, credential, or provider data. No cross-conversation retrieval, summaries, or long-term
memory was added.

The actor seam still uses validated `request.user_id`, but its production contract is now the
verified Cognito `sub` inserted by the Amplify SSR proxy before IAM-authenticated AgentCore
invocation. Python intentionally does not repeat JWT verification. Live Sandbox proof of the full
proxy-to-persistence chain remains **[NOT VERIFIED]** until a non-production Amplify Hosting branch
exists and receives the approved Compute role. DynamoDB history retention/deletion policy remains
infrastructure-owned and **[NOT VERIFIED]** in this runtime change.

## Validation

- The current structural-refactor validation evidence is recorded in
  `skills/features/classification-pipeline.md`.
- Generic correction and bounded re-solve behavior is covered by automated tests and the
  2026-07-25 live Dev smoke evidence in `classification-pipeline.md`. Selection remains
  probabilistic among semantically similar candidates and is not claimed as general intelligence.
- Conversation-understanding tests cover the reported percentage typo, English/Hindi/Hinglish
  variants, numeric selection across two turns, unrelated questions, no-history clarification,
  streaming generation, persistence, and readable-log ordering.
- Python Ruff: passed for all changed Python files.
- Terminal stream tests cover normal, Practice, replay, and transport serialization paths. The
  normal completion assertion verifies persistence returns before the public terminal frame and the
  exact persisted identity is serialized.
- The 2026-08-16 incident regression covers a post-answer `ValueError` before persistence I/O and
  verifies the emitted diagnostic is content-free while the public error contract remains typed.
- AgentCore CDK TypeScript build: passed.
- AgentCore policy/source-staging synthesis tests: 3 passed.
- Earlier full-gate counts are superseded by the final gate below.
- `agentcore validate`: passed.
- Read-only Dev smoke: all four fixed SSM parameters exist; ConversationHistory,
  ConversationSession, `ConversationHistoryByConversation`, and `ConversationSessionByUser` are
  active; an exact-conversation GSI query succeeded.
- Dev CloudFormation stack `AgentCore-meritRankerTutor-Dev` is `CREATE_COMPLETE`.
- Dev Memory `meritranker_short_term_memory_dev` is `ACTIVE`, has 30-day expiration, and has no
  strategies. Runtime is `READY` and its control-plane configuration contains the generated Memory
  ID.
- A live Simple Interest request logged History `succeeded`, Session `succeeded`, and Memory
  `succeeded`; direct content-free inspection found one valid USER/ASSISTANT event with the expected
  actor, session, and turn token.
- A controlled Memory timeout used the live exact-conversation DynamoDB GSI and returned
  `source=dynamodb_fallback`, one usable item.
- A content-free live `ListEvents` check through `ConversationPersistenceService` returned
  `source=agentcore_memory`, one usable turn, `dynamodb_status=not_attempted`, and zero history
  repository calls.
- Live User A/User B checks returned one own event each and zero events for both guessed
  cross-user/cross-conversation combinations.
- The historical 2,255-test gate is superseded by the current gate recorded in
  `skills/features/classification-pipeline.md`.
- Final deployed Runtime-version-9 smoke used a fresh AgentCore session and conversation
  `f1d17cf5-dba1-46a5-9008-0795aff15451`. The first percentage request and exact
  `how did u calculated 75%` follow-up both returned HTTP 200 and `success=true`.
- The follow-up selected the previous percentage turn and resolved to
  `Explain how or why the referenced value or result 75% is obtained in: percentage.` Academic
  classification then returned `subject=math`, `topic=PERCENTAGE`, and
  `requires_recent_conversation=false`.
- CloudWatch recorded context load 81 ms, relation and selection stages, resolution 0 ms,
  generation 7 ms, persistence 87 ms, and total 192 ms for the follow-up.
- Direct content-free inspection returned `source=agentcore_memory`, two completed pairs,
  `memory_status=succeeded`, and `dynamodb_status=not_attempted`.
- An unrelated Rajasthan-capital request in the same conversation remained independent,
  preserved the submitted query, and returned `requires_recent_conversation=false`.
- `How did you calculate acceleration from force and mass?` also remained independent after the
  percentage history, despite its contextual-looking wording.
- The deployed generator remains `answer_source=mock`; a semantically complete percentage
  explanation from the real provider remains **[NOT VERIFIED]**.
- The 2026-07-27 reference-resolution tests verify local pronouns do not trigger reads, external
  references use bounded candidates, incompatible history is not selected, and clarification
  responses remain non-substantive under the existing persistence policy.
- The grounded-reference hardening keeps legacy clarification and missing-referent records in both
  durable stores while excluding them from read-time academic candidate projections. Typed hygiene
  reasons distinguish clarification, unresolved reference, acknowledgement, technical failure,
  failed quality, and other non-substantive responses.
- Persistence eligibility now evaluates response function as well as accepted quality. A polished
  missing-reference clarification cannot be written as an academic question-answer pair even when
  its quality status is accepted. The existing clarification skip leaves History, Session, and
  Memory unwritten; substantive persistence ordering and idempotency are unchanged.
- Synthetic live conversation `FA1D7049-5BBB-45AB-8018-0B9EB0C11C98` contained an intentionally
  seeded missing-reference Memory event alongside persisted substantive Akbar answers. The event
  remained listable in AgentCore Memory, was rejected only from candidate projection, and did not
  prevent later History, Session, and Memory writes from succeeding for grounded answers.

## Deployment security incident

During the first deployment attempt, the upstream alpha CodeZip packager included `app/.env.local`.
The runtime failed before creation, and the subsequent in-progress stack was stopped before Runtime
creation. Both affected S3 asset hashes, their retained object versions, and delete markers were
permanently removed. The clean artifact was inspected before successful redeployment and contains
no `.env*` or log files. Credentials that were present in `app/.env.local` at the time of upload
must still be rotated as a precaution.
