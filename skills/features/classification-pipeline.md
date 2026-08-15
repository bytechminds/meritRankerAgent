# Classification Pipeline

## Status

The 2026-07-28 unified-primary migration routes normal text classification through the native
Google Gen AI SDK and `gemini-3.1-flash-lite`. The former Azure GPT-4.1-mini classifier alias is
retained only as a configuration rollback target and is not referenced by the active primary
route. Azure GPT-4.1 remains the sole bounded strong route; it has no mini/native-OpenAI
model-level fallback.

Text and image modes now compose `classification_semantics.md` with separate minimal overlays.
Gemini text uses the `QueryClassification` JSON schema through native structured output; internal
provenance fields are excluded from the wire schema and restored/derived locally. The one-call
image path also uses the native Google SDK and its fixed multimodal Pydantic wire schema. Prompt
text no longer embeds the image JSON schemas.

Image routing is evaluated with the existing materiality policy after extraction. A text-only
GPT-4.1 strong call is permitted only when the extracted question is complete, non-visual, has no
extraction warnings, and the remaining problem is a material routing conflict. Unreadable,
cropped, visual, symbol, or relationship uncertainty remains fail-closed and never claims text
verification of the original image.

Local validation on 2026-07-28:

- focused classifier/context/image/usage gate: 271 passed;
- complete repository gate: 2,480 passed, one credential-gated live image test skipped;
- Ruff: passed;
- `agentcore validate`: passed;
- native Gemini/Azure live comparison: not verified because the execution safety gate blocked
  sending the repository prompt/schema and test questions to an external provider without a direct
  user approval. No provider call was made, so no live accuracy, token, cost, latency, or fallback
  rate is claimed.

Conditional-context implementation completed locally on 2026-07-25. Automated validation and live
Dev evidence are tracked below. Release remains evidence-gated: only the tested generic
follow-up/correction/re-solve behaviors are claimed.

The stabilization pass recognizes complete commandless word problems structurally, distinguishes
bare topic phrases from solvable problems, preserves explicit operation/action references, and
keeps image classification ahead of all text-conversation work.

The 2026-07-27 demand-based search inspection confirmed text and image classifications preserve the
same `need_web_search`, `web_search_reason`, and `web_search_query` mapping. The shared context
retrieval boundary now evaluates direct web demand before choosing the configured KB/vector
provider, so default `s3_vector` retrieval can no longer bypass a classifier-required search.

The live current-affairs investigation found the classifier and graph mapping were correct. Tavily
was invoked but rejected requests that combined an explicit July 2026 `start_date`/`end_date` with
relative `time_range=month`. Absolute date windows now omit `time_range`. The shared
post-classification demand normalizer also conservatively covers current-affairs misspellings,
explicit online-search requests, and current-affairs contextual actions while excluding static
definitions. Explicit month searches use a compact canonical current-affairs query without relative
`latest` terms. It runs after either existing classifier boundary and adds no provider call or
cache.

Required-web generation is now fail-closed at the evidence boundary: selected context must retain
at least one URL, and the final answer must cite only selected URLs. Freshness-sensitive Practice
requires a compact, selected evidence bundle with at least one distinct usable source per accepted
question; absent, weak, or insufficient evidence prevents the asynchronous task launch. Graph and
SSE paths use the same checks; required current facts cannot use live token delivery even under an
`always_live` local policy.

The text-classifier prompt explicitly treats English, Hindi, Hinglish, mixed-script,
transliterated, informal, and misspelled student messages as input to understand. Response language
remains outside the classifier contract: `QueryClassification` contains no language field, and the
validated frontend request preference passes independently to generation. This clarification adds
no classifier/model call and does not change the existing structured output schema.

## Classifier prompt and token audit (2026-07-27)

The text classifier retains the same academic, relation/action, web-demand, multilingual, and
practice-intent contract. Its system prompt was compacted from 19,085 to 9,596 characters by
removing repeated JSON-only, subject, method, relation/action, web-demand, and boundary
instructions. No Pydantic JSON schema was appended on this text path before or after the change.
Standalone requests still omit candidate placeholders. Primary and strong classifiers use the same
compact prompt; strong input adds only the existing bounded rejected-result fields and reason.

Normal candidate cards remain at most three, and correction/re-solve remains at most five. Their
total formatted budget is now 3,200 characters instead of 6,200; latest question/answer clues are
bounded to 450/300 characters and older clues to 350/240. Cards still contain the supplied turn ID,
query clue, answer/entity clue, subject/topic/difficulty, and compatibility indicators. No full
history or full prior answer is sent.

Development DEBUG logs emit content-free `CLASSIFIER PROMPT BUDGET` section character counts plus
the provider-reported total input-token count. No tokenizer, prompt text, candidate content, query,
answer, or provider payload is logged.

Live Azure evidence:

- Standalone `Explain photosynthesis`: 4,445 -> 2,353 primary input tokens (47.1% reduction);
  classification remained `explain_concept/general/basic/NEW_QUESTION`.
- Akbar pronoun follow-up: primary 2,456 and strong 2,511 input tokens; selected `akbar-turn` with
  `FOLLOW_UP + ANSWER_WITH_CONTEXT`.
- July 2026 current-affairs practice: primary 2,361 and strong 2,414 input tokens;
  `practice_question`, `need_web_search=true`.
- Forced threshold 0.99: exactly one primary (2,353) and one strong (2,406) call; the maximum-one
  strong fallback policy and threshold logic are unchanged.

## Primary acceptance and material strong fallback (2026-07-28)

The primary/strong coordinator now returns an internal typed `PrimaryClassificationDecision`.
Schema/provider failures and material conflicts use explicit reasons, including invalid schema,
missing fields, unsupported enums, low material confidence, context selection, relation/action,
web demand, practice intent, correction/re-solve, subject/intent, and reference resolution.
Normal logs include schema validity, primary confidence, routing fields, material-conflict count,
strong trigger, and typed reason without emitting classifier JSON or request content.

Valid standalone classifications and fully resolved one-candidate follow-ups may be accepted at the
existing 0.85 material-confidence floor when no routing-critical conflict exists. The configured
0.93 threshold remains unchanged and still applies to correction/re-solve, transformation,
ambiguity, and multiple-candidate context. Invalid contracts and material cross-field conflicts
always require the existing maximum-one strong fallback. Optional topic, tags, retrieval hints, and
minor difficulty uncertainty do not independently justify GPT-4.1.

The compact prompt now states that confidence is routing/classification certainty rather than
answer certainty. It adds one rank/order-to-reasoning boundary and explicit correction/re-solve
enum pairing while compacting repeated safety wording. Prompt size is 10,074 characters, 4.98%
above the 9,596-character compact baseline and still 47.2% below the earlier 19,085-character
prompt. No model assignment, output schema, route, or token budget changed.

The exact Riya baseline returned a valid primary `math/solve_question/basic` result at 0.90 while
its own `SEATING_ARRANGEMENT` topic and `REASONING` family contradicted the subject. The old policy
reported only `primary_low_confidence` and invoked strong GPT-4.1. After the prompt and policy fix,
the final live primary returned `reasoning/solve_question/basic` at 0.90 and was accepted with
`PRIMARY_ACCEPTED`; two additional live samples produced the same one-call result. A generic
two-sided ordinal-position structural guard still escalates a future wrong-subject result rather
than accepting it solely because its confidence exceeds the material floor.

The pre-compaction prompt did not contain an explicit rank/order rule, and no pre-compaction live
Riya output was retained. Prompt compaction therefore is not proven to have caused the behavior.
The confirmed defects were missing rank/order guidance, confidence coupled to non-routing
uncertainty, an unconditional global threshold, and untyped fallback reasons.

Final live Azure matrix: 12 required families, 15 actual classifier calls, three strong fallbacks
(Akbar low material confidence, genuine two-entity ambiguity, and correction invalid-contract
recovery). Replaying the old global-threshold decision over the same primary outputs would require
10 strong fallbacks and 22 total calls. Final p50 classifier provider time was 2,826 ms and p95 was
5,087 ms. Riya used 2,515 input and 195 output tokens in one call; its observed pre-fix execution
used 4,823 input and 392 output tokens across primary plus strong.

## Pronoun and contextual-reference resolution (2026-07-27)

The existing query classifier now resolves external personal pronouns, demonstratives, implicit
object/method references, event references, ordinal references, and bounded Hindi/Hinglish variants
against supplied candidate cards. Local antecedents remain standalone. Candidate compatibility is
evaluated before recency; recency is only a tie-breaker among compatible candidates.

`FOLLOW_UP + ANSWER_WITH_CONTEXT` represents a new factual or academic question whose subject comes
from one selected previous turn. `EXPLAIN_PREVIOUS` remains limited to explaining an earlier
answer, step, formula, or method. The internal selected-context contract may carry one bounded
`resolved_reference` string such as `he → Akbar`; it is not persisted and is not exposed to clients.

The deterministic validator rejects unresolved external references returned as `NEW_QUESTION`,
unknown or incompatible selected IDs, invalid relation/action mappings, and factual person
follow-ups mislabeled as explanation. The existing strong classifier may run once with the same
bounded cards and conflict reason. If both attempts fail, exactly one compatible candidate permits a
bounded contextual fallback; multiple compatible candidates or no compatible candidate produce a
context-aware clarification.

No classifier, model route, Memory adapter, DynamoDB field, cache, retrieval path, image call, graph
node, or transport contract was added. Normal text classification remains one model call; invalid or
conflicting output remains capped at primary plus one strong call.

### Grounded candidates and polluted-Memory handling

Read-time Memory hygiene now classifies each loaded response by its primary function. Clarification,
missing-reference, acknowledgement, technical-failure, failed-quality, and non-substantive records
remain in AgentCore Memory and ConversationHistory but are rejected before candidate-card
construction. Rejection is typed and content patterns are phrase/function based rather than
single-token based.

Person compatibility requires both a validated grounded label and person evidence in the candidate;
the current query's pronoun and a candidate's unresolved pronoun cannot establish grounding.
Equivalent evidence gates apply to operations/formulas, ordered items, objects/concepts, and events.
Entity labels come from explicit candidate text or validated topic metadata and reject pronouns,
auxiliaries, response headings, sentence-initial artifacts, and generic academic terms. Missing
labels remain null.

Compatible cards are grouped by grounded identity. Recency selects the newest substantive card only
inside one identity chain. Multiple distinct grounded identities produce labeled clarification;
labels are validated and deduplicated before rendering. No grounded compatible identity produces a
generic referent clarification.

The write boundary uses the same primary-response classifier in addition to finalization, language,
and quality checks. A quality-passed missing-reference response is therefore recorded as a
clarification persistence skip rather than a substantive academic pair. History/Session/Memory
ordering, idempotency, schemas, and durable legacy records are unchanged.

Final synthetic Dev evidence used conversation
`FA1D7049-5BBB-45AB-8018-0B9EB0C11C98`. AgentCore Memory returned the substantive Akbar event plus
an intentionally seeded missing-reference event. Read-time hygiene retained both in the load result,
rejected only the polluted projection as `unresolved_reference_response`, and supplied only grounded
academic cards downstream. Request `e163a632-621e-4f54-9530-e601eeadce41` selected the Akbar turn
with `FOLLOW_UP + ANSWER_WITH_CONTEXT`; request `5105a18a-5c0e-4165-a8f1-89a81d2abbf3` answered both
the wives and residence subquestions through the newest Akbar chain turn. A later substantive Shah
Jahan turn caused the exact `Do you mean Akbar or Shah Jahan?` clarification, while a complete
algebra request remained current-only.

## Git baseline

- Branch: `master`
- Current committed baseline: `4e90dbca0144b677a7489733b15a365fcf6dab1b`
  (`0.1.0@4e90dbc`, `added dynamodb`)
- Earlier observed runtime: `9e6348b` (`added exam level prompt, image classification`)
- `9e6348b` is not a complete stable rollback target because it predates the Memory/persistence
  foundation added by `4e90dbc`.
- The regression was in uncommitted experimental work layered on `4e90dbc`: a separate
  conversation-classifier role, six-turn expansion, canonical/root lineage, action-aware generator
  overlays, and a task-alignment gate.
- Baseline verification before this refactor: Ruff passed; pytest 2,251 passed and one skipped;
  `agentcore validate` passed.

## Conditional-context inspection baseline (2026-07-25)

This map records the implementation observed before the conditional-context source changes.

| Stage | Current owner before this task | Observed behavior / issue |
|---|---|---|
| `understand_conversation` | `ConversationUnderstandingService.understand` | Always calls `load_recent_context(..., 3)`, filters turns, deterministically decides relation/action, selects the newest turn, and builds a canonical query before academic classification. This is authoritative behavior and therefore conflicts with the target single-classifier ownership. |
| `classify` | `ClassificationCoordinator` -> existing `classify_query` | Owns academic subject/intent/topic/difficulty and existing primary/strong fallback, but receives no candidate cards and returns no selected-turn/action contract. |
| `prepare_follow_up` | graph closure plus streaming branch | Accepts the precomputed compatibility result, but also retains a second legacy resolver path that can load context, resolve a query with another model boundary, and reclassify. This duplicates relation interpretation. |
| `collect_context` | `ContextRetrievalService` | Performs academic KB/web context retrieval after classification. It does not own conversation selection and should remain unchanged. |
| Memory read decision | conversation-understanding compatibility service | Memory is read before the request is known to need context, including clear standalone text. `requires_recent_conversation` is also independently inferred by the query-classifier prompt and deterministic regex fallback. |
| Recent context | `RecentConversationContextService` | AgentCore Memory is primary; controlled empty/failure falls back to exact-conversation DynamoDB. Actor/session scoping and bounded reads are already correct. |
| Clean-turn filtering | `memory_hygiene.filter_substantive_turns` | Deterministic compatibility filtering exists, but it runs only after the unconditional read and includes experimental root/canonical fields that are not persisted. |
| Graph / streaming parity | shared academic wrapper, separate surrounding control flow | Both call the shared academic wrapper, but conversation preparation and legacy follow-up branches are separately implemented and can diverge. |
| Image classification | image classifier -> image adapter -> coordinator | A classified image bypasses the text classifier correctly. The entrypoint still invokes conversation understanding for non-stream image requests, causing an unnecessary context read. |
| Generation context | `format_recent_conversation` | Up to two root/reference/selected turns may be sent. It is not action-specific and is derived from compatibility root/canonical logic rather than a validated classifier-selected turn. |
| Persistence | `ConversationPersistenceService` | Accepted substantive final answers are written to History, Session, and Memory; clarification/failed-quality/cancelled results are skipped. This policy is independent of context reads and must remain unchanged. |
| Readable logs | observability event collector | Relation, hygiene, Memory source, classification, generation, and persistence events exist. Gate decision and selected action-specific context size are not represented as distinct sections. |

Current classifier schema fields are the academic `QueryClassification` fields plus the legacy
`requires_recent_conversation` boolean. Experimental conversation schemas separately contain
relation, requested action, selected/reference/root IDs, canonical task, and clarification flags.
The target implementation will retain only the minimal classifier-owned relation/action/selected
turn contract and derive all other behavior in code.

Uncommitted work at this inspection point consists of the shared classification package, image
adapter, deterministic conversation compatibility cleanup, Memory/persistence compatibility
changes, observability/tests, and matching feature documentation layered on committed baseline
`4e90dbc`. The protected stabilization tests cover shared text/image mapping, one text call, zero
text calls for image classification, malformed primary -> one strong fallback, Memory-first and
DynamoDB fallback, streaming/non-streaming execution, quality, persistence, and idempotency.

## Final pipeline

```text
request validation
  -> modality boundary
     -> text: deterministic ContextNeedGate
        -> standalone: skip Memory
        -> contextual/uncertain: Memory first -> DynamoDB fallback -> clean candidate cards
        -> existing query classifier (academic + relation/action/one selected ID)
           -> primary -> strong once when invalid/low-confidence/conflicting -> typed fallback
     -> image: existing image extraction + academic classification (cache/single-flight/retry)
  -> shared ClassificationCoordinator validation and downstream mapping
  -> SelectedContextBuilder (current query only, or one action-specific selected turn)
  -> retrieval policy -> generator policy -> quality -> persistence
```

Image input does not call the text classifier or read conversation context after a valid image
result. A later text follow-up can select the stored image-derived question-answer pair. Graph and
streaming text paths call the same typed classification stage and selected-context builder.

## Ownership matrix

| Decision | Authoritative owner | Supporting components |
|---|---|---|
| Request modality | `DoubtSolverRequest` / `main.invoke` | image request validator |
| OCR/image extraction | existing `ImageQuestionClassifier` | Gemini image provider |
| Product feature/task mode | existing request mode and query classifier | request schema |
| Context read need | `ContextNeedGate` | normalized phrase families and complete-request safeguards |
| Conversation relation/action/selected turn | existing query classifier | compact clean candidate cards and strict validation |
| Academic subject/topic/intent/difficulty | existing query classifier or existing image classifier | classification validator |
| Strong fallback | existing query-classifier coordinator | shared academic adapter |
| Clarification | derived relation/action policy | gate state, valid selected ID, technical-failure separation |
| Retrieval route | retrieval policy | final mapped classification |
| Generator route | LLM route resolver | final mapped classification |
| Persistence eligibility | existing persistence policy | final answer quality |

The graph, streaming transport, repositories, and logger consume decisions; they do not reinterpret
academic fields.

## Complete call graph

| Function/class | File | Called by | Input | Output | Model call | Authority | Duplicate responsibility |
|---|---|---|---|---|---|---|---|
| `invoke` | `app/main.py` | AgentCore entrypoint | validated request | response/stream | no direct provider call | modality/orchestration only | none |
| `ImageQuestionClassifier.classify` | `app/services/image_question_classification/classifier.py` | `main.invoke` for image only | validated image + optional instruction | `ImageClassificationResult` | one image call on cache miss; one retry for transient failure | image extraction and image academic result | none |
| `adapt_image_classification` | `app/services/classification/image_classification_adapter.py` | `main.invoke` | classified image result | `ClassificationStageResult` | no | convergence adapter | none |
| `ClassificationCoordinator.accept_image_classification` | `app/services/classification/coordinator.py` | image adapter | normalized query + image classification | validated mapped stage result | no | image result validation/mapping | none |
| `ClassificationCoordinator.classify_text` | same | graph/shared streaming wrapper | normalized text | validated mapped stage result | delegates once to existing classifier | stage sequence and technical-failure separation | none |
| `run_existing_classifier` | `app/services/classification/academic_classifier.py` | coordinator | text + request metadata | `QueryClassification` | delegates to existing classifier | adapter only | none |
| `ContextNeedGate.evaluate` | `app/services/conversation/context_need_gate.py` | conversation preparation | current text only | required/not-needed/uncertain | no | context read only | none |
| `ConversationUnderstandingService.understand` | `app/services/conversation/conversation_understanding.py` | graph/stream | actor/session/current text | gate + optional load + clean cards | no | candidate preparation only | none |
| `build_candidate_cards` | `app/services/conversation/candidate_builder.py` | preparation service | clean bounded turns | delimited clue cards | no | classifier input shaping | none |
| `classify_query` | `app/services/query_classifier_service.py` | academic adapter and legacy graph | current text + optional cards | `QueryClassification` | zero in deterministic mode; otherwise primary and optional strong | sole model-based academic and conversation authority | none |
| `_classify_with_llm_orchestrated_or_fallback` | same | `classify_query` | text | `ClassifierRunResult` | primary; strong at most once | primary/strong/fallback policy | none |
| `validate_raw_classification` | `app/services/classification/classification_validator.py` | coordinator | `QueryClassification` | validated classification | no | structural validation | none |
| `map_academic_classification` | `app/services/classification/academic_classifier.py` | coordinator and compatibility graph mapping | raw classification | graph-safe mapping | no | sole academic-to-graph mapping | none |
| `orchestrated_classify_query_with_delivery_signals` | `app/graphs/doubt_solver_graph.py` | graph node + streaming | text | mapping/confidence/fallback | through coordinator | thin orchestration wrapper | none |
| `_orchestrated_classify_node` | same | LangGraph | graph state | classification update | through shared wrapper | graph transition only | none |
| `build_selected_generation_context` | `app/services/conversation/selected_context_builder.py` | graph/stream prepare stage | validated classifier result + eligible turns | one bounded action-specific context | no | generation context policy | none |
| `resolve_follow_up_with_recent_context` | `app/services/conversation/follow_up_query_resolver.py` | legacy compatibility graph only | current query + exact conversation context | resolved query + recent context | existing resolver call when needed | legacy non-orchestrated compatibility | excluded from the new graph/stream text path |
| `RecentContextService.load` | `app/services/conversation/recent_context_service.py` | persistence facade | actor/conversation/limit | typed context load | no | Memory-first source selection | none |
| `AgentCoreShortTermMemory.read_recent_completed_turns` | `app/services/conversation/short_term_memory.py` | recent-context service | exact actor/session | completed turns | no | Memory read only | none |
| `ConversationHistoryRepository.list_recent_completed_turns` | `app/services/conversation/history_repository.py` | recent-context fallback | exact actor/conversation | completed turns | no | DynamoDB fallback read only | none |

## Final classifier contract

- Input: validated current query text, request ID, optional compact candidate cards, gate decision,
  supplied candidate ID allowlist, and optional strong-classifier status hook.
- Output: Pydantic `QueryClassification`, extended internally with `relation`,
  `selected_turn_id`, and `requested_action`.
- Prompt: shared `app/prompts/classification_semantics.md` plus text overlay
  `app/prompts/query_classifier_text.md`; the strong route adds the JSON-only compatibility
  contract in `app/prompts/query_classifier.md`.
- Primary route: `general.classifier.default`.
- Primary alias: `doubt_solver_classifier_gemini`; native Gemini
  `gemini-3.1-flash-lite`, with no model-level fallback.
- Strong route: `general.classifier_strong.default`.
- Strong alias: `doubt_solver_classifier_strong`; Azure OpenAI `gpt-4.1`, with no model-level
  fallback.
- Primary settings: temperature `0.1`, maximum output `650`, model timeout `20s`.
- Strong settings: temperature `0.0`, maximum output `800`, model timeout `25s`.
- Confidence threshold: configured by `DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD`.
- Retry/fallback:
  - primary valid and confident with no signal conflict -> accept;
  - contextual primary requires confidence `>=0.85`, compatible relation/action, and a supplied
    selected ID where required;
  - malformed, unknown-ID, incompatible, low-confidence, or conflicting primary -> strong exactly
    once with the same cards plus primary result and rejection reason;
  - exceptional five-card correction expansion forces the strong route once;
  - strong malformed/fails -> typed deterministic fallback;
  - a clear `CONTEXT_REQUIRED` reference may use the newest supplied clean turn in that fallback;
    an `UNCERTAIN` topic phrase remains `AMBIGUOUS/ASK_CLARIFICATION`;
  - a generic correction/re-solve must select the newest substantive candidate; an older selection
    is rejected with `generic_latest_reference_selected_older_turn`.
- Parsing: strict JSON parsing followed by Pydantic validation.
- Cache: no academic-classifier response cache.
- Logging: route ID, validity, confidence, fallback reason, strong-use flag, and latency; no prompt or
  full user content.

`QueryClassification` decides intent, subject, topic hints, response style, confidence, difficulty,
retrieval need, current-web need, relation, action, and at most one supplied turn ID. Code derives
`requires_recent_conversation`, clarification, and generation context policy. It does not decide
request modality, Memory reads, OCR, persistence, generator model ID, or final answer quality.

Required values cannot remain blank at the shared boundary. A technical exception becomes a typed
`technical_fallback`. A clear standalone request continues through the existing safe academic path.
An uncertain request whose model result cannot be validated becomes
`AMBIGUOUS/ASK_CLARIFICATION`; no invented turn is used. A short phrase is rejected as an
unsupported numerical solve only when it is classified as `NEW_QUESTION`; valid correction and
re-solve actions may retain `solve_question`.

## Image path

- Validation/loading: existing image loader and validator.
- OCR and image academic classification are combined in the existing multimodal provider request.
- Provider temperature: `0.0`.
- Default model: `gemini-3.1-flash-lite`.
- Default timeout: `15s`.
- Default maximum output: `1,400` tokens.
- Cache: process-local SHA-256 key, default TTL `300s`, maximum `256` entries.
- Duplicate suppression: process-local single-flight.
- Retry: one retry only for transient provider failures.
- A valid image result supplies normalized question, academic classification, visual metadata, and
  confidence. The image adapter validates and maps it without invoking the primary text
  classifier. A material non-visual routing conflict may use the strong-only text route once.
- The Gemini fixed wire schema and prompt version v5 carry the shared bounded web-search reason and
  query fields. Static image questions set no search demand; current or explicit-online image
  questions enter the same downstream search decision as text.
- Text requests never invoke the image classifier.

## Model-call and latency budget

| Scenario | Before experimental calls | Final normal calls | Configured classification latency bound |
|---|---:|---:|---|
| First standalone text | academic primary | academic primary | primary model timeout `20s` |
| Standalone text with history | conversation classifier + academic primary | gate + academic primary; zero Memory reads | primary `20s` |
| Known follow-up | conversation classifier + academic primary; possible resolver | gate + one bounded Memory read + academic primary | Memory latency + primary `20s` |
| Correction | conversation classifier + academic primary | gate + bounded Memory + academic primary; strong only on validation/expansion | primary `20s`; optional strong `25s` |
| Image question | image classifier | image classifier | image default `15s` |
| Text follow-up to stored image turn | conversation classifier + academic primary | gate + Memory + academic primary | Memory latency + primary `20s` |
| Malformed primary | primary + strong | primary + strong once | primary `20s` + strong `25s` |
| Both malformed | primary + strong + deterministic fallback | same | bounded by the two configured calls |

The experimental conversation-classifier call was removed. Normal call count does not increase.
Observed local-runtime/Dev-service classification timings were approximately `2.5-3.6s` per Azure
classifier call. The verified standalone request completed in `4.97s`; the verified image request
completed in `8.38s`, including `2.68s` image-stage latency (`2.64s` provider latency).
These are smoke observations, not load-test percentiles. Unit tests assert call counts and zero
text-classifier calls for image convergence.

## Context, caching, and persistence boundaries

- Normal classifier context cap: at most three clean completed question-answer pairs.
- Explicit correction/re-solve recovery may exceptionally load up to five pairs so an original
  substantive question remains selectable through a short correction chain. No other request type
  may exceed three. Five cards force the existing strong classifier once.
- Candidate cards use question previews of 500 characters (700 latest) and answer clues of
  350 characters (500 latest), with a hard 6,200-character normal input budget.
- Answer clues rank exact phrase, number/percentage, formula, option, lexical, conclusion, and
  adjacent-sentence matches. Logs expose only bounded match-indicator names and sizes.
- Selected generator context is capped at 7,600 characters and contains only the selected turn.
- AgentCore Memory remains first. DynamoDB is attempted only for controlled Memory empty/failure.
- Exact actor/conversation isolation is preserved.
- Clearly empty, error-like, clarification-only, acknowledgement, failed-quality, correction-only,
  or meta pairs may be excluded by deterministic hygiene rules.
- Correction and re-solve meta queries are excluded from later candidate selection even when their
  answer contains a full generated solution, preventing a meta query from replacing the original
  substantive problem during a correction chain.
- No AI Memory-hygiene classifier exists.
- No new answer cache exists.
- Image classification caching is unchanged.
- ConversationHistory, ConversationSession, idempotency, and Memory write contracts are unchanged;
  experimental root/canonical/reference metadata is not written.

## Graph and streaming structure

No new node directory was created because the existing graph already has one meaningful
classification transition. Creating separate normalize/validate nodes would add state transitions
without independent orchestration value.

```text
START
 -> understand_conversation
 -> classify
 -> prepare_follow_up
 -> intent=practice + explicit creation signal + feature enabled:
    create/dispatch durable assessment -> END
 -> otherwise: collect_context -> generate
 -> END
```

The `classify` node is thin. Validation, fallback handling, and mapping live in
`services/classification`. Streaming calls the same coordinator-backed classification wrapper and
therefore does not maintain a second classification implementation.

The 2026-07-28 practice-generation addition does not change classifier ownership or its public
schema. The normalized `practice` intent is only eligible when a deterministic predicate also finds
an explicit create/generate/count/similar-question/quiz/test signal. Advice and strategy questions
remain in normal doubt solving, and the launcher defensively applies the same predicate before any
assessment write. Disabled configuration preserves the previous doubt-answer practice behavior;
eligible enabled requests return the created assessment ID without holding a long generation
stream.

## Removed or disabled experimental code

- Separate `conversation_classifier` task role, route, model permissions, prompt, and runtime call.
- Six-turn context expansion and second conversation-classifier call.
- Action-aware generator route fields and correction/re-solve prompt overlays.
- Task-alignment quality gate.
- Root/canonical/reference persistence metadata.
- Duplicate subject/intent maps in the graph.
- Fail-closed bypass of the previously verified follow-up resolver fallback.

- Removed the deterministic final relation/turn/canonical-task helper and unused root/reference
  conversation schemas.
- Removed duplicate graph/stream resolver reinterpretation from the new orchestrated text path.
- Retained the legacy resolver only for the separate non-orchestrated compatibility path.
- Correction marks prior conclusions untrusted; re-solve omits prior reasoning; similar and
  transformation omit the prior answer unless a method excerpt is explicitly requested.

## Validation

- Practice-routing regression: both compiled graph and SSE paths branch on the existing mapped
  `practice` intent, call the launcher once, skip normal answer generation, and preserve the
  existing response envelopes. See `skills/features/practice-generation.md` for the preserved graph
  validation and unverified live/deployment items.

- Primary-acceptance, prompt, context, web, practice, multilingual, image-bypass, streaming, and
  usage focused gate after materiality changes: 579 passed.
- Narrow classifier/policy/usage gate: 176 passed.
- Token/usage/classifier/context/image/stream focused gate after prompt compaction: 348 passed.
- Characterization baseline: 2,251 passed, one skipped before the structural refactor.
- Focused stabilization gate: 417 tests passed across context preparation, classifier,
  coordinator, graph/streaming, image, and persistence suites.
- Added coverage for commandless complete problems, required operation references, the full
  Hindi/Hinglish phrase list, numeric collisions, query-aware clue indicators, short-topic
  solve-conflict fallback, latest-turn correction validation, correction/re-solve hygiene, and
  clear-required versus uncertain deterministic fallback.
- Final repository gate after the material fallback-policy change: Ruff passed; pytest reported
  2,466 passed and one credential-gated test skipped. `agentcore validate` and `git diff --check`
  also passed.
- Live conversation `context-stabilization-20260725-2249` verified:
  - standalone acid problem: `CONTEXT_NOT_NEEDED`, no context read, answer `30%`, all persistence
    components succeeded;
  - explicit replacement-step follow-up: `CONTEXT_REQUIRED`, AgentCore Memory selected context,
    useful explanation, all persistence components succeeded;
  - bare `water acid solution`: `UNCERTAIN`, bounded Memory read, safe concept explanation with no
    unsupported numerical solve;
  - `Highlight the water-acid solution.`: `CONTEXT_REQUIRED`; Azure rejected both classifier calls
    with its content filter, and the bounded clear-follow-up fallback still produced a contextual
    explanation without a third model call;
  - unrelated algebra: `CONTEXT_NOT_NEEDED`, no old acid context, correct result;
  - correction and re-solve: latest substantive algebra turn selected, prior conclusion treated as
    untrusted, fresh `x = 4` derivation after correction/re-solve meta turns were filtered;
  - image: one Gemini call, zero text-classifier calls, generated `x = 6`, all persistence
    components succeeded;
  - image text follow-up: `CONTEXT_REQUIRED`, AgentCore Memory returned the image-derived pair and
    the answer explained the exact subtraction step.
- Live SSE request `cffd52e4-bfa5-4ee6-b353-96b26a3597ac` emitted terminal `complete` only after
  History, Session, and Memory all logged `succeeded`.
- Pronoun/reference focused coverage includes the exact Akbar regression, person/female/plural
  pronouns, object/method/formula/event/ordinal references, Hindi/Hinglish and malformed grammar,
  local antecedents, incompatible candidates, ambiguity, explicit comparison, no compatible
  candidate, one-turn generation context, and the existing image/streaming suites.
- Final 2026-07-27 repository gate: Ruff passed; pytest reported 2,346 passed and one
  credential-gated test skipped. The focused classifier/context/graph/stream/image/persistence gate
  reported 540 passed.
- Live conversation `pronoun-ref-20260727-1119` verified the exact
  `Does he ruled longest?` request as `FOLLOW_UP + ANSWER_WITH_CONTEXT`, selected
  `pronoun-ref-1119-turn-1`, answered directly about Akbar, and persisted History, Session, and
  Memory successfully. `What happened after his death?` retained the Akbar chain; a complete
  algebra request skipped recent context and returned `x = 5`.
- Live conversation `pronoun-ambiguity-20260727-1127` stored Akbar and Shah Jahan turns, then
  returned `Do you mean Akbar or Shah Jahan?` for `Did he rule longer?`.
- Live image conversation `pronoun-image-20260727-1128` logged one
  `gemini-3.1-flash-lite` image call, `text_classifier_bypassed=true`, solved `x + 5 = 12`, and a
  later text reference selected the image-derived equation and explained subtracting 5 from both
  sides.

## Known issues and deferred work

## 2026-07-28 classification and correction reliability

- Explicit grammar, narration/direct-indirect speech, vocabulary, comprehension, and error-spotting
  signals are structural English guards even for practice generation.
- Explicit seating/ranking guards map to reasoning; explicit history, polity, and science-family
  guards map to general. A contradictory high-confidence primary result is a material conflict and
  may use the existing single strong-classifier fallback.
- Counted or explicitly named academic practice/problem requests are self-contained and skip recent
  conversation reads. Complete two-operand percentage questions are also standalone across English,
  Hindi, and Hinglish phrasing.
- Generic correction/re-solve phrases select the newest hygienic substantive candidate
  deterministically before the primary contract is accepted. The selected candidate's academic
  subject, topic, and difficulty replace conflicting older-turn classifier hints; an older target
  is allowed only when the request explicitly identifies it.
- A complete `solve this ...` statement is treated as locally self-contained when the academic
  object and problem are present; standalone demonstratives no longer force a Memory read.
- Genuine object ambiguity prefers bounded academic labels from candidate topics or the candidate
  question phrase, producing labels such as `Simple Interest` and `Quadratic Equations` instead
  of answer fragments such as `Rate`, `Solutions`, or `Any`.
- The shared classifier prompt grew from 7,355 to 7,500 bytes (+1.97%); redundant subject and
  recency language was replaced rather than duplicated.

- `[AI RISK]` Model selection among semantically similar recent candidates is probabilistic; strict
  ID/action validation prevents invented references but cannot prove the best semantic choice.
- `[RELIABILITY RISK]` Azure content filtering can reject a benign classifier prompt containing
  acid-mixture candidate text. The existing one-strong-call bound and typed deterministic fallback
  contain the failure; provider policy tuning is outside this runtime change.
- `[NOT VERIFIED]` A forced live DynamoDB recent-context fallback was not exercised in this
  stabilization run; deterministic integration coverage remains green.
- `[AI RISK]` A correction/re-solve chain whose original substantive question is older than five
  persisted completed pairs cannot recover it without clarification. This is the deliberate bound;
  no semantic store or lineage field was added.
- No persistent root/reference lineage, summarizer, Redis, semantic recent-turn search, or frontend
  conversation metadata was added.
- `[AI RISK]` The compatibility safeguard is deliberately bounded and conservative. It does not
  claim broad coreference intelligence; unusual entities or phrasing may ask for clarification.
