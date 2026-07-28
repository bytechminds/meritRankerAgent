# Feature: Image Question Classification

## Purpose

Add a secure image-specific entry path for student doubt requests while preserving the existing
text classifier and downstream `QueryClassification` contract.

## Current Status

**Implemented locally; one synthetic live Gemini smoke test verified.** Feature flag defaults to
disabled.

## User Flow

1. A web or mobile client submits an image and optional instruction.
2. The backend validates and normalizes the image before any provider call.
3. One multimodal call extracts and classifies without solving.
4. A valid classification enters the same retrieval/generation path as text; a rejection returns a
   controlled actionable message.

Final response language is applied only after Gemini extraction/classification. An English image
question can produce English, Hindi, or Roman-script Hinglish according to the validated frontend
request without changing the Gemini route or adding an image-language classifier. Non-English SSE
answers use the shared private verification/replay boundary before answer chunks are emitted.

## Entrypoints And Graph

| File | Boundary | Description |
|---|---|---|
| `app/main.py` | `invoke()` | Explicit modality routing and controlled rejection response |
| `app/graphs/image_question_classifier_node.py` | `image_classifier_node()` | Provider-neutral image node |
| `app/graphs/doubt_solver_graph.py` | classifier nodes | Reuse a prevalidated classification or run the existing text classifier |
| `app/services/doubt_solver/streaming_doubt_solver_service.py` | `StreamDoubtSolverInput` | Accepts the same optional prevalidated lean classification |

No graph node imports a provider SDK or reads credentials.

## Schemas

| File | Models |
|---|---|
| `app/schemas/image_input.py` | `ImageInput` |
| `app/schemas/image_question_classification.py` | statuses, provider request/output, visual and parse metadata, validated image, application result |
| `app/schemas/doubt_solver.py` | optional `image`; requires query or image |

`ImageProviderOutput.classification` uses the existing `QueryClassification`; no second subject,
intent, difficulty, or retrieval taxonomy exists.

## Services And Prompt

| File | Responsibility |
|---|---|
| `classifier.py` | validation orchestration, one retry, confidence gate, cache, safe telemetry |
| `provider.py` | replaceable provider protocol |
| `gemini_provider.py` | one Gemini OpenAI-compatible multimodal call, compatible wire schema, and authoritative domain mapping |
| `image_loader.py` | inline payload loading; storage/URL deny-by-default boundary |
| `image_validator.py` | MIME/signature/decode/frame/dimension/orientation/resize validation |
| `cache.py` | bounded TTL cache and concurrent duplicate suppression |
| `prompt_builder.py` | dedicated prompt plus authoritative live schemas |
| `app/prompts/image_question_classifier.md` | extraction/classification instructions; prohibits solving and invention |

## Config / Env

`IMAGE_CLASSIFIER_ENABLED=false`, `IMAGE_CLASSIFIER_PROVIDER=gemini`,
`IMAGE_CLASSIFIER_MODEL=gemini-3.1-flash-lite`, `GOOGLE_GEMINI_API_KEY`, timeout, image-size,
output-token, cache-TTL, confidence, and resize limits are documented in
`docs/dev/backend-env.md`.

## Tests

| File | Covers |
|---|---|
| `app/tests/test_image_question_classification.py` | contracts, image validation, controlled fixtures, provider mapping, retry, cache, concurrency, redaction, startup config |
| `app/tests/test_image_question_entry_integration.py` | disabled behavior, image routing, text regression, downstream stop on rejection |
| `app/tests/test_image_question_classification_live.py` | explicit opt-in live Gemini smoke test |

## Security And Performance Notes

- One provider call per cache miss, with one retry only for retryable failures.
- Safe diagnostics record the actual provider attempt count and
  `text_classifier_bypassed=true`; validation, cache-hit, and single-flight follower paths report
  zero new provider calls.
- No retrieval, Pattern data, chat history, or answer generation is sent to the image classifier.
- No private payload, signed URL, key, prompt, or provider response is logged.
- Successful and deterministic provider outcomes use a bounded process-local cache keyed by image,
  instruction, provider/model, and prompt/schema versions.
- [SECURITY RISK] The in-memory cache temporarily retains extracted question data. It is bounded,
  non-persistent, short-lived, and must not be replaced with a shared cache without encryption,
  tenancy, and retention review.
- [AUTH TODO] The runtime still has no authenticated student identity boundary.

## Known Limitations

- [NOT VERIFIED] Representative production-image quality, timeout behavior, cost, and regional availability.
- [BLOCKER] A live Hindi algebra-image request was extracted and classified correctly, but the
  downstream generator returned formula-only output with an English heading twice. Language
  validation failed closed to a Hindi reliability response; a successful Hindi image answer was
  not demonstrated.
- [PROD BLOCKER] Confirm provider data-processing terms for student images before production use.
- [DEFER] No approved storage upload resolver exists; `storageKey` and `signedUrl` fail closed.
- [DEFER] No tracing backend or metrics collector exists; structured logs expose counter-ready
  success/rejection/error, total/provider latency, cache-hit, confidence, and version metric fields.
- [DEFER] Runtime cancellation is limited by the synchronous AgentCore/provider call boundary.

## Latest Changes

- `2026-07-27` - Verified the active `gemini-3.1-flash-lite` standard Developer API token rates
  and added exact-match local pricing. A live synthetic equation image used one Gemini call,
  bypassed text classification, reported 4,484 input and 391 output tokens, and produced a
  $0.0017075 application estimate. Provider invoices remain authoritative.
- `2026-07-27` - Added provider-reported Gemini token usage to the shared request-scoped LLM
  observability collector without changing the image call, retry, cache, classifier bypass, prompt,
  or response contract. Image usage is transferred into the SSE-owned request collector before
  generation; cache hits and single-flight followers correctly record zero new model calls.
- `2026-07-27` - Verified that image-derived questions carry the frontend language through the
  existing downstream generator. Hindi and Hinglish image SSE responses now use the same private
  language verification/replay boundary as text; Gemini extraction, classification, call count,
  cache, schema, and text-classifier bypass remain unchanged. A live Hindi image recheck failed
  language compliance after its one repair and returned the localized Hindi reliability response;
  successful live Hindi image generation remains blocked.
- `2026-07-27` - Demand-based web-search inspection confirmed the image classifier remains before
  all text classification and conversation handling. The Gemini prompt now defines the same
  current/explicit-search demand policy as the text classifier, and its fixed wire schema restricts
  `web_search_reason` to the shared enum-like values. Prompt version
  `image-question-classifier-v4` invalidates older cached classifications. A live current RBI image
  used one Gemini call, bypassed the text classifier, mapped `need_web_search=true`, and completed
  through the shared Tavily/retrieval/generation path. A live static equation image used one Gemini
  call and no web search.
- `2026-07-27` - Grounded-reference and polluted-Memory safeguards are confined to text
  conversation candidate projection and completed-answer persistence eligibility. Standalone image
  routing, its single Gemini call, text-classifier bypass, and later text selection of a substantive
  image-derived turn remain unchanged. Synthetic live request
  `8af94e87-3fef-43e8-be07-e28287c276c1` logged `provider_call_count=1` and
  `text_classifier_bypassed=true`, solved `x + 5 = 12`, and its later text follow-up selected the
  persisted image-derived equation.
- `2026-07-27` - Pronoun/reference action and validator extensions preserve the normal image route:
  image results still bypass the text classifier and recent text context. Live request
  `72c9901d-6f06-45f9-afe8-8ca1bf3c0b0f` logged one Gemini call and
  `text_classifier_bypassed=true`; its later text reference selected the persisted image-derived
  equation and correctly explained subtracting 5 from both sides.
- `2026-07-25` - A real local Dev image request logged
  `provider_call_count=1`, `text_classifier_bypassed=true`, generated `x = 6`, and persisted
  History, Session, and Memory. A later text operation follow-up loaded the image-derived pair.
- `2026-07-25` - Preserved the one-call image classification path while making text conversation
  reads conditional. A classified image bypasses both the text classifier and recent-context read;
  its accepted normalized question-answer pair is still persisted and was live verified as
  selectable by a later text follow-up.
- `2026-07-16` - Replaced the provider-facing schema with a fixed-shape Gemini-compatible wire
  contract, mapped it back through existing domain validators, and verified one synthetic live call.
- `2026-07-16` - Added the isolated image entry route, Gemini adapter, validation, cache, tests, and
  documentation without changing text classification, retrieval, planning, generation, or Pattern
  ingestion behavior.

## Next Steps

- Run the opt-in live fixture suite with approved credentials and representative production images.
- Measure current-image source-selection quality across a representative fixture set; a live RBI
  query required the existing reputed-source retry after the authoritative attempt was weak.
- Add an allowlisted storage loader after the web/mobile upload architecture is defined.
- Connect safe telemetry fields to the production metrics/tracing backend.
