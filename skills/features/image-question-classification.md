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
- [PROD BLOCKER] Confirm provider data-processing terms for student images before production use.
- [DEFER] No approved storage upload resolver exists; `storageKey` and `signedUrl` fail closed.
- [DEFER] No tracing backend or metrics collector exists; structured logs expose counter-ready
  success/rejection/error, total/provider latency, cache-hit, confidence, and version metric fields.
- [DEFER] Runtime cancellation is limited by the synchronous AgentCore/provider call boundary.

## Latest Changes

- `2026-07-16` - Replaced the provider-facing schema with a fixed-shape Gemini-compatible wire
  contract, mapped it back through existing domain validators, and verified one synthetic live call.
- `2026-07-16` - Added the isolated image entry route, Gemini adapter, validation, cache, tests, and
  documentation without changing text classification, retrieval, planning, generation, or Pattern
  ingestion behavior.

## Next Steps

- Run the opt-in live fixture suite with approved credentials and representative production images.
- Add an allowlisted storage loader after the web/mobile upload architecture is defined.
- Connect safe telemetry fields to the production metrics/tracing backend.
