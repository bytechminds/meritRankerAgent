# Image Question Classification

## Architecture

```text
AgentCore invoke
  -> explicit image presence check
  -> image_classifier_node
  -> ImageQuestionClassifier protocol
  -> image validation / normalization / versioned cache
  -> configured ImageQuestionClassificationProvider
  -> Gemini adapter
  -> existing QueryClassification
  -> existing retrieval and answer-generation path
```

The image path is additive. Text-only requests still use `classify_query()` unchanged. The image
node and application service do not import Gemini or the OpenAI SDK; provider-specific code exists
only in `services/image_question_classification/gemini_provider.py`.

## Request Contract

`DoubtSolverRequest` accepts `query`, `image`, or both. `query` is the complete question for text
requests and an instruction such as `Solve question 12 only` for image requests.

```json
{
  "mode": "doubt_solver",
  "query": "Solve question 12 only",
  "image": {
    "source": "camera",
    "mimeType": "image/png",
    "base64": "<base64>"
  }
}
```

Exactly one image payload locator is required: `bytes`, `base64`, `storageKey`, or `signedUrl`.
The current runtime loader supports inline bytes/base64. Because this repository has no approved
upload/storage resolver, storage keys and signed URLs fail closed and are never fetched. Add an
allowlisted server-side `ImageReferenceLoader` before enabling storage references.

## Validation And Security

- Supported decoded formats: JPEG, PNG, WebP.
- Declared MIME must match decoded image format; extensions are not trusted.
- Empty, corrupt, oversized, undersized, animated, and multi-frame payloads are rejected before a
  provider call.
- EXIF orientation is applied and metadata is removed by re-encoding.
- Only images above the configured maximum dimension are resized; aspect ratio and high-quality
  resampling preserve equations and diagram labels.
- Arbitrary URL fetching is denied, preventing SSRF in the default implementation.
- Logs contain request ID, status, counter-ready outcome class/count, total/provider latency, cache
  hit, confidence, prompt/schema version, and model only. They exclude bytes, base64, URLs, keys,
  prompts, and private question text.

The Gemini key is backend-only. Enabling the feature without `GOOGLE_GEMINI_API_KEY`, a supported
provider, a model, or valid limits fails during startup.

## Output And Rejections

A successful provider result contains one `QueryClassification`; that exact existing contract is
passed to the current downstream graph. Extracted image diagnostics remain in the separate
`ImageParseMetadata` application result and are not exposed in the public success response.

Controlled image statuses are:

- `CLASSIFIED`
- `REJECTED_UNREADABLE_IMAGE`
- `REJECTED_NO_QUESTION_FOUND`
- `REJECTED_AMBIGUOUS_QUESTION`
- `REJECTED_INCOMPLETE_CONTEXT`
- `REJECTED_UNSUPPORTED_IMAGE`
- `PROVIDER_TEMPORARILY_UNAVAILABLE`

Rejections stop before retrieval and answer generation. Client responses contain actionable text,
not provider names, raw errors, thresholds, or model output.

## Provider And Prompt Contract

The dedicated prompt is `app/prompts/image_question_classifier.md`. Prompt composition appends the
live `QueryClassification` and image output JSON schemas, so there is no copied classifier taxonomy.
Gemini uses one multimodal request, temperature `0`, bounded output, no chat history, and OpenAI SDK
Pydantic structured parsing. The adapter uses a fixed-shape Gemini-compatible wire schema and maps
it through the authoritative `QueryClassification` and `ImageProviderOutput` validators before
routing. Prompt/schema version `v2` invalidates cache entries created before this wire contract.

To add another provider:

1. Implement `ImageQuestionClassificationProvider.classify()`.
2. Translate provider errors to `ImageProviderTemporaryError` or `ImageProviderResponseError`.
3. Return a locally validated `ImageProviderOutput`.
4. Register the provider in `factory.py` and add fail-fast configuration tests.
5. Do not modify graph routing or the downstream classification contract.

## Retry, Cache, And Cost

Invalid inputs make zero provider calls. Retryable timeout, connection, throttling, and 5xx failures
receive one retry; invalid output and deterministic rejection states are not retried. Transient
provider failures are not cached.

The bounded process-local cache key includes the normalized image hash, hashed instruction,
provider, model, prompt version, schema version, and composed-prompt hash. This prevents stale reuse
after prompt/schema/model changes. Single-flight coordination suppresses concurrent duplicate calls.
The cache is memory-only with a short TTL; it is not a distributed or persistent student-data store.

## Tests

Default offline suite:

```bash
cd app
uv run pytest tests/test_image_question_classification.py \
  tests/test_image_question_entry_integration.py
```

Optional live Gemini smoke test:

```bash
cd app
IMAGE_CLASSIFIER_ENABLED=true \
GOOGLE_GEMINI_API_KEY=<secret> \
RUN_LIVE_GEMINI_IMAGE_TEST=true \
LIVE_GEMINI_IMAGE_PATH=/absolute/path/to/question.png \
uv run pytest tests/test_image_question_classification_live.py
```

The live smoke test has been verified with one synthetic image and the configured stable model.
Representative production-image quality, provider data-processing terms, and production
latency/cost remain `[NOT VERIFIED]` by the offline suite.
