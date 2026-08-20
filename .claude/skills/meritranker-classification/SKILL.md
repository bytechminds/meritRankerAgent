---
name: meritranker-classification
description: Use for any change to text or image academic query classification — intent/subject/difficulty classification, image question extraction, materiality routing between text-only and image paths, or the classifier-to-generation handoff. Covers app/services/classification/, app/services/image_question_classification/, app/graphs/image_question_classifier_node.py.
---

# MeritRanker Classification Pipeline

Owns text/image academic classification and the fallback pipeline that routes a
student query into Doubt Solver or Practice.

## Current status

2026-07-28 unified-primary migration: normal text classification routes through
the native Google Gen AI SDK / `gemini-3.1-flash-lite`; Azure GPT-4.1-mini is
retained only as a config rollback target. Azure GPT-4.1 is the sole bounded
strong route with no mini/native-OpenAI fallback. Full detail:
`skills/features/classification-pipeline.md` and
`skills/features/image-question-classification.md` (image entry path —
"Partially Implemented" per `skills/features/README.md`).

## Owning code (verified against current tree)

- `app/services/classification/` — `academic_classifier.py`,
  `classification_validator.py`, `contracts.py`, `coordinator.py`,
  `image_classification_adapter.py`, `web_search_demand.py`.
- `app/services/image_question_classification/` — `classifier.py`,
  `provider.py`, `gemini_provider.py`, `image_loader.py`, `image_validator.py`,
  `prompt_builder.py`, `cache.py`, `factory.py`, `errors.py`.
- `app/graphs/image_question_classifier_node.py` — graph entry node.
- `app/schemas/image_question_classification.py`.

## Invariants

- Unreadable, cropped, visual, symbol, or relationship-uncertain images stay
  fail-closed — never claim text verification of the original image.
- A text-only strong (GPT-4.1) call is permitted only when the extracted
  question is complete, non-visual, has no extraction warnings, and the
  remaining problem is a material routing conflict — not by default.
- Internal provenance fields on the structured classification output are
  excluded from the wire schema and restored/derived locally, not sent to the
  model or the client.
- Image classification results are cached; a cache hit adds no new provider
  call (see AI usage metering invariant).

## Tests

`app/tests/test_classification_coordinator.py`, `test_classifier_json.py`,
`test_difficulty_classification.py`, `test_image_question_classification.py`,
`test_image_question_classification_live.py` (credential-gated, live),
`test_image_question_entry_integration.py`,
`test_orchestrated_classifier_routing.py`,
`test_classification_answer_reliability.py`.

## Regression impact to check

Doubt Solver entry routing, Practice-creation gate (classifier recognizes
Practice intent), web-search demand signal, language foundation.
