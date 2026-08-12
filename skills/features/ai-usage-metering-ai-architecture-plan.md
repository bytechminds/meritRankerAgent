# AI Architecture Plan: Centralized AI Usage Metering V1

## AI Boundary

No prompt, model selection, output validation, graph edge, fallback condition, or retry policy is
changed. Metering observes the normalized provider result only after the provider call completes.

## Flow

`provider usage → record_llm_call → active operation accumulator → terminal Decimal calculator → safe shadow summary`

The same accumulator is carried from the request context into the existing stream wrapper and,
when Practice launches, into the existing task launcher keyed by the durable test ID. Parallel
Practice groups receive context copies that reference the same locked accumulator.

## Risks

- [NOT VERIFIED] Current Azure and DeepSeek deployment prices are not project-supplied; their
  summaries must stay incomplete rather than infer native-provider prices.
- [AI RISK] A provider can omit usage on failed/streamed responses; the result remains incomplete,
  never estimated from text or token caps.

## Test Strategy

Use fake provider results and direct central-sink tests for primary/fallback usage, Decimal math,
disabled metering, missing profiles, stream transfer, Practice handoff, parallel isolation, and
READY/FAILED terminal summaries. No real provider calls are required.
