# Business Requirements: Practice Planner Reliability

## Acceptance Criteria

1. A multi-topic request accepts comma, semicolon, pipe, or plus separation only when each topic
   canonicalizes to a non-empty identifier.
2. Five fallback slots cover both requested topics and differ in allocation by at most one.
3. Planner output receives one repair attempt at most; initial valid output is not retried.
4. Safe diagnostics expose no model content or student content.
5. An invalid fallback ends as `PRACTICE_PLANNER_FALLBACK_INVALID`, never a generic unhandled
   Pydantic error.
6. Configuration, prompt, route, and unexpected runtime failures do not silently become fallback.
7. No provider model/routing/token, generator, Pattern, billing, API, data, infrastructure, or
   production deployment behavior changes.
