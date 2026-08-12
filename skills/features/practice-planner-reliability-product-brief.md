# Product Brief: Practice Planner Reliability

## Goal

Ensure an invalid planner response for a multi-topic Quick Practice cannot leave the assessment
in `PRACTICE_GENERATION_FAILED` when a deterministic valid blueprint is available.

## Scope

- One initial planner call and at most one schema-directed repair.
- Validated deterministic multi-topic fallback, safe diagnostics, and controlled failure when
  fallback construction itself is invalid.

## Non-goals

- Model routing, token budgets, generator, verifier, Pattern, billing, infrastructure, API, or
  production deployment changes.

## Success Criteria

The exact five-question SSC_GD/PRE multi-topic request reaches a valid plan or a typed terminal
failure without an unhandled validation exception.
