# AI Architecture Plan: Practice Planner Reliability

The planner remains authoritative when it produces a valid blueprint. On invalid structured
output, the existing bounded repair receives only phase, reason, schema, field paths, and types;
it never receives or logs raw prior output. A Quick Practice alone may use validated deterministic
fallback after the bounded attempts. Configuration, route, prompt, and unexpected runtime errors
remain loud and are never converted to academic fallback.
