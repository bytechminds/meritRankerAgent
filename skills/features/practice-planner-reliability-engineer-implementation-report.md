# Implementation Report: Practice Planner Reliability

## Changed Files

- `app/features/practice_generation/planning.py`: canonical deterministic multi-topic fallback,
  strict topic coverage, bounded safe diagnostics, and typed fallback failure.
- `app/features/practice_generation/orchestration.py`: safe planner/fallback events and controlled
  planner terminalization.
- `app/features/practice_generation/providers.py` and planner prompts: explicit repair phase and
  corrective instruction.
- `app/observability/events.py`: registered the two emitted fallback event names.
- Focused planner, orchestration, prompt, and observability tests; this feature context.

## Behaviour

The original comma-separated topic is never used as a canonical slot ID. A five-question fallback
uses `time_and_work`, `number_system`, `time_and_work`, `number_system`, `time_and_work` and is
validated before generation. The retry limit remains one repair (two calls maximum). Provider
execution failure may fallback for allowed Quick Practice; configuration/prompt/route and other
unexpected errors do not.

## Validation

`uv run pytest -q tests/test_observability.py tests/practice_generation/test_practice_core.py
tests/practice_generation/test_practice_orchestration.py tests/practice_generation/test_practice_prompts.py`
passed 175 tests. Scoped Ruff and `compileall` passed. `agentcore validate` and `git diff --check`
passed. The full `make check` gate is blocked by two unrelated existing lint errors in
`repositories.py` and `test_practice_repository.py`.
