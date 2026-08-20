---
name: meritranker-demo-agent
description: Use only for the local AgentCore + LangGraph + Pydantic foundation demo — the two-node StateGraph proving the base runtime stack works end-to-end with no real LLM, storage, or auth. Covers app/graphs/demo_graph.py.
---

# MeritRanker Demo Agent

Foundation-only proof that the local dev stack works: AgentCore HTTP via
`BedrockAgentCoreApp`, a two-node LangGraph `StateGraph`, Pydantic v2 validation,
Rich logging, and tests that run without AWS credentials.

## Current status

**Demo** — local dev foundation, not connected to any real model or data store.
Full detail: `skills/features/demo-agent.md`.

## Owning code

`app/graphs/demo_graph.py`, `app/main.py` (AgentCore entrypoint — currently
modified in the working tree).

## When this skill actually applies

Only when the task explicitly concerns the foundation demo graph or the
AgentCore entrypoint wiring itself — not for Doubt Solver, Practice, or any
other production feature graph, even though they also use LangGraph. For those,
use the matching feature skill instead.

## Tests

`app/tests/test_demo_graph.py`, `test_main_orchestrated_entrypoint.py`,
`test_main_routing.py`.
