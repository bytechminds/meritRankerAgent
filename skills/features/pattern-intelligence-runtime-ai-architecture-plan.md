# AI Architecture Plan: Pattern Intelligence Runtime

> Role: AI Solution Architect

## AI Workflow Plan

No model chooses Pattern runtime routing. Existing planner/classifier output supplies bounded hints; deterministic code builds a retrieval plan, validates authoritative PatternGraph records, and projects compact method context. Existing generator/solver and verifier remain unchanged model authorities.

## Graph Design

| Node | Type | Purpose |
|---|---|---|
| Existing planner | Existing model node | Defines ideal practice slots before availability lookup |
| Pattern runtime | Deterministic service | Candidate discovery, hydration, compatibility, context projection |
| Existing generator/solver | Existing model service | Produces current-question answer or new practice item |
| Existing verifier | Existing model service | Retains academic acceptance authority for practice |

**Edges:**

```text
Practice planner → Pattern runtime → existing reuse/generation → verifier
Doubt classifier → Pattern runtime → existing solver/generator
```

## Prompt / Model / Retrieval Boundaries

| Step | Type | Model / Source | Input | Output | Boundary |
|---|---|---|---|---|---|
| Candidate discovery | Retrieval | Direct S3 Vectors | Compact deterministic query | Pattern IDs only | Pattern runtime service |
| Pattern hydration | Retrieval | DynamoDB Pattern | Bounded IDs | Validated canonical Pattern | Pattern runtime store |
| Doubt references | Retrieval | `questionsByPattern` | Selected Pattern ID | Answer-redacted candidates | Pattern runtime store |
| ColBERT | Rerank | Existing RAGatouille adapter | Bounded linked references | Reordered references | Pattern runtime adapter |
| Generator/solver | LLM | Existing routed provider | Existing request + compact context | Existing output | Existing services |

All externally sourced data is Pydantic-adapted before it reaches trusted runtime state. No new model output is introduced.

## Tool Boundaries

No new LangGraph tools are used.

## State / Schema Additions

| Name | Type | Location | Purpose |
|---|---|---|---|
| `PatternRetrievalPlan` | Pydantic | Pattern runtime | Deterministic source selection |
| `PatternGenerationContext` | Pydantic | Pattern runtime/practice provider | Safe method hint only |
| `DoubtPatternContext` | Pydantic | Pattern runtime / prompt boundary | Safe PatternGraph + optional reference projection |
| Public graph state | Unchanged | Existing graphs | No raw Pattern record is persisted to public output |

## Hallucination Risks

| Risk | Mitigation | Label |
|---|---|---|
| Vector result implies same answer | Candidate only; authoritative guard; guidance never final-answer authority | [AI RISK] |
| Generator copies reference | Anti-copy prompt rule; one bounded reference; existing verifier | [AI RISK] |
| SolveFlow reveals old answer | Include only approved step fields, never answer fields | [AI RISK] |

## Prompt-Injection Risks

| Prompt section | Content source | Injection risk | Mitigation |
|---|---|---|---|
| System prefix | Local prompt files | Low | Existing stable developer-owned prompt |
| Dynamic Pattern context | DynamoDB record | Medium | Typed compact allowlist, delimiters, no instruction-following authority |
| Reference question | PatternQuestion record | High | Question/options only; no answer/solution; bounded and clearly untrusted |

## Output Validation Strategy

| Node | Output type | Validation method | On validation failure |
|---|---|---|---|
| Pattern runtime | Internal Pydantic models | `model_validate()` + graph guard | `IGNORE`/existing fallback |
| Practice generator | Existing structured model | Existing parser/verifier | Existing repair/regenerate/fail policy |
| Doubt generator | Existing output guards | Existing answer safety path | Existing fallback |

## Evaluation / Test Strategy

| Test | File | Type | Mock used |
|---|---|---|---|
| Candidate score cannot authorize Pattern | `app/tests/test_pattern_intelligence_runtime.py` | Unit | Fake vector/store |
| No answer fields in context | `app/tests/test_pattern_intelligence_runtime.py` | Unit | Canonical fixtures |
| Practice and doubt fallback | focused existing/new tests | Unit/graph | Injected fake services |

**[NOT VERIFIED]** Real model and live vector evaluation require a controlled Dev run.

## Model / Provider Requirements

| Requirement | Value | Status |
|---|---|---|
| New model provider | None | Active |
| New model call | None | Active |
| Context target | Approx. 3–4k input tokens where feasible | [ASSUMPTION] |
| Tool-calling support | No | Active |

## Latency / Cost Notes

| Factor | Estimated value | Label |
|---|---|---|
| Practice retrieval | Per compatible group, not slot | [NOT MEASURED] |
| Doubt retrieval | One embedding/vector batch/hydration; optional linked query/rerank | [NOT MEASURED] |
| LLM calls | No added model call | Pass |
| Context | Explicit bounded projection | [MEASURE] |

## Observability Plan

| Signal | Implementation | Status |
|---|---|---|
| Pattern decision/tier/fallback | Safe reason codes, counts, and warning codes | Implemented locally |
| Hydration/vector/ColBERT latency | Service-level timing fields | [NOT VERIFIED] no live instrumentation baseline |
| Prompt budget | Safe component token-count logs/metadata | Implemented locally; execution [NOT VERIFIED] |

## Clarification / Escalation Plan

- If live metadata lacks `patternId` or direct S3 Vector support, keep the feature disabled and escalate to the runtime owner.
- If a Pattern-to-playable-question mapping is proposed, require Solution Architect and security review before enabling `REUSE_SAFE`.
- If compact context must include answer or solution fields, block the change.

## Open Issues

| # | Issue | Label | Status |
|---|---|---|---|
| 1 | Live S3 Vector document/index contract | [NOT VERIFIED] | Open |
| 2 | Verified playable Question mapping | [BLOCKER] | Open |

## Implementation Review — 2026-08-10

The local implementation retains the planned deterministic boundary and adds stronger guards than
the initial plan: low vector scores are negative-gated, vector version hashes must match the
canonical current version when supplied, graph source values/prose are not admitted, and SolveFlow
uses an approved replay gate plus strict identifier/method-text projection. Practice sends source
metadata only for exact parity and a structured reasoning-operation identity; doubtful records
fail closed. Prompt rendering drops references first and logs explicit fallback rather than cutting
the student question or silently mutating core guidance.

The architecture remains **not ready for activation**: no live S3 Vectors producer contract or
verified Pattern-to-playable QuestionBank map exists, and current local test/lint execution is
[NOT VERIFIED] because the environment rejected the required `uv` invocation after final edits.
