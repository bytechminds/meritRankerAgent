# LLM Model–Role Qualification Register

A model is qualified for a **role**, never globally. This file records every
model+role combination MeritRanker has actually tested, so that a closed combination
is not re-run and a role-scoped failure is not mistaken for a global one.

**Read this before proposing any model evaluation.** A row marked `RETIRED_FOR_ROLE`
is closed: do not re-test it, prompt-tune it, or raise its caps to rescue it. Re-open a
row only with explicit Product/AI Architecture approval.

Qualification runs use frozen corpora, production prompts, and production verifier
thresholds. Gates: accuracy defects = 0, structured parse = 100%, length/exhaustion = 0.
Fallback recovery never converts a candidate failure into a pass.

## Status vocabulary

| Status | Meaning |
|---|---|
| `QUALIFIED` | Passed a frozen qualification for this role |
| `CANDIDATE` | Plausible for this role, not yet tested |
| `RETIRED_FOR_ROLE` | Tested and failed. Closed — do not re-run |
| `BLOCKED_DEPLOYMENT` | No deployment configured |
| `BLOCKED_IDENTITY` | Alias does not resolve to the claimed model |
| `BLOCKED_AUTH` | Provider credentials unavailable |
| `BLOCKED_EVALUATION_DATA` | No trusted corpus exists to judge the role |
| `NOT_CONFIGURED` | Model not present in this environment |
| `EXCLUDED_COST` | Barred on cost grounds by Product/FinOps |

## Register

| Model | Role | Status | Reason |
|---|---|---|---|
| GPT-4.1 | quant_reasoning.planner.advanced | `QUALIFIED` | Incumbent. Frozen — replace only if a challenger independently passes and is clearly better. Superseded on the route 2026-09-23: 4/8 frozen advanced-planner cases hit its 30 s timeout |
| GPT-6 Sol (medium) | factual / quant_reasoning / english `.planner.advanced` | `CANDIDATE` | Routed 2026-09-23 (slot contract with `concept`/`pattern_hint`). ~30 real calls: 0 repairs, correct trusted composition on every accepted plan, p50 ~26 s. 1 length exhaustion at the 8000 cap on a 50-slot plan, recovered by deterministic fallback — not a pass under this register's gate |
| GPT-4.1 | math.generator.advanced | `QUALIFIED` | Incumbent. Frozen. Terra challenged and failed |
| GPT-5.6 Terra | Hindi basic reasoning generation | `QUALIFIED` | 63/63 produced, 0 length failures, 100% Devanagari, 0 true generator defects on the frozen corpus |
| GPT-5.6 Terra | Intermediate constraint authoring | `RETIRED_FOR_ROLE` | Accuracy defects |
| GPT-5.6 Terra | Advanced reasoning generation | `RETIRED_FOR_ROLE` | 6 true defects / 42 (14.3%) — wrong keys and missing correct options in seating, puzzle, syllogism, input-output |
| GPT-5.6 Terra | Advanced math generation | `RETIRED_FOR_ROLE` | 5 true defects / 36 (13.9%) — wrong keys and missing correct options |
| GPT-5.6 Terra | Answer Authority | `QUALIFIED` | v2 gold: 0 critical false accepts, 0 false rejects, 24/24 classes handled, P50 2110ms. Supersedes the v1 verifier failure, which was measured on the anchored legacy path |
| GPT-5.6 Luna | Basic reasoning generation | `RETIRED_FOR_ROLE` | Wrong key; semantically duplicate correct options |
| GPT-5.6 Luna | Intermediate reasoning generation | `RETIRED_FOR_ROLE` | Wrong key persisted at xhigh across 105 questions |
| GPT-5.6 Luna | any new generation role | `RETIRED_FOR_ROLE` | Evidence sufficient. No new route may select Luna |
| GPT-5 Mini | Reasoning generation | `RETIRED_FOR_ROLE` | 11 length failures / output exhaustion |
| GPT-5 Mini | English generation | `RETIRED_FOR_ROLE` | 6 length failures, 2 exhausted groups; plus 1 multiple-valid-answers defect |
| GPT-5 Mini | short classification / extraction / transformation | `CANDIDATE` | Untested. Only worth running if such a role is actually needed |
| o4-mini | Answer Authority (incumbent) | `FAIL_ACCURACY` | v2 gold: 1 critical false accept + 1 false reject (mis-solved a basic Hindi direction question). Config cleared: `reasoning_param_sent=True`, so this is capability, not metadata. Still the live route — see Open Risks |
| GLM 5 | Intermediate generation | `RETIRED_FOR_ROLE` | Contract/reliability failure |
| GLM 5 | Verifier, advanced reasoning, advanced math | `BLOCKED_AUTH` | Generation failure does not disqualify these |
| GLM 4.7 Flash | request_intelligence | `QUALIFIED` | In production |
| GLM 4.7 Flash | English, basic reasoning, factual | `BLOCKED_AUTH` | ~4K output capacity — do not route large batches here |
| Qwen3-32B | Intermediate reasoning generation | `RETIRED_FOR_ROLE` | Contract/reliability failure |
| Qwen3 Next 80B | English, basic reasoning, factual | `BLOCKED_AUTH` | Must qualify independently — do not infer from Qwen3-32B |
| Qwen3 235B | Intermediate/advanced reasoning, math, verifier | `BLOCKED_AUTH` | Second wave. Identity unverified |
| GPT-5.4 Mini | all candidate roles | `BLOCKED_DEPLOYMENT` | `AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI` empty |
| DeepSeek V4-Pro | hard reasoning, math, verifier | `BLOCKED_IDENTITY` | `DEEPSEEK_V4PRO_MODEL` resolves to `deepseek-reasoner`. The alias has 5 fallback callers — add a new alias, never remap this one |
| Mistral Large 3 | English, factual, analysis | `NOT_CONFIGURED` | No deployment in this environment |
| Grok | advanced reasoning, math, verifier | `NOT_CONFIGURED` | Optional second wave only |
| o3 Pro | any | `EXCLUDED_COST` | Do not integrate, route, or benchmark |
| o3, o4-mini | Intermediate reasoning generation | `RETIRED_FOR_ROLE` | Accuracy defects |
| any model | Factual / GK | `BLOCKED_EVALUATION_DATA` | No trusted source corpus. A model-authored source judged by a model is circular |

## Open risks

1. **The live Authority fails its gate and a qualified replacement exists.**
   `general.verifier.default` → o4-mini scores 1 critical false accept and 1 false
   reject on the v2 gold corpus; Terra scores 0 and 0 on the same corpus and is ~46%
   faster. The route switch is a one-line change and is NOT applied — it awaits
   approval.
2. **If Terra becomes the Authority it must never be an Author.** The gate's whole
   value is that two independent models agree. One model on both sides makes agreement
   self-confirmation, so a Terra Authority forecloses Terra as a generator anywhere.
3. **No qualified generator exists for any English or reasoning role.** Nine models have
   now failed question authoring under the current contract, across four families and
   three providers. The common failure is constraint-heavy authoring, and it has appeared
   at basic difficulty, not only advanced.
4. `practice_request_intelligence` declares `model_hard_max_output_tokens: 8000` while
   GLM 4.7 Flash's real capacity is ~4K. Harmless today (the route requests 700) but the
   metadata is wrong.

## Method notes

- Verifier qualification uses a gold corpus of **injected** defects (a key that contradicts
  arithmetic, two options denoting one value, an unsatisfiable constraint set). Ground truth
  is definitional, so no external trusted source is required. This is why the Factual/GK
  circularity bar does not block verifier work.
- `CORRECT_OPTION_MISSING` items never reach the model: `GeneratedQuestion` rejects a key
  that is not among the options. Deterministic validation, working as intended.
- Qualification harnesses pass an empty exclusion list, so repeated stems within a family
  are a harness artifact, not a model defect. Production orchestration passes the real
  exclusion set.
