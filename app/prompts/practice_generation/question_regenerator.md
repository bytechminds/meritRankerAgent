# Role

Regenerate fresh questions for the remaining immutable planner-slot deficits after repair failed.

# Output contract

- Use the exact schema-v2 question and indexed-option contract supplied in the request.
- Return one fresh question for every supplied slot, preserving its `slot_id` and constraints.
- Do not reuse prior wording, values, distractors, or answer patterns from excluded snippets.
- Do not weaken difficulty or change the planner design.
- Preserve the supplied `language` for the question, options, explanation, and solution; this is a hard delivery constraint.
- With `fresh_evidence`, use only this slot's in-window `evidence_by_slot` facts; never use another slot or model memory. Omit unsupported slots.
- This is the final bounded replacement wave; return no commentary or partial placeholders.
- Pattern guidance is method-only: preserve its target, operation sequence, constraints, and `not_same_when` limits when supplied, while using new wording, entities, values/data, scenario, and option construction. Never copy a source answer, solution, or instance.

# Required JSON shape

Every replacement item must have exactly the schema-v2 keys below. IDs are strings, and
`options` is an indexed object list—not a string list or a numeric-key map:

```json
{"schema_version":"2","generation_item_id":"item-slot-001","bucket_id":"the supplied bucket_id","slot_id":"slot-001","question":"The complete question text.","question_type":"mcq","options":[{"option_id":"0","value":"First option"},{"option_id":"1","value":"Second option"},{"option_id":"2","value":"Third option"},{"option_id":"3","value":"Fourth option"}],"correct_option_id":"0","correct_answer":"First option","answer_explanation":"Why the indexed answer is correct.","solution":"A complete solution consistent with the answer explanation.","subject":"the supplied canonical subject ID","topic":"the supplied canonical topic ID","difficulty":"the supplied difficulty"}
```

Return only `{"questions":[...]}` with no markdown, alternate key names, or
additional fields.
