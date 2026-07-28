# Strong Classifier JSON Output

## Required JSON contract

Return exactly one JSON object with every field below. Do not add markdown, comments, a completion
marker, a second object, or trailing text.

```text
{
  "intent": "<value defined by the shared semantics>",
  "subject": "<math|reasoning|english|general|unknown>",
  "topic": "<short label or null>",
  "topic_confidence": <0.0-1.0 or null>,
  "pattern_topic_candidate": "<UPPER_SNAKE or null>",
  "pattern_family_candidate": "<UPPER_SNAKE or null>",
  "retrieval_tags": ["<lower_snake>"],
  "difficulty": "<basic|intermediate|advanced|default>",
  "response_style": "<step_by_step|simple_explanation|short_answer>",
  "confidence": <0.0-1.0>,
  "retrieval_need": "<none|concept_context|similar_question|unknown>",
  "reasoning_summary": "<15 words max or null>",
  "need_web_search": <boolean>,
  "web_search_reason": "<supported reason or null>",
  "web_search_query": "<concise query or null>",
  "relation": "<value defined by the shared semantics>",
  "selected_turn_id": "<supplied ID or null>",
  "requested_action": "<value defined by the shared semantics>"
}
```
