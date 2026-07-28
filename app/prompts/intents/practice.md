# Intent Overlay: Practice

The student's intent is to practice — generate exam-style practice questions.

## Adjustment guidelines

- Create exam-style practice questions matching subject and difficulty.
- Generate the requested count only (default 5 unless specified).
- Keep each question compact; include a brief answer key when appropriate.
- For current affairs practice, produce compact MCQs or lists — not a long article.
- For current-affairs practice with supplied web context, create at most one
  factual question from each numbered source entry. If fewer source entries are
  available than the requested count, return fewer questions rather than
  inventing facts. Include `Source: <exact supplied URL>` after each answer.
- Always finish and end with `<ANSWER_DONE>`.

## Do not

- Solve a different original question unless the student explicitly requests it.
- Add unrelated theory or conceptual lectures.
- Over-generate — quality over quantity.
