# Q1 Second-Task Stage-A Failure Audit

## 1. Historical Stage-A result

Stage A1 remains permanently `Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED` at closeout
commit `00ab87c386c72a7c88fac438e94240619555d629`. The sealed 128-row journal has
SHA-256 `5b0fec6960ac414f56995d91a43c3b41c49a06b5fb868156a8e24d037b9281b1`.
Its independent historical forensic classification remains
`Q1_SECOND_TASK_STAGE_A_FORENSIC_CLEAN`.

The frozen counts are unchanged: BASELINE had 53/64 commitment-valid, 52/64
evaluable, and 39/64 correct rows; TEXTUAL_CAREFUL had 62/64 commitment-valid,
61/64 evaluable, and 50/64 correct rows. Stage B remains
`NOT_AUTHORIZED_NOT_OPENED`, with zero meaningful-controller and zero
activation-null LiveCodeBench trajectories.

## 2. Exact failure taxonomy

All 128 sealed outputs were inspected by one condition-symmetric mechanical
classifier. Raw text and references remain only in the sealed journal and a
private ignored audit artifact; the repository row audit contains identities,
hashes, statuses, categories, types, and booleans but no benchmark or output
text.

| Category | BASELINE | TEXTUAL_CAREFUL |
|---|---:|---:|
| VALID_AS_FROZEN | 52 | 61 |
| MISSING_FINAL_MARKER_UNIQUE_LITERAL_PRESENT | 0 | 0 |
| FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT | 0 | 0 |
| UNIQUE_LITERAL_IN_CODE_BLOCK | 0 | 0 |
| UNIQUE_LITERAL_WITH_TRAILING_PROSE | 0 | 0 |
| MULTIPLE_IDENTICAL_COMMITMENTS | 0 | 0 |
| MULTIPLE_CONTRADICTORY_COMMITMENTS | 1 | 0 |
| MALFORMED_PYTHON_OR_JSON_LITERAL | 0 | 0 |
| TYPE_MISMATCH | 0 | 0 |
| NO_RECOVERABLE_LITERAL | 0 | 0 |
| TRUNCATED_GENERATION | 0 | 0 |
| MECHANICAL_REPETITION | 0 | 0 |
| UNFINISHED_REASONING | 0 | 0 |
| AMBIGUOUS_OUTPUT | 0 | 0 |
| REFERENCE_OR_PROMPT_AMBIGUITY | 0 | 0 |
| OTHER: unique terminal typed FINAL after empty nonliteral final-style headings | 11 | 3 |

The BASELINE invalid set contains 12 rows: 11 have one mechanically unique
terminal typed literal and one has a competing distinct typed literal. The
TEXTUAL_CAREFUL invalid set contains three rows, all with a unique terminal
typed literal. There were no token-cap, repetition, truncation, unclosed
reasoning, missing-answer, type-mismatch, or benchmark-reference-ambiguity
failures.

## 3. BASELINE failure mechanism

The best-supported primary mechanism is `PARSER_LIMITATION`. In 11/12
non-evaluable BASELINE rows, the response ended in exactly one explicit typed
`FINAL` literal, but an earlier empty “Final Answer” or heading-derived final
section caused external-semantic-v3 to treat the response as a malformed or
multiple commitment. Seven of those 11 terminal candidates match the objective
reference and four do not. Correctness was not used to define or select the
repair.

The remaining 1/12 row contains a distinct competing typed literal and remains
fail-closed. It is the only supported `MULTIPLE_CONTRADICTORY_COMMITMENTS`
case. There is no evidence that the BASELINE gate failure was driven by token
cap, runaway generation, mechanical repetition, unfinished reasoning, or task
ambiguity.

## 4. TEXTUAL_CAREFUL comparison

All three non-evaluable TEXTUAL_CAREFUL rows exhibit the same parser-coverage
pattern and contain one mechanically unique terminal typed literal; all three
match the reference. The smaller frequency, 3/64 versus 12/64, is associated
with condition, but Stage A1 does not isolate why: TEXTUAL_CAREFUL jointly adds
meticulous tracing, verification, and a repeated final-answer instruction.

The comparison therefore does not identify an internal reasoning mechanism and
does not establish that prompt repetition alone fixed the channel. It does show
that the parser limitation was condition-symmetric in form even though its
observed incidence differed by condition.

## 5. Semantic versus answer-channel contribution

The frozen +11 correct-row difference decomposes descriptively into +3 among
51 matched family-rollout pairs where both conditions were evaluable and +8
among 13 pairs where at least one condition was invalid or non-evaluable.

Under the locked post-hoc parser repair, BASELINE adds seven correct rows and
TEXTUAL_CAREFUL adds three, reducing the diagnostic accuracy difference from
11/64 (+0.171875) to 7/64 (+0.109375). Thus four of the 11 net correct-row
advantage is attributable to differential answer-channel exclusion under this
mechanical diagnostic. The remaining difference is not promoted to a new
confirmatory estimate. All quantities in this section are
`POST_HOC_DIAGNOSTIC_NOT_STAGE_A1_RECLASSIFICATION`.

## 6. Current parser/prompt audit

The frozen BASELINE user prompt already requires one final line in the form
`FINAL: <exact Python or JSON literal>` and forbids text after it. BASELINE has
no system prompt. TEXTUAL_CAREFUL uses the same user prompt plus the historical
system policy that asks for meticulous program tracing, verification, and one
final line.

The frozen parser removes closed thinking blocks, scans all visible final-style
markers globally, and treats an empty final-style heading as a commitment
section. This is what converted earlier presentation headings into conflicts
with a later exact terminal literal. The typed evaluator itself is exact and
safe: `ast.literal_eval`/JSON only, no execution, no fuzzy matching, and no LLM
judge.

Overall cause: `PARSER_LIMITATION` primary, with one genuine fail-closed
ambiguity. `NEUTRAL_OUTPUT_CONTRACT_LIMITATION`, `GENERATION_INSTABILITY`, and
`BENCHMARK_AMBIGUITY` are not supported as the primary cause. Because the
baseline prompt was already explicit, no prompt change is selected for A2.

## 7. Candidate repairs

The initial conservative parser extension for harmless marker variants, unique
standalone literals, code-block literals, and repeated identical commitments
recovers 0/15 failures and is insufficient.

The selected parser-only repair is
`TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1`. It accepts a
row only when:

- the last non-empty visible line is one inline `FINAL` marker with one supported
  typed literal;
- every earlier final-style marker is an empty `FINAL_ANSWER` or heading-derived
  `FINAL_SECTION` marker;
- no distinct supported standalone or fenced literal is visible;
- the candidate type matches the objective reference type;
- extraction uses neither condition, item exception, reference value,
  correctness, fuzzy matching, nor semantic judgment.

Competing inline commitments, a distinct typed literal, malformed or nonterminal
answers, trailing prose, truncation, and unclosed thinking fail closed. A neutral
prompt repair is not selected because the existing prompt already states the
contract and the parser-only repair is sufficient. A combined repair is
therefore unnecessary.

## 8. Adversarial validation

Thirty-three focused Stage-A failure/A2 tests cover integers, lists, booleans,
strings, whitespace, hidden thinking, code blocks, trailing prose, identical
and contradictory commitments, reasoning literals versus final literals,
malformed and malicious payloads, quoted strings, nested containers, type
mismatch, truncation, repetition, no answer, terminality, unique seeds, and the
20-family gate. The focused suite passes 33/33; the broader directly relevant
Q1 second-task suite passes 42/42.

An independent low-level implementation reproduced the frozen 53/52/39 and
62/61/50 counts, the 11/3 repair eligibility counts, and the repaired 63/63/46
and 64/64/53 counts with zero row-field discrepancies. Classification:
`Q1_SECOND_TASK_STAGE_A_FAILURE_AUDIT_FORENSIC_CLEAN`.

An implementation erratum records one preliminary noncanonical diagnostic that
scanned closed `<think>` content. It was discarded and corrected by applying
the canonical visible-text transformation. The taxonomy and repair eligibility
rules were not changed by that correction.

## 9. Post-hoc Stage-A1 diagnostic re-score

`POST_HOC_DIAGNOSTIC_NOT_STAGE_A1_RECLASSIFICATION`

| Condition | Frozen validity | Diagnostic validity | Frozen evaluability | Diagnostic evaluability | Frozen accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|---:|
| BASELINE | 0.828125 | 0.984375 | 0.812500 | 0.984375 | 0.609375 | 0.718750 |
| TEXTUAL_CAREFUL | 0.968750 | 1.000000 | 0.953125 | 1.000000 | 0.781250 | 0.828125 |

Both conditions would cross the historical 0.95 answer-channel thresholds
under the candidate parser. This does not reclassify Stage A1, alter its journal,
or convert its positive textual result into a Stage-A pass.

## 10. Is a fresh Stage-A2 justified?

Yes. The failure mechanism is specific; the parser-only repair is generic,
condition-symmetric, exact, adversarially tested, and fail-closed; it does not
inject CAREFUL semantics or use correctness to resolve ambiguity; it produces
substantial Stage-A1 diagnostic recovery; and 20 untouched reserve families
remain. A fresh Stage-A2 is therefore scientifically justified, but it is only
a draft and is not authorized for execution.

## 11. Stage-A2 sample-size comparison

| Property | Option 1: 20 reserve | Option 2: 30 (20 reserve + 10 Stage B) |
|---|---:|---:|
| Stage-A2 trajectories | 80 | 120 |
| Rows/condition | 40 | 60 |
| Validity step | 0.0250 | 0.0167 |
| Invalid rows allowed at 0.95 | 2 | 3 |
| Wrong-both family minimum | 2 | 3 |
| Correct-at-least-once minimum | 4 | 6 |
| Stage-B families retained | 130 | 120 |
| Full-transfer Stage-B joint planning power | 0.81514 | 0.78552 |
| 75%-transfer Stage-B joint planning power | 0.55722 | 0.52326 |
| Empirical-mix runtime with 25% margin | 1.00 h | 1.50 h |
| Estimated raw journal | 0.39 MB | 0.58 MB |

Option 1 is selected. Its validity gate is coarser, but permits two invalid rows
per condition; Stage A1 already passed every opportunity gate comfortably; and
using only the 20 reserve families preserves all 130 Stage-B families and the
prospectively planned Stage-B power. Both options consume all untouched reserve.

## 12. Prospective Amendment 2

`AMENDMENT2_LOCK_DRAFT.json` freezes the parser-only repair, unchanged prompts
and generation, 20 untouched reserve families, one deterministic row per
family, 80 unique Stage-A2 logical keys and seeds, unchanged thresholds with
count gates translated to 2 and 4 families, the existing retry/resume policy,
and the original Spark-2 environment fingerprint.

The Stage-A1 schedule is marked `EXECUTED_AND_FAILED_AS_FROZEN`. Stage-A2 has
zero outcomes and status `DRAFT_AWAITING_PRINCIPAL_RESEARCHER_FREEZE_NOT_EXECUTED`.
Its family manifest and schedule are distinct from A1 and disjoint from all 130
Stage-B families; all seeds are unique and disjoint from both older schedules.

## 13. Stage-B status

`NOT_AUTHORIZED_NOT_OPENED`

Meaningful-controller LiveCodeBench trajectories: 0. Activation-null
LiveCodeBench trajectories: 0. The fixed L27-D75 controller and all nulls remain
sealed.

## 14. Repository and resource state

- new model inference: 0
- Spark 1 used: NO
- Spark 2 scientific inference: NO
- RunPod used: NO
- Q2 outputs inspected: NO
- Stage-B controller trajectories: 0
- Stage-B null trajectories: 0
- historical Stage-A1 classification modified: NO
- Stage-A2 executed: NO

`Q1_SECOND_TASK_STAGE_A2_DESIGN_READY_FOR_PRINCIPAL_REVIEW`
