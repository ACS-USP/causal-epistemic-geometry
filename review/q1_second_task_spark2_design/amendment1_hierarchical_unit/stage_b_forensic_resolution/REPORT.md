# Q1 LiveCodeBench Stage-B Parser Forensic Resolution

## 1. Frozen primary result

The frozen primary classification remains
`Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY`. The meaningful-controller
estimate is `C = 0.006879844961240278` with 95% family-bootstrap interval
`[-0.010640373434704862, 0.0244633273703041]`. The meaningful-minus-null-mean
contrast is `0.011801207513416796`, interval
`[-0.0005907824612403139, 0.02443910256410256]`.

No raw row, primary parsed record, primary metric, bootstrap result, gate, or
historical classification was changed.

## 2. Parser implementations compared

The authoritative primary path was the prospectively pinned
`TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1` parser in
`src/epistemic_geometry/experiments/q1_second_task_stage_a_failure.py`, whose
frozen SHA-256 is
`0557447697a1f75caab0f9680863982fe2636393335e3fffd94223e6d92e50d9`.
The historical independent path used
`scripts/audit_q1_second_task_stage_a2.py::score` through
`scripts/audit_q1_second_task_stage_b.py`.

The resolution procedure was frozen first at commit `38c3575`. The historical
audit artifact remains immutable.

## 3. Static semantic differences

The static comparison found four historical-auditor deviations:

1. 27 rows: repair was skipped whenever a direct commitment payload existed,
   even when that payload was not literal-evaluable. The frozen primary applies
   its repair whenever the direct path is not semantically evaluable.
2. One row: an unsupported tuple was treated as a competing repair literal.
   The frozen repair candidate domain is only `bool`, `int`, `list`, and `str`.
3. One row: the auditor omitted the frozen mechanical-repetition precedence
   that blocks repair of a non-evaluable direct path.
4. One row: reference type was used to invalidate a direct, parsed commitment.
   Under the frozen direct path this is evaluable-wrong; expected type gates a
   repaired candidate only after mechanical selection.

The complete rule-by-rule table is in
`STATIC_SEMANTIC_DIFFERENCE_TABLE.md`. The primary source hash matches the lock;
no primary drift or prose/executable conflict was found.

## 4. All 30 disagreement cases

All 30 were audited. `DISAGREEMENT_CASES_SAFE.jsonl` contains only hashed
logical keys and structural metadata; it contains no raw model output,
benchmark text, reference value, item ID, or family ID.

Counts by condition were: BASELINE 3, TEXTUAL_CAREFUL 5, meaningful 2, R0 0,
R1 1, R2 1, R3 3, R4 0, R5 1, R6 10, and R7 4. Neither parser receives
condition identity, so this imbalance is descriptive only.

## 5. Normative ruling by root cause

Every realized difference is
`PRIMARY_MATCHES_LOCK_AUDIT_DEVIATES`. Correctness and reference value were not
used to decide the ruling. No item-specific exception, fuzzy matching, or LLM
judge was introduced.

## 6. Synthetic equivalence tests

Twenty-five synthetic tests cover the prospectively required marker, thinking,
literal, ambiguity, type, repetition, truncation, and repair-eligibility cases.
All 25 pass. The resolved auditor remains independently written and does not
import or invoke the primary Stage-A2 parser.

## 7. Final high-level resolution

`AUDIT_IMPLEMENTATION_NON_EQUIVALENT`

The disagreement was confined to the historical independent-audit parser. The
primary implementation is hash-correct and matches the frozen executable and
normative contract.

## 8. Corrected independent audit

The corrected independent parser was run over all 5,720 sealed rows. It has:

- parser-field disagreements: 0;
- maximum primary/audit metric difference: 0.0;
- classification agreement: true;
- forensic state: `Q1_SECOND_TASK_STAGE_B_FORENSIC_RESOLVED_PRIMARY_CONFIRMED`.

It reproduces the frozen P1/P2/split-half/safety decisions and the frozen
primary intervals exactly. The first disagreement artifact remains preserved
as provenance.

## 9. P2a sensitivity decomposition

This section is `POST_HOC_FORENSIC_SENSITIVITY_ONLY`.

The historical auditor changed 17 correctness/error indicators: 16 primary
correct→audit error and one primary error→audit correct. Counts were BASELINE
2, meaningful 2, R2 1 in the opposite direction, R3 2, R6 7, R7 1, and
TEXTUAL_CAREFUL 2. Other conditions had no correctness change.

The historical-audit P2a lower bound moved by `+0.0022445447599880736`, from
`-0.0005907824612403139` to `+0.0016537622987477594`. The pooled point contrast
moved by `+0.0018215190816934998`. Its frozen counterfactual decomposition is:

- baseline main contribution: `+0.0011935375670840739`;
- meaningful-condition main contribution: `+0.0012820512820512775`;
- eight null main contributions combined: `-0.0003577817531305854`;
- interaction residual: `-0.00029628801431126617`.

The maximum scalar discrepancy, `0.019230769230769273`, was the R6 semantic
evaluability summary (primary `0.9692307692307692`, historical audit `0.95`).
Condition-level and split-half details are frozen in
`P2A_SENSITIVITY_DECOMPOSITION.json`.

## 10. Terminal-class robustness

Across the primary and historical audit: P1 fails, P2b passes, split-half A
fails, split-half B passes, commitment-validity safety fails,
semantic-evaluability safety fails, accuracy safety passes, and the terminal
class is negative. P2a alone was parser-sensitive.

Thus the terminal transfer conclusion is parser-robust, while the historical
P2a discrepancy was parser-sensitive.

## 11. Scientific interpretation

The fixed controller did not satisfy the complete prospectively frozen
LiveCodeBench transfer criterion at N=130/R=4. This does not establish zero
transfer or exclude smaller effects. The corrected forensic result strengthens
provenance for the frozen negative classification; it does not create a new
scientific result or reclassify Stage B.

## 12. Repository and resource state

- new model inference: 0
- Stage-B rows modified: 0
- primary parser/results modified: NO
- Q2 inspected: NO
- Q3 run: NO
- Spark 1 scientific/GPU use: NO
- Spark 2 scientific/GPU/model use: NO
- Spark 2 read-only sealed-artifact access: YES

The last line distinguishes access to the machine hosting the private sealed
journal from scientific execution: no runner, model, GPU, or new trajectory was
started.

`Q1_SECOND_TASK_STAGE_B_FORENSIC_RESOLVED_PRIMARY_CONFIRMED`
