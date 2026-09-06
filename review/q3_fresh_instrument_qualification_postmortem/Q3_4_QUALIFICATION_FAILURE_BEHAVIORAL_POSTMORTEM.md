# Q3.4 Qualification Failure: Behavioral Postmortem

Status: `POST_HOC_DESCRIPTIVE_ONLY`

The immutable historical result remains:

```text
Q3_FRESH_INSTRUMENT_NOT_QUALIFIED
Q3_FRESH_INSTRUMENT_QUALIFICATION_FORENSIC_CLEAN
```

This audit used only the sealed 6,000-row qualification journal, the frozen
300-family qualification dataset, and the already persisted private score
table. It performed no model inference, did not inspect raw model text
manually, and did not access the confirmation or reserve populations with
Qwen. Its purpose is explanation and closure, not reclassification.

## 1. Coincidence of correct answers

Seven fixed policies, including the champion, each scored 80/600. Six of them
had exactly the same 80 correct logical rows: 40 families correct in both
rollouts. `Q2_OOS_V2_DIRECTION_03_MEDIUM` differed only at six rows: it shared
77 correct rows with the common set, omitted three, and added three. Across all
seven policies, the intersection was 77 rows and the union was 83 rows.

| Correct-set quantity | Count |
|---|---:|
| Correct under all seven policies | 77 rows |
| Correct under exactly six policies | 3 rows |
| Correct under exactly one policy | 3 rows |
| Union across seven policies | 83 rows |
| Family intersection | 39 families |
| Family union | 43 families |
| Pairwise Jaccard among the common six | 1.0000 |
| Pairwise Jaccard between `DIRECTION_03` and each common policy | 0.9277 |

The common six were `Q2_OOS_V2_DIRECTION_13_MEDIUM`,
`Q2_OOS_V2_DIRECTION_16_MEDIUM`, `V4_DIRECTION_02_MEDIUM` (the champion),
`V4_DIRECTION_10_MEDIUM`, `V4_DIRECTION_19_MEDIUM`, and
`V4_DIRECTION_31_MEDIUM`.

The overlap was driven almost entirely by output type. The common six had all
80 correct rows in the Boolean stratum and zero correct rows for integer,
string, list, tuple, or dictionary outputs. `DIRECTION_03` had 78 Boolean, one
list, and one string success.

| Output type | Rows per policy | Common-six correct (each) | `DIRECTION_03` correct |
|---|---:|---:|---:|
| Boolean | 128 | 80 | 78 |
| Integer | 128 | 0 | 0 |
| String | 128 | 0 | 1 |
| List | 88 | 0 | 1 |
| Tuple | 64 | 0 | 0 |
| Dictionary | 64 | 0 | 0 |

Within the common six, the Boolean prediction was one constant literal on all
128 Boolean rows. That single-literal behavior mechanically produced the same
80 correct rows for all six policies. `DIRECTION_03` produced two Boolean
values, but its modal value still occupied 123/126 evaluable Boolean rows
(97.62%). No literal answer value is reproduced in this report.

The apparent archetype differences are not independent evidence because
output type, archetype, and complexity are coupled by the deterministic
generator allocation. Descriptively, each common policy's 80 correct rows
split as follows:

| Archetype | Correct | Complexity | Correct |
|---|---:|---|---:|
| Arithmetic | 12 | 6 | 18 |
| Branching | 6 | 8 | 20 |
| Sequence aliasing | 4 | 10 | 20 |
| Text | 14 | 12 | 22 |
| Mapping | 10 |  |  |
| Nested control | 12 |  |  |
| Bounded recursion | 10 |  |  |
| Mixed | 12 |  |  |

Across the seven policies, 357--395 distinct canonical predicted values were
present among 592--600 evaluable rows. The global modal value occupied
20.67%--21.62% of evaluable rows, the top five occupied 30.00%--35.30%, and
simple constants occupied 21.18%--24.25%. Thus the striking agreement is a
Boolean-collapse phenomenon, not evidence that every output type collapsed to
one constant or that only a tiny set of families received all distinct values.

## 2. Routing and cost

The online router selected only two of the eight bank policies:

| Selected policy | Selections | Share | Correct | Evaluable | Mean / median tokens | P95 tokens | Mean / median seconds | P95 seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `V4_DIRECTION_32_MEDIUM` | 232 | 38.67% | 30 | 232 | 10.58 / 9 | 23 | 1.13 / 0.95 | 2.27 |
| `V4_DIRECTION_35_STRONG` | 368 | 61.33% | 51 | 332 | 637.12 / 23 | 2,707.6 | 57.60 / 2.21 | 246.11 |

No fallback occurred. The champion is outside the eight-policy argmax order,
so a champion selection would have been the mechanically identifiable
non-finite-state fallback; its count was zero. Every one of the router's 36
invalid or unevaluable rows occurred when `V4_DIRECTION_35_STRONG` was
selected. Those selections consumed 98.96% of routed generated tokens and
98.78% of routed generation time.

Against the separately sampled fixed-policy observation for the same family,
rollout index, and selected policy, the router comparison was:

| Outcome pair | Rows |
|---|---:|
| Both correct | 71 |
| Router only correct | 10 |
| Fixed policy only correct | 9 |
| Neither correct | 510 |

This is a net difference of one correct row, matching the descriptive
router-minus-champion gain of +0.001667. The router and fixed conditions used
different frozen seeds, so answer identity is neither expected nor required:
among 535 pairs evaluable on both sides, 143 (26.73%) had the same canonical
value. This disagreement is not an execution defect.

The full per-condition cost distribution was strongly right-skewed:

| Condition | Mean / median tokens | P95 / P99 tokens | Mean / median seconds | P95 / P99 seconds |
|---|---:|---:|---:|---:|
| `ONLINE_ROUTED` | 394.86 / 14 | 2,442.3 / 3,225.0 | 35.77 / 1.44 | 221.01 / 294.81 |
| `Q2_OOS_V2_DIRECTION_03_MEDIUM` | 158.55 / 9 | 1,751.4 / 3,577.3 | 14.60 / 1.01 | 158.64 / 329.94 |
| `Q2_OOS_V2_DIRECTION_13_MEDIUM` | 12.58 / 9 | 23.0 / 24.0 | 1.28 / 0.99 | 2.28 / 2.38 |
| `Q2_OOS_V2_DIRECTION_16_MEDIUM` | 15.40 / 9 | 23.0 / 256.0 | 1.53 / 1.01 | 2.29 / 22.32 |
| `V4_DIRECTION_02_MEDIUM` | 11.92 / 9 | 23.0 / 24.0 | 1.23 / 1.00 | 2.26 / 2.37 |
| `V4_DIRECTION_10_MEDIUM` | 15.34 / 9 | 23.0 / 256.0 | 1.52 / 1.01 | 2.29 / 22.41 |
| `V4_DIRECTION_19_MEDIUM` | 12.42 / 9 | 23.0 / 24.0 | 1.27 / 0.99 | 2.22 / 2.41 |
| `V4_DIRECTION_31_MEDIUM` | 11.64 / 9 | 23.0 / 24.0 | 1.20 / 0.98 | 2.22 / 2.35 |
| `V4_DIRECTION_32_MEDIUM` | 12.98 / 9 | 24.0 / 27.0 | 1.32 / 0.99 | 2.29 / 2.52 |
| `V4_DIRECTION_35_STRONG` | 451.40 / 19 | 2,299.6 / 3,209.5 | 40.86 / 1.80 | 207.93 / 293.88 |

Across all 6,000 rows, summed generation time was 60,346.6 seconds and
generated tokens were 658,254. The median row was only 9 tokens and 1.04
seconds, but P95 was 709.3 tokens and 62.67 seconds. This explains why a
median-based forecast would materially understate the campaign tail.

## 3. Focal interface and evaluation audit

The aggregate audit found no evidence of a prompt-interface, family-swap, or
typed-comparison defect:

- all 300 private prompts reconstructed byte-for-byte from the frozen complete
  source program and input;
- public-manifest, schedule, and journal prompt hashes had zero mismatches;
- all 6,000 schedule indices, seeds, and logical keys matched;
- rendered-prompt hashes were deterministic within every family and unique
  across all 300 families;
- all 600 online rows selected exactly once and retained the same policy
  through decoding;
- the backend tokenization path used no truncation argument.

Input-token counts were not persisted, so prompt-token truncation cannot be
audited directly from the journal. The exact prompt-byte and code-path evidence
is consistent with complete delivery and contains no positive evidence of
truncation, but it is not equivalent to a persisted token-count proof.

| Frozen status | Rows | Interpretation |
|---|---:|---|
| `VALID_CORRECT` | 799 | Mechanically correct typed commitment |
| `VALID_WRONG` | 5,075 | Valid/evaluable but unequal typed value |
| `INVALID_FORMAT` | 76 | No mechanically acceptable final commitment |
| Repetition stop | 44 | Frozen terminal failure |
| Hard cap | 6 | Frozen terminal failure |

Among 5,874 evaluable rows, 5,872 had the expected reference type and two had a
different type; those two remained valid wrong answers under exact typed
comparison. All 50 mechanical terminal rows were forced incorrect exactly as
frozen. No answer was manually adjudicated and no parser rule changed.

Dual-reference agreement and the independent forensic scorer establish
consistency with the frozen implementations. They do not logically exclude a
shared specification error. This focal audit found no concrete evidence of
one.

## 4. Distribution change from closed CRUXEval development

The new qualification population was much longer and structurally denser than
the closed 300-item CRUXEval panel:

| Aggregate measure | Q3.4 qualification | Closed CRUXEval |
|---|---:|---:|
| Prompt characters, median (mean) | 2,136 (2,158.2) | 340.5 (348.7) |
| Source characters, median (mean) | 1,876.5 (1,897.5) | 130.5 (135.1) |
| Source lines, median (mean) | 57 (57.9) | 5 (5.4) |
| AST nodes, median (mean) | 600 (603.1) | 39 (41.5) |
| Statements, median (mean) | 45 (45.6) | 5 (5.2) |
| Branches, median (mean) | 5 (5.55) | 0 (0.55) |
| Loops, median (mean) | 5 (5.43) | 0 (0.54) |
| Input characters, median (mean) | 62 (62.7) | 16 (18.6) |

Output-type composition also shifted. Q3.4 deliberately allocated 64 Boolean,
64 integer, 64 string, 44 list, 32 tuple, and 32 dictionary families. The
closed CRUXEval panel contained 20 Boolean, 47 integer, 132 string, 70 list, 6
tuple, 22 dictionary, 2 bytes, and 1 float reference.

Q3 operation labels are generator-IR categories, whereas CRUXEval composition
can only be summarized here through AST-node proxies; they are not a shared
semantic taxonomy. No prompt representation was captured on the new
population, so this audit cannot measure feature-space shift or explain how
the frozen router scores changed internally.

## 5. Ruling and lessons

### Facts supported by the sealed data

1. The fixed bank provided almost no useful policy differentiation: six
   policies had identical correct rows and the frozen oracle headroom was only
   0.015, below the 0.05 gate.
2. Their apparent competence was almost entirely one constant Boolean answer;
   no common-six policy solved a non-Boolean row.
3. The router selected only two policies, favored the costly STRONG policy on
   61.33% of rows, and achieved no material gain (+0.1667 percentage points).
4. The STRONG selections account for all router invalidity and almost all
   routed compute.
5. The new generated population was substantially longer and structurally
   denser than the closed CRUXEval development panel.
6. The focal checks found no evidence of prompt omission, family exchange,
   schedule mismatch, or scoring-interface failure.

### Best-supported explanation

The qualification failed because the frozen portfolio had inadequate
competence and inadequate complementary opportunity on the specified
restricted-Python population. The bank's common behavior collapsed onto a
frequent Boolean commitment, while the router concentrated execution on a
high-cost STRONG policy without creating material utility. The marked
distribution shift is a plausible contributor to this loss of competence,
but it is descriptive rather than a causal attribution.

### What cannot be concluded

The data do not identify whether the underlying cause is task length,
particular operation mixtures, prompt-feature shift, steering sensitivity, or
another shared model behavior. No unsteered condition was run, so this audit
does not show that unsteered Qwen failed. Subgroups discovered after closeout
are not new endpoints, and the Boolean stratum does not define a favorable
confirmation population.

Confirmation should remain closed because three mandatory scientific gates
failed: champion accuracy 0.1333 was below the 0.25 floor, bank oracle headroom
0.015 was below 0.05, and router validity/evaluability 0.94 were below 0.95.
Opening 1,000 confirmation families would therefore test utility with an
instrument already shown to lack adequate difficulty calibration,
opportunity, and answer-channel reliability. It would also incur an estimated
20.5--22.3 hours at R=2 under the qualification-derived forecast.

The practical lessons are to require pre-confirmation competence and oracle
opportunity gates, retain tail-aware runtime forecasts, and test whether a
frozen portfolio has genuinely differentiated successes before committing a
large confirmation campaign. This report authorizes no replacement system,
new benchmark, new tournament, or next candidacy.

## Sources and release boundary

- Historical closeout:
  [`Q3_FRESH_INSTRUMENT_QUALIFICATION_CLOSEOUT.md`](../q3_fresh_instrument_qualification_closeout/Q3_FRESH_INSTRUMENT_QUALIFICATION_CLOSEOUT.md)
- Frozen aggregate result:
  [`Q3_FRESH_QUALIFICATION_RESULT.json`](../q3_fresh_instrument_qualification_closeout/Q3_FRESH_QUALIFICATION_RESULT.json)
- Independent audit:
  [`Q3_FRESH_QUALIFICATION_FORENSIC_AUDIT.json`](../q3_fresh_instrument_qualification_closeout/Q3_FRESH_QUALIFICATION_FORENSIC_AUDIT.json)
- Qualification schedule and system lock:
  [`Q3_FRESH_QUALIFICATION_SCHEDULE.json`](../q3_fresh_instrument_qualification/Q3_FRESH_QUALIFICATION_SCHEDULE.json),
  [`Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json`](../q3_fresh_instrument_qualification/Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json)
- Public family metadata:
  [`QUALIFICATION_FAMILY_MANIFEST.json`](../q3_fresh_instrument_qualification/QUALIFICATION_FAMILY_MANIFEST.json)
- Closed CRUXEval panel:
  [`SEMANTIC_PANEL_MANIFEST.json`](../q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json)
- Deterministic aggregate-only analyzer:
  [`analyze_q3_fresh_qualification_behavioral_postmortem.py`](../../scripts/analyze_q3_fresh_qualification_behavioral_postmortem.py)

Private source identities used by the analyzer were: raw journal SHA-256
`2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431`,
score table SHA-256
`c3b4ab47cf2422afb311fa978496e2abfbe5485ac76040ee3dcead2986ace533`,
and qualification dataset SHA-256
`c791e38c29d36a43fbac8ce00412e4c77d533665e0b8cb9eef8fa12fb918ac1d`.
The release-safe aggregate generated by this analysis has SHA-256
`5dd5c2775735218ecd856ce8af7cf8c843dce33008b2632861943b089d48c974`.
Raw prompts, source programs, references, model outputs, item-level scores,
family IDs, and literal answer values are not reproduced in this report.

No confirmation or reserve inference was performed. The historical
qualification classification is unchanged, and Q3 remains `NOT_RUN`. The
aggregate computation used Spark 1 CPU-only with CUDA disabled; Qwen was not
loaded, and Spark 2 and RunPod were not used.
