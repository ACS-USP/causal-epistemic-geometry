# Q1 Second-Task Spark-2 Design Review

## A. Why Q2 must not be parallelized

Q2 V4.1 is already open under a frozen 37,800-row schedule on Spark 1. Moving,
sharding, or duplicating it onto Spark 2 would introduce a new backend and
scheduler after opening, complicate duplicate prevention, and risk outcome
leakage. It also would not address Q1's principal remaining limitation: whether
the fixed Qwen controller transfers beyond CRUXEval. This sprint did not read a
Q2 output path, inspect a Q2 outcome, or modify the Q2 process.

## B. LiveCodeBench model-free instrument audit

Classification: `LIVECODEBENCH_OUTPUT_INSTRUMENT_MODEL_FREE_PASS`.

- Official code source: LiveCodeBench commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (MIT code license).
- Official dataset: `livecodebench/test_generation`, revision
  `6f3ac40bbecf81eba15899139d279b077f2816fd`; pinned parquet SHA-256
  `4826aa00c059d6d47a099606ceed2d0e51d3aeeb1868f1bcf349a038bb64b4b1`.
- Pool: 442 exact test-output rows from 182 question families. Every row has
  one functional test and a deterministic JSON reference. Stable IDs are
  `question_id:test_id`.
- References: 343 integers, 45 lists, 33 booleans, and 21 strings. All 442
  round-trip through the frozen exact typed evaluator.
- Evaluator: external-semantic-v3 final commitment followed by
  `ast.literal_eval`/JSON parsing and exact typed comparison. No LLM judge,
  fuzzy matching, or execution of model-generated code is used.
- The upstream evaluator was audited but calls Python `eval`; it is not used.
  Consequently this is a mechanically exact CEG instrument over official
  references, not a claim of byte-identical leaderboard scoring.
- The complete 182-question pool was compared with all 800 rows of the pinned
  CRUXEval test revision. Exact normalized-code collisions: 0. Five-token
  shingle pairs at Jaccard >=0.80: 0; maximum observed: 0.0.
- The historical 2048-token run remains
  `LOW_CAP_DIAGNOSTIC_NOT_SCIENTIFIC_EVIDENCE` and supplied no outcomes or
  thresholds to this design.

Licensing caveat: the dataset card records only `cc`, without a CC variant.
No benchmark text, code, or answer is redistributed in Git; only IDs and
hashes are sealed. Publication redistribution needs a separate license review.
This does not prevent the internally reproducible, hash-pinned instrument.

## C. Exact fixed Qwen controller provenance

The meaningful arm is unchanged from frozen Q1:

| Field | Frozen value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model/tokenizer revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Layer | zero-based L27 |
| Dose | D75, eta `9.637427952852196` |
| Reference scale | `10.153299177386142` |
| Effective delta norm | `97.85168930581241` |
| Canonical vector hash | `e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838` |
| Vector-file hash | `b1630039fcbb829028a0e8f9f521d7e87bb24e831bc81c74a1591a6c39f40772` |
| Timing | sustained current token |
| Scope | final prompt token at prefill; current token once per decode forward |

No LiveCodeBench activations, correctness labels, prompts, or outcomes entered
controller construction.

## D. Spark-2 native engine qualification

Classification: `SPARK2_NATIVE_ENGINE_QUALIFIED`.

The qualified scientific claim is explicitly
`SPARK2_NATIVE_CROSS_BACKEND_REPLICATION`, not A40-to-GB10 numerical
equivalence. The environment is one NVIDIA GB10, aarch64, Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13.0, Transformers 4.57.6, BF16, SDPA, and the exact
Qwen revision. The environment fingerprint is
`306d65af9643cc1144d344ae57141ac96ffbbcf70520f67e9276a907d29660bc`.

All frozen checks passed: tokenization, alpha-zero token identity, seed
repeatability, vector identity, hook cleanup, current-token/cache semantics,
one application per intended forward, equal meaningful/null delta norms,
parser roundtrip, and synthetic journal resume. The maximum observed shift
error was 0.5 BF16 epsilon; non-current-token change was 0.0; 73 intervention
applications matched 73 forwards.

The smoke generated 177 synthetic tokens in 15.354 seconds (11.528 token/s),
persisted no text, processed zero benchmark items, and inspected no
correctness. Four pre-output engineering incidents and their prospective or
deterministic repairs are preserved in `ENGINE_INCIDENT_AND_AMENDMENT.json`.

## E. Stage-A opportunity/textual-careful design

Stage A is a disjoint 50-item, two-rollout DEVELOPMENT split selected by whole
question family. It has exactly two conditions: `BASELINE` and
`TEXTUAL_CAREFUL`; no activation vector is opened.

The gate is conjunctive except for the stated textual-usefulness alternatives:

- baseline validity and evaluability >=0.95;
- pooled baseline accuracy in [0.25, 0.90];
- repeated-baseline B00 >=0.05;
- at least 5 items wrong twice and at least 10 correct at least once;
- textual validity and evaluability >=0.95;
- textual accuracy >= baseline -0.03;
- and at least one of: accuracy gain >=0.03, mean tokens >=1.5x baseline, or
  median tokens >= baseline +10.

Exactly 200 logical rows are frozen. Stage A failure returns
`Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED`; the meaningful controller remains
unopened. Stage A requires a separate principal authorization.

## F. Stage-B fixed-controller design

Stage B is a fresh 150-item holdout, disjoint from Stage A and reserve at the
question-family level. It uses four independent rollout seeds and 11 conditions:
baseline, textual CAREFUL, the fixed meaningful L27-D75 controller, and eight
nulls. Total: `150 x 11 x 4 = 6,600` frozen logical trajectories.

The primary endpoint is pooled-R4 competence-adjusted complementarity `C`.
Pass requires a positive 95% item-bootstrap lower bound for meaningful C, a
positive lower bound for meaningful-minus-eight-null-mean C, meaningful C
strictly above every null point estimate, both predesignated R2 halves positive
on C and the null-mean contrast, and all frozen safety guards. Secondary
quantities are G, D, rescue, damage, and accuracy. Items are never filtered;
invalid terminal outcomes are errors.

## G. Four-rollout estimator derivation

The pooled estimator is the U-statistic specified in
`R4_ESTIMATOR_DERIVATION.md`. Within-condition products exclude self-products;
crossed products for D use `r != s`; between-item competence terms use
`t != u`. At R=2 the implementation is exactly equal to the independent
canonical Q1 estimator, not only equal in expectation. Split A is rollouts
`{0,1}` and split B is `{2,3}`. The item is the 50,000-resample bootstrap unit,
all conditions/rollouts move together, and negative D estimates are retained.

## H. Random-bank design

R0-R3 are reused byte-for-byte from the Q1 confirmatory lock. R4-R5 are new
isotropic Gaussian L27 directions; R6-R7 use the canonical
construction-matched sign-shuffled source-pair procedure. Their seeds were
frozen once before benchmark inference, and each new vector was Gram-Schmidt
projected against the meaningful vector and all earlier nulls. There was no
redraw or behavioral selection. The maximum absolute off-diagonal cosine over
the meaningful-plus-eight-null bank is `8.88e-16`. Every condition uses the
same layer, timing, scope, eta, and effective delta norm.

## I. Item/split provenance and contamination audit

The deterministic whole-question split is:

| Role | Items | Question families | Outcomes opened? |
|---|---:|---:|---|
| Stage A development | 50 | frozen manifest | No |
| Stage B holdout | 150 | frozen manifest | No |
| Unallocated reserve | 242 | 99 | No |

All 442 item hashes are unique. Stage A, Stage B, and reserve have no shared
question ID. Schedule keys and seeds are unique both within and across stages.
The manifests contain IDs and hashes, not benchmark content.

## J. Power and precision table

Planning uses only frozen Qwen Q1 C/null estimates and intervals. The estimated
item ICC is 0.8767; therefore R=4 yields modest precision improvement over R=2
but adds independently frozen split-half reliability. Values below use eight
nulls and 100,000 deterministic Monte Carlo replicates per cell.

| N | R | Expected C CI width | Expected delta-C CI width | Joint power, 100% transfer | Joint power, 75% transfer | Null false-positive |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2 | 0.0622 | 0.0518 | 0.698 | 0.430 | 0.0021 |
| 100 | 4 | 0.0612 | 0.0509 | 0.712 | 0.442 | 0.0023 |
| 120 | 2 | 0.0568 | 0.0472 | 0.772 | 0.511 | 0.0021 |
| 120 | 4 | 0.0558 | 0.0465 | 0.786 | 0.523 | 0.0024 |
| 150 | 2 | 0.0508 | 0.0423 | 0.853 | 0.609 | 0.0025 |
| **150** | **4** | **0.0499** | **0.0416** | **0.862** | **0.623** | **0.0025** |
| 200 | 2 | 0.0440 | 0.0366 | 0.926 | 0.730 | 0.0025 |
| 200 | 4 | 0.0433 | 0.0360 | 0.932 | 0.745 | 0.0022 |

At N=150/R=4 the full historical effect exceeds the approximately-80% planning
target. At 75% transfer the design has only 62% joint power, and weaker transfer
can be missed; this limitation is explicit. Eight rather than four nulls makes
the point-max criterion harder and improves specificity rather than maximizing
the chance of a pass. The final N=150 choice was made prospectively from this
table and the synthetic throughput, before benchmark inference.

## K. Runtime and storage estimate

The synthetic sequential smoke measured 11.528 generated token/s. The table is
for 6,600 Stage-B rows and includes a 25% safety margin; prompt-length and
long-tail effects may make real execution slower.

| Mean generated tokens/row | Projected Stage-B wall time |
|---:|---:|
| 128 | 25.5 h |
| 256 | 50.9 h |
| 512 | 101.8 h |
| 1,024 | 203.6 h |
| 4,096 worst case | 814 h |

Stage A adds about 1.5 h at 256 tokens/row with the same margin. Thus N=150 is
weekend-feasible only if the realized mean remains roughly <=300 tokens; this
is a projection, not a promise or a stop rule. Reserve at least 1.5 GB for raw
JSONL, token IDs, manifests, seals, and audits under a worst-case token cap.

## L. Frozen classification table

| State | Mechanical meaning |
|---|---|
| `Q1_SECOND_TASK_FIXED_CONTROLLER_PASS` | Stage A and engine qualify; P1, both P2 checks, both split halves, and all safety guards pass |
| `Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL` | Scientific complementarity conjunction passes but any validity, evaluability, or competence guard fails |
| `Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY` | Instrument/execution qualify but the frozen complementarity conjunction fails |
| `Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED` | Model-free, Spark-2, or Stage-A opportunity/textual gate fails |
| `Q1_SECOND_TASK_EXECUTION_INCOMPLETE` | Frozen Stage-B logical keys cannot be completed under the retry policy |

Safety reuses Q1 exactly: meaningful validity no more than 5 points below
baseline, evaluability no more than 5 points below baseline, and accuracy no
more than 10 points below baseline. The single conjunctive primary hypothesis
uses a 50,000-resample item bootstrap; no post-hoc endpoint switching is allowed.

## M. Comparison with the three alternatives

| Option | Scientific value | Protocol risk | Complexity/runtime | Paper contribution |
|---|---|---|---|---|
| Parallelize open Q2 onto Spark 2 | Low incremental | Unacceptably high after opening | High coordination risk | Does not address Q1 task specificity |
| Add rollouts to opened 57-item Q1 holdout | Low; post-opening precision only | High; changes confirmatory design | Moderate | Cannot establish second-task transfer |
| Fresh same-task CRUXEval precision replication | Moderate | Low under a new lock | Lower than LiveCodeBench | Narrows uncertainty but leaves domain/task limitation |
| Fixed-controller LiveCodeBench replication | Highest for current limitation | Controlled by two-stage lock | 200 + 6,600 future rows | Tests transfer to a second objective program-execution task |

Alternatives A-C were not executed.

## N. Scientific claim boundary

A positive result would support transfer between CRUXEval output prediction and
one second objective program-execution task on Qwen3-8B under a Spark-2-native
backend. It would not establish domain-general, model-general, architecture-
general, or cross-backend-equivalent control. A negative result would bound the
fixed controller as more task-specific, conditional on Stage A having qualified
the benchmark opportunity. It would not rewrite the frozen Q1 confirmatory pass
or the character-count boundary result.

## O. Repository state

The work is isolated on `research/q1-second-task-spark2-design`, based on the
then-current canonical `origin/main` at
`0e6b40d3b7bcc9dbfe430a8721644a84eed2bf89`. Historical Q1/Q2 classifications,
the current Research OS state, and frozen reports are unmodified. The design
has its own manifests, schedules, vector bank, estimator lock, governance lock,
engine qualification, power grid, and independent model-free audit.

## P. Resource state

- Spark 1 used: **NO**
- Spark 2 scientific inference: **NO**
- Spark 2 synthetic engine smoke: **YES**, one GB10
- RunPod used: **NO**
- Q2 outputs inspected: **NO**
- Q2 process modified: **NO**
- Q1 historical result modified: **NO**
- New scientific benchmark outcomes: **0**
- Correctness inspected: **NO**
- Stage A executed: **NO**
- Stage B executed: **NO**
- Q3: **NOT RUN**

The design is complete and awaits principal review. Neither Stage A nor Stage B
is authorized by these artifacts.

`Q1_SECOND_TASK_DESIGN_READY_FOR_PRINCIPAL_REVIEW`
