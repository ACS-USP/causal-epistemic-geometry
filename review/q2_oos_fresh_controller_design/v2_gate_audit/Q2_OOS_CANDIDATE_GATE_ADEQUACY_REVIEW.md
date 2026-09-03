# Q2 OOS Candidate-Gate Adequacy Review

## 1. V1 historical result

Q2 OOS V1 remains permanently `Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED` at
commit `249543e044f3d07713ac90dc6b68988e237f5119`. Its one immutable
19-candidate normalized-Gaussian PCG64DXSM stream had rank 8, entropy effective
rank 5.915762425524043, condition number 3.591939479153766, and maximum
absolute pair cosine 0.8843979744859711. It failed the frozen effective-rank
and condition-number gates before Spark safety qualification. The 19 vectors
remain `HISTORICAL_FAILED_STREAM`, `EXCLUDED_FROM_V2`, and
`NEVER_SEMANTICALLY_EXECUTED`.

## 2. Exact gate definitions

All algebra uses a `candidates × 8` coefficient matrix. If its singular values
are `s_1 >= ... >= s_8`, entropy effective rank is
`exp(-sum p_j log p_j)` with `p_j=s_j²/sum s²`; stable rank is
`sum s_j²/s_1²`; condition number is `s_1/s_8`; rank uses tolerance `1e-10`;
and pair cosine is the maximum absolute off-diagonal entry of `CCᵀ` for unit
rows. The V1 stream gate required rank 8, effective rank at least 6, condition
number at most 3, maximum absolute pair cosine below 0.98, and unit-norm error
at most `1e-12`.

The selected first-10-safe gate required rank 8, effective rank at least 4.8,
condition number at most 10, maximum pair cosine below 0.98, fresh-by-reference
A0 `q90-q10` at least 0.20, and shell-amplitude CV at most 0.03. A0 is
`1-cosine(c_fresh,c_reference)` against the exact frozen 31-reference bank.
Regression tests reproduce the public V1 values exactly and show that the
eight nonzero singular values of `C` and `CQᵀ` agree within `1e-12`.

## 3. N=19 realized-stream percentiles

Against 100,000 independently generated `n=19` streams:

- effective rank was at the 0.722nd percentile;
- condition number was at the 79.668th percentile (20.332% upper tail);
- maximum pair cosine was at the 76.911th percentile;
- 52.818% of streams failed at least one frozen stream gate;
- 1.463% failed both effective rank and condition number.

Thus V1's low effective rank was unusual, but the broader gate failure was not:
the prospectively selected `n=19` had only 47.182% probability of passing the
joint stream gate. This calibration result does not reopen V1.

## 4. Algebraic-gate pass probability by candidate count

Each row uses 100,000 streams; intervals are Wilson 95% Monte Carlo intervals.

| n | Joint pass | 95% CI | Median eff. rank | Median condition | Median max |cos| |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.01972 | [0.01888, 0.02060] | 5.9257 | 5.1139 | 0.7954 |
| 16 | 0.20842 | [0.20591, 0.21095] | 6.4062 | 3.5430 | 0.8287 |
| 19 | 0.47182 | [0.46873, 0.47492] | 6.6406 | 3.0397 | 0.8463 |
| 22 | 0.70999 | [0.70717, 0.71279] | 6.8168 | 2.7311 | 0.8594 |
| 24 | 0.82182 | [0.81944, 0.82418] | 6.9083 | 2.5854 | 0.8667 |
| 28 | 0.94355 | [0.94210, 0.94496] | 7.0603 | 2.3610 | 0.8785 |
| 32 | 0.98399 | [0.98319, 0.98475] | 7.1725 | 2.2068 | 0.8879 |
| 36 | 0.99435 | [0.99387, 0.99480] | 7.2615 | 2.0905 | 0.8954 |
| 40 | 0.99631 | [0.99591, 0.99667] | 7.3335 | 2.0014 | 0.9020 |
| 48 | 0.99622 | [0.99582, 0.99658] | 7.4420 | 1.8691 | 0.9119 |
| 56 | 0.99443 | [0.99395, 0.99487] | 7.5195 | 1.7736 | 0.9195 |
| 64 | 0.99292 | [0.99238, 0.99342] | 7.5791 | 1.7043 | 0.9257 |

The first audited sizes exceeding 0.90, 0.95, and 0.99 are 28, 32, and 36.
The slight decline after 40 is caused by the chance that at least one pair in a
large pool exceeds the fixed 0.98 cosine ceiling.

## 5. Safety-reserve probability by candidate count

Exact binomial probabilities of at least ten safe candidates are:

| n | p=.60 | p=.65 | p=.70 | p=.75 | p=.775 | p=.80 |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 0.81391 | 0.91253 | 0.96745 | 0.99110 | 0.99600 | 0.99842 |
| 22 | 0.94489 | 0.98204 | 0.99567 | 0.99930 | 0.99977 | 0.99994 |
| 24 | 0.97834 | 0.99451 | 0.99902 | 0.99989 | 0.99997 | 0.99999 |
| 28 | 0.99733 | 0.99960 | 0.99996 | 1.00000 | 1.00000 | 1.00000 |
| 32 | 0.99974 | 0.99998 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

The historical 40-candidate sample estimated `p_safe=0.775`. Its strongest
single-coordinate association gave a univariate standardized logistic slope
1.162 (bootstrap 95% interval 0.440 to 2.446), while the omnibus centroid
permutation test was not significant (`p=0.1785`). This small sample supports
sensitivity analysis, not a selection model.

## 6. Joint qualification probability

Under algebraic-safety independence, `P(stream pass and >=10 safe)` at
`p_safe=.60` is 0.384 for n=19, 0.804 for n=24, 0.941 for n=28, 0.984 for
n=32, and 0.994 for n=36. Exact Fréchet bounds are reported alongside these
products and do not assume independence. Direct weak/moderate axis-dependent
simulations closely matched independent reserve rates, but do not remove the
uncertainty from only 40 historical safety observations.

The full V1 stream gate and the original selected-bank condition gate together
remain poorly calibrated: increasing `n` cannot raise final qualification much
above the approximately 72% probability that a random `K=10,d=8` selected
bank satisfies condition number at most 10.

## 7. Selected-bank qualification simulation

Each cell used 20,000 streams and the exact first-ten-safe policy. At
`p_safe=.60`:

| n | Reserve | Original selected gate, unconditional | Inference-aligned gate, unconditional |
|---:|---:|---:|---:|
| 19 | 0.8106 | 0.5883 | 0.7900 |
| 22 | 0.9419 | 0.6808 | 0.9166 |
| 24 | 0.9776 | 0.7049 | 0.9523 |
| 28 | 0.9974 | 0.7227 | 0.9746 |
| 32 | 0.9998 | 0.7270 | 0.9747 |
| 36 | 1.0000 | 0.7254 | 0.9753 |

The inference-aligned column retains rank, effective rank, pair-cosine, and
cross-block-spread requirements but reports condition number descriptively.
At n=24 it was 0.9549 under the moderate historical axis association. Even the
deliberately localized top-axis ten had 0.9693 inference-aligned pass
probability. Cross-block A0 spread passed in every simulated cell; effective
rank passed about 97.5%; condition number alone passed only about 73%.

Crucially, complete streams can fail condition number at most 3 while their
selected ten remain usable: depending on n/scenario, roughly 56%--67% of
selected banks from full-stream condition failures passed the original
selected gate, and substantially more passed the inference-aligned gate.

## 8. Gate-to-inference alignment

| Gate | Intended protection | Alignment with K×31 test | Ruling for V2 draft |
|---|---|---|---|
| Full-stream rank | catastrophic dimension loss | indirect but basic | retain |
| Full-stream effective rank >=6 | spherical pool coverage | redundant with selected bank | descriptive |
| Full-stream condition <=3 | spherical pool conditioning | weak; unused reserves drive it | descriptive |
| Full-stream max cosine <.98 | near-duplicate candidates | gross integrity | retain |
| Selected rank | identifiable 8-D support | direct | retain |
| Selected effective rank >=4.8 | gross multidimensional collapse | direct | retain |
| Selected condition <=10 | numerical aesthetics of 10×8 C | did not predict cross-block power | descriptive |
| Selected max cosine <.98 | duplicate analyzed rows | direct | retain |
| Cross-block A0 spread >=.20 | rank dynamic range | direct | retain |
| Shell amplitude CV <=.03 | implemented-dose comparability | direct, model-measured | retain |

The full-stream condition gate was primarily heuristic and duplicated a
stricter-than-necessary concern already assessed on the selected bank. The
selected condition gate rejected about 27% of otherwise random banks without
detectable protection of the actual relational test.

## 9. Cross-block power analysis

Across 1,200 moderate-safety-selected banks, the frozen fresh-row permutation
test at half the closed A0 association (`rho=0.281909`) had planning power 1.0
and all-LOFO-positive frequency 1.0 with 999 maps. Observed synthetic rho had
95% range 0.2652--0.3027. Binary power was therefore ceiling-saturated.

Using continuous achieved rho to avoid that ceiling, standardized multivariable
coefficients after simultaneous adjustment were 0.029 for A0 spread, 0.003 for
row diversity, 0.006 for effective rank, and 0.019 for log condition number.
No material incremental relation between selected condition/effective rank and
achieved association remained after cross-block diagnostics. This does not
claim universal irrelevance; it shows the old condition threshold was not
calibrated to protect power in the frozen planning regime.

## 10. Route A — existing stream

Rejected. V1 failed frozen gates before safety, and no first-ten-safe bank
exists. Reinterpreting its gates after seeing 5.915762 and 3.591939 would be a
retroactive rescue even though this audit shows those gates were poorly
calibrated. V1 remains blocked and its vectors remain public historical
development evidence only.

## 11. Route B — larger stream, same gates

A one-shot n=36 stream reaches 0.994 candidate-stream pass probability and
essentially certain safety reserve at plausible rates. However, the original
selected condition gate still caps final selected-bank qualification near
72%. Route B spends additional safety GPU time while retaining the principal
inference-misaligned bottleneck. It is defensible but inefficient and is not
recommended.

## 12. Route C — inference-aligned gates

Recommended. A wholly new, one-shot n=24 stream reaches 0.978 probability of
ten safe candidates even at `p_safe=.60` and 0.952--0.955 unconditional
qualification under independent/moderate safety scenarios when the selected
gate protects the actual K×31 geometry rather than unused-pool conditioning.
The full-stream and selected-bank condition numbers remain mandatory reported
diagnostics, not pass/fail gates. A frozen per-bank synthetic power check adds
a direct presemantic guard.

## 13. Route D — deterministic design

A spherical code, maximin bank, or low-discrepancy design would improve
coverage predictably, but it changes the target from random fresh-controller
generalization to designed-intervention geometry. It should be treated as a
different future experiment, not a replacement for this OOS replication.

## 14. Recommended V2 protocol

The draft fixes `K=10`, `n=24`, namespace
`Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V2`, PCG64DXSM normalized Gaussian draws,
and seed derivation from a future committed PRELOCK. Exactly one stream may be
generated. It must be publicly committed before model use, exclude all V1
IDs/coefficients/hashes, and cannot be redrawn.

The stream stops only for corruption, rank loss, near duplication, or
historical overlap. Spark-1 safety then evaluates all 24 candidates at both
frozen shells and selects the first ten safe in immutable order. The selected
bank must pass rank 8, effective rank 4.8, max pair cosine 0.98, cross-block A0
spread 0.20, amplitude CV 0.03, and a frozen synthetic power/LOFO gate.
Condition number and row diversity are required reports. A1 and A2 retain the
V1 qualification rules. No semantic execution is authorized by this draft.

Projected future scale is 1,152 safety trajectories and, only after a later
prediction lock and authorization, 12,000 semantic trajectories. Linear
historical extrapolation gives about 1.53 Spark-1 GPU-hours for safety and
27.40 GPU-hours for semantic collection; A2 requires a new outcome-free
throughput preflight because no canonical elapsed-time artifact exists.

## 15. Draft frozen objects

- `AUDIT_PRECHECK.json`: seeds, counts, scenarios, and Monte Carlo precision;
- `V2_PROTOCOL_DRAFT.json`: complete prospective V2 design;
- `ALGEBRAIC_GATE_CALIBRATION.csv`: 1.2 million stream summaries;
- `SAFETY_RESERVE.csv` and `JOINT_ALGEBRAIC_SAFETY_FEASIBILITY.csv`;
- `SELECTED_BANK_CALIBRATION.csv`: independent, weak, moderate, and adversarial selection;
- `CROSS_BLOCK_POWER.csv` and `CROSS_BLOCK_POWER_SUMMARY.json`;
- `V1_REALIZED_PERCENTILES.json` and `HISTORICAL_SAFETY_GEOMETRY.json`.

No V2 seed value, coefficient, intervention vector, safety schedule, A2 capture,
or semantic schedule was materialized.

## 16. Repository/resource state

- new controller stream generated: NO
- V1 vectors altered: NO
- model inference: 0
- semantic trajectories: 0
- correctness inspected: NO
- Spark 1 used: NO
- Spark 2 used: NO
- RunPod used: NO
- LiveCodeBench raw outputs inspected: NO
- Q3 run: NO

`Q2_OOS_V2_INFERENCE_ALIGNED_DESIGN_READY_FOR_PRINCIPAL_REVIEW`
