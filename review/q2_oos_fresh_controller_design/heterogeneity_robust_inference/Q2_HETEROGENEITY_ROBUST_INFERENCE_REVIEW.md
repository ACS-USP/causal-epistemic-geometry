# Q2 Heterogeneity-Robust Inference Review

## 1. Why row-QAP failed the stress null

The historical V2 stop remains `Q2_OOS_V2_NULL_CALIBRATION_BLOCKED`. Its strict
exchangeable-null FPR was 0.0512, whereas the unchanged reviewer-hardening stress
null reached 0.0758. The new frozen ablation independently reproduced the same
failure mode: row-QAP was calibrated under strict exchangeability (0.0496, Wilson
95% interval 0.0439--0.0560) and anti-conservative under the exact frozen stress
generator (0.0872, 0.0797--0.0953).

The cause is not a defective permutation implementation. Row permutation is exact
when complete fresh rows are exchangeable. The scientific null “aggregate rho is
zero”, however, is weaker: fresh controllers may have different latent response
profiles, intercepts, and geometry-related nuisance relations. Their positive and
negative row associations can average to approximately zero while the row labels
still have different conditional distributions. Permuting those nonexchangeable
labels then constructs the wrong randomization distribution.

## 2. Nonexchangeability ablation

| Synthetic scenario | FPR | Wilson 95% | Mean global rho | Row-rho q05 / q50 / q95 | Rows exchangeable? |
|---|---:|---:|---:|---:|---|
| strict exchangeable | 0.0496 | 0.0439--0.0560 | -0.0001 | -0.258 / -0.0004 / 0.257 | yes |
| fresh safety conditioning only | 0.0452 | 0.0398--0.0513 | 0.0003 | -0.258 / 0.0010 / 0.255 | no |
| geometry-related nuisance only | 0.0530 | 0.0471--0.0596 | 0.0119 | -0.242 / 0.0127 / 0.265 | no |
| safety + geometry nuisance | 0.0568 | 0.0507--0.0636 | 0.0122 | -0.242 / 0.0131 / 0.267 | no |
| exact frozen stress | 0.0872 | 0.0797--0.0953 | 0.0111 | -0.242 / 0.0122 / 0.267 | no |
| safety + geometry, no controller intercepts | 0.0518 | 0.0460--0.0583 | 0.0118 | -0.243 / 0.0121 / 0.266 | no |
| safety + geometry, independent shells | 0.0570 | 0.0509--0.0638 | 0.0118 | -0.240 / 0.0121 / 0.264 | no |

Safety conditioning alone did not inflate FPR. Fresh/reference latent
heterogeneity, shared item factors, controller intercepts, and two-shell coupling
were also harmless under the strict generator. Material inflation emerged from the
stress interaction: geometry-related heterogeneous response profiles plus
controller-specific mean-difficulty effects and coupled shared structure. Removing
controller intercepts returned FPR to 0.0518; decoupling shells reduced it to
0.0570. The exact stress generator remained the strongest combined case.

Thus global-zero association and row exchangeability do not coincide. A mixture of
controllers with positive and negative conditional relations may have aggregate
rho near zero, yet each row retains its own association distribution. The proper
sampling unit is the fresh controller, not its 31 dyads.

## 3. Scientific estimand

For fresh controller \(i\), define

\[
r_i = \frac{1}{2}\sum_{s\in\{M,S\}}
\operatorname{Spearman}_{j=1}^{31}\left(A0_s(i,j),D^{shape}_s(i,j)\right).
\]

The population question is whether relational alignment is positive across the
population of prospectively sampled, safety-conditioned fresh controllers. The
primary estimand is the population median/sign tendency of \(r_i\). The complete
\(16\times31\) cross-block Spearman remains a useful descriptive effect size, but
496 dyads per shell are not 496 independent generalization units.

## 4. Candidate inference methods

| Method | Scientific unit / estimand | Calibration result | Role |
|---|---|---|---|
| A: original row-QAP | permuted fresh-row labels; global rho | failed stress null (0.0746) | secondary diagnostic only |
| B: exact sign test | fresh controller; median/sign of \(r_i\) | passed every null | **selected primary** |
| C: studentized mean | fresh controller; mean \(r_i\) | passed every null | robust sensitivity |
| D: row-cluster bootstrap | fresh controller clusters; global rho | FPR passed, interval coverage slightly low in some cases | sensitivity only |
| E: clustered rank regression | fresh-controller cluster slope | failed stress null (0.0842) | not retained |

Method B discards exact zero row associations prospectively, reports their count,
and uses an exact one-sided Binomial test of \(P(r_i>0)\le0.5\). Nonfinite row
associations fail closed. With 16 non-tied rows, at least 12 positives are required
at alpha 0.05.

## 5. Null calibration by method

| Method | Strict | Frozen stress | Row heterogeneity | Safety-conditioned | Heavy heterogeneity |
|---|---:|---:|---:|---:|---:|
| original row-QAP | 0.0540 | **0.0746** | 0.0006 | 0.0490 | 0.0008 |
| exact sign | 0.0318 | 0.0432 | 0.0000 | 0.0364 | 0.0000 |
| studentized mean | 0.0426 | 0.0434 | 0.0000 | 0.0514 | 0.0000 |
| cluster bootstrap, percentile | 0.0340 | 0.0220 | 0.0000 | 0.0445 | 0.0000 |
| cluster bootstrap, basic | 0.0405 | 0.0245 | 0.0000 | 0.0475 | 0.0000 |
| clustered rank regression | 0.0534 | **0.0842** | 0.0000 | 0.0522 | 0.0000 |

The sign and studentized-mean procedures met the frozen FPR/Wilson rules in all
five nulls. Studentized-mean null coverage was 0.9486--1.0000. The bootstrap
variants met the FPR rule but missed the predeclared 0.925 coverage floor in some
settings (strict basic 0.9160; safety percentile/basic 0.9150/0.9110), so they are
not preferred inferential gates. A nested bootstrap-t variant was not run because
it was not computationally feasible at the frozen outer-replicate scale; no claim
of its calibration is made.

## 6. Power by method

| Method | 25% closed A0 | 50% closed A0 | rho-like 0.15 |
|---|---:|---:|---:|
| original row-QAP | 0.9480 | 1.0000 | 0.9640 |
| **exact sign** | **0.7200** | **0.9973** | **0.7597** |
| studentized mean | 0.8847 | 1.0000 | 0.9243 |
| cluster bootstrap, percentile | 0.8513 | 1.0000 | 0.8953 |
| cluster bootstrap, basic | 0.8500 | 1.0000 | 0.8900 |
| clustered rank regression | 0.9180 | 1.0000 | 0.9437 |

Power did not override calibration. Method B exceeded the frozen 0.60 preferred
power floor and came first under the predeclared precedence because it directly
targets controller-population generalization, is exact and auditable, and makes
fewer assumptions than the more powerful alternatives.

## 7. Selected robust primary inference

The prospective replacement primary is `B_ROW_SPEARMAN_SIGN`:

1. compute one equal-shell row Spearman \(r_i\) for each of the 16 fresh
   controllers against the fixed 31-controller atlas;
2. require all 16 row associations to be finite;
3. report their median and complete distribution;
4. test \(P(r_i>0)>0.5\) with the exact one-sided sign test at alpha 0.05;
5. discard exact zeros only from the Binomial denominator and report them;
6. retain global cross-block rho as descriptive effect size;
7. report studentized-mean, row-cluster-bootstrap, original row-QAP, item
   bootstrap, and LOFO results only as predeclared sensitivities that cannot
   change the primary classification.

The frozen design remains K=16, n=34 candidates, one future stream, 300 items,
two shells, two rollouts, and 19,200 potential semantic trajectories. No stream
was generated here.

## 8. Fresh×fresh secondary inference

The original conjugation QAP happened to be conservative in the tested node
heterogeneity scenarios (FPR 0.0518 strict, 0.0392 heterogeneous, 0.0278 heavy),
but it still relies on node exchangeability. The selected formal secondary is a
node-jackknife pseudovalue t test: FPR 0.0132 under strict exchangeability and
0/5000 under both heterogeneity nulls; power was 0.726 at the rho-like 0.15
alternative. It is conservative, remains `SECONDARY_ONLY`, and cannot rescue the
fresh×old primary. Node bootstrap was not retained because duplicated sampled
nodes create structural zero dyads.

## 9. Historical Q2 QAP calibration audit

The model-free protocol was committed before historical `Dshape` was accessed.
With K=31 and exact pre-outcome A0/A1/A2 geometries, the historical
controller-label QAP calibrated under strict symmetric exchangeability for all
three metrics (FPR 0.0480--0.0484). Under node heterogeneity it remained calibrated
for A0 (0.0422) and A1 (0.0438) but reached 0.0698 for A2 (Wilson
0.0631--0.0772). Therefore complete controller-label exchangeability is not a
uniformly safe explanation of the historical inference.

The preselected node-jackknife alternative was conservative across all synthetic
nulls (FPR 0--0.0104; coverage 0.9896--1.0) and was frozen before the historical
semantic matrix was opened.

## 10. Historical Q2 robust post-hoc sensitivity

Exactly one analysis labeled
`POST_HOC_HETEROGENEITY_ROBUST_Q2_SENSITIVITY` was run. It does not change
`Q2_V4_1_G2`, `RS+`, or `RT+`.

| Metric | Historical full rho | Jackknife pseudovalue mean | Robust 95% CI | Holm p | All LOO positive |
|---|---:|---:|---:|---:|---|
| A0 | 0.563818 | 0.554993 | [0.451206, 0.658780] | 8.49e-12 | yes |
| A1 | 0.556311 | 0.547656 | [0.444810, 0.650503] | 8.49e-12 | yes |
| A2 | 0.441128 | 0.434434 | [0.307282, 0.561586] | 4.71e-08 | yes |

Result: `Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT`. Positive A0/A1/A2 relational
support survives controller/node-level inference; A2 still does not outperform
A0/A1, and the historical G2 classification is immutable.

## 11. Historical runtime autopsy

The sealed journal hash matched
`d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99` before
the allowlisted runtime fields were read. No text, correctness, or semantic metric
was accessed.

| Quantity | Result |
|---|---:|
| rows | 37,800 |
| total generation time | 86.3159 h |
| elapsed mean / median | 8.221 s / 0.623 s |
| elapsed p90 / p95 / p99 | 1.985 / 3.032 / 363.895 s |
| capped rows | 721 (1.9074%) |
| capped share of total time | 84.4473% |
| capped / noncapped mean | 363.952 / 1.303 s |
| top 1% / 5% / 10% time share | 44.30% / 90.14% / 91.67% |
| MEDIUM / STRONG mean | 5.161 / 11.398 s |
| retries / runtime errors | 0 / 0 |

The campaign was dominated by a small 4096-token tail, not by normal rows.
Elapsed time and generated tokens correlated at Pearson 0.999997 and Spearman
0.994157.

## 12. Future runtime distribution

One hundred thousand paired-controller pseudo-campaigns preserved each historical
controller's MEDIUM/STRONG profile.

| 16-controller / 19,200-row estimate | Mean | P50 | P80 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| finite 16-of-31 bank | 44.18 h | 44.20 h | 47.48 h | 49.10 h | 50.35 h | 52.46 h |
| controller superpopulation | 44.15 h | 43.84 h | 48.52 h | 51.17 h | 53.42 h | 57.86 h |

The exact worst historical 16-controller subset would take 58.81 h. In a 1.5x
cap-propensity stress the superpopulation P50/P95 become 62.25/76.78 h; at 2x,
80.66/100.17 h.

Early-prefix ETA backtests show why the original campaign was hard to forecast.
At 256 rows, naive median absolute percentage error was 17.6%; a tail-aware
estimator reduced it to 7.5%. At 1024 rows the tail-aware median error was 6.0%
and its P95 upper bound covered 98.5% of pseudo-campaigns. Runtime planning cannot
affect controller or inference-method selection.

## 13. Revised OOS protocol recommendation

Recommendation: retain the scientific experiment but replace the inferential
sampling unit. The draft `REVISED_OOS_PROTOCOL_DRAFT.json` keeps K=16, n=34, one
future stream, all original safety/instrument gates, and the same claim boundary.
It replaces primary row-QAP with controller-level exact sign inference, demotes
global rho and row-QAP to descriptive/diagnostic roles, and uses node jackknife
for the fresh×fresh secondary.

A future stream requires a new principal-approved prospective PRELOCK. No stream,
seed, controller, model inference, or semantic execution is authorized by this
review.

## 14. Repository/resource state

- V1/V2 historical classifications altered: **NO**
- new controller stream: **0**
- Qwen inference: **0**
- new semantic trajectories: **0**
- historical Q2 sensitivity labeled post-hoc: **YES**
- raw text inspected: **NO**
- correctness used for method selection: **NO**
- Spark 1 CPU-only used: **YES**, CUDA disabled
- Spark 2 used: **NO**
- Q3 run: **NO**

`Q2_OOS_HETEROGENEITY_ROBUST_INFERENCE_READY`
