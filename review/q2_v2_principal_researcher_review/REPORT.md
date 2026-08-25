# Q2 V2 Principal Researcher Review

Status: **POST-HOC / EXPLORATORY**. This review uses only frozen Q2 V2 artifacts.
It creates no model output and cannot alter the historical decision.

## Executive decision

The frozen conclusion remains `Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`.
M2 finite-secant geometry did show a broad, family-consistent association and a
real predictive improvement over M0/M1. However, the strongest new diagnosis is
unfavorable to a specifically relational-geometry interpretation: four simple
pre-outcome strength features (dose and delta-norm means/differences) match the
raw M2 rank association and explain nearly all of M2's incremental held-out RMSE
gain.

The recommended next step is therefore **not** to rescue or exactly replicate
Q2 V2. It is to prepare one stronger, falsification-oriented Q2 V3 in which:

1. every controller family is genuinely new;
2. predictions are frozen before behavioral outcomes;
3. the bank must pass a pre-outcome magnitude-confounding design gate; and
4. M2 must improve prospectively over a frozen dose/norm nuisance model.

The Q2 V3 protocol in this directory is a draft awaiting principal-researcher
freeze. It has not run. If its pre-outcome design-adequacy gate cannot be met,
the recommendation is to stop Q2 rather than collect an uninterpretable panel.

## Frozen conclusion versus new evidence

### Frozen Q2 V2 conclusion

Q2 V2 did not pass its prespecified composite criterion. M2's held-out RMSE
ratio was 0.9067437, above the frozen maximum of 0.90. No rounding, tolerance,
endpoint change, or post-hoc sample extension is applied here.

### Developmental observation already present at closeout

M2 was stronger than M0/M1 on held-out rank and error:

| Metric | Mean family-held-out rho | RMSE ratio | QAP p |
|---|---:|---:|---:|
| M0 flat | 0.201325 | 0.986710 | 0.000700 |
| M1 whitened | 0.190178 | 0.988152 | 0.000900 |
| M2 finite secant | 0.427920 | 0.906744 | 0.002200 |

### Exploratory findings introduced after seeing Q2 V2

Everything below this heading is post-hoc and descriptive.

## A1. Independent reproduction

The new analysis path reproduced all fifteen frozen scalar values (rho, RMSE,
constant RMSE, RMSE ratio, and QAP p for each metric) with maximum absolute
difference exactly 0.0. The historical independent audit remains
`Q2_V2_FORENSIC_CLEAN`, with its earlier maximum cross-path discrepancy of
1.5543e-15.

## A2. Family decomposition

M2's point-estimate rho was positive in every held-out family and its RMSE beat
the constant predictor in every family. This is broad rather than a one-family
artifact.

| Held-out family | M0 rho / ratio | M1 rho / ratio | M2 rho / ratio |
|---|---:|---:|---:|
| Counterfactual checking | -0.005 / 1.026 | 0.026 / 1.017 | 0.379 / 0.960 |
| Decompose then solve | 0.379 / 0.975 | 0.363 / 0.980 | 0.419 / 0.885 |
| Explicit state tracking | 0.265 / 0.974 | 0.246 / 0.977 | 0.524 / 0.900 |
| Independent verification | 0.196 / 0.985 | 0.170 / 0.988 | 0.332 / 0.901 |
| Invariant checking | 0.248 / 0.974 | 0.224 / 0.975 | 0.386 / 0.898 |
| Type/representation discipline | 0.125 / 0.996 | 0.113 / 1.000 | 0.528 / 0.908 |

Each fold has four held-out controllers and 80 directed cross-family test
edges. Item-bootstrap intervals remain wide: the M2 rho interval crosses zero
for independent verification and approaches zero for two other families. With
only six families, no family-characteristic regression is scientifically
credible.

M0 is locally competitive in decompose-then-solve rank (0.379 versus 0.419),
but not in calibrated RMSE. No family reverses the aggregate M2 advantage.

## A3. Paired item bootstrap

Ten thousand resamples moved each item with all conditions and both rollouts.
The paired advantages favored M2 in most, but not all, resamples:

| Contrast | Median | Descriptive 95% interval | Fraction positive |
|---|---:|---:|---:|
| rho(M2)-rho(M0) | 0.1734 | [-0.0238, 0.3468] | 0.9567 |
| rho(M2)-rho(M1) | 0.1822 | [-0.0161, 0.3557] | 0.9632 |
| RMSE(M0)-RMSE(M2) | 0.000852 | [-0.000299, 0.003133] | 0.8857 |
| RMSE(M1)-RMSE(M2) | 0.000872 | [-0.000261, 0.003149] | 0.8938 |

These are not new confirmatory p-values. The intervals crossing zero show that
item sampling alone can erase the apparent advantage, especially for RMSE.

## A4. Ranking versus calibration

Across the 480 directed held-out fold edges, M2 predicted observed distances
with Spearman 0.405 and Pearson 0.414. A pooled diagnostic recalibration gave
intercept 0.00218 and slope 0.8358. A quadratic term did not improve materially:
RMSE 0.0148439 versus 0.0148442 for linear recalibration. Residual rank against
prediction was only -0.109.

This does **not** support a simple story in which M2 has the right ordering but
fails mainly because of a remediable nonlinear calibration curve. M2 contains
ordinal information and has reasonably mild global scale bias, but substantial
edge-level noise remains.

The held-out scatter is duplicated by scientific role: a cross-family edge is
tested once when each endpoint family is held out. This matches the frozen fold
definition; unique-edge summaries are reported separately and never treated as
independent observations.

## A5-A6. Dose and magnitude confounding

This is the review's most consequential result.

Four deliberately simple nuisance features were fixed for this review:

- absolute delta-norm difference;
- mean delta norm;
- absolute dose-fraction difference;
- mean dose fraction.

They use controller metadata that existed before common-panel outcomes. No
model-family search was performed.

| Predictor | Held-out rho | RMSE ratio |
|---|---:|---:|
| delta-norm difference | 0.314 | 0.967 |
| mean delta norm | 0.403 | 0.925 |
| dose-fraction difference | 0.169 | 0.991 |
| mean dose fraction | 0.281 | 0.960 |
| four-feature nuisance model | 0.443 | 0.929 |
| M2 alone | 0.428 | 0.907 |

Adding M2 to the four-feature model reduced mean held-out RMSE only from
0.0151511 to 0.0150674, an augmented-to-nuisance ratio of 0.9945. Mean residual
rho was 0.0544. A 10,000-permutation family-block QAP gave exploratory
one-sided p=0.3739 (null mean 0.0126).

Within equal-dose pairs, M2 still had rho 0.295; among pairs no more than one
dose step apart it had rho 0.513. Those restrictions show that M2 is not
literally identical to the discrete dose label. They do not overcome the more
complete nuisance analysis: calibrated displacement magnitude, not dose bin
alone, explains much of the aggregate relation.

The strongest supported interpretation is therefore:

> M2 is a good finite-displacement strength proxy in this bank, with at most
> weak unresolved evidence for relational/directional information beyond
> strength.

## A7. Null controllers

The frozen primary analysis used only the 24 meaningful controllers. Therefore
the M2 primary association cannot depend on the four nulls.

Secondary pair-class diagnostics found:

| Pair class | Pairs | Mean behavioral D | M2 rho |
|---|---:|---:|---:|
| meaningful-meaningful | 276 | 0.01428 | 0.433 |
| meaningful-null | 96 | 0.00790 | 0.457 |
| null-null | 6 | 0.00000 | 0.516 |

The null-null estimate is too small-sample to interpret. Meaningful-null M2
prediction argues that the random controllers are not behaviorally
"geometry-free"; they are interventions outside the semantic source span whose
finite downstream displacement can still be structured. M0/M1 did not predict
meaningful-null D. This ablation strengthens the finite-displacement reading,
but does not establish semantic relational geometry beyond magnitude.

## A8. Direction, dose, and local regime

M2 was not best for the shortest secants. Within the lowest quartile of M2
distance, rho was 0.004; in quartiles three and four it was 0.301 and 0.347.
Pairs with mean dose 0.25 had rho -0.166, while larger-dose groups were mostly
positive. This is compatible with an item-noise floor at weak displacements and
with a signal that becomes visible only after a sufficiently large causal move.

Same-family and cross-family unique-edge rho were nearly identical (0.432 and
0.431). Same source-direction-base pairs had rho 0.500 but only 12 pairs.
Consequently there is no evidence that M2 works only within a family, and no
credible evidence for a bounded *local* metric regime. The opposite concern is
more immediate: weak/local secants are poorly resolved.

Leave-one-controller M2 rho ranged from 0.394 to 0.510 and remained positive in
all 24 ablations. RMSE ratio ranged from 0.857 to 0.933. No single controller
determines the sign, although calibrated loss is visibly controller-sensitive.

## A9. M0 versus M1 versus M2

M0 and M1 distance matrices are almost the same: Spearman 0.9903 and Pearson
0.9951 over unique meaningful pairs. The particular covariance whitening used
in V2 therefore barely reordered this controller bank. Its failure to improve
prediction is unsurprising.

Mathematically:

- M0 uses the angular distance between unit source directions;
- M1 uses the same angular construction after one fixed inverse-covariance
  inner product;
- M2 applies each calibrated finite intervention to the frozen model on 12
  label-free probes and measures the square root of mean full-vocabulary
  Jensen-Shannon divergence across four teacher-forced checkpoints.

M2 consequently contains controller magnitude, sign, nonlinear network gain,
token context, and finite downstream distributional response. It is not an
exact local JVP, Fisher, pullback, or Riemannian metric. The data support that
this richer finite causal response predicts D better than either tested static
angle. They do **not** show that correcting activation anisotropy is generally
insufficient, because only one whitening realization was tested. They also do
not show that the extra information is specifically angular or relational,
because magnitude accounts for most of the incremental signal.

## Negative and inconvenient findings

1. The frozen composite gate failed.
2. Paired item-bootstrap contrast intervals cross zero.
3. A four-scalar nuisance model slightly exceeds M2's held-out rho.
4. M2 adds only about 0.55% RMSE improvement over that nuisance model.
5. Residualized M2 is not supported by family QAP (p=0.374).
6. Weak/short secants have essentially no within-stratum association.
7. The empirical V2-based V3 planning simulation produces zero proxy passes at
   N=120, 160, 200, or 240 under the proposed incremental gate. More items do
   not solve missing family-level relational signal.

## Remaining uncertainty

The frozen data cannot determine whether:

- a deliberately magnitude-deconfounded controller bank would reveal a
  residual M2 relation;
- the relation generalizes to genuinely new source families;
- the V2 result is task-specific to the exhausted CRUXEval pool;
- a larger set of genuinely independent families would stabilize calibration;
- finite secants predict useful complementarity, rather than movement size.

## Principal-researcher recommendation

**Prepare, but do not yet freeze, one stronger prospective Q2 V3.** It should be
treated as a decisive falsification attempt, not as a likely-success extension.
Its pre-outcome bank must contain enough M2 variation orthogonal to dose/norm
features. If that design gate fails, stop Q2. If the gate passes, run once on a
fresh objective panel and require M2 to beat the nuisance model prospectively.

An exact Q2 V2 replication is not recommended: it would reproduce the same
confound. Q3 remains `NOT RUN`.

## Reproducibility map

- exact reconstruction: `PRIMARY_REPRODUCTION.json`;
- family folds: `FAMILY_DECOMPOSITION.csv`;
- paired bootstrap: `BOOTSTRAP_CONTRASTS.json` and
  `ITEM_BOOTSTRAP_SAMPLES.npz`;
- held-out predictions: `HELDOUT_PREDICTIONS.csv`;
- calibration: `CALIBRATION_DIAGNOSTICS.json`;
- magnitude tests: `NUISANCE_BASELINES.json`;
- local/dose tests: `DOSE_LOCAL_VALIDITY_ANALYSIS.json`;
- null ablation: `NULL_PAIR_ANALYSIS.json`;
- robustness: `ROBUSTNESS_SENSITIVITY.json`;
- complete analysis enumeration: `EXPLORATORY_ANALYSIS_ENUMERATION.json`.

The figures are diagnostic views only and carry no new decision threshold.

