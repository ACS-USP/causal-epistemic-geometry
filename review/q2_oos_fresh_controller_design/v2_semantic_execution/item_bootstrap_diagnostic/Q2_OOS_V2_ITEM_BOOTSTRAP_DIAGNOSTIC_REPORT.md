# Q2 OOS V2 Item-Bootstrap Diagnostic

Status: `CLOSED`

Label: `POST_HOC_DIAGNOSTIC_ONLY`

Ruling: `Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED`

The frozen primary result remains `Q2_OOS_V2_A0_PASS`. This audit found no defect in the sealed scored rows, binary error arrays, `Dshape`, `Dtotal`, fresh/reference item ordering, or any of the 16 primary controller-level associations. The historical item-bootstrap artifact is preserved byte-for-byte; only its publication interpretation changes.

## 1. Frozen diagnostic boundary

The precheck was committed and pushed before the implementation audit. It fixed the source hashes, eight synthetic scenarios, candidate methods, seeds, simulation counts, screening rule, and the prohibition on selecting a replacement because it produced a favorable interval. No new semantic trajectory was generated, Qwen was not loaded, and no raw model text was manually inspected.

## 2. Exact implementation reconstruction

The historical 50,000-resample computation was reproduced with maximum archived-quantile difference `0.0`. The implementation does exactly what the frozen analysis specified:

- the item is the resampling unit;
- each resample contains 300 draws with replacement;
- all 32 fresh conditions, both rollouts, both shells, and the paired 31-reference profiles move together;
- `Dshape` is recomputed inside every resample;
- row and global Spearman statistics are recomputed inside every resample;
- negative unbiased distances remain untruncated;
- there is no complete-case filtering;
- normalization uses 300 sampled occurrences, not the number of unique items;
- NumPy's default linear quantile implementation and the frozen seed are reproduced exactly.

Primary-object checks were exact:

| Object | Maximum absolute difference |
|---|---:|
| binary error arrays | 0.0 |
| `Dshape` | 0.0 |
| `Dtotal` | 0.0 |
| 16 primary `r_i` values | 0.0 |

Fresh and historical item ordering also matched exactly. There is therefore no basis for changing the primary OOS result.

## 3. Structural diagnosis on the sealed panel

For an ordinary size-300 bootstrap sample, the analytic expected number of unique items is 189.8204. The frozen 50,000 draws contained a mean of 189.8224 unique items; an independent structural RNG audit obtained 189.8357. Multiplicity further reduced the mean Kish-style effective support to 150.4904 items. The maximum multiplicity had median 5 and ranged from 3 to 10.

The exact historical resampling distribution had no rank-degenerate replicate, but it was displaced:

| Statistic | Full panel | Resampling q50 | q50 − full |
|---|---:|---:|---:|
| global equal-shell A0 association | 0.643055 | 0.540056 | −0.102999 |
| median fresh-controller `r_i` | 0.725174 | 0.609613 | −0.115561 |

This is not caused by accidental IID resampling of dyads, mismatched order, unique-item normalization, clipping, or a stale seed. Duplicate weights do not create new rollout information. They perturb the empirical item panel while reducing unique and effective support. That matters because the scientific statistic is compound and nonlinear:

```text
two R=2 binary-error profiles
  -> unbiased pairwise Dshape estimates
  -> 16×31 pairwise distance block
  -> row/global ranks with data-dependent ties
  -> Spearman association
```

At only two rollouts, latent propensity differences are already attenuated by finite-sample binary noise. Repeating item indices changes pairwise covariance terms and their leverage but does not supply independent Bernoulli information. Rank transformation then propagates changes in distance order and ties nonlinearly. Ordinary percentile-bootstrap centering is therefore not guaranteed for this estimator.

The appropriate interpretation of the archived percentiles is an **ordinary item-resampling panel-perturbation sensitivity distribution**, not a conventional 95% confidence interval for a latent item population.

## 4. Synthetic calibration

The frozen model-free simulation used `N=300`, `R=2`, 16 fresh controllers, 31 references, and coupled MEDIUM/STRONG shells. It ran 96 outer panels per scenario and 127 resamples per candidate method and panel.

| Scenario | Latent truth, global | Mean full-panel estimate | Ordinary-bootstrap centering | Ordinary coverage |
|---|---:|---:|---:|---:|
| NULL | −0.004 | 0.002 | −0.001 | 0.625 |
| WEAK_POSITIVE | 0.532 | 0.137 | −0.037 | 0.000 |
| MODERATE_POSITIVE | 0.860 | 0.378 | −0.092 | 0.000 |
| OBSERVED_LIKE | 0.973 | 0.640 | −0.121 | 0.000 |
| HETEROGENEOUS_CONTROLLER_EFFECTS | 0.566 | 0.475 | −0.051 | 0.000 |
| HETEROGENEOUS_ITEM_DIFFICULTY | 0.967 | 0.613 | −0.122 | 0.000 |
| SPARSE_BLIND_SPOT_DIFFERENCES | 0.991 | 0.436 | −0.110 | 0.000 |
| HEAVY_TIES_NEAR_DEGENERATE | 0.390 | 0.003 | −0.001 | 0.000 |

Here “centering” is the mean resampling median minus the full-panel estimate. Coverage targets the known latent item-population statistic. The observed-like scenario deliberately matches the observed full-panel association, not the unattainable latent truth; it reproduces the approximately −0.12 second-stage bootstrap displacement.

The simulations distinguish two attenuation layers:

1. the full `N=300`, `R=2` compound estimator can be attenuated relative to latent propensity geometry;
2. ordinary item reweighting can attenuate the recomputed rank association further relative to that full-panel estimate.

No tested item-level alternative passed all 16 frozen statistic-by-scenario screens. Pass counts were 0/16 for the ordinary bootstrap, Bayesian multiplier bootstrap, all four subsampling fractions, and delete-25%; delete-10% passed 1/16. The controller-cluster bootstrap was not eligible as an item-level replacement because it targets fresh-controller-population uncertainty conditional on this observed item panel.

The absence of a selected replacement is intentional. Narrowness and favorable significance were not selection criteria.

## 5. Publication ruling

The primary exact sign test is unaffected: its independent scientific unit is one prospectively sampled fresh controller, and all 16 frozen `r_i` values remain exact and positive.

Publication treatment:

- preserve the original item-bootstrap artifact and historical values;
- stop calling its percentiles a conventional 95% confidence interval;
- describe it as an item-panel perturbation sensitivity distribution;
- report the centering anomaly and calibration failure in the supplement;
- do not substitute an unvalidated item-level interval;
- retain the controller-cluster interval only with its distinct controller-population interpretation;
- do not use any post-hoc uncertainty display to strengthen the confirmatory primary claim.

## 6. Provenance and firewall

The implementation audit verifies private artifacts only by their frozen SHA-256 identities. The committed diagnostic outputs contain aggregate statistics and synthetic results, not raw prompts, benchmark text, model generations, or item-level correctness.

- new semantic trajectories: 0
- Qwen loaded: NO
- raw text manually inspected: NO
- correctness used for method selection: NO
- Spark 1 GPU used: NO
- Spark 2 used: NO
- RunPod used: NO
- Q3 run: NO

`Q2_OOS_V2_A0_PASS` and `Q2_OOS_V2_FORENSIC_CLEAN` remain immutable.
