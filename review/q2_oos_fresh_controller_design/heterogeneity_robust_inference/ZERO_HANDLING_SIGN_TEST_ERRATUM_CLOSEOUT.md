# Q2 OOS V2 zero-handling sign-test erratum closeout

The prospective precheck was committed and pushed at
`9e3151820e25d7ff871952fb79c709d24f8f6561` before this recalibration. Only
`B_ROW_SPEARMAN_SIGN_ZERO_INCLUSIVE` was rerun. No competing method was rerun
for selection, no historical semantic outcome was accessed, and no V2 stream
or V2 seed existed.

Every finite fresh controller remains in the exact Binomial denominator:

- `r_i > 0`: success;
- `r_i == 0`: non-success;
- `r_i < 0`: non-success;
- nonfinite `r_i`: `Q2_OOS_V2_INFERENCE_DEGENERATE`.

The tested estimand is the controller-population positive-sign tendency,
`P(r_i > 0)`, not the population median. Median and mean `r_i` remain
descriptive effect-size summaries.

| Scenario | Replicates | FPR / power | Wilson 95% | Exact-zero rows | Zero fraction |
|---|---:|---:|---:|---:|---:|
| STRICT_EXCHANGEABLE_NULL | 5,000 | 0.0318 | [0.02728, 0.03703] | 26 | 0.000325 |
| FROZEN_NONEXCHANGEABLE_STRESS_NULL | 5,000 | 0.0432 | [0.03791, 0.04919] | 35 | 0.000438 |
| ROW_HETEROGENEITY_NULL | 5,000 | 0.0000 | [0, 0.000768] | 0 | 0 |
| SAFETY_CONDITIONED_NULL | 5,000 | 0.0364 | [0.03155, 0.04196] | 54 | 0.000675 |
| HEAVY_HETEROGENEITY_NULL | 5,000 | 0.0000 | [0, 0.000768] | 3 | 0.000038 |
| POSITIVE_25_PERCENT_CLOSED_A0 | 3,000 | 0.7200 | [0.70366, 0.73578] | 22 | 0.000458 |
| POSITIVE_50_PERCENT_CLOSED_A0 | 3,000 | 0.9973 | [0.99475, 0.99865] | 12 | 0.000250 |
| POSITIVE_RHO_LIKE_0_15 | 3,000 | 0.7597 | [0.74405, 0.77462] | 13 | 0.000271 |

All five null scenarios pass the previously frozen calibration requirement.
Both mandatory alternative scenarios exceed the previously frozen 0.60 power
floor. There were zero degenerate panels. The nonpositive-count median was
eight in every null scenario, four at 25% historical A0, one at 50% historical
A0, and three at rho-like 0.15.

Artifacts:

- `ZERO_HANDLING_SIGN_TEST_RECALIBRATION.json` SHA-256
  `3f6ef29caa0ae3107abbe37fcaa30a39e0a4b422e599f905e4dea70dffc82cb9`;
- `ZERO_HANDLING_SIGN_TEST_RECALIBRATION.csv` SHA-256
  `32c257bc7d0a00ff3514608cdf9c9c88f560b963fd95e7703929de525de612cf`.

Classification: `Q2_OOS_V2_SIGN_TEST_CALIBRATED`.
