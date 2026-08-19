# Metrics and statistics

This is the canonical statistical specification. Historical reports may retain
old names for reproducibility; new analyses use the definitions here.

## Paired deterministic outcomes

For the same item under baseline and treatment, the 2×2 table is:

| | treatment correct | treatment wrong |
|---|---:|---:|
| baseline correct | `n_cc` | `n_cw` (damages) |
| baseline wrong | `n_wc` (rescues) | `n_ww` (double faults) |

With `N` items:

```text
rescue_rate     = n_wc / (n_wc + n_ww)     # conditional on baseline error
damage_rate     = n_cw / (n_cc + n_cw)     # conditional on baseline success
rescue_fraction = n_wc / N
damage_fraction = n_cw / N
net_flip_fraction = (n_wc - n_cw) / N = delta_accuracy
```

`rescue_rate - damage_rate` is not a net effect: its terms have different
denominators. The old `rescue_minus_damage_interval` key remains a deprecated
compatibility alias and is explicitly labeled as conditional-rate subtraction.
New reports use the interval for `net_flip_fraction`, which is identical to the
paired delta-accuracy interval.

Every complementarity summary must show baseline and treatment accuracy,
delta accuracy, rescues, damages, disagreement, double fault, error Jaccard,
error phi (with undefined status for constant vectors), and pair-oracle
headroom together. Pair-oracle accuracy is an upper bound requiring oracle
selection; it is not an implementable ensemble.

## Stochastic policy estimands

For item `t`, condition `j`, and independent rollout `r`:

```text
e_tjr ~ Bernoulli(p_tj)
```

The item-level propensity `p_tj` separates a stochastic policy's persistent
blind spots from one sampled hard-error vector. Across the item distribution:

```text
mu_j = E[p_tj]
O_ij = 1 - E[p_ti p_tj]
O_00 = 1 - E[p_t0^2]
G_j  = O_0j - O_00 = E[p_t0^2 - p_t0 p_tj]
```

The repeated-baseline oracle is the natural resampling null. Its decomposition
is:

```text
G_j = mu_0 (mu_0 - mu_j) + Var(p_t0) - Cov(p_t0, p_tj)
```

The competence-adjusted complementarity estimand is:

```text
C_j = G_j - mu_0 (mu_0 - mu_j)
    = Var(p_t0) - Cov(p_t0, p_tj)
```

`G_j` mixes mean competence change with covariance change. `C_j` removes the
mean-accuracy component, but must still be reported beside `mu_0`, `mu_j`, and
the raw paired outcomes. The historical name `excess_pair_oracle` is retained
for compatibility and documented as this mixed quantity.

## Unbiased two-rollout propensity distance

With two independent rollouts per condition, an unbiased estimator of
`E_t[(p_ti-p_tj)^2]` is:

```text
D_hat = mean_t[
    e_ti1 e_ti2 + e_tj1 e_tj2
  - e_ti1 e_tj2 - e_ti2 e_tj1
]
```

It is valid only for independent rollout banks with matched item identity. It
must reject common-random-number coupling, missing or reordered item
provenance, and shapes other than exactly two independent draws per condition.
It is unbiased but can be high variance and can be negative in a finite sample.

## Seed regimes

`INDEPENDENT_PRIMARY` uses separate, explicit baseline and intervention seed
banks. It estimates operational repeated-agent complementarity and supports the
propensity estimands above.

`MATCHED_COUPLING_SECONDARY` uses the same RNG seed as a common-random-number
coupling. The intervention remains a controlled condition, but the observed
paired answer is coupling-dependent. It is a useful sensitivity or causal
diagnostic, not a substitute for independent operational draws.

Every future scientific row and config records `seed_regime`.

## Propensity correlation and reliability

With only two rollouts, the plug-in propensity takes values `{0, 0.5, 1}`.
Correlations are severely discretized and attenuated. They are not primary.
The implementation requires at least four rollouts per condition for the normal
descriptive status, or an explicit opt-in that labels a two-rollout result
`LOW_RESOLUTION_TWO_ROLLOUT_PLUGIN_ATTENUATED`.

Likewise, splitting two rollouts produces one binary observation per half, not
a smooth propensity reliability estimate. Four or more rollouts are required
unless the output is explicitly labeled low resolution.

## Rank and matrix association

Spearman correlation uses average ranks for ties. The V4 sequential-tie bug and
offline correction are documented in
[Q1_V4_GEOMETRY_REANALYSIS.md](Q1_V4_GEOMETRY_REANALYSIS.md).

For Q2, pairwise matrix entries are dyadically dependent. The primary null
permutes direction labels and recomputes the whole distance matrix
(QAP/Mantel-style). Treating `n(n-1)/2` pairs as independent observations is
forbidden. Dyadic regression may be secondary if its dependence assumptions
and uncertainty procedure are explicit.

## Dense code hierarchy

Executable tests are nested:

```text
problem -> generated program -> rollout -> intervention -> test case
```

Five problems with one hundred tests each are not 500 independent problems.
Future dense-code analysis must preserve nested IDs and use problem-clustered
bootstrap or an explicit hierarchical model. Diagnostics should include test
duplication, per-test marginal failure rates, failure-vector effective rank,
within-problem redundancy, dense-vector disagreement, and problem-clustered
pair-oracle summaries.

`cluster_bootstrap_mean` first reduces nested test outcomes to an equally
weighted problem mean, then resamples problems. It intentionally prevents a
problem with many redundant tests from dominating uncertainty or effective
sample size.

## Uncertainty

Development intervals are descriptive. Paired item bootstraps resample the
scientific unit; multiple views or tests from one latent problem move together.
Confirmatory uncertainty must match the frozen sampling design. Undefined
correlations remain `null`/`NaN` with a reason rather than becoming zero.
