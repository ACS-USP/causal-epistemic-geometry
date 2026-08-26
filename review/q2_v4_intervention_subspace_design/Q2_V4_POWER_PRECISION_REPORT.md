# Q2 V4 — dependence-aware power and precision report

Status: CPU-only synthetic planning. No historical or V4 correctness outcome
was loaded.

The simulation uses an 8-dimensional isotropic coefficient bank, shared
controller embeddings, shared item logits, two shells, two independent
Bernoulli rollout blocks, the corrected shape estimator, and shell-coupled
controller-label QAP. Every pairwise matrix entry is recomputed from shared
controller/item data. Planning used 120 repetitions and 499 QAP maps per cell;
the final protocol proposes 50,000 maps.

The primary scenario makes A0/A1/A2 correlated, as expected for a metric
ladder. Rates at target rho=0.25 are:

| K | N | omnibus FPR at rho=0 | omnibus power | A2 attribution | radial power | A2 rho 95% MC width |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 200 | 0.042 | 0.433 | 0.408 | 0.583 | 0.201 |
| 16 | 300 | 0.075 | 0.525 | 0.458 | 0.750 | 0.204 |
| 20 | 200 | 0.067 | 0.508 | 0.442 | 0.592 | 0.214 |
| 20 | 300 | 0.033 | 0.558 | 0.483 | 0.717 | 0.215 |
| 24 | 200 | 0.058 | 0.583 | 0.542 | 0.717 | 0.203 |
| 24 | 300 | 0.042 | 0.717 | 0.658 | 0.875 | 0.199 |
| 32 | 200 | 0.067 | 0.842 | 0.825 | 0.792 | 0.148 |
| 32 | 300 | 0.042 | 0.942 | 0.942 | 0.942 | 0.118 |

The achieved mean true A2 rho at the nominal 0.25 cell was about 0.23–0.24.
At K=32/N=300, omnibus/A2 attribution power was 0.767/0.742 near true rho
0.19, 0.942/0.942 near 0.24, 0.950/0.950 near 0.29, and 1.000/1.000 near 0.38.
The null omnibus rate was 0.042 and A2-attribution FPR 0.025.

The K effect is decisive. Moving from K=24 to 32 at N=300 raises nominal-0.25
omnibus power from 0.717 to 0.942. Increasing N improves item precision and
radial power but cannot substitute for vertices. K=32/N=200 already clears 80%
omnibus power; N=300 reduces the A2 Monte Carlo width from 0.148 to 0.118 and
raises radial power from 0.792 to 0.942.

## Finite-response superiority

A separate synthetic scenario gives A2 genuinely non-static quadratic
controller structure; it is a power stress test, not a proposed metric. For
K=32/N=300:

| achieved true A2 rho | mean observed A2-best-static | omnibus | A2 attribution | full A2 superiority |
|---:|---:|---:|---:|---:|
| 0.203 | 0.062 | 0.467 | 0.458 | 0.183 |
| 0.244 | 0.069 | 0.600 | 0.567 | 0.250 |
| 0.320 | 0.094 | 0.800 | 0.792 | 0.417 |
| 0.390 | 0.110 | 0.958 | 0.958 | 0.567 |
| 0.471 | 0.137 | 0.992 | 0.992 | 0.833 |

Thus G3 is well powered only when finite response adds substantially more than
the frozen 0.10 point margin. V4 is strongly informative for G0/G1/G2 at
moderate relational effects, but a marginal G3 effect remains hard. All metric
results must be reported even if the composite label is lower.

## Decision

Recommend exactly `K=32, N=300`. This is the smallest candidate K that robustly
clears 80% omnibus/A2-attribution power near rho 0.25; N=300 is justified by the
large precision and radial gains over N=200. N=400 was not needed: the recent
V3 simulations already established diminishing item returns, while V4 directly
repairs the vertex bottleneck.

Full tables are in `POWER_SIMULATION.csv` and
`SUPERIORITY_POWER_SIMULATION.csv`; assumptions and seeds are in
`POWER_SIMULATION_METADATA.json`.
