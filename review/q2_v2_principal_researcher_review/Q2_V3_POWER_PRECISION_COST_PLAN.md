# Q2 V3 draft power, precision, and cost plan

Status: `DESIGN ONLY — NOT AN EXECUTION AUTHORIZATION`

## Dependence-aware empirical planning

The planning simulation uses Q2 V2 as a development prior only. For each of
2,000 replicates at N=120, 160, 200, and 240 it:

- resamples item IDs as clusters, moving both rollouts and every controller;
- selects five of six complete V2 families without replacement;
- keeps all four controllers in each selected family;
- recomputes the dyadic D matrix;
- evaluates fixed nuisance and nuisance+M2 mappings.

It never treats controller pairs as independent. It is still optimistic because
the mappings and controller effects come from V2, not genuinely new families.

| N | median residual rho | 95% interval | median incremental RMSE ratio | 95% interval | proxy pass |
|---:|---:|---:|---:|---:|---:|
| 120 | 0.071 | [-0.131, 0.248] | 0.990 | [0.965, 1.017] | 0.0% |
| 160 | 0.072 | [-0.116, 0.243] | 0.989 | [0.963, 1.015] | 0.0% |
| 200 | 0.075 | [-0.105, 0.238] | 0.988 | [0.961, 1.015] | 0.0% |
| 240 | 0.076 | [-0.095, 0.233] | 0.986 | [0.963, 1.014] | 0.0% |

The proxy pass applies the draft V3 thresholds without QAP/bootstrap gates. Its
zero rate is an inconvenient planning result, not a reason to weaken thresholds.
It says that adding items alone cannot manufacture relational M2 information
absent in V2. New-family design adequacy is the dominant uncertainty.

N=200 is selected for the draft because it reduces item noise materially versus
V2 while avoiding the pretense that N=240 solves family novelty. Five genuinely
new families provide the real scientific replication unit.

## Expected trajectories

| Phase | Draft rows |
|---|---:|
| baseline-only instrument qualification, 40 items x 2 | 80 |
| source construction/qualification | approximately 480 |
| 20-controller four-dose matched calibration plus baseline | approximately 970 |
| fresh common panel, 200 x 25 x 2 | 10,000 |
| total scientific/model trajectories | approximately 11,530 |

Teacher-forced covariance/secant captures are accounted separately from free
generation. Exact counts and schedules must be frozen before execution.

## Runtime and cost

Q2 V2 collected 6,960 panel rows in 2.6602 billed A40 GPU-hours at US$1.1853,
about 2,616 rows/hour and US$0.000170/row. A direct same-engine extrapolation of
11,530 rows is about 4.4 GPU-hours and US$1.96.

V3's proposed fresh instrument can have a different length tail, so the draft
does not budget at that lucky point estimate:

- expected A40 runtime: 6-10 GPU-hours;
- conservative runtime: 20 GPU-hours;
- expected cost: US$3-5 including startup and captures;
- conservative cost: US$9;
- proposed hard autonomous ceiling: US$12;
- wallet gate before remote start: verified balance at least US$15 and projected
  full cost, with 50% tail margin, no more than US$12.

A non-scientific preflight must sample the exact future engine/condition mix and
recompute the projection before any scientific output. If the projection exceeds
US$12, stop for principal review. Do not reduce items, families, controllers,
rollouts, or the token cap to fit budget.

The historical 16.7% cap-hitting tail that consumed about 96% of preflight time
must be modeled explicitly. Q2 V2's much lower realized cost is evidence that
the old projection was pessimistic, not a guarantee that V3 will be equally
cheap.

## Decision implication

This is a high-risk, decisive falsification design. The expected scientific
value comes from the pre-outcome magnitude-deconfounding gate and genuinely new
families, not from increasing N until a threshold is crossed.

Source data: `Q2_V3_PRECISION_SIMULATION.json`.

