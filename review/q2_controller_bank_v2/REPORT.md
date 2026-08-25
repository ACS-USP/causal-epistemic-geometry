# Q2 V2 — Calibrated controller-family-held-out geometry

## Outcome

Q2 V2 retains a prospectively qualified controller bank and now also has a
qualified execution engine, but the frozen 120-item common panel remains
untouched because the available RunPod wallet is insufficient.

- Operational status: `Q2_V2_ENGINE_QUALIFIED_COMMON_PANEL_BLOCKED_INSUFFICIENT_WALLET`
- Scientific classification: DEVELOPMENT; no predictive Q2 result exists
- Q1: unchanged
- Q3: not run
- Common-panel correctness outcomes read: no
- Common-panel rows: 0 / 6,960
- RunPod active GPUs after benchmark recovery: 0

## Frozen bank and predictive design

The previously qualified bank remains unchanged:

- source axes proposed / qualified: 6 / 6;
- signed/location meaningful controllers: 24 across six conceptual families;
- controllers satisfying the frozen causal rule: 12;
- selected dose bins represented: 4;
- fresh random controls: 4, projected against an SVD orthonormal basis of the
  meaningful span;
- maximum absolute null-to-span cosine: `8.413408858487514e-17`;
- common panel: 120 items, 29 conditions, two independent rollouts, 6,960 rows;
- primary prediction: leave one source family out;
- metrics: M0 cosine, M1 covariance-whitened, and M2 finite secant;
- target: canonical unbiased two-rollout error-profile distance.

No controller, source axis, dose, null, family fold, item, seed, generation
setting, estimator, or threshold changed during execution qualification.

## Budget chronology

The original US$25 ceiling correctly stopped Q2 V2 before the common panel.
After that clean stop, and still at zero common-panel rows, the principal
authorized an operational amendment with a preferred cumulative projection of
US$30 and a hard cumulative ceiling of US$45. The amendment did not alter the
scientific design.

## A40 cost-tail qualification

The exact serial A40 reference runner processed the same 174 non-scientific
fixture rows. The persisted distribution was:

| Statistic | Runtime (s) | Generated tokens |
|---|---:|---:|
| p50 | 0.594274 | 16 |
| p75 | 4.214919 | 114 |
| p90 | 174.216478 | 4,096 |
| p95 | 174.747427 | 4,096 |
| p99 | 176.230632 | 4,096 |
| max | 177.583943 | 4,096 |

Exactly 29/174 rows reached 4,096 generated tokens. Those 16.67% of rows
accounted for 96.22% of fixture runtime and 95.69% of generated tokens. No row
was excluded and `max_new_tokens=4096` remains frozen.

## GPU bakeoff and equivalence

The platform selection rule was frozen before benchmark results: choose the
qualified platform with the lowest projected complete common-panel dollar cost,
breaking ties by wall-clock time.

| Platform | Price | Mean s/row | Tokens/s | Projected hours | Projected panel cost | Qualification |
|---|---:|---:|---:|---:|---:|---|
| A40 secure | US$0.44/h | 30.246668 | 23.5871 | 73.0961 | US$32.1623 | PASS, serial reference |
| RTX 6000 Ada secure | US$0.84/h | 18.349891 | 38.8013 | 44.3456 | US$37.2503 | FAIL, not exact-equivalent |
| H100 SXM secure | US$3.29/h | not run | not run | not run | not run | Not economically plausible for bounded bakeoff |

The RTX engineering hook audit passed, but exact discrete equivalence failed:
116 compared-field mismatches occurred across 21 of 174 fixture-condition
rows, including generated token sequences and parser results. The RTX platform
was therefore rejected. No equivalence criterion was weakened.

Microbatching was not introduced because exact per-logical-row autoregressive
RNG and hook equivalence was not a straightforward bounded change. The frozen
serial reference runner was retained.

## Engine and wallet decision

Selected engine: `A40_SECURE_REFERENCE`, serial, no batching.

- observed cumulative Q2-V2 RunPod billing snapshot: US$3.9482;
- projected common-panel cost with frozen 25% margin: US$32.1623;
- projected cumulative cost: US$36.1105;
- preferred US$30 projection: FAIL;
- hard US$45 ceiling: PASS;
- principal-reported available wallet: US$5.42;
- minimum additional wallet for the projected panel: US$26.7423;
- reasonable 10% operational buffer: US$3.2162;
- recommended top-up: US$29.9585, practically US$30.00.

Because the wallet gate failed, no common-panel process was started. This is an
operational funding block, not a scientific or predictive geometry result.

## Predictive results

| Metric | Family-held-out rho | Permutation | RMSE |
|---|---|---|---|
| M0 flat | NOT RUN | NOT RUN | NOT RUN |
| M1 whitened | NOT RUN | NOT RUN | NOT RUN |
| M2 finite secant | NOT RUN | NOT RUN | NOT RUN |

No correctness, D matrix, G, C, controller accuracy, geometry association,
family ranking, or predictive common-panel outcome has been computed or read.

## Evidence vector

- Controller bank: QUALIFIED.
- Meaningful source-family breadth: six families, 24 controllers.
- Null geometry: QUALIFIED.
- A40 serial execution engine: QUALIFIED.
- RTX 6000 Ada alternative: REJECTED by exact-equivalence gate.
- Hard cumulative cost ceiling: PASS.
- Available wallet gate: FAIL.
- Common panel: SCIENTIFICALLY UNTOUCHED, 0 / 6,960.
- Predictive M0/M1/M2 evidence: DOES NOT EXIST.

## Infrastructure incidents

The first RTX attempt stopped before fixture generation because the cloned
environment lacked `accelerate`. The exact A40 version, `accelerate==1.14.0`,
was restored, the engineering gate was rerun, and the RTX benchmark restarted
from zero. No scientific item was involved. The historical A40 preflight had
persisted summary statistics but not row-level timing arrays; the exact frozen
non-scientific fixture schedule was therefore rerun prospectively to persist the
cost tail. Neither incident changed scientific semantics.

The temporary RTX Pod was terminated after artifact recovery. The A40 Pod was
stopped immediately after its benchmark artifact was recovered, then terminated
only after local artifact verification and branch push. The dedicated RunPod
MCP subsequently verified zero Pods, zero active GPUs, and zero network volumes.

## Next action

`PRINCIPAL_RESEARCHER_REVIEW`: top up the RunPod wallet before authorizing an
immediate continuation from the frozen A40 engine lock. Q3 and all further
experiments remain unauthorized.
