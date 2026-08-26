# Q2 V4 — prospective intervention-subspace geometry protocol draft

Status: `Q2_V4_DESIGN_FREEZE_READY / NOT FROZEN / NOT AUTHORIZED / NOT RUN`.

## Question

Within a development-defined, prospectively reconstructed and label-free-
qualified L27 intervention subspace, does pre-outcome angular geometry predict
semantic blind-spot shape for unseen sampled directions after matching
implemented intervention amplitude?

## Fixed design recommendation

- backend: native Spark 1, one GB10, no Spark 2/multi-node;
- model: Qwen/Qwen3-8B at exact revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- layer/operator: zero-based L27 sustained current-token intervention;
- source concepts: the four historical surviving V3 concepts, two locations;
- subspace: SVD basis with relative singular threshold 1e-6;
- directions: K=32 isotropic PCG64DXSM sphere samples from one future-lock seed;
- shells: implemented radii 0.25 and 0.50, alpha solved per direction;
- semantic conditions: baseline plus 64 meaningful controllers; no semantic
  random/null controller;
- panel: N=300 provenance-Class-C items in the inherited deterministic order,
  rebound under a new V4 manifest;
- rollouts: exactly two `INDEPENDENT_PRIMARY` blocks;
- rows: 39,000;
- endpoint: N/(N-1)-corrected blind-spot-shape distance;
- secondary endpoint decomposition: total distance and squared global mean
  error-rate shift;
- geometries: A0 coordinate angle, A1 covariance-whitened angle, A2 baseline-
  centered finite-response angle; D2 total finite response secondary;
- inference: 50,000 shell-coupled controller-label QAP maps, maxT across
  A0/A1/A2, 10,000 item-cluster bootstraps, delete-one-controller stability;
- radial: 50,000 paired shell swaps plus item bootstrap;
- M3/Q3: excluded/not run.

## Pre-semantic stop gates

Stop before semantic outcomes if any of the following fails:

1. exact Spark-native model/engine qualification;
2. all four textual source policies and all eight representation directions;
3. Spark-native subspace rank >=6, condition <=10, and concept leverage;
4. the single K=32 coefficient bank's algebraic gate;
5. implemented-radius root error <=0.5% for all 64 controllers;
6. each shell controller validity/evaluability >=0.90 and no more than 0.05
   below baseline, truncation <=0.05, raw-sequence movement >=0.10 medium and
   >=0.15 strong;
7. M1 fit integrity and A2 baseline radius/Hilbert consistency;
8. complete prediction-lock hash, panel provenance, schedule, and independent
   seed audit;
9. frozen resource projection or storage headroom.

No failed direction is replaced and no alternate random seed is tried.

## Outcome classification

- `Q2_V4_G0_NO_QUALIFYING_RELATIONAL_GEOMETRY`;
- `Q2_V4_G1_STATIC_ANGULAR_GEOMETRY_SUPPORTED`;
- `Q2_V4_G2_FINITE_RESPONSE_ANGULAR_GEOMETRY_SUPPORTED`;
- `Q2_V4_G3_FINITE_RESPONSE_ANGULAR_GEOMETRY_REQUIRED`.

Append independent `R+` or `R-`. Report A0/A1/A2, both shells, total/mean/
shape decomposition, all corrected p-values, and all uncertainty regardless of
classification. Never use “no geometry” beyond the frozen subspace/design.

## Collection discipline

Freeze and hash all logical keys before model output. Journal append/flush/fsync
per row. Never regenerate completed keys; invalid/truncated outcomes remain
errors. During collection inspect only counts, duplicates, process/GPU/disk,
journal integrity, and cost/runtime. Do not inspect condition metrics before
all 39,000 rows complete.

## Claim boundary

The direction population is isotropic in the qualified Spark-native subspace,
conditional on symmetric pre-outcome qualification. Success supports
generalization to unseen directions from that design distribution, not new
semantic families or arbitrary activation directions. A later experiment would
be needed for subspace transfer.

This draft does not authorize model download, Spark inference, source
reconstruction, shell calibration, geometry capture, semantic collection, or
Q3.
