# Q2 V3 — Prospective Out-of-Bank Finite-Secant Geometry

Status: `DRAFT / AWAITING PRINCIPAL_RESEARCHER_FREEZE`  
Execution: `NOT RUN`  
Inference authorization: `NONE`

## 1. Scientific question

Does the exact Q2 V2 finite-secant construction predict semantic error-profile
distance for genuinely new controller families, beyond predictions available
from dose and intervention magnitude alone?

This is not a rescue of Q2 V2. The frozen V2 classification remains
`Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`.

## 2. Prospective chronology

The future lock must enforce this order:

1. qualify one fresh exact objective instrument;
2. freeze five new semantic source families and all source prompts;
3. construct and label-free qualify every direction;
4. calibrate doses without correctness, G, C, D, rescue, or damage;
5. compute the exact frozen M0/M1/M2 geometries;
6. pass the pre-outcome magnitude-confounding design gate;
7. fit both prediction mappings using Q2 V2 development outcomes only;
8. persist every V3 pair prediction and hash the prediction lock;
9. commit and push the complete prospective lock;
10. only then collect V3 behavioral outcomes;
11. analyze once, audit independently, and stop.

No V3 behavioral outcome may alter a controller, dose, metric, nuisance model,
mapping, item, threshold, or classification.

## 3. Frozen model and operator

- model: `Qwen/Qwen3-8B`;
- revision/tokenizer: `b968826d9c46dd6066d109eabc6255188de91218`;
- BF16, no quantization, SDPA, `enable_thinking=false`;
- sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0;
- maximum new tokens: 4096;
- layer: zero-based block 27;
- construction: paired mean difference;
- source locations: prompt boundary and execution boundary;
- signs: plus and minus;
- intervention: sustained current-token, final prompt token at prefill and
  current token exactly once per decode forward.

## 4. Genuinely new source families

Exactly five source families are proposed. None is a renamed V2 family.

1. `CONTROL_FLOW_PATH_COVERAGE`: enumerate every reachable branch/path versus
   follow the first plausible path.
2. `MUTATION_ALIAS_CAUSALITY`: track object identity, mutation, and aliasing
   versus treat values as independent snapshots.
3. `API_CONTRACT_EXACTNESS`: apply exact Python built-in/method contracts versus
   rely on rough familiar behavior.
4. `LOOP_BOUNDARY_ACCOUNTING`: derive exact iteration domains and update order
   versus summarize loops qualitatively.
5. `HYPOTHESIS_BRANCH_ELIMINATION`: maintain candidate outputs and eliminate
   them by constraints versus commit to the first plausible output.

The future lock must include exact positive/negative prompts. No fallback axis
may be invented after source outputs. All five must pass prospectively frozen
behavioral separation and held-out activation-projection rules. Failure of any
axis returns `Q2_V3_SOURCE_BANK_NOT_QUALIFIED` before common-panel inference.

Each family produces four controllers:

- prompt-boundary plus/minus;
- execution-boundary plus/minus.

Primary meaningful bank: exactly 20 controllers across five new families.
Four new span-orthogonal random controllers are secondary controls.

## 5. Label-free dose calibration

Each signed direction receives the frozen four-bin V2 dose ladder relative to
its own source reference scale: 0.25, 0.50, 0.75, and 1.00. Selection uses only:

- commitment validity/evaluability;
- raw sequence movement;
- semantic change without correctness;
- token movement;
- truncation;
- exact engineering integrity.

Accuracy, G, C, D, rescue, damage, and future-panel outcomes are forbidden.
The dose rule must be frozen before calibration. All 20 controllers remain in
the bank after dose selection, including weak but mechanically safe members.

## 6. Exact geometry definitions

### M0

Normalized Euclidean distance between unit controller directions:

\[
d^{(0)}_{ij}=\sqrt{2-2\langle \hat v_i,\hat v_j\rangle}.
\]

### M1

The exact V2 covariance-whitened normalized angle, including lambda=0.1 and
`Sigma_lambda=(1-lambda)Sigma+lambda*mean_variance*I`. The covariance capture,
regularization, and normalization are unchanged.

### M2

The exact V2 finite behavioral secant:

\[
d^{(2)}_{ij}=\sqrt{\operatorname{mean}_{p,t}
  JS(P_{i,p,t},P_{j,p,t})}.
\]

Use the same 12 frozen label-free probes, teacher-forced text, four checkpoint
rules, full vocabulary, equal weighting, and float64 output-side JS reduction.
No V3 behavior may modify this definition. M2 is a finite-displacement response,
not a local JVP/Fisher/pullback metric.

## 7. Frozen nuisance model

The nuisance feature vector for pair `(i,j)` is exactly:

1. `abs(delta_norm_i-delta_norm_j)`;
2. `(delta_norm_i+delta_norm_j)/2`;
3. `abs(dose_fraction_i-dose_fraction_j)`;
4. `(dose_fraction_i+dose_fraction_j)/2`.

Fit two affine mappings on all eligible cross-family Q2 V2 meaningful pairs:

\[
\widehat D_N=\beta_0+\beta_N^T x_N,
\]

\[
\widehat D_{N+M2}=\gamma_0+\gamma_N^T x_N+\gamma_2 d^{(2)}.
\]

The fitted coefficients and numerical implementation must be frozen in the V3
prediction lock. V3 outcomes are never used for calibration.

M0+N and M1+N are fully reported negative-control mappings, fit on V2 only.

## 8. Pre-outcome design-adequacy gate

Before any V3 behavioral output, the 20-controller bank must satisfy all:

1. all controller identities, doses, hashes, source families, and M2 values are
   frozen;
2. all four dose bins are represented;
3. no source family contributes fewer than four controllers;
4. `R^2(M2 ~ four nuisance features) <= 0.75` across cross-family pairs;
5. the standardized augmented feature matrix has condition number <=30;
6. at least 60 cross-family dyads can be matched within 0.25 pooled SD on each
   nuisance feature while spanning at least the middle 80% of M2 distance;
7. at least 40 such matched dyads cross distinct source-family pairs.

These are design-identifiability rules, not outcome thresholds. If any fail,
classify `Q2_V3_MAGNITUDE_DECONFOUNDING_DESIGN_FAILED` and stop. Do not redraw
controllers or alter doses after behavioral outcomes—there will be none.

## 9. Fresh objective instrument

The 800-item CRUXEval pool is exhausted for a genuinely fresh same-benchmark
panel: the final 57 IDs were consumed by Q1 confirmatory evaluation. Q2 V3 must
not pretend otherwise.

Primary draft candidate: official LiveCodeBench test-output prediction at one
future frozen revision, using its exact objective evaluator and only IDs never
used by the historical external-benchmark campaign. Before scientific freeze:

- an offline provenance audit must establish at least 240 untouched eligible
  items;
- 40 items are allocated to a disjoint baseline-only instrument qualification;
- exactly 200 later items are allocated to V3 evaluation;
- no item replacement is allowed after outputs.

Instrument qualification uses two baseline rollouts and requires:

- commitment validity and semantic evaluability >=95%;
- pooled accuracy in [0.25, 0.85];
- repeated-baseline B00 >=0.05;
- at least 10 items wrong in both rollouts;
- at least 20 items correct in at least one rollout.

If the official source, exact evaluator, freshness, or opportunity gate fails,
return `Q2_V3_FRESH_INSTRUMENT_NOT_QUALIFIED`. No alternative benchmark may be
substituted without a new principal-reviewed draft.

This choice adds a domain-generalization burden. It is unavoidable if no fresh
same-domain pool exists and must be explicit in interpretation.

## 10. Common-panel design

- fresh evaluation items: 200;
- conditions: baseline + 20 meaningful + 4 fresh nulls = 25;
- rollouts: exactly two independent primary rollouts;
- trajectories: 10,000;
- seeds: unique hash of experiment, item, condition, and rollout;
- condition schedule: deterministic, interleaved, and frozen;
- journal: append/flush/fsync, immutable logical key, exact-seed resume;
- model invalid/truncated outcomes remain errors and are never retried;
- infrastructure failures alone may retry the same key and seed;
- no scientific peeking before all 10,000 rows are complete.

The primary target remains the canonical unbiased estimator:

\[
D_{ij}=\frac1N\sum_n(e_{in0}-e_{jn0})(e_{in1}-e_{jn1}).
\]

Primary dyads are cross-family pairs among the 20 meaningful controllers. Null
and within-family dyads are secondary.

## 11. Primary prospective metrics

For each new family, score edges between its four controllers and all sixteen
controllers outside that family. Predictions were frozen before outcomes.

Primary rank statistic:

\[
\rho_{res}=\operatorname{mean}_{family}
\operatorname{Spearman}
(\widehat D_{N+M2}-\widehat D_N, D-\widehat D_N).
\]

Primary predictive-error statistic:

\[
R_{inc}=\frac{\operatorname{mean}_{family}RMSE(\widehat D_{N+M2},D)}
{\operatorname{mean}_{family}RMSE(\widehat D_N,D)}.
\]

Use 10,000 family-block controller-label QAP permutations and 10,000 item-cluster
bootstraps. The item bootstrap moves every condition and both rollouts together.
Families, not dyads, are the unit for sign consistency.

## 12. Independently justified gate

`Q2_V3_OUT_OF_BANK_FINITE_SECANT_INCREMENTAL_SIGNAL` requires all:

1. instrument and design-adequacy gates pass;
2. every meaningful controller has validity/evaluability >=90%, and bank median
   validity/evaluability >=95%;
3. every meaningful controller accuracy is at least baseline minus 10 points;
4. mean residual rho >=0.25 (a prospectively meaningful moderate ordinal
   increment, not chosen from V2's 0.9067);
5. one-sided family-QAP p<=0.05;
6. incremental RMSE ratio <=0.90, interpreted as at least a 10% RMSE reduction
   beyond a strong nuisance predictor (about 19% reduction in squared error);
7. item-bootstrap lower bound >0 for paired nuisance RMSE minus augmented RMSE;
8. at least four of five families have residual rho >0;
9. at least four of five families have augmented RMSE below nuisance RMSE;
10. M2+N mean RMSE is at least 5% below both M0+N and M1+N.

The 0.90 value is retained for an independent reason: it is the minimum
predictive improvement judged scientifically worthwhile over a much stronger
reference predictor. It is not a tolerance around V2's 0.9067.

Multiplicity: the M2+nuisance composite above is the sole primary test. M0/M1
comparisons are prespecified controls and conditions 10, not separately mined
hypotheses. All secondary results are reported; no best metric selection.

## 13. Frozen classifications

- `Q2_V3_OUT_OF_BANK_FINITE_SECANT_INCREMENTAL_SIGNAL`: every primary rule
  passes.
- `Q2_V3_ORDINAL_PARTIAL_SIGNAL`: rules 1-5 and 8 pass, but calibrated error
  rules fail.
- `Q2_V3_CALIBRATED_PARTIAL_SIGNAL`: rules 1-3, 6-7, and 9 pass, but residual
  rank/QAP rules fail.
- `Q2_V3_MAGNITUDE_ONLY_GENERALIZATION`: nuisance predicts D, but M2 has
  residual rho<=0.10 or incremental RMSE ratio>=0.98.
- `Q2_V3_NO_OUT_OF_BANK_GEOMETRY_SIGNAL`: safety/design pass but no coherent
  prospective prediction.
- `Q2_V3_CONTROLLER_BANK_DESTRUCTIVE`: controller safety rules fail.
- `Q2_V3_FRESH_INSTRUMENT_NOT_QUALIFIED`.
- `Q2_V3_MAGNITUDE_DECONFOUNDING_DESIGN_FAILED`.
- `Q2_V3_ENGINE_OR_INTEGRITY_FAILURE`.

Falsification of the present M2 interpretation is specifically supported if
`R_inc>=1`, `rho_res<=0`, or no more than two families have positive incremental
direction. Preserve the full evidence vector even if a composite category fires.

## 14. Independent audit

An independent path must recompute raw-row completeness, seeds, parser,
controller identity, all predictions and their pre-outcome hashes, D, nuisance
and M2 scores, QAP, item bootstrap, and classification without importing the
primary high-level analysis.

## 15. Construction stretch — design only

Only after a successful V3 may a separate protocol fit frozen predictors for
complementarity and safety, then choose once:

\[
v^*=\arg\max_v \widehat C(v)
\]

subject to frozen lower bounds on competence and validity/evaluability, an upper
bound on semantic movement, and an intervention-magnitude budget. Candidate
directions, geometry, objective, constraints, tie-breaks, and `v*` must be
committed before fresh behavioral evaluation. No fallback candidate is allowed.

This is a future Q2 construction stretch, not Q3. Routers, committees, voting,
or collective utility remain forbidden.

## 16. Firewall

- Q2 V2 classification: immutable;
- Q2 V3: draft, not frozen, not run;
- Q1: unchanged;
- Q3: `NOT RUN`;
- confirmatory holdout: not reused;
- RunPod/DGX authorization: none in this draft task.

