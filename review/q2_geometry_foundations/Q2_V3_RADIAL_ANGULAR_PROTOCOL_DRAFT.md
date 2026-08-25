# Q2 V3 — Prospective radial/angular out-of-bank geometry

Status: `DRAFT / AWAITING PRINCIPAL_RESEARCHER_FREEZE`

Execution: `NOT RUN`

Inference authorization: `NONE`

## Primary scientific question

After matching physical intervention strength, does precomputed
output-sensitive internal geometry prospectively predict which semantic blind
spots differ across genuinely unseen causal intervention families?

## Frozen history

- Q2 V2: `Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`;
- Q2 V2 forensic: `Q2_V2_FORENSIC_CLEAN`;
- M2's post-hoc residual signal beyond dose/magnitude: unresolved/weak;
- Q1 unchanged;
- Q3 not run.

## Controller bank

Use five genuinely new conceptual families from the prior draft:

1. `CONTROL_FLOW_PATH_COVERAGE`;
2. `MUTATION_ALIAS_CAUSALITY`;
3. `API_CONTRACT_EXACTNESS`;
4. `LOOP_BOUNDARY_ACCOUNTING`;
5. `HYPOTHESIS_BRANCH_ELIMINATION`.

Each family supplies two prospectively oriented base directions: one
prompt-boundary source and one execution-boundary source. Orientation is the
positive-minus-negative source contrast frozen before downstream outcomes.
Antipodal plus/minus duplication is excluded from the primary bank because it
would inject trivial angles of pi.

Each of the ten base directions is deployed at two shells:

\[
5\text{ families}\times2\text{ directions}\times2\text{ shells}=20
\text{ meaningful controllers}.
\]

Four secondary null controllers use two fresh SVD-span-orthogonal random base
directions at both shells. They are not part of the primary meaningful
regression.

## Geometry-neutral shell variable

On a disjoint label-free calibration set define physical injection strength

\[
r_{phys}(\delta)=
\sqrt{\frac{\mathbb E_{x,k}\|\operatorname{BF16}(\delta)\|_2^2}
{\mathbb E_{x,k}\|h_{x,k}\|_2^2}}.
\]

The numerator is the actual per-forward injected residual shift after frozen
BF16 casting; the denominator is the background current-token residual energy
at layer 27. It uses no logits, metric candidate, correctness, or semantic
outcome, so it does not select in favor of M1/M2/M3. A deterministic dose grid
chooses the nearest safe value to each of two target shells. The targets must be
frozen from historical label-free safe/manipulation evidence before new-family
calibration outputs are inspected. Exact target values remain `TBD BEFORE
PRINCIPAL FREEZE`.

Accuracy, G/C/D, rescue, damage, and common-panel outcomes are forbidden during
shell calibration. Validity/evaluability and label-free sequence movement act
only as safety/manipulation gates; they do not rank directions within a shell.

## Pre-outcome identifiability gate

Before semantic-panel collection all must hold:

1. ten oriented directions and two shell targets are frozen;
2. within each shell, physical-radius CV `<=0.03`;
3. each family's shell median differs from the global shell median by `<=3%`;
4. all 40 cross-family angular dyads per shell are present;
5. effective rank of each retained direction Gram is `>=5`;
6. non-antipodal absolute cosine is `<0.95`;
7. angular-distance 90th-minus-10th percentile is `>=0.20`;
8. affine `R^2(candidate angular distance ~ absolute radius difference + mean
   radius) <=0.10` within each shell for every primary metric;
9. no family contributes more than 30% of total angular leverage;
10. standardized prediction-feature condition number `<=30`;
11. null physical radii satisfy the same shell tolerances;
12. if M3 is included, its separate engine classification is
   `M3_DIRECTIONAL_ENGINE_QUALIFIED`.

These values are design tolerances, not behavioral thresholds. Geometry-only
simulation shows exact shelling eliminates radial explainability; the 0.10
limit is deliberately far below the superseded 0.75 proposal. If the bank
fails, return `Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED` before semantic outcomes.
Do not redraw after any semantic output.

## Candidate geometries

- **M0:** exact normalized coordinate-space angular chord.
- **M1:** exact regularized covariance-whitened angular chord with lambda 0.10.
- **M2:** exact finite output-response JS pair distance, plus a new explicitly
  prospective baseline capture. Because sqrt(JS) is Hilbert-embeddable, baseline
  distances permit an implicit response angle through the law of cosines. The
  frozen V2 artifact itself had no such angle.
- **M3:** teacher-forced multi-checkpoint categorical-Fisher Gram and its angle,
  only if independently qualified.

For every metric, freeze pairwise distances, radii where defined, angular
predictors, code hashes, probes, checkpoints, and arrays before common-panel
outcomes. No Q2 V3 semantic outcome may fit a primary predictor coefficient.

## Instrument firewall

The finite official CRUXEval pool is exhausted, including the final Q1
confirmatory set. The preferred primary panel remains the same scientific task
distribution, but the draft is blocked from freeze until one of these is
proven prospectively:

1. an official CRUXEval generation procedure can create at least 240 genuinely
   fresh, distribution-faithful items with exact objective references; or
2. another independently justified same-distribution source is accepted by the
   principal researcher after provenance and instrument qualification.

Forty disjoint items qualify baseline opportunity; 200 are then frozen for the
primary panel. Existing CRUXEval items and holdout data cannot be reused. If no
same-domain source qualifies, return `Q2_V3_SAME_DOMAIN_INSTRUMENT_UNAVAILABLE`
rather than silently replacing the primary question.

LiveCodeBench test-output prediction is an optional, separately authorized
secondary transfer panel after the primary result. It does not substitute for
same-domain identification.

## Common panel

- primary items: 200;
- conditions: baseline + 20 meaningful + 4 null = 25;
- rollouts: two independent primary draws;
- rows: 10,000;
- schedule: deterministic interleaving frozen before outcomes;
- invalid/truncated model outputs remain errors;
- infrastructure retries reuse the exact logical key and seed;
- no metric inspection until all rows are complete.

The target remains the canonical unbiased two-rollout estimator

\[
D_{ij}=\frac1N\sum_t(e_{it0}-e_{jt0})(e_{it1}-e_{jt1}).
\]

Primary dyads are the 40 cross-family direction pairs within each shell, 80
shell-stratified dyads total. Same-direction cross-shell pairs form the radial
analysis. Null and within-family pairs are secondary.

## Claim hierarchy and inference

### Claim A: radial causal intensity (secondary control)

For each direction compare (D(i,strong;baseline)) with
(D(i,medium;baseline)). Report family-balanced signs and magnitudes.
`RADIAL_INTENSITY_REPLICATED` requires:

- strong-minus-medium median > 0;
- at least 8/10 directions positive;
- item-cluster bootstrap lower bound > 0;
- family-block sign-permutation p `<=0.05`.

### Claim B: relational geometry beyond physical magnitude (primary)

Within each shell, correlate each prospectively frozen angular/directional
predictor with (D_{ij}). Average family-balanced Spearman values across shells.
Use controller-family-aware max-statistic QAP permutations across all included
metrics and item-cluster bootstrap moving all conditions/rollouts together.

A metric has `RELATIONAL_SIGNAL` only if:

- mean family-balanced rho `>=0.25`;
- family-wise-max QAP p `<=0.05` after multiplicity control;
- item-bootstrap lower bound for rho > 0;
- at least 4/5 held-out-family summaries are positive;
- both shell-specific rho values are positive;
- leave-one-direction-out never reverses the aggregate sign.

The 0.25 threshold denotes a prospectively meaningful moderate rank relation;
it is unrelated to Q2 V2's 0.9067 RMSE ratio.

### Claim C: which information source is required?

Use a closed hierarchy after Claim B:

1. test whether either M0 or M1 qualifies;
2. test M2 only if its max-statistic corrected relational gate qualifies;
3. test M3 only if qualified before the experiment and its corrected relational
   gate qualifies.

“M2 required” additionally requires paired family-balanced rho improvement
`>=0.10` over both M0 and M1, bootstrap lower bound >0 for each contrast, and
step-down family-QAP p `<=0.05`.

“M3 required” requires the same `>=0.10` improvement and corrected evidence
against the best of M0/M1/M2. All metric results are reported regardless of the
hierarchy. Predictive calibration and RMSE are secondary because no Q2 V3
semantic outcome may fit the primary mapping.

## Classification vocabulary

- `Q2_V3_NO_STABLE_CONTROL_GEOMETRY`: radial and relational claims fail.
- `Q2_V3_RADIAL_ONLY_CONTROL_GEOMETRY`: Claim A passes; no relational metric passes.
- `Q2_V3_GENERIC_RELATIONAL_GEOMETRY`: relational gate passes, but M0/M1 suffice
  or no complex metric proves superiority.
- `Q2_V3_FINITE_RESPONSE_GEOMETRY_REQUIRED`: M2 passes its superiority gate.
- `Q2_V3_LOCAL_OUTPUT_GEOMETRY_REQUIRED`: qualified M3 passes its superiority gate.
- `Q2_V3_ANGULAR_ASSOCIATION_PREDICTIVE_ADEQUACY_LIMITED`: rank gate passes but
  prospectively scaled secondary error/calibration criteria fail.
- `Q2_V3_ANGULAR_IDENTIFIABILITY_FAILED`: pre-outcome bank gate fails.
- `Q2_V3_SAME_DOMAIN_INSTRUMENT_UNAVAILABLE`: fresh primary source cannot qualify.
- `Q2_V3_CONTROLLER_BANK_DESTRUCTIVE`: safety gate fails.
- `Q2_V3_INSTRUMENT_FAILURE` / `Q2_V3_ENGINE_FAILURE`.

No classification implies geometry-guided controller construction. That is a
later, separately authorized stage. Q3 remains not run.
## Falsification map

- stronger physical shells move behavior but no metric predicts within-shell
  dyads: radial-only geometry;
- M0/M1 predict within-shell error relations: generic relational geometry;
- M2 adds corrected, stable superiority: finite function-aware geometry;
- qualified M3 uniquely adds corrected superiority: local output-induced
  geometry;
- neither radial nor angular relations replicate: no stable candidate geometry;
- local M3 predicts token movement but not semantic error relation: output
  sensitivity and epistemic geometry dissociate.
