# Q2 V2 — Calibrated controller-family-held-out geometry

## Outcome

Q2 V2 qualified the prospective controller bank but did not open the common
panel because the frozen cost gate failed. This is an operational cost block,
not a predictive geometry null.

- Classification: `Q2_V2_BANK_QUALIFIED_COMMON_PANEL_BLOCKED_PROJECTED_COST`
- Role: DEVELOPMENT
- Q1: unchanged
- Q3: not run
- Common-panel correctness outcomes read: no
- Common-panel rows: 0
- RunPod active GPUs: 0

## V1 postmortem

The immutable 348 Q2 V1 qualification trajectories were analyzed offline
without correctness, G, C, D, rescue, damage, or common-panel outcomes. At the
historical common norm, 2/12 meaningful controllers reached movement 0.25.
Seven failures were at 1/12 and three at 2/12. Sensitivity was sign-, axis-,
and source-location-dependent, motivating per-direction calibration.

## V2 source and controller bank

- Source axes proposed / qualified: 6 / 6
- Signed/location directions: 24
- Source qualification rows: 576
- Dose-calibration rows: 1,164
- Dose grid: D_LOW, D_MEDIUM, D_HIGH, D_VERY_HIGH
- Meaningful controllers retained: 24
- Distinct conceptual families: 6
- Controllers satisfying the frozen causal rule: 12
- Selected dose bins represented: 4
- Selected raw-sequence movement range: 0.0 to 0.25
- Causal selected-controller movement range: 1/6 to 1/4
- Accuracy, G, C, D, rescue, and damage used for bank construction: no

The final bank therefore satisfies the frozen bank-level dynamic-range rule;
weak but mechanically valid controllers remain intentionally represented.

## Null geometry

Four fresh random controllers were constructed by projecting against an SVD
orthonormal basis of the meaningful span and then normalizing. The maximum
absolute null-to-span cosine is `8.413408858487514e-17`, below the frozen
`1e-6` tolerance. Pairwise null orthogonality and unit-norm checks pass.

## Frozen predictive design

The final lock freezes:

- the 120-item untouched DEVELOPMENT common panel;
- 29 conditions and two independent rollouts, totaling 6,960 rows;
- leave-one-source-family-out prediction;
- M0 normalized Euclidean/cosine geometry;
- M1 activation-covariance-whitened geometry with frozen regularization;
- M2 finite behavioral secant geometry;
- the unbiased two-rollout target `D_ij = E[(p_i-p_j)^2]`;
- family-aware permutation and held-out RMSE procedures.

No JVP, Fisher, pullback, manifold geometry, geometry-guided controller, or Q3
analysis was added.

## Engineering and cost gate

The sustained intervention engineering gate passed. Label-free geometry inputs
were captured for M1 and M2 without correctness labels. The synthetic
throughput preflight then processed 174 non-scientific fixture rows:

- mean time per row: 32.69949623119506 seconds;
- median time per row: 0.6018590814783238 seconds;
- mean generated tokens: 713.4310344827586;
- maximum generated tokens: 4,096;
- projected common-panel runtime with 25% margin: 79.0237825587214 hours.

The dedicated RunPod MCP reports US$0.44/hour for the A40. The projected
common-panel cost is US$34.770464325837416. Adding the observed pre-common
Q2 V2 spend of US$2.579460696550086 gives a projected cumulative cost of
US$37.3499250223875, above the frozen US$25 hard ceiling.

The common panel was not started. No N, controller, dose, null, rollout,
metric, or threshold was reduced to fit the wallet.

## Predictive results

### M0 flat

- Family-held-out rho: NOT RUN
- RMSE: NOT RUN
- Permutation: NOT RUN

### M1 whitened

- Family-held-out rho: NOT RUN
- RMSE: NOT RUN
- Permutation: NOT RUN

### M2 finite secant

- Family-held-out rho: NOT RUN
- RMSE: NOT RUN
- Permutation: NOT RUN

No predictive Q2 result exists. In particular, the cost block must not be
reported as evidence for or against flat, whitened, or finite-secant geometry.

## Infrastructure closeout

The dedicated RunPod MCP reports Pod `2gy9k0axzzzp6u` in `EXITED` state and
zero active Pods/GPUs. The 20 GB Pod disk remains retained; no network volume
is attached. DGX Spark was not used.

## Next scientific question

Can the already-frozen, qualified controller bank support family-held-out
prediction of semantic error-profile distance when the complete 6,960-row
common panel can be executed under a newly authorized cost envelope or a
prospectively qualified lower-cost execution plan without changing the
scientific design?

No further inference is authorized by this closeout. Return for principal
review.
