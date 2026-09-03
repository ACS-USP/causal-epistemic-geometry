# Q2 Fresh-Controller Out-of-Bank Design Review

## 1. Scientific rationale

Q2 V4.1 established moderate relational association within one prospectively
fixed 31-controller bank (`Q2_V4_1_G2`, with `RS+` and `RT+`).  The proposed
next test asked whether the simplest and strongest prior geometry, A0,
generalizes to entirely fresh controller identities drawn from the same fixed
rank-8 subspace.  It did not reopen V4.1 or claim new-item, task, model,
subspace, manifold, or utility generalization.

## 2. What is genuinely out-of-sample

The frozen fresh namespace, seed, and PCG64DXSM stream are absent from the
historical 40-candidate stream, 31-controller bank, semantic outcomes,
prediction matrices, and QAP maps.  Exact coefficient overlap, vector-hash
overlap, and candidate-ID overlap were all false.  The maximum absolute
cross-stream coefficient cosine was 0.923036.  The shared model, panel,
subspace, layer, timing, and shells were intentionally held fixed.

## 3. Reference bank and frozen lineage

The design pins the exact 31-controller immutable manifest, rank-8 Q basis,
Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, L27
sustained-current-token intervention, N=300 panel, two rollouts, MEDIUM=0.25,
STRONG=0.50, and the sealed A0/A1/A2 artifacts from Q2 V4.1.  The reference
result remains `Q2_V4_1_G2`; it is not modified.

## 4. Fresh-controller generation

The PRELOCK commit `c774ef57b0247024d866c6efd8b0ab2aaa5c67d0` froze K=10,
19 candidates, the namespace `Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V1`,
big-endian 128-bit SHA-256 seed derivation, normalized Gaussian coefficients
in R8, `v=Qc`, generation order, algebraic gates, safety gates, first-safe
selection, and no-redraw rule.  The derived seed was
`184636762849226582434755075854859606661`.

The one realized stream retained rank 8, unit norms, finite values, and maximum
absolute pair cosine 0.884398.  It failed two frozen stream gates:

- effective rank 5.915762 < 6.0;
- condition number 3.591939 > 3.0.

No threshold was changed and no second stream was drawn.

## 5. K power/runtime comparison

The dependence-aware planning grid used 240 synthetic N=300 panels per cell
and evaluated K={6,8,10,12,16}.  At K=10, planning FPR was 0.0583; permutation
power was 0.979 at 50% of the prior A0 association and 1.000 at 75% and 100%;
full-effect sampling width was 0.136 and all-LOFO-positive frequency was 1.000.
Moderate controller-profile nuisance reduced achieved true rho to 0.216 but
retained 0.946 power and 0.996 LOFO stability.  Severe nuisance reduced true
rho to 0.079 and power to 0.371.  K=10 was the smallest design passing the
frozen utility rule and implied 12,000 future trajectories.  Nineteen
candidates gave P(at least 10 safe)=0.967 at safety rate 0.70 and 0.996 at the
historical 0.775 rate.

## 6. Cross-block estimand

The primary future matrix would have shape 10×31 in each shell, with
`A0_new,old = 1 - cosine(c_new,c_old)` and the unchanged superpopulation
`Dshape = N/(N-1) * (Dtotal - m0*m1)`.  The primary statistic was frozen as the
equal-weight mean of MEDIUM and STRONG cross-block Spearman correlations.
Pair entries were never to be treated as IID.

## 7. Dependence-aware permutation test

The frozen null permutes complete fresh-controller geometry rows against the
semantic rows, keeps all 31 reference columns fixed, and applies the same map
to both shells and all metrics.  This conditions on the sealed reference bank
and directly tests fresh-identity alignment.  The schedule was specified as
identity plus 49,999 unique PCG64DXSM maps, with
`p=count(T_perm>=T_obs)/50000`.  A0 is the single primary metric; A1/A2 were
secondary with Holm correction.  No final map artifact was generated because
the candidate stream gate stopped the experiment first.

## 8. Item bootstrap

The prospective design froze 10,000 percentile item-cluster resamples.  Each
sampled item moves all old and fresh controllers, both shells, and both rollout
blocks together; repeated sampled items retain multiplicity and negative
finite-sample Dshape values remain unclipped.  This panel uncertainty is
distinct from controller-label randomization.  No bootstrap schedule was
materialized after the candidate-stream failure.

## 9. Controller stability checks

Every leave-one-fresh-controller-out A0 aggregate rho had to remain strictly
positive.  Leave-one-reference-controller-out values were frozen as a complete
descriptive stability report but not an additional terminal gate.  No semantic
stability statistic exists because no fresh semantic outcome was opened.

## 10. Secondary radial design

The design would reuse the sealed V4.1 baseline rather than add a redundant
baseline block.  Secondary radial replication required positive median
`Dshape(BASELINE,STRONG)-Dshape(BASELINE,MEDIUM)`, exact one-sided sign p<=0.05
(at K=10, at least 9 positives), and a positive item-bootstrap lower bound.
This was not a continuous dose-response claim and was not executed.

## 11. A0/A1/A2 roles

A0 was prospectively primary because it was simpler and numerically strongest
in closed V4.1.  A1 and A2 were secondary replications.  A2 was explicitly
pre-semantic-outcome but not pre-intervention.  No A2-superiority or G3 test
was proposed.

## 12. Presemantic Spark-1 qualification

Spark 1 was not used.  The frozen algebraic gate failed before occupancy,
environment, safety-shell, or label-free A2 execution was scientifically
necessary.  Spark 2 and RunPod were not used.

## 13. Safe-bank result

`NOT_RUN_PREEMPTED_BY_FROZEN_CANDIDATE_STREAM_GATE`.

Safety status is unknown and must not be described as safe-bank insufficiency.
No safety trajectory, correctness label, or semantic trajectory was opened.

## 14. Frozen artifacts and hashes

- PRELOCK: `4c705fc1c6498d0d227aabfdc1092909d437c79e402311af25b1fd45ef22ce7c`
- protocol lock: `63e74bfc1411f4d6bfe75161ffb4439241f7904da09c95978aaf34f34d580f80`
- power table: `746dd9cd4e29f5a76cae8b292f4a9f09ea9294a2064a8a7cf7935165bc1ba81c`
- dependence sensitivity: `427b2270b665f20692956225c60f70a7ac07d8c91fcbae719c9dab3a627eaf4d`
- candidate manifest: `61f3fbcc4a48b413fa6feeef6bdc865f9b3ec06f6c767cd6681e1401a538d66b`
- historical-overlap audit: `4225ad8b2270379fc39e19a71356f518cd34da98c97bf6b43036430366c8f007`

The candidate manifest contains all 19 coefficient vectors, ambient L27 vector
paths, and per-vector hashes.

## 15. Future semantic schedule

The qualified design would have required 10×2×300×2=12,000 new trajectories,
with no rerun of the old 31 controllers and no baseline rerun.  Because the
candidate stream failed its frozen algebraic gate, no future semantic schedule
was generated and semantic execution remains unauthorized.

## 16. Prediction lock and terminal states

The protocol prospectively defined A0 predictions P1-P4 and the terminal
states `A0_PASS`, `ASSOCIATION_INCOMPLETE`, `NO_REPLICATION`,
`INSTRUMENT_NOT_QUALIFIED`, and `EXECUTION_INCOMPLETE`.  A final prediction
lock was not opened because the bank never reached safety/A1/A2 qualification.
The present terminal design state is
`Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED`.

## 17. Matched random-subspace design memo

The model-free memo freezes the requirements for a later matched random rank-8
L27 subspace: dimension, norm, covariance policy, controller count, shells,
safety attrition, semantic panel, and inference must match.  No random subspace
was generated or run in this sprint.

## 18. Repository/resource state

- branch: `research/q2-fresh-controller-oos-design`
- PRELOCK commit: `c774ef57b0247024d866c6efd8b0ab2aaa5c67d0`
- new semantic Q2 trajectories: 0
- new correctness inspected: NO
- LiveCodeBench outputs inspected: NO
- Q3 run: NO
- Spark 1 presemantic use: NO
- Spark 2 used: NO
- RunPod used: NO
- old Q2 result modified: NO

`Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED`
