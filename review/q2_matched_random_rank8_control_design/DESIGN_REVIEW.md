# Q2 matched random rank-8 specificity-control design review

## Executive ruling

`Q2_MATCHED_RANDOM_RANK8_CONTROL_REQUIRES_FURTHER_THEORY`

The proposed control has a clear scientific target and a calibrated
subspace-level reference-tail test in the model-free planning simulation.
However, the experiment is not ready for a prediction prelock. The unresolved
issue is not whether a random orthonormal basis can be drawn; it is how to
define and sample the relevant population of *safety-conditioned subspace
orientations* while retaining all 47 fixed coefficient identities. The most
defensible paired design is also large and underpowered for modest specificity
advantages.

This is a design-only ruling. No final random basis or experimental seed was
generated; no safety or semantic inference was run.

## 1. Specificity estimand

Let `Q_L` be the fixed learned 4096-by-8 orthonormal basis, and let
`c_1,...,c_47` be the exact frozen coefficient identities comprising 31
historical-reference and 16 fresh controllers. For any candidate orthonormal
random basis `Q`, define the matched activation directions by

```text
v_k(Q) = Q c_k.
```

For fresh identity `i`, define `r_i(Q)` exactly as in Q2 OOS V2: the
equal-shell average of the two Spearman associations, over the 31 reference
identities, between coefficient geometry A0 and semantic blind-spot-shape
distance. The natural subspace statistic is

```text
T(Q) = median_i r_i(Q),  i = 1,...,16.
```

The proposed specificity objects are

```text
Delta = T(Q_L) - E_Q[T(Q) | Q qualifies]

tau = P_Q(T(Q) >= T(Q_L) | Q qualifies).
```

The independent external-validity unit is one subspace orientation `Q`.
Controllers and dyads are nested measurements and cannot be substituted for
independent orientations. One random subspace would therefore be descriptive,
not evidence about arbitrary random orientations.

The target population must be stated conditionally: rank-8 orientations drawn
from a prospectively specified law that pass a prospectively specified,
semantic-outcome-free safety rule. Without that law, “random subspace” is not
a complete estimand.

## 2. Exact coefficient-geometry preservation

The audit reconstructed the exact 47-by-8 coefficient matrix in frozen order:

- 31 historical-reference identities from the closed V4.1 bank;
- 16 fresh identities from the closed OOS V2 bank;
- coefficient dimension 8;
- all coefficient norms numerically equal to one;
- no semantic outcome used.

An orthonormal map preserves every coefficient inner product:

```text
(Q c_a)^T (Q c_b) = c_a^T c_b.
```

It therefore preserves the complete coefficient-space Gram matrix and A0
geometry, not merely marginal norms. The ordered matrix, Gram, and A0 hashes
are frozen in `COEFFICIENT_GEOMETRY_AUDIT.json`. Selecting replacement
coefficient identities after safety would break this match and is not an
acceptable shortcut.

## 3. Candidate random-subspace families

### A. Ambient Haar-random rank-8

Draw a Gaussian 4096-by-8 matrix and take an orientation-corrected QR factor,
with the exact construction fixed before drawing. This directly addresses the
claim that an arbitrary ambient local rank-8 orientation would show similar
relational structure. It does not match activation anisotropy, and its
post-safety target law must be defined carefully.

### B. Haar-random rank-8 in the learned orthogonal complement

Draw in the 4088-dimensional orthogonal complement of the learned span. This
rules out accidental overlap with the learned subspace and is useful as a
secondary, sharper separation control. It is deliberately not representative
of the ambient Haar population and could make the learned orientation look
more distinctive by construction. It should not be the sole control.

### C. Activation-covariance-matched rank-8

This is the most relevant response to a reviewer who attributes the result to
residual-stream anisotropy rather than learned semantic construction. The
repository contains a label-free covariance allocation, but not a complete
frozen ambient covariance operator plus whitening, regularization, rank-floor,
and orientation-sampling rule sufficient to define this control. Projected A1
objects inside the learned span are not enough. This family remains promising
but underspecified.

No family should be chosen because it is expected to produce the weakest
semantic association.

## 4. Shell and safety matching

Every candidate orientation would need the same layer, timing, model,
intervention scope, and implemented-amplitude targets:

- MEDIUM = 0.25;
- STRONG = 0.50;
- the same frozen label-free safety panel and guards;
- a fixed one-shot orientation stream and a finite reserve specified before
  any semantic outcome;
- no semantic screening, redraw, controller replacement, or amplitude rescue.

To preserve the complete 47-identity coefficient geometry, all 47 identities
must remain present. If “qualified subspace” means that every one of those
identities passes both shells, feasibility is presently unknown. As a rough
sensitivity calculation only, treating per-controller safety as independent
gives:

| Per-controller pass probability | P(all 47 pass) | Expected orientations for 20 qualified |
| ---: | ---: | ---: |
| 0.7647 (fresh-bank observed) | 0.00000334 | 5.98 million |
| 0.7750 (historical-bank observed) | 0.00000627 | 3.19 million |
| 0.9000 | 0.00707 | 2,829 |
| 0.9500 | 0.0897 | 223 |
| 0.9900 | 0.6235 | 32.1 |

These are not estimates of the true subspace-level pass probability because
controller safety may be strongly dependent within an orientation. That is
precisely the missing quantity. Assuming favorable dependence would not be a
valid prelock.

Selecting the first 47 safe controllers from a larger coefficient stream would
change the frozen coefficient identities and A0 geometry. Lowering amplitudes,
dropping identities, or adapting the reserve after seeing attrition would also
change the scientific control. A separate label-free orientation-feasibility
design is needed before a semantic prelock.

## 5. Comparison of primary routes

### Route 1 — closed learned result versus S new random subspaces

The fixed learned statistic is compared with a future empirical distribution
over random orientations. This is the least expensive route and directly
estimates the random-orientation tail probability. Its weakness is imperfect
pairing: the learned and random arms are separated by campaign time and may not
share all stochastic execution noise. Reusing the same items is not the same
as rerunning all arms together.

### Route 2 — common-panel rerun of learned and S random subspaces

This is the preferred scientific route if feasibility is solved. Run all 47
coefficient identities in the learned orientation and in every random
orientation with common new item identities and homologous seeds. Compare
subspace statistics, not dyads. Common items/seeds reduce measurement noise and
permit paired item-level sensitivity analyses. The learned arm is still a
single prespecified orientation; the random orientation remains the
generalization unit.

### Route 3 — nonselective pilot followed by confirmation

A small pilot can qualify engine behavior, estimate orientation-level safety
attrition, and check runtime. It cannot select “promising” subspaces using
semantic associations. If all pilot subspaces advance and a disjoint item
panel is reserved for confirmation, pilot cost does not increase confirmatory
N and the finite CRUXEval supply becomes more restrictive. This route helps
only if the pilot targets label-free feasibility, not semantic effect.

## 6. Subspace-level inference

For `S` independently sampled qualified random orientations, a transparent
primary reference-tail statistic is

```text
p_MC = (1 + number of random T_s >= T_learned) / (S + 1).
```

This estimates where the prespecified learned orientation lies relative to the
specified safety-conditioned random-orientation law. The wording should be an
empirical/Monte Carlo reference-tail comparison, not an assertion that the
learned orientation was exchangeably drawn from the random law.

At alpha 0.05, `S=19` gives a minimum attainable p-value of 0.05, but deleting
one orientation makes significance arithmetically impossible. `S=20` is the
minimum that preserves 0.05 resolution after one deletion. `S=39` and `S=79`
improve tail resolution and stability but do not create many independent
learned orientations.

Secondary uncertainty may include:

- the mean/median learned-minus-random subspace contrast with subspace-level
  bootstrap over random orientations;
- nested item resampling with each complete subspace/controller block moving
  together;
- leave-one-subspace-out sensitivity;
- controller-level summaries within each subspace, explicitly nested;
- a separate sensitivity to the safety-conditioning mechanism.

No controller-level, dyad-level, or item-level secondary may rescue a failed
subspace-level primary comparison.

## 7. Model-free calibration and power

The prechecked simulation used 20,000 replicates per cell. It varied
`S in {20,39,79}`, `N in {100,150,300}`, `R in {2,4}`, two subspace-heterogeneity
levels, three levels of generic random alignment, and the complete 31/16
controller split plus one explicitly non-authorized planning sensitivity.
No ambient basis was generated.

Across complete-design null cells, the subspace-tail rejection rate ranged
from 0.04245 to 0.05320, below the frozen 0.06 calibration ceiling. Thus the
inferential idea is calibrated in the stated scalar planning model.

For the preferred Route 2 at `S=20, N=300, R=2`, full 31/16 identity design,
subspace SD 0.10, and generic random mean 0.35:

| Learned specificity advantage | Planning power |
| --- | ---: |
| 0.05 | 10.7% |
| 0.10 | 20.2% |
| 25% of observed learned median (0.1813) | 44.1% |
| 50% of observed learned median (0.3626) | 91.1% |

With heavier subspace heterogeneity (SD 0.20), power fell to 19.5% and 47.8%
for the 25% and 50% scenarios. Raising `S` to 79 at SD 0.10 increased those
powers to about 50.3% and 95.1%; it did not solve low power for 0.05--0.10
advantages. Increasing N or R had limited benefit once orientation
heterogeneity dominated.

These simulations are structural planning approximations, not evidence about
the actual distribution of random subspace effects. Their purpose is to expose
what must be learned before prelock.

## 8. Compute

Using the validated OOS rate of 19,200 rows in 9.4233 hours, one complete
47-identity subspace at `N=300, R=2`, two shells requires 56,400 trajectories,
or about 27.68 Spark-1 hours.

| Route | S | N | R | Semantic trajectories | Projected Spark-1 hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 100 | 2 | 376,000 | 184.5 |
| 2 | 20 | 100 | 2 | 394,800 | 193.8 |
| 1 | 20 | 300 | 2 | 1,128,000 | 553.6 |
| 2 | 20 | 300 | 2 | 1,184,400 | 581.3 |
| 2 | 39 | 300 | 2 | 2,256,000 | 1,107.2 |
| 2 | 79 | 300 | 2 | 4,512,000 | 2,214.5 |

The Route-2, `S=20, N=300, R=2` minimum serious design is roughly 24.2
continuous Spark-1 days before safety attrition. Evaluating the frozen 24-item,
two-shell safety panel for all 47 identities costs 2,256 rows per candidate
orientation, or at least 45,120 rows for 20 orientations if every orientation
qualifies. Unknown attrition can dominate that cost.

The smaller `N=100` route remains approximately eight continuous days and is
poorly powered for modest specificity. Reducing N therefore does not create an
obviously worthwhile confirmatory design.

## 9. Recommended next design work

Do not generate bases or open safety. First resolve four prelock questions:

1. Define the safety-conditioned orientation population and a finite,
   one-shot stream rule that preserves all 47 coefficient identities.
2. Run a separate model-free or label-free feasibility study of complete-bank
   orientation qualification; it must not inspect semantic outcomes.
3. Freeze or reject a covariance-matched ambient construction, including the
   covariance operator, regularization, sampling measure, and numerical gates.
4. Set a prospective compute ceiling and minimum practically important
   specificity advantage. The current grid shows that a modest advantage is
   not realistically resolvable at the minimum S.

If those questions are resolved, Route 2 with ambient Haar as the primary
family and orthogonal-complement Haar as a secondary family is the strongest
current candidate. Covariance matching should replace or accompany ambient
Haar only after its sampling law is fully specified. No final protocol is
prepared now because the design is not ready for prelock.

## 10. Scientific and resource boundary

- historical Q2 V4.1 changed: NO;
- Q2 OOS V2 primary changed: NO;
- final random bases generated: 0;
- experimental seeds generated: 0;
- safety inference: 0;
- new semantic trajectories: 0;
- Qwen loaded: NO;
- Spark 1 GPU used: NO;
- Spark 2 used: NO;
- RunPod used: NO;
- Q3 run: NO.

`Q2_MATCHED_RANDOM_RANK8_CONTROL_REQUIRES_FURTHER_THEORY`

