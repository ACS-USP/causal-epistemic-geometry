# Q2 V4 — prospective intervention-subspace geometry design

Classification: `Q2_V4_DESIGN_FREEZE_READY`.

This CPU-only sprint opens a new Q2 design after the V3 family instrument
reached an outcome-free stopping point. V3's relational hypothesis remains
untested; its four-family instrument was underpowered. V4 replaces sparse
semantic-family vertices with 32 prospectively sampled directions inside an
8-dimensional source-supported intervention subspace.

## Effective subspace

The eight historical unit directions from four source-qualified concepts and
two source locations have exact and retained rank 8. Their singular spectrum is
`[1.586876, 1.149407, 1.060146, 1.025535, 0.824119, 0.706604,
0.685159, 0.580647]`; condition number is 2.732943 and entropy effective rank
6.593573. A relative 1e-6 numerical rank rule retains all components.

The final basis must be reconstructed and requalified natively on Spark 1. The
historical A40 SVD is design evidence, not a cross-backend equivalence claim.

## Direction population and scale

The recommended generator draws one frozen PCG64DXSM Gaussian coefficient
stream, normalizes each row on S^7, and maps it through the orthonormal basis.
No metric or behavior optimizes the bank. Symmetric coefficient-space gates are
defined before generation; failure stops without redraw.

Power simulation selects K=32 and N=300. Baseline plus medium/strong deployment
of all 32 directions yields 65 conditions and 39,000 two-rollout semantic
trajectories. Implemented radii 0.25 and 0.50 remain recommended, conditional on
native label-free calibration and all-controller safety/manipulation passage.

## Endpoint correction

The proposed `D_total - m0*m1` estimator is exactly unbiased for variance on a
fixed finite panel. For the same-domain item-population variance it is biased by
`(N-1)/N`; V4 therefore uses the exact N/(N-1) correction. Negative estimates
are retained. Total distance and global mean-shift squared remain reported.

## Geometry and inference

A0 is coordinate angular dissimilarity; A1 is regularized covariance-whitened
angular dissimilarity; A2 is a baseline-centered finite output-response angle.
The latter is mathematically valid because natural-log equal-weight JS is a
squared Hilbert distance and equal checkpoint averaging is a scaled direct-sum
embedding. It requires adding the unsteered baseline and a prospective
zero-radius/repeatability gate. D2 total finite-response distance is secondary.

The primary test is shell-mean Spearman under 50,000 controller-label QAP maps,
using the same permutation across both shells and A0/A1/A2. Single-step maxT
controls metric attribution. A2-required G3 additionally needs two >=0.10
superiority contrasts and corrected evidence. Radial evidence uses paired
medium/strong shell swaps over 32 directions and is returned independently.

At nominal rho 0.25, K=32/N=300 produced planning omnibus and A2-attribution
power 0.942/0.942, omnibus FPR 0.042, radial power 0.942, and A2-rho Monte Carlo
width 0.118. A dedicated finite-specific scenario reached 0.833 G3 power only
when observed A2 advantage averaged about 0.137; marginal superiority remains
hard and must not be overinterpreted.

## Backend

Decision: `V4_NATIVE_SPARK1`. Use Spark 1 only, one GB10 per job, dstack for
frozen jobs, SSH for debugging, no Spark 2, no multi-node. Exact model presence,
intervention engine, source reconstruction, subspace, shell safety, M1/M2, and
throughput all require a separately authorized prospective qualification.

Estimated future scale is 15–30 Spark GPU-hours, 18–36 wall-clock hours, and
roughly 5–15 GiB artifacts with 20 GiB operational headroom. These are planning
ranges, not qualified throughput.

## Evidence firewall

- new semantic inference: NONE;
- V4 semantic outcomes: 0;
- Spark scientific model inference: NONE;
- RunPod/GPU provisioning: NONE;
- V4 M0/M1/M2 matrices: NOT CONSTRUCTED;
- source reconstruction/shell calibration: NOT RUN;
- Q3: NOT RUN;
- confirmatory Q1 result: unchanged.

The design is ready for principal review and a separate prospective backend-
qualification/freeze authorization. It is not frozen and not executable by
this closeout.
