# Q2 V4 independent design audit

Classification: `Q2_V4_DESIGN_AUDIT_CLEAN`.

The audit independently checked the eight source-vector paths and hashes,
recomputed unit-column SVD/rank/conditioning, verified deterministic
coefficient generation, enumerated the finite-panel and item-population
expectations of the shape estimator on exact Bernoulli fixtures, recovered
angles from synthetic Hilbert squared distances, and checked shell-coupled
controller permutations.

The prompt's shape estimator is correct for a fixed panel but requires the
documented N/(N-1) correction for a superpopulation target. The M2 angle is
valid only after adding baseline captures and a zero-radius gate. These are
prospective design corrections, not post-outcome repairs.

The audit found no semantic journal import in the V4 module or design script,
no model/backend runner call, no GPU use, no V4 outcome, no Q3 action, and no
merge from `infra/dgx-spark-bringup`.
