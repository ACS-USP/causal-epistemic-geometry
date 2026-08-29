# Four-rollout latent-propensity estimators

This is the prospective mathematical contract for Stage B.  It generalizes the
canonical Q1 two-rollout estimators without changing their latent targets.

For item `t`, let `b[t,r]` and `x[t,r]` be binary errors for baseline and one
non-baseline condition, with `r = 0,...,R-1`.  Invalid, unevaluable, malformed,
or terminally truncated model outputs have error one. Infrastructure failures
do not create an error record and are handled only by the locked retry policy.

Define itemwise rollout means

`bbar[t] = sum_r b[t,r] / R` and `xbar[t] = sum_r x[t,r] / R`.

The unbiased within-condition products are

`B00 = mean_t sum_(r != s) b[t,r] b[t,s] / (R(R-1))`

and analogously for the condition.  The baseline-condition product is

`B0x = mean_t bbar[t] xbar[t]`.

The repeated-baseline gain is `G = B00 - B0x`.  Between-item U-statistics are

`U00 = sum_(t != u) bbar[t] bbar[u] / (N(N-1))`

and

`U0x = sum_(t != u) bbar[t] xbar[u] / (N(N-1))`.

Competence-adjusted complementarity is

`C = B00 - B0x - U00 + U0x`.

The profile-movement estimator uses independent, off-diagonal rollout pairs:

`D = mean_t sum_(r != s) (b[t,r]-x[t,r]) (b[t,s]-x[t,s]) / (R(R-1))`.

This form is unbiased for the same latent squared propensity difference as the
Q1 estimator.  At `R=2`, it is algebraically identical to the frozen canonical
formula `b0*b1 + x0*x1 - b0*x1 - b1*x0`; this identity is unit-tested against
the independent historical implementation.

Rescue and damage are

`rescue = mean_t bbar[t] (1-xbar[t])`

and

`damage = mean_t (1-bbar[t]) xbar[t]`.

They obey `rescue - damage = accuracy_x - accuracy_baseline`.

The primary estimator pools all four rollouts.  Two predesignated exact-R2
replications use `{0,1}` and `{2,3}`.  Both halves are reported and enter the
frozen consistency gate; neither may be selected after outcomes.  Items are
the bootstrap unit, and every condition plus all four rollouts for an item move
together. Negative finite-sample values of `D` are retained.
