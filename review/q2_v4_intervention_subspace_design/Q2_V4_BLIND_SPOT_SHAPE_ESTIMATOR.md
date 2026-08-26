# Q2 V4 — blind-spot-shape estimator

Let q_i(t) be controller i's error probability on item t and define
delta_t=q_i(t)-q_j(t). With two independent rollout blocks, let

\[
d_{t,r}=e_{i,t,r}-e_{j,t,r},\qquad r\in\{0,1\}.
\]

Block independence implies

\[
E[d_{t,0}d_{t,1}\mid t]=\delta_t^2.
\]

Thus

\[
\widehat D^{total}_{ij}=\frac1N\sum_t d_{t,0}d_{t,1}
\]

is conditionally unbiased for the mean squared propensity difference. Define

\[
\hat m_r=\frac1N\sum_t d_{t,r}.
\]

Because the complete rollout-0 vector is independent of the rollout-1 vector,
including across the same item,

\[
E[\hat m_0\hat m_1\mid t_1,\ldots,t_N]
=\left(\frac1N\sum_t\delta_t\right)^2.
\]

Therefore the estimator proposed in the authorization,

\[
\widehat D^{shape,panel}_{ij}
=\widehat D^{total}_{ij}-\hat m_0\hat m_1,
\]

is exactly unbiased for the variance under the uniform distribution on the
fixed frozen panel:

\[
\frac1N\sum_t(\delta_t-\bar\delta)^2.
\]

It is not exactly unbiased for the superpopulation item variance when the N
items are regarded as IID draws. Averaging over item sampling gives the usual
factor `(N-1)/N`. The primary V4 same-domain distributional estimand therefore
uses the finite-N correction

\[
\boxed{
\widehat D^{shape}_{ij}
=\frac{N}{N-1}
\left(\widehat D^{total}_{ij}-\hat m_0\hat m_1\right)
}.
\]

Equivalently,

\[
\widehat D^{shape}_{ij}
=\frac1N\sum_t d_{t,0}d_{t,1}
-\frac{1}{N(N-1)}\sum_{t\ne s}d_{t,0}d_{s,1}.
\]

The uncorrected panel estimand remains an algebraic audit output. The global
mean-shift component is `hat m_0 hat m_1`, and total distance is reported, so
the decomposition is never hidden.

## Assumptions

- the two rollout blocks are independent as vectors;
- seeds are unique across item/condition/rollout and the complete block-0 seed
  bank is disjoint from block 1;
- controllers and items are frozen before outcomes;
- items receive equal weight;
- invalid, unevaluable, truncated, and model-runtime outcomes remain errors;
- no complete-case filtering occurs.

Within-block coupling across controllers would not invalidate the derivation;
cross-block independence is the key requirement. The implementation does not
clip negative sample estimates. Exact finite-state enumeration in
`tests/test_q2_v4_design.py` verifies both expectations.
