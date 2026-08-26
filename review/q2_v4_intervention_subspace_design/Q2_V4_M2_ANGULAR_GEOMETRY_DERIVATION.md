# Q2 V4 — baseline-centered finite-response angular geometry

Classification: `VALID_WITH_MODIFICATIONS`.

For the project's equal-weight, natural-log Jensen-Shannon divergence on
finite categorical distributions, JS is of negative type. Consequently there
is a real Hilbert space H and map phi such that

\[
JS(P,Q)=\lVert\phi(P)-\phi(Q)\rVert_H^2.
\]

This is stronger than the triangle inequality for `sqrt(JS)`. The relevant
primary references are Fuglede and Topsøe, *Jensen-Shannon Divergence and
Hilbert Space Embedding* ([DOI 10.1109/ISIT.2004.1365067](https://doi.org/10.1109/ISIT.2004.1365067)),
and Endres and Schindelin, *A New Metric for Probability Distributions*
([DOI 10.1109/TIT.2003.813506](https://doi.org/10.1109/TIT.2003.813506)).

Historical M2 averages over R fixed probe/checkpoint indices. For fingerprint
`P_i=(P_{i,1},...,P_{i,R})`, define the direct-sum embedding

\[
\Phi(P_i)=R^{-1/2}(\phi(P_{i,1}),\ldots,\phi(P_{i,R}))\in H^{\oplus R}.
\]

Then

\[
\lVert\Phi(P_i)-\Phi(P_j)\rVert^2
=\frac1R\sum_r JS(P_{i,r},P_{j,r}).
\]

Thus the exact historical aggregation `sqrt(mean JS)` remains Hilbertian. Equal
weights are essential; fixed nonnegative weights summing to one would also work
through square-root coordinate scaling.

V4 adds the unsteered baseline fingerprint P_0 to the same capture. Let

\[
r_i^2=d(i,0)^2,\qquad d_{ij}^2=d(i,j)^2.
\]

The Hilbert law of cosines gives

\[
\langle i,j\rangle
=\frac{r_i^2+r_j^2-d_{ij}^2}{2}
\]

and, for positive radii,

\[
\boxed{
\cos\theta^{A2}_{ij}
=\frac{r_i^2+r_j^2-d_{ij}^2}{2r_ir_j}
}.
\]

The primary A2 matrix is `1-cos(theta)`. Total `sqrt(mean JS)` distance is
retained as secondary D2 because it deliberately mixes radius, direction, and
finite nonlinearity.

## Zero-radius and numerical gate

Before prediction lock, repeat the deterministic baseline technical capture.
Freeze

```text
tau_squared = max(1e-12, 100 * maximum_repeat_baseline_mean_JS)
```

using only repeatability evidence. If any primary controller has
`r_i^2 <= tau_squared`, A2 is undefined and V4 stops before semantic outcomes;
the controller is not replaced. A cosine outside [-1,1] by at most 1e-8 may be
clamped as roundoff. A larger violation fails the Hilbert-distance consistency
check.

This construction is a finite output-response angle around a baseline origin.
It is not a local JVP, Fisher metric, pullback metric, or Riemannian geometry.
M3 remains excluded.
