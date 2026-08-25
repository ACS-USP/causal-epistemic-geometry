# Q2 geometry foundations

Status: `MATHEMATICAL FOUNDATIONS COMPLETE — NO SCIENTIFIC INFERENCE`

Q2 V2 remains `Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`, with forensic
classification `Q2_V2_FORENSIC_CLEAN`. Nothing in this memo changes that
result. The purpose is to define what each candidate geometry actually means
and to redesign Q2 V3 around the distinction between intervention strength and
intervention orientation.

## 1. Intervention and behavior spaces

Let the intervention-site activation be (h\in\mathcal H\subseteq\mathbb R^d).
For prompt or teacher-forced context (x), the downstream model defines a map

\[
F_x:\mathcal H\rightarrow\mathcal Y.
\]

For a next-token categorical output, write

\[
h\mapsto z_x(h)\mapsto p_x(h)=\operatorname{softmax}(z_x(h)).
\]

The scientific target is different. Controller (i) induces an itemwise
semantic correctness propensity

\[
p_i(t)=P(\text{semantically correct on item }t\mid i),
\]

and Q2's behavioral target is

\[
D^{error}_{ij}=\mathbb E_t[(p_i(t)-p_j(t))^2].
\]

Thus a token-distribution geometry is not, by definition, an epistemic-error
geometry. Q2 asks whether the former predicts the latter through the chain

\[
\text{activation perturbation}\rightarrow
\text{token distributions}\rightarrow
\text{autoregressive trajectories}\rightarrow
\text{semantic correctness}.
\]

Every arrow may discard information, amplify a small local change, or cross a
discrete semantic decision boundary.

## 2. Exact audit of Q2 V2 geometries

The implementation in `src/epistemic_geometry/analysis/q2_geometries.py` and
the frozen Q2 V2 lock are authoritative.

### M0: normalized coordinate-space angular chord

For nonzero controller vectors (v_i,v_j), Q2 V2 computed

\[
d^{(0)}_{ij}
=\left\|\frac{v_i}{\|v_i\|_2}-\frac{v_j}{\|v_j\|_2}\right\|_2
=\sqrt{2-2\langle\hat v_i,\hat v_j\rangle}.
\]

This is a metric on the normalized unit-sphere images and a pseudometric on raw
nonzero vectors because positive rescaling is discarded. It is induced by the
Euclidean inner product after normalization. It contains orientation and sign,
but no dose or radial information. It is invariant to orthogonal coordinate
changes and a common scalar rescaling, not to arbitrary invertible linear
reparameterizations. Its cost is (O(K^2d)).

Correct prospective name: **normalized coordinate-space angular chord
distance**. Calling it the complete flat distance between deployed
interventions is misleading because the deployed displacement norms were
removed.

### M1: regularized covariance-whitened angular chord

From label-free prompt-boundary activations, Q2 V2 estimated the sample
covariance \(\Sigma\), then froze

\[
\Sigma_\lambda=(1-\lambda)\Sigma+lambda\bar\sigma^2 I,
\qquad \lambda=0.10,
\]

where \(\bar\sigma^2=\operatorname{tr}(\Sigma)/d\). With
\(G_1=\Sigma_\lambda^{-1}\), it computed

\[
d^{(1)}_{ij}=
\sqrt{2-2\frac{v_i^TG_1v_j}
{\sqrt{(v_i^TG_1v_i)(v_j^TG_1v_j)}}}.
\]

This is an angular chord under a positive-definite, regularized Mahalanobis
inner product. It is not the unnormalized Mahalanobis distance between deployed
interventions because each vector is normalized in the (G_1)-norm. The pure
inverse-covariance construction transforms covariantly under invertible linear
changes, but the isotropic ridge based on (I) breaks full (GL(d)) invariance;
orthogonal/common-scale invariance remains. M1 is a global second-moment
correction. It is not a local density metric and not a learned activation
manifold.

Correct prospective name: **regularized covariance-whitened angular chord
distance**. “Manifold geometry” is incorrect. “Mahalanobis angular distance” is
acceptable only with the normalization and regularization stated explicitly.

### M2: finite output-response Jensen-Shannon pseudometric

For each controller, Q2 V2 persisted full-vocabulary logits at four checkpoints
over 12 label-free probes under a fixed teacher-forced continuation and a
finite sustained intervention. It computed

\[
d^{(2)}_{ij}=\sqrt{\frac1R\sum_{r=1}^{R}
JS(P_{i,r},P_{j,r})},\qquad R=48.
\]

The output reduction uses natural-log Jensen-Shannon divergence in float64.
Because \(\sqrt{JS}\) is Hilbert-embeddable and an equal-weight product of such
distances remains Hilbert-embeddable, this is a legitimate distance between
captured output-response fingerprints. It is a pseudometric on controllers:
distinct interventions can have identical captured responses.

M2 uses model intervention outputs but no correctness or semantic labels. It is
global and finite, not differential. Its cost is (O(K^2RV)), with vocabulary
size (V). The frozen capture did not include a no-intervention baseline, so it
did not identify an origin, response radius, or response angle. It also did not
construct an activation-space quadratic form.

Correct prospective name: **finite output-response Jensen-Shannon
pseudometric**. “Finite behavioral secant” is tolerable shorthand only if
“behavior” means the captured token distribution. It must not be called a
finite pullback approximation without a demonstrated small-displacement limit.

## 3. Local output-induced M3

For a categorical distribution (p=\operatorname{softmax}(z)), the Fisher
matrix in logit coordinates is

\[
F_p=\operatorname{diag}(p)-pp^T.
\]

If (J_z=\partial z/\partial h), the pullback to activation coordinates is

\[
G_x(h)=J_z(h)^TF_pJ_z(h).
\]

For direction (v), let (r=J_zv). Then

\[
v^TG_xv=r^TF_pr=\operatorname{Var}_{k\sim p}(r_k),
\]

so constant shifts of every logit have zero energy. Under the natural-log
convention and smoothness around (h),

\[
KL(p(h)\|p(h+\epsilon v))
=\frac{\epsilon^2}{2}v^TG_xv+O(\epsilon^3).
\]

Using Wurgaft et al.'s Hellinger convention

\[
H^2(p,q)=\frac12\|\sqrt p-\sqrt q\|_2^2,
\]

the corresponding local identities are

\[
H^2(p(h),p(h+\epsilon v))
=\frac{\epsilon^2}{8}v^TG_xv+O(\epsilon^3),
\]

and, for equal-weight natural-log JS,

\[
JS(p(h),p(h+\epsilon v))
=\frac{\epsilon^2}{8}v^TG_xv+O(\epsilon^3).
\]

The constants were independently verified in 24 CPU-only linear-softmax
fixtures. This establishes the mathematics, not the Qwen numerical engine.

## 4. Sequence-level behavior map recommended for M3

A single next-token distribution is mathematically clean but too remote from
the scientific endpoint. The recommended candidate uses the same frozen,
label-free probe contexts and teacher-forced checkpoint positions as M2. Let

\[
z_{x,k}(a)
\]

be logits at probe (x), checkpoint (k), under a sustained intervention
\(\delta(a)=\sum_i a_i u_i\) in the prospective controller span. At
zero intervention define

\[
r_{x,k,i}=\left.\frac{\partial z_{x,k}(a)}{\partial a_i}\right|_{a=0},
\]

and

\[
\Gamma_{ij}=\mathbb E_{x,k}
[r_{x,k,i}^TF_{p_{x,k}}r_{x,k,j}].
\]

This is the dataset-averaged local output-sensitive Gram matrix on the bank
span. It avoids constructing a (d\times d) tensor. Averaging loses
state-dependence: two directions can have the same mean energy while acting on
different items or checkpoints. Item-conditional Gram summaries must therefore
be retained as diagnostics.

Because the deployed operator injects the same direction at the final prompt
token and at every decode forward, this Jacobian is with respect to the shared
intervention coefficient (a), not one isolated residual vector (h). It
therefore includes all downstream effects of the prospectively frozen sustained
operator. Calling it a pullback to one activation state would be inaccurate;
it is a pullback to the low-dimensional intervention-parameter space.

For deployed displacement coefficients, the Gram gives

\[
r_i=\sqrt{\Gamma_{ii}},\qquad
\cos\theta_{ij}=\frac{\Gamma_{ij}}
{\sqrt{\Gamma_{ii}\Gamma_{jj}}},
\]

and

\[
d_{ij}^2=\Gamma_{ii}+\Gamma_{jj}-2\Gamma_{ij}.
\]

The scientific test is not that this is the “true geometry” of correctness. It
is

\[
d^{control}_{ij}\stackrel{?}{\longrightarrow}D^{error}_{ij},
\]

and, with intervention magnitude experimentally matched,

\[
\theta^{control}_{ij}\stackrel{?}{\longrightarrow}D^{error}_{ij}.
\]

## 5. Relation to Wurgaft et al.

Wurgaft et al. distinguish:

\[
G_I=I,
\]

a conformal activation-density metric

\[
G_E(h)=(\alpha e^{-E(h)}+\beta)^{-1}I,
\]

and a behavior pullback

\[
G_F(h)=J_F(h)^Tg_y(F(h))J_F(h)+\epsilon I.
\]

Their language-model behavior space is an open simplex over a small set of
task-relevant concept tokens plus an “other” class. They embed it using square
root probabilities and Hellinger distance, then study paths/geodesics tied to
explicit conceptual manifolds.

The analogy is exact only at the differential construction level: proposed M3
also pulls an output-distribution metric through a Jacobian. It breaks at the
behavior-space interpretation. Our full-vocabulary, teacher-forced checkpoint
distributions are proxies along a multi-token computation; our endpoint is
semantic correctness of complete stochastic trajectories. M1 is not their
density geometry, and M2 is not their pullback geometry.

## 6. Is the ladder defensible?

Yes, with corrected labels and a non-monotone epistemic interpretation:

1. **M0:** assumes coordinate-space orientation is meaningful and ignores the
   activation distribution and downstream function.
2. **M1:** adds one global, label-free second-moment model of where activations
   vary, but no model-output sensitivity.
3. **M2:** adds finite downstream response under the deployed operator, thereby
   mixing radial gain, direction, context, and curvature.
4. **M3:** adds a mathematically local output-distribution metric on the bank
   span, provided its numerical engine qualifies.

This is a hierarchy of information sources, not a theorem that M3 must predict
semantic errors best. M2 and M3 are not nested estimators at finite dose.

## 7. Density/manifold baseline decision

Wurgaft et al.'s density geometry is scientifically distinct from M1: it is a
state-dependent conformal metric derived from a fitted local activation density,
whereas M1 is one global covariance quadratic form. A faithful density baseline
would require a prospectively chosen density estimator, bandwidth, intrinsic
dimension, path integration rule, and enough activation support around every
new intervention. Q2 V2 provides none of those choices.

The revised Q2 V3 should therefore **not** add a density/manifold metric to its
primary ladder. Adding one now would expand researcher degrees of freedom and
multiplicity without a separately qualified estimator. It remains a future
mechanistic hypothesis if M0/M1 fail while a function-aware metric succeeds.

## 8. Adversarial alternatives

Q2 V3 must distinguish these falsifiable possibilities:

- semantic correctness is discontinuous across token-level decision boundaries;
- autoregressive compounding destroys local output geometry;
- item-specific metrics average into an uninformative global Gram;
- output geometry predicts response magnitude but not error identity;
- angular structure exists only in a small local region;
- the stable object is radial gain alone;
- blind spots are governed by combinatorial program states rather than a smooth
  Riemannian geometry.

These are candidate scientific conclusions, not caveats to be explained away.

## Reference inspected

Daniel Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Neural
Network Representation and Behavior* (2026), local reference SHA-256
`8dc25353e089b97e8f4a4474df3670f57fde0568a3a2ec08a51c234261051ca8`.
