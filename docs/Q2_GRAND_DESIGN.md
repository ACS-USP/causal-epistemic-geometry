# Q2 grand design: geometry that predicts and engineers complementarity

Status: design only. Q2 scientific outcomes have not been accessed and no Q2
experiment is authorized by this document.

## Grand question

**Can intervention geometry predict, and eventually engineer, epistemic
complementarity?**

Q1 established that fixed internal interventions can move semantic error
profiles, but also showed that readout, causal first stage, safety, and
cross-domain utility are distinct. Q2 should therefore test a predictive map,
not search retrospectively for an attractive correlation.

## Two spaces and a map

Within a fixed model, inference policy, task, evaluator, and intervention
operator, define an intervention/control space

\[
\mathcal H = \{h=(v,\ell,\alpha,s,\tau): v\text{ direction},\ell\text{ layer},
\alpha\text{ dose},s\text{ sign},\tau\text{ timing/scope}\}.
\]

The coordinates are chart-dependent. A vector at layer 12 is not naively
commensurate with one at layer 27, and vectors from different architectures do
not inhabit one Euclidean space. Cross-layer or cross-model comparison requires
an explicitly defined alignment or a behavioral metric.

For item population \(X\), let

\[
p_h(x)=\Pr(e=1\mid x,h)
\]

under the frozen stochastic generation policy, where invalid, unevaluable,
truncated, and model-runtime outcomes retain their canonical error status. The
epistemic error-profile space is

\[
\mathcal E \subseteq [0,1]^{|X|}, \qquad \Phi:\mathcal H\to\mathcal E,
\quad \Phi(h)=p_h.
\]

Q2 asks whether a metric measured in \(\mathcal H\) predicts held-out geometry
after mapping through \(\Phi\).

## Canonical error-space targets

### Propensity distance

The primary target is

\[
D_{ij}=\mathbb E_x[(p_i(x)-p_j(x))^2].
\]

With exactly two independent rollouts per condition, the canonical product
estimator is unbiased for the squared propensity difference and must replace a
plug-in square. Finite-sample estimates may be negative; they are reported, not
clipped.

### Error covariance and overlap

Secondary targets include double-fault overlap, pair-oracle gain, covariance,
and error-set correlation. They answer different questions and should not be
collapsed into one “diversity” score.

### Competence-adjusted complementarity

C distinguishes useful error reorganization from movement caused by changing
individual competence. It is a utility-relevant target, not a metric tensor on
intervention space. Q2 may ask whether geometry predicts C, but D should remain
the cleaner primary geometry target.

## Candidate intervention-space metrics

### 1. Normalized Euclidean/cosine geometry

For directions in the same layer and coordinate system,

\[
d_{\mathrm{cos}}(v_i,v_j)=1-\frac{v_i^Tv_j}{\|v_i\|\|v_j\|}.
\]

This is the baseline, not the presumed truth. It is not intrinsically
cross-layer or cross-model.

### 2. Activation-covariance geometry

Using a covariance estimate \(\Sigma\) frozen on source/calibration activations,

\[
\langle v_i,v_j\rangle_{\Sigma^{-1}}=v_i^T(\Sigma+\lambda I)^{-1}v_j.
\]

Regularization, source population, layer, and whitening rank must be frozen
before behavioral outcomes. A whitened metric can outperform Euclidean distance
without being a causal metric.

### 3. Finite behavioral secant geometry

For a fixed prompt/continuation fixture distribution, compare intervention-
induced output distributions or hidden trajectories at their actual doses:

\[
d_{\mathrm{sec}}^2(h_i,h_j)=\mathbb E_{x,t}
[\operatorname{JS}(P_{i,x,t},P_{j,x,t})].
\]

This is a finite-displacement control diagnostic. It is attractive for the
first Q2 experiment because it uses the exact deployed operator and avoids an
exact-derivative engine. It is not a local pullback/Fisher metric.

### 4. Local Jacobian/Fisher/pullback geometry

For continuous local coordinate \(a\) and model output distribution \(P_a\), a
directional Fisher object has the form

\[
g_h(u,v)=u^TJ_h^TF_hJ_hv.
\]

This is scientifically valuable only after the derivative engine and its
relationship to historical BF16 execution are prospectively qualified. Gate
12/12.1 established exact identities in an FP32 computational lift but did not
qualify the complete scientific bridge. Exact local work is a later Q2 branch,
not the first experiment.

### 5. Manifold/path metrics

Geodesics, learned manifolds, or nonlinear controller optimization require
evidence that local metrics predict held-out behavior and compose across a
path. They are not justified at Q2 entry.

## Predictive generalization design

Retrospective correlation across all controller pairs is inadequate because
dyads share controllers and items. The unit of predictive generalization must
be an unseen controller (or entire source family), not an unseen pair formed
from controllers already used for fitting.

A valid design should:

1. freeze a controller bank independently of Q2 semantic outcomes;
2. measure candidate geometries before revealing behavioral profiles;
3. split by controller/source family into train, validation, and held-out test;
4. estimate error profiles with independent two-rollout schedules on one common
   item panel;
5. fit metric calibration only on train controllers;
6. predict distances from held-out controllers to frozen anchors;
7. score prediction with controller-blocked loss and controller-label
   permutations;
8. bootstrap items as clusters and never treat dyads as independent samples;
9. compare candidate metrics under one frozen model-selection rule;
10. preserve all safety and competence outcomes without post-treatment
    filtering.

Primary predictive scores should be an absolute held-out loss (for example
mean squared prediction error for D) and improvement over a constant/base-rate
predictor. Rank correlation is useful but insufficient because it can hide
calibration failure. Metric comparisons use paired held-out controllers and a
permutation scheme that respects shared-controller dependence.

## Identification threats

- **Bank degeneracy:** many doses/signs of one vector can create a spurious
  one-dimensional relation.
- **Scale leakage:** dose determines both representation distance and behavior;
  source-family-held-out tests are needed.
- **Outcome-selected geometry:** choosing layers, whitening, tokens, or metrics
  after D/C is visible invalidates prediction.
- **Pair leakage:** random pair splits put the same controller in train and
  test.
- **Task saturation/floor:** D cannot be learned without baseline and
  controller error opportunity.
- **Validity mediation:** invalid outputs are part of p and must also be reported
  separately; complete-case D is not primary.
- **Architecture mixing:** model families need separate charts and analyses
  before any meta-level comparison.
- **Source/utility conflation:** careful-like movement can help one task and harm
  another, as Gate 10/11 showed.

## Staged Q2 program

### Stage Q2-A — finite-secant predictive benchmark

Build a diverse, frozen Qwen controller bank; compare Euclidean, whitened, and
finite-secant distances under controller-held-out prediction of D. No exact JVP
is required.

### Stage Q2-B — source-family and domain generalization

If Q2-A predicts unseen controllers, hold out whole source axes and repeat on a
second objective domain with an independently verified source policy.

### Stage Q2-C — exact local geometry

Only if justified, qualify a continuous engine on the exact scientific runtime
or freeze an explicitly different FP32-lift object. Compare exact local energy
with finite secants and held-out behavioral sensitivity.

### Stage Q2-D — prospective controller selection

Use the best frozen metric to choose one controller from candidates without
semantic outcome access, then evaluate its predicted D/C on fresh outcomes.

### Stage Q2-E — constrained construction

Optimize a controller under predicted complementarity and safety constraints,
freeze it, and evaluate once. This is the first stage that genuinely
*engineers* complementarity.

## Claim boundary

Q2 begins only when a prospective protocol, bank, common item panel, metric
family, controller split, and inferential procedure are frozen. Existing
readout, KL/JS, hidden-displacement, accuracy, G/C/D, and engineering-JVP results
are design evidence; none is a Q2 predictive outcome.
