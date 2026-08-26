# Q2 V4 — controller-label QAP and radial inference

Status: preliminary prospective design.

## Primary relational statistic

For each shell s and metric m in {A0,A1,A2}, compute Spearman rho between the
upper triangles of the pre-outcome geometry matrix G_m,s and the corrected
semantic shape matrix E_shape,s. The metric statistic is the equal-weight mean
of medium- and strong-shell rho. All K choose 2 dyads enter, but no dyad is
treated as independent.

The sampled coefficient rows are IID from one isotropic sphere rule and every
qualification/degeneracy gate is controller-label symmetric. Under the null
that geometry labels are arbitrary relative to semantic shape, controller
identities are exchangeable within this qualified design distribution.

## Permutation scheme

- K=32 controller labels are permuted as one permutation pi;
- the same pi is applied to both shells and every metric;
- E is held fixed and `G -> P_pi G P_pi^T`;
- 50,000 maps total: identity first plus 49,999 unique Monte Carlo maps;
- RNG: NumPy PCG64DXSM;
- seed: first 128 bits of SHA-256 over
  `Q2-V4-QAP-V1|<prospective-lock-commit>`;
- one-sided p-value: `count(T_pi >= T_observed) / 50000`, identity included;
- no individual-dyad permutation.

The 50,000-map choice gives resolution 0.00002 and Monte Carlo standard error
about 0.001 at p=0.05. It is feasible at K=32 and materially sharper than
10,000 without making the matrix computation dominant.

## Multiplicity and classifications

The primary family contains A0 coordinate angle, A1 covariance-whitened angle,
and A2 finite-response angle. Use single-step maxT over the three rho statistics
for the omnibus and adjusted attribution p-values. D2 total finite-response
distance is secondary and cannot create a primary classification.

A metric is relationally supported only if:

- maxT-adjusted p <= 0.05;
- shell-mean rho >= 0.20;
- both shell-specific rho values are strictly positive;
- the item-cluster bootstrap 95% lower bound is above zero;
- deleting any one controller does not reverse the aggregate sign.

Outcome classification is hierarchical:

- V4-G0: no primary metric qualifies;
- V4-G1: A0 and/or A1 qualifies, while A2 does not establish a separate claim;
- V4-G2: A2 qualifies;
- V4-G3: A2 qualifies and both `rho_A2-rho_A0` and
  `rho_A2-rho_A1` are at least 0.10, have positive item-bootstrap lower bounds,
  and pass a two-contrast single-step maxT QAP at 0.05.

G3 is intentionally stronger than “A2 significant and A0/A1 not significant.”
The power simulation shows that K=32/N=300 has excellent omnibus/A2 attribution
power at moderate effects but needs a genuinely large additional A2 signal for
high G3 power. That limitation must be reported, not repaired after outcomes.

## Radial inference

For direction k define

\[
R_k=D^{shape}(BASELINE,k_{STRONG})-D^{shape}(BASELINE,k_{MEDIUM}).
\]

Primary radial statistic: median over 32 directions. Use 50,000 Monte Carlo
paired shell swaps, swapping medium/strong labels within each direction and
recomputing the complete statistic with the same baseline. This preserves
shared-baseline and item dependence. Identity is included and the same
count/50,000 convention is used with an independently derived PCG64DXSM seed.

Use a 10,000 item-cluster bootstrap moving all conditions and both rollout
blocks together. Return `R+` only if median R>0, permutation p<=0.05, the
bootstrap lower bound is above zero, and at least 22/32 direction effects are
positive. Otherwise return `R-`. Radial status is independent of G0–G3.

## Additional uncertainty

Report the complete delete-one-controller series, direction-level effects, and
an induced-subgraph controller bootstrap descriptively. QAP remains the primary
randomization test; item bootstrap measures panel/rollout precision and does
not pretend dyads are IID.
