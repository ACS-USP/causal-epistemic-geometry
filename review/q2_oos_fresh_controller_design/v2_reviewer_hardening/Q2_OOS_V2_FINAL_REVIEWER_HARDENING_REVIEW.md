# Q2 OOS V2 Final Reviewer-Hardening Review

## 1. Fixed V1/V2 history

Q2 OOS V1 remains permanently
`Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED` at
`249543e044f3d07713ac90dc6b68988e237f5119`. Its 19 candidates remain
`HISTORICAL_FAILED_STREAM`, `EXCLUDED_FROM_V2`, and
`NEVER_SEMANTICALLY_EXECUTED`. The accepted model-free gate audit remains
`Q2_OOS_V2_INFERENCE_ALIGNED_DESIGN_READY_FOR_PRINCIPAL_REVIEW` at
`ff7ede3785e3e4a203cf64f4260e7cc6b819918b`. This review preserves Route C
and does not reopen either historical ruling.

Methods, simulation seeds, precision, and the K-selection rule were committed
at `6e7a4356409a193212ed70ecc65ff0dd1144841d` before the final simulations. No
V2 controller stream or future
V2 seed was materialized.

## 2. K=10 vs 12 vs 16

Each qualification cell used 40,000 streams. Each power cell used eight
moderate-safety planning banks, 600 independent synthetic N=300 binary panels,
two rollout banks, both shells, 31 fixed reference columns, and 999 unique
fresh-row maps including identity. Width is the empirical 2.5%--97.5% panel
sampling width; the future experiment retains its separately frozen 10,000-map
item bootstrap.

| K | n | P(>=K safe), p=.60 | Qual., independent | Qual., moderate axis | Half-prior power | Quarter-prior power | Power at rho=.15 | Null FPR | All-LOFO+ at half | Width at rho=.15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 24 | 0.9783 | 0.9553 | 0.9537 | 0.985 | 0.610 | 0.685 | 0.0117 | 1.000 | 0.1272 |
| 12 | 27 | 0.9663 | 0.9639 | 0.9627 | 0.993 | 0.600 | 0.835 | 0.0483 | 1.000 | 0.1204 |
| 16 | 34 | 0.9556 | 0.9528 | 0.9544 | 1.000 | 0.805 | 0.870 | 0.0733 | 1.000 | 0.1135 |

Monte Carlo standard error for the K=16 null FPR is 0.0106, so its point
estimate just below the frozen 0.075 ceiling is not evidence that the true FPR
differs from nominal 0.05. It is reported as simulation uncertainty, not
hidden or converted into a new gate.

## 3. Independent fresh-controller sample-size rationale

The independent generalization units are the K fresh controller identities,
not the K×31 dyads. K=10 is statistically powered for moderate effects but is
the most reviewer-vulnerable: one identity is 10% of the external sample and
there are only 45 entirely new-new edges. K=12 reduces single-identity
leverage to 8.3% and improves small-effect power, but remains a modest
identity sample. K=16 gives 60% more identities than K=10, 120 new-new edges,
the narrowest planning distribution, and materially better power around
rho=.15. It is still a bounded within-laboratory identity sample, not a claim
over arbitrary intervention spaces.

The frozen decision rule first required reserve and unconditional qualification
probabilities of at least 0.95 in both safety scenarios, null FPR at most
0.075, half-prior power at least 0.80, rho=.15 power at least 0.60, half-prior
LOFO stability at least 0.80, width at most 0.40, at most 20,000 semantic
trajectories, and at most 50 projected safety-plus-semantic GPU-hours. Among
passing designs it selected the largest K unless the adjacent increase cost
more than 50%. All three passed; K=16 costs 33.0% more than K=12 and was
therefore selected.

## 4. Final candidate reserve n

The final recommendation is K=16 and n=34. At p_safe=.60, the exact reserve
probability is 0.95556. The unconditional inference-aligned qualification
probability was 0.95278 under independent safety (Wilson 95% interval
[0.95065, 0.95481]) and 0.95438 under moderate historical-axis sensitivity
([0.95229, 0.95638]). The next smaller n is not eligible because it fails the
exact 0.95 safety-reserve requirement.

All 34 future candidates must be evaluated; the first 16 passing both frozen
shells are selected in generation order. No redraw, replacement, extra
candidate, or subset optimization is allowed.

## 5. Fresh×old primary

The primary endpoint remains A0 versus Dshape over a 16×31 cross block in
each shell. The statistic is the equal-weight mean of shell-specific Spearman
correlations. A primary pass requires positive aggregate rho, one-sided
fresh-row QAP p<=0.05, a positive lower 95% item-bootstrap bound, and every
leave-one-fresh-controller aggregate rho strictly positive. A1 and A2 remain
Holm-corrected secondary metrics and create no A2-superiority claim.

This uses 496 fresh×historical-reference dyads per shell but keeps inferential
language at the 16-controller identity level.

## 6. Fresh×fresh secondary

The predeclared secondary compares A0 and Dshape among all 120 strict
upper-triangle fresh×fresh dyads per shell. It uses the equal-weight shell mean
of upper-triangle Spearman correlations, 50,000 controller-label QAP maps,
the same item-cluster bootstrap, and complete leave-one-fresh-controller-out
stability.

For each map, geometry is conjugated as `P_pi A0 P_pi^T` while Dshape remains
fixed; the same pi is used in both shells. K=10 has 3,628,800 possible maps,
so exact enumeration is technically feasible but unnecessarily costly. K=12
has 479,001,600 maps and K=16 has 20,922,789,888,000. The final convention is
therefore identical for all K: identity plus 49,999 unique sampled
non-identity maps.

The outcome-free classification precedence is:

1. `FRESH_FRESH_NO_ASSOCIATION` if rho<=0 or QAP p>0.05;
2. `FRESH_FRESH_ASSOCIATION_POSITIVE` if rho>0, p<=0.05, bootstrap lower
   bound>0, and every LOFO rho>0;
3. `FRESH_FRESH_ASSOCIATION_INCOMPLETE` for remaining positive,
   permutation-supported results lacking bootstrap or LOFO support.

This resolves an overlap in the precheck prose before any semantic outcome.
The secondary can never rescue, upgrade, or change a failed fresh×old primary.

## 7. Predicted-power gate ruling

`RETAIN_AS_MANDATORY_DIAGNOSTIC_ONLY`.

Across the 24 eligible planning banks (eight per K), half-prior bank-level
permutation power ranged from 0.9467 to 1.0 and all-LOFO-positive frequency was
1.0 for every bank. The accepted earlier audit was also ceiling-saturated
across 1,200 K=10 banks. Rank, effective rank, near-duplication, A0 cross-block
spread, and explicit geometry diagnostics already protect the relevant gross
failure modes. A terminal predicted-power gate would therefore add no observed
discrimination while making qualification depend on an arbitrary synthetic
response model. The calculation remains mandatory and reported because it is
useful planning evidence, but it has no pass/fail threshold.

## 8. Permutation-null mathematical review

For the primary, fresh controller is the scientific exchangeability unit.
Let `G` and `D` be K×31 geometry and semantic matrices. The frozen statistic
uses `T(G,D)`, and a map evaluates `T(P_pi G,D)`. The same `P_pi` is used in
both shells. The 31 historical columns stay fixed because the experiment asks
whether newly sampled controller identity labels align with their semantic
rows conditional on the sealed reference bank.

The tested null is: conditional on the frozen references, shared N=300 panel,
shells, and safety-conditioned fresh bank, the assignment of fresh-controller
geometry rows to fresh-controller semantic rows is exchangeable. Complete-row
permutation preserves all 31 shared-reference edges, item-panel dependence,
within-row dependence, and cross-shell dependence. Safety is presemantic,
condition-symmetric, and never uses correctness, so it does not assign rows
using the semantic endpoint.

Double row+column permutation is a different test for the rectangular primary:
it would additionally randomize the historical reference identities, assume
exchangeability of a fixed, previously observed reference bank, and no longer
condition on the exact reference geometry that defines the replication. It is
therefore inappropriate for the fresh×old question. Conjugating both axes is
correct only for the fresh×fresh secondary, where the same fresh identity
occupies both axes of a symmetric matrix.

## 9. Claim boundary

The maximum positive claim is: within the same frozen Qwen3-8B / CRUXEval /
rank-8 intervention laboratory, the previously discovered A0 relational
geometry generalizes to prospectively sampled, safety-conditioned fresh
controller identities. A positive fresh×fresh secondary would strengthen only
that identity claim.

The protocol does not test cross-task, cross-model, cross-subspace, universal
metric, global smoothness, manifold, Riemannian, utility, or precise-pairwise
prediction claims. Broad dispersion association is the intended construct.

## 10. Final V2 protocol recommendation

Freeze K=16, n=34, Route C, one normalized-Gaussian PCG64DXSM stream from a
future committed PRELOCK, first-16-safe selection, inference-aligned bank
gates, diagnostic-only predicted power, the unchanged fresh×old A0 primary,
and the new fresh×fresh A0 secondary. The actual seed must be derived only
from that future PRELOCK commit. This review does not authorize PRELOCK
materialization, safety execution, A2 capture, or semantic execution.

## 11. Projected compute

| Component | Scale | Projected Spark-1 GPU-hours |
|---|---:|---:|
| Safety | 34×2 shells×12 items×2 rollouts = 1,632 | 2.163 |
| Future semantic | 16×2 shells×300 items×2 rollouts = 19,200 | 43.843 |
| Safety + semantic | 20,832 trajectories | 46.006 |

A2 capture still requires an outcome-free throughput preflight because no
canonical elapsed-time artifact supports a defensible extrapolation. It is not
silently included in the 46.006 GPU-hour estimate.

## 12. Repository state

- branch: `research/q2-fresh-controller-oos-design`
- starting audit HEAD: `ff7ede3785e3e4a203cf64f4260e7cc6b819918b`
- reviewer-hardening precheck commit: `6e7a4356409a193212ed70ecc65ff0dd1144841d`
- new V2 stream generated: NO
- actual V2 seed derived: NO
- model inference: 0
- semantic trajectories: 0
- correctness inspected: NO
- Spark 1 used: NO
- Spark 2 used: NO
- V1 changed: NO

`Q2_OOS_V2_FINAL_PROTOCOL_READY_FOR_PRELOCK`
