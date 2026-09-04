# Q3.1 Prompt-Representation Selectability Review

## 1. Scientific purpose

Q3 remains `NOT_RUN`. This DEVELOPMENT-only stage asks whether an unsteered,
label-free prompt representation can make one-call policy selection stable on
the already closed 300-family Q2/Q3.0 panel, and whether the true rank-8
controller geometry adds value beyond a capacity-matched learned policy-ID
representation. No fresh holdout was allocated or inspected.

The terminal development ruling is:

```text
Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL
```

## 2. Precheck

Before capture, commit `b9251a5b7819e5a79d4a6edc4e212aad94876985`
froze the 300 families, Qwen/tokenizer revision, prompt rendering, capture site,
5×4 nested family folds, fold-local preprocessing, A0-maximin K=8 primary bank,
A1/A2 secondary banks, models, controls, grids, seeds, gates and terminal
rulings. The precheck SHA-256 is
`aa78b37d730b1bfe7b0ff3e8bb4b2b47b67784a4f912485e7d9a30592caab8e1`.

Primary realization required gain ≥0.03, ≥25% oracle headroom, ≥4/5 positive
outer folds and worst-fold gain ≥−0.02. Incremental geometry required true
geometry to exceed both learned policy identity and fixed permuted coordinates
by ≥0.01, with ≥4/5 nonnegative foldwise contrasts against each. Selection
also required no policy share above 0.60 and at least three selected policies.

## 3. Representation capture

The authorized capture used Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16 and SDPA in the qualified
Spark-1 environment. It performed exactly 332 prompt-only forwards: 300 full
captures plus two deterministic forensic passes over 16 prompts. There was no
steering, candidate answer, semantic generation, reference loading, correctness
loading, or benchmark execution.

The representation is float32 after capture, shape 300×4096, at the layer-27
block input for the final non-padding prompt token. Its private matrix SHA-256
is `3612a645e3739e3cf7bf4d32f1f808034b15604a1e7f99e784c45e04b49d81ac`.

## 4. Capture forensics

All 300 item identities and prompt hashes matched the frozen private manifest.
Every vector was finite with width 4096. The 16-family deterministic repeat
subset had maximum absolute difference `0.0`. Layer-26 output and layer-27
input had maximum absolute difference `0.0` across all audited forwards. The
capture used the precheck commit and exact model-byte manifest
`cedc88ba2f732baea6bb71f5e6d7f6bc3aad00d302c3456d208a21687c9e069c`.

The capture is `Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSIC_CLEAN`.
The private prompt manifest, row metadata, full itemwise development result and
representation values are hash-pinned but not tracked.

## 5. Single-forward deployment feasibility

Single-forward deployment is mechanically feasible. A layer-27 pre-hook can
read the current prompt representation and select a frozen controller before
the same layer invocation; the pre-registered output hook can then apply the
selected current-token intervention without a second full prefill. The
synthetic hook-order test passed, and the real capture verified exact equality
of the proposed upstream and intervention-boundary sites. This does not itself
test steered semantic generation. See
[Q3 Single-Forward Routing Feasibility](Q3_SINGLE_FORWARD_ROUTING_FEASIBILITY.md).

## 6. Representation preprocessing

Raw 4096-dimensional vectors were never fitted directly. PCA dimensions
{8,16,32}, centering and component scaling were fitted inside each outer- or
inner-training fold. Outer-test rows never influenced PCA or normalization.
Inner CV selected interaction rank {1,2,4} and L2 {0.1,1,10}; tie-breaking
favored higher inner accuracy, then lower representation dimension, lower rank
and stronger regularization order as frozen. The model used deterministic
full-batch Adam for 400 steps.

## 7. Primary A0 K=8 bank

The primary bank was exactly A0-maximin, K=8, no baseline, selected separately
inside each outer-training fold under the Q3.0 rules. Its cross-fitted global
champion accuracy was 0.4533 and its oracle headroom was 0.1683. The five
banks differed only in the prospectively allowed outer-training shell choice
for one historical controller; no outer-test result selected a bank.

## 8. Geometry-aware router

The true-geometry low-rank logistic router achieved:

| Quantity | Value |
|---|---:|
| Routed accuracy | 0.5067 |
| Cross-fitted champion accuracy | 0.4533 |
| Absolute gain | +0.0533 |
| Oracle headroom | 0.1683 |
| Oracle fraction realized | 0.3168 |
| Positive outer folds | 5/5 |
| Worst-fold gain | +0.0083 |
| Fold-gain SD | 0.0375 |
| Commitment validity | 0.9783 |
| Semantic evaluability | 0.9783 |
| Maximum policy share | 0.4933 |
| Distinct selected policies | 6 |

All realization, validity/evaluability, concentration and diversity gates
passed. Calibration was Brier 0.1992, log loss 0.6219 and 10-bin ECE 0.0910.

## 9. Geometry-blind and permutation controls

The capacity-matched learned policy-ID router achieved accuracy 0.5033 and
gain +0.0500, with 4/5 positive folds and worst-fold gain 0.0000. The fixed
permuted-coordinate router achieved accuracy 0.4883 and gain +0.0350, with
4/5 positive folds and worst-fold gain 0.0000. The frozen Q3.0 deterministic
prompt-structure control gained only +0.0200, with 3/5 positive folds.

The precheck named a policy-prior random router but did not fix the probability
construction needed to instantiate it. It was therefore not invented after
outcomes. This omission does not affect the terminal ruling because neither
the realization gate nor the incremental-geometry gate depends on that
descriptive control, but it is a protocol-completeness limitation.

## 10. Nested development results

| Model | Accuracy | Gain vs champion | Oracle fraction | Positive folds | Worst fold |
|---|---:|---:|---:|---:|---:|
| True A0 geometry | 0.5067 | +0.0533 | 0.3168 | 5/5 | +0.0083 |
| Learned policy identity | 0.5033 | +0.0500 | 0.2970 | 4/5 | 0.0000 |
| Permuted coordinates | 0.4883 | +0.0350 | 0.2079 | 4/5 | 0.0000 |
| Q3.0 prompt structure | 0.4733 | +0.0200 | 0.1188 | 3/5 | −0.0167 |

The main development advance over Q3.0 is stable selectability from hidden
prompt features. It is not a geometry-specific result.

A1 and A2 were secondary and could not rescue A0. The true A1 router gained
+0.0250 (4/5 positive; worst −0.0250). The true A2 router gained +0.0533
(5/5 positive; worst +0.0083), but its learned policy-ID control gained more,
+0.0633. These secondary results reinforce the attribution boundary.

## 11. Fold stability

The primary true-geometry gains by frozen outer fold were:

```text
+0.0083, +0.1000, +0.0833, +0.0333, +0.0417
```

All were positive and the worst exceeded the −0.02 gate. The geometry-blind
gains were `+0.0417, +0.0917, +0.0750, 0.0000, +0.0417`; hence stable routing
was not unique to true controller coordinates.

## 12. Incremental geometry attribution

True geometry exceeded learned policy identity by only +0.0033, failing the
prospective +0.01 minimum. It exceeded permuted coordinates by +0.0183, but
only 3/5 foldwise true-minus-permuted contrasts were nonnegative, below the
required 4/5. True-minus-blind was nonnegative in 4/5 folds, but failed the
effect-size threshold.

Accordingly, the correct interpretation is that pre-generation prompt
representations make the diversity of causal policies selectable on this
closed development panel, while the actual controller geometry has not shown
incremental value beyond knowing policy identity.

## 13. Compute accounting

- Spark-1 prompt-only forwards: 332.
- Main representation captures: 300.
- Forensic repeats: 32.
- Capture-runner elapsed time: 141.951 seconds.
- New semantic generations: 0.
- Candidate answers: 0.
- Future deployment mechanism: one ordinary prefill/answer call, with no
  second full prefill mechanically required.

All routing fits and nested evaluation were CPU-only. Spark 1 was idle after
capture; Spark 2 and RunPod were not used.

## 14. Fresh-holdout implications

No fresh holdout should be allocated from this result. Stable selectability is
now plausible, but the stronger geometry-to-utility attribution gate failed,
and Q3.0 already established that the current CRUXEval inventory lacks a
sufficiently large fresh confirmatory population.

Future supply could be sought through model-free construction of executable
CRUXEval-like program-tracing families, a family-disjoint compatible benchmark
with an exact evaluator, or a separately frozen objective program-execution
instrument. Any route requires provenance, licensing, deduplication, difficulty
and evaluator qualification before model inference. No source was selected and
no dataset was created here.

## 15. Reviewer/fragility audit

- **Evidence level:** development only on outcomes already known from closed
  Q2 campaigns; no confirmatory claim is available.
- **Central attribution weakness:** +0.33 p.p. over learned policy identity is
  below the prespecified +1 p.p. minimum.
- **Fold attribution:** true geometry failed the required fold consistency
  against permuted coordinates.
- **Calibration:** true A0 geometry had slightly worse Brier/log-loss/ECE than
  the geometry-blind model despite marginally higher routed accuracy.
- **Sample/model selection:** 300 families and nested CV constrain leakage, but
  remain modest for selecting PCA dimension, rank and regularization.
- **Control completeness:** the descriptive random-router control was named but
  under-specified and therefore not post-hoc instantiated.
- **Scope:** Qwen3-8B, CRUXEval, the learned rank-8 subspace and the closed policy
  population only. No cross-task, cross-model, matched-random-subspace or
  realized-utility conclusion follows.

## 16. Repository/resource state

- Branch: `research/q3-route-a-prompt-representation`.
- Parent Q3.0 closeout: `da90220311ad710794233745677430067bf30d75`.
- Precheck commit: `b9251a5b7819e5a79d4a6edc4e212aad94876985`.
- New semantic trajectories: **0**.
- Fresh evaluation outcomes inspected: **NO**.
- Qwen prompt-only forwards: **332**.
- Representation site: **layer-27 block input, final non-padding prompt token**.
- Single-forward deployment feasible: **YES**.
- Primary routed gain: **+0.0533**.
- Oracle fraction realized: **0.3168**.
- Positive folds: **5/5**.
- Worst-fold gain: **+0.0083**.
- Geometry-aware minus geometry-blind gain: **+0.0033**.
- Geometry-aware minus permuted-coordinate gain: **+0.0183**.
- Q1/Q2 classifications changed: **NO**.
- Q3 confirmatory experiment run: **NO**.
- Spark 1 GPU: **used only for authorized prompt-only capture; now idle**.
- Spark 2: **NO**.
- RunPod: **NO**.
- Personal handbook/paper workspace modified: **NO**.

`Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL`
