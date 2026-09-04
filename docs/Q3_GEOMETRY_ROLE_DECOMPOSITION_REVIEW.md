# Q3.2 Geometry-Role Decomposition Review

## 1. Immutable Q3.1 result

Q3.1 remains permanently
`Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL`.
It showed stable prompt-representation policy selectability (+0.0533 over its
cross-fitted champion; 5/5 positive folds), while true A0 coordinates added
only +0.0033 over learned policy identity, below the frozen +0.01 criterion.
Q3 remains `NOT_RUN`; Q3.2 is closed-data DEVELOPMENT only.

## 2. Prospective development precheck

The precheck was frozen and pushed before outcomes were opened. Its SHA-256 is
`1754b7ad4b842752498f23a7c9b2ce7be0e57f2ad15754ff8babc19cff0f85ee`. A pre-result implementation-only
amendment cached the identical fold PCA and changed no scientific object; its
SHA-256 is `5394b82ab8d89c1cab2d75737f3ad0c526d1b5c48f60be444d4c9a38921bf6bd`. The full
private result was reproduced byte-for-byte twice and has SHA-256
`d1913c4e2b4f500ecece62da83598f5ca157455a788f6b7fc3a45f166c86c71e`.

## 3. A0-maximin bank performance

With the Q3.1 geometry-blind learned-policy-identity router, A0-maximin K=8
achieved routed accuracy 0.5033, versus its own
outer-training-selected champion at 0.4633: gain
+0.0400. Oracle headroom was 0.1583, of
which 0.2526 was realized. Fold gains were
`+0.0250, +0.0500, +0.0500, +0.0000, +0.0750`; 4/5 were positive
and the worst was +0.0000. Validity and evaluability were
both 0.9783.

## 4. Matched random-bank distribution

There were 512 prospectively frozen competence-matched
random-bank procedures. Their routed-gain q2.5/median/q95/q97.5 values were
`-0.0233, 0.0083, 0.0350, 0.0383`. A0's
gain percentile was 0.9863; its
plus-one upper-tail diagnostic p was 0.021442.
A0 exceeded the matched median gain by +0.0317.
These banks share items/controllers and are a paired development diagnostic,
not IID scientific replications.

## 5. Low-diversity and alternative-geometry banks

| Bank/design | Routed accuracy | Gain | Headroom | Positive folds | Worst fold |
|---|---:|---:|---:|---:|---:|
| A0-maximin | 0.5033 | +0.0400 | 0.1583 | 4/5 | +0.0000 |
| A1-maximin | 0.4650 | +0.0100 | 0.1250 | 3/5 | -0.0333 |
| A2-maximin | 0.5167 | +0.0383 | 0.1283 | 3/5 | -0.0250 |

The low-A0-diversity distribution had median gain
0.0083; the
unmatched deterministic-random distribution had median gain
0.0167.
The outcome-optimized bank remains an oracle upper bound, not a deployable bank.

## 6. Geometry bank-selection attribution

A0 passed every frozen gate: ≥0.03 realization gain, ≥95th-percentile gain and
headroom, both plus-one p-values ≤.05, ≥0.01 above matched median gain, and
nonnegative fold contrast in at least 4/5 folds (observed 5/5). Part A is:

`GEOMETRY_BANK_SELECTION_SUPPORTED`

This supports geometry's role in constructing the portfolio on closed data; it
does not show that coordinates improve routing within a fixed bank.

## 7. Historical-to-fresh controller transfer design

The model was trained on the fixed 31 historical controllers and evaluated on
the fixed 16 fresh OOS controllers, with simultaneous five-fold item-family
cross-fitting. PCA, scaling, hyperparameters and all model fitting used
historical-controller outer-training data only. The primary descriptor was
`[amplitude × unit rank-8 coordinates, amplitude]`; MEDIUM=0.25 and STRONG=0.50.
No policy-identity embedding or fresh-controller outcome entered fitting.

## 8. True-coordinate controller-OOS results

True coordinates achieved routing accuracy 0.4667,
against uniform fresh-policy accuracy 0.4548, for
gain +0.0119. Fold gains were
`+0.0141, -0.0203, +0.0495, +0.0185, -0.0023`: 3/5 positive,
with worst fold -0.0203. Log loss was
0.5762, Brier 0.1877,
and mean itemwise policy-ranking Spearman 0.0900.

## 9. Permuted/random/agnostic controls

| Representation | Routing gain | Positive folds | Worst fold | Log loss |
|---|---:|---:|---:|---:|
| True coordinates | +0.0119 | 3/5 | -0.0203 | 0.5762 |
| Permuted coordinates | -0.0031 | 3/5 | -0.0286 | 0.5965 |
| Random coordinates | -0.0031 | 2/5 | -0.0440 | 0.5954 |
| Controller-agnostic | +0.0061 | 4/5 | -0.0018 | 0.5940 |

True-minus-control routing-gain differences were
+0.0150 (permuted),
+0.0150 (random), and
+0.0057 (agnostic). The first
two crossed +0.01; the agnostic contrast did not. True coordinates improved log
loss over every control, but this predictive advantage did not satisfy the
routing-utility gates.

## 10. Controller-OOS routing utility

The true model failed the ≥0.03 realization gate, the ≥4/5 positive-fold gate,
and the −0.02 worst-fold gate. It also failed routing and fold-consistency
attribution against the agnostic control. The descriptor-only A0 kernel prior
gained -0.0048; the historical
global shell prior gained +0.0090.
Part B is:

`CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED`

## 11. Geometry-role ruling

Part A supports geometry for bank construction. Part B does not support useful
controller-OOS routing transfer. Q3.1 already did not support incremental
geometry for fixed-bank routing. The high-level ruling is:

`Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING`

## 12. Q3 narrative implication

The surviving development claim is narrow: A0 geometry may design a
complementary portfolio, while prompt representations plus learned policy
identity perform routing for known policies. True coordinates did not support
deployment to unseen controller identities. This is not realized Q3 utility,
and Q3 remains `NOT_RUN`.

## 13. Fresh-evaluation instrument roadmap

No future item was generated, selected or scored. The minimum proposed supply
is 800 family-independent units. The roadmap compares: newly authored
CRUXEval-like executable traces; a family-disjoint public exact-evaluator
benchmark; and a separately generated deterministic program-execution
benchmark. See [the instrument roadmap](Q3_FRESH_EVALUATION_INSTRUMENT_ROADMAP.md).

## 14. Reviewer/fragility audit

- Closed-data development analysis cannot establish prospective utility.
- Part-A banks reuse controllers/items and are not IID bank replications.
- Competence matching is finite-pool and cannot remove all bank-composition
  confounding.
- The 31→16 split is provenance-defined, but the fresh population is
  safety-conditioned and from the same Qwen/CRUXEval/rank-8 laboratory.
- Part B has only 16 held-out controller identities and five item folds.
- True-coordinate prediction improved log loss, but routing benefit was small
  and fold-unstable; this must not be narrated as successful transfer.
- The result does not address cross-model/task generalization or a fresh Q3
  holdout.

## 15. Repository/resource state

- New semantic trajectories: **0**.
- New Qwen forwards: **0**.
- Closed historical/fresh controller outcomes used: **YES, development only**.
- Future fresh-evaluation outcomes inspected: **NO**.
- A0 bank percentile among matched random banks: **0.986328**.
- Part-A ruling: **GEOMETRY_BANK_SELECTION_SUPPORTED**.
- True-coordinate controller-OOS gain: **+0.011875**.
- Part-B ruling: **CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED**.
- Minimum proposed fresh-family count: **800**.
- Q1/Q2/Q3.1 classifications changed: **NO**.
- Q3 confirmatory experiment: **NOT_RUN**.
- Spark 1 GPU used: **NO**.
- Spark 2 used: **NO**.
- RunPod used: **NO**.
- Personal handbook/paper workspace modified: **NO**.

`Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING`
