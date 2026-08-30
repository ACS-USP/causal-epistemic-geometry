# Start Here: the science in five minutes

## The problem

Two language models can have the same average accuracy while failing on
different problems. Those itemwise failure patterns matter: a change in *which*
problems fail may create complementarity even when mean competence barely
changes. Causal Epistemic Geometry asks whether such blind spots can be moved by
intervening in a frozen model and whether the resulting movements have
predictable geometric structure.

## The three questions

1. **Q1 — controllability:** can an activation intervention causally reorganize
   semantic blind spots without simply degrading the model?
2. **Q2 — geometry:** once blind spots move, do distances and directions among
   interventions predict distances among error profiles?
3. **Q3 — utility:** can a realizable selector or committee turn that structure
   into improved performance?

The current evidence answers Q1 narrowly and Q2 within one fixed experimental
subspace. Q3 has not been run.

## What Q1 established

The confirmatory experiment froze architecture-specific controllers, matched
random controls, 57 held-out CRUXEval items, two rollouts, estimators, bootstrap
rules, and safety guards before outcomes. The fixed Qwen3-8B L27-D75 controller
passed the complete rule: positive competence-adjusted complementarity, a
positive meaningful-minus-null contrast, superiority to every matched random
control, and preserved commitment/evaluability.

The same confirmatory program did **not** produce a Ministral model-level pass.
Its complementarity components were positive, but commitment validity and
semantic evaluability were 0.88596 and failed the frozen guards. A separate
fixed-controller transfer to long character counting was also negative. The
supported claim is therefore Qwen3-8B + CRUXEval, not universal steering.

Start with the [Q1 closeout](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md) and
the [Q1 visual evidence package](Q1_VISUAL_EVIDENCE.md).

## What Q2 established

Q2 V4.1 used all 31 directions that survived a prospectively frozen safety
gate. For each pair of controllers, the experiment compared semantic
blind-spot-shape distance with three pre-outcome geometries:

- **A0:** static flat geometry in the intervention coefficient space;
- **A1:** static geometry after the frozen covariance/whitening transform;
- **A2:** finite-response geometry built from label-free output-distribution
  responses.

All three geometries had significant, sign-stable relational association with
semantic blind-spot geometry. Aggregate Spearman correlations were 0.563818 for
A0, 0.556311 for A1, and 0.441128 for A2. A2 carried real signal but did not
outperform A0 or A1; both observed A2 superiority contrasts were negative.
The frozen classification is therefore **G2**, not G3.

The independent radial tests were also positive. **RS+** means every one of the
31 directions had greater blind-spot-*shape* displacement at STRONG than at
MEDIUM amplitude. **RT+** means the analogous total displacement was greater
in all 31 directions. These are local experimental results, not proofs of
global smoothness, linearity, manifold structure, or Riemannian geometry.

Read the [short Q2 pointer](Q2_V4_1_SEMANTIC_EXECUTION_CLOSEOUT.md) and then the
[canonical closeout](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.md).

## What is open now

A separate DEVELOPMENT transfer experiment is collecting the frozen
LiveCodeBench Stage B on Spark 2. It tests the exact fixed Qwen controller and
eight frozen null directions on 130 independent question families. Collection
is blind; no partial correctness, controller comparison, or semantic metric is
part of the current evidence. Its branch is
`research/q1-second-task-spark2-design` at authorization commit
`91c3db4ba41f8ad60f89920b605cbd09fba6dff9`.

## Evidence vocabulary

- **Confirmatory:** prospectively locked test on a held-out sample.
- **Development:** prospective or frozen scientific evidence used to build or
  assess instruments; informative but not upgraded to confirmatory.
- **Negative boundary:** a completed test that bounds generality.
- **Instrument non-qualification:** a stop before the target hypothesis was
  tested; it is not a scientific null.
- **Forensic clean:** an independent path reproduced the sealed result and its
  classification.

## What to trust and where to look

Use frozen raw/hash-pinned artifacts first, then independent forensic audits,
then `project_state.yaml` and `experiments/registry.yaml`. Narrative documents
are navigation, not a license to override a frozen result. The
[Claim–Evidence Matrix](CLAIM_EVIDENCE_MATRIX.md) encodes the current headline
claims and their allowed scope.

For exact numbers, read [Scientific Results](SCIENTIFIC_RESULTS.md). For the
full methodological genealogy, read the [Experiment Index](EXPERIMENT_INDEX.md).
For data and command boundaries, read [Reproducibility](REPRODUCIBILITY.md).

