# Q3 concept note — from complementarity to collective utility

## Status and boundary

Q3 is **not run**. This note states an end-state question; it is not an
implementation protocol and does not authorize data allocation, model calls,
selector training, or holdout access.

## End-state question

Can measured or predicted epistemic complementarity be converted into
realizable collective utility without oracle access?

Q1 asks whether an intervention can change a model's error profile while
preserving competence. Q2 asks whether internal control geometry predicts that
change. Q3 would ask whether those differences can be exploited at inference
time when the correct answer is unknown.

## Oracle headroom is not deployable utility

For two systems, pair-oracle accuracy answers a diagnostic question: how often
would at least one system be correct if an oracle could choose after seeing the
reference answer? It is an upper bound on opportunity, not an operational
method. High oracle headroom can coexist with a useless router.

A deployable system instead needs a selector, router, verifier, or aggregation
rule whose inputs are available before correctness is known. Its net value must
include routing mistakes, extra compute, latency, abstention, invalid outputs,
and any safety cost. Q3 must therefore distinguish:

- epistemic opportunity: complementary error sets exist;
- predictability: observable features forecast which system is more reliable;
- realization: a frozen mechanism converts that forecast into higher utility;
- economics: the gain survives inference cost and operational constraints.

## Candidate realizable mechanisms

1. **Confidence-calibrated router.** A frozen router uses pre-answer or
   answer-internal, non-oracular features to choose baseline versus a steered
   variant. Evaluation must be item-held-out and include routing calibration,
   abstention, and invalidity.

2. **Verifier-mediated selective committee.** Both variants answer, and an
   independently frozen deterministic or learned verifier selects, requests a
   tie-break, or abstains. The verifier must be trained without evaluation
   labels and compared with equal-compute controls.

3. **Budgeted adaptive policy.** Begin with the cheaper baseline and invoke the
   controller only when a prospectively validated uncertainty signal crosses a
   threshold. Utility is accuracy or task reward minus compute, latency, and
   safety penalties.

None is licensed by pair-oracle gain alone.

## How Q2 should inform Q3

Q2 should supply more than a ranking of controller pairs. Ideally it identifies
regions or directions in control space that predict distinct, safe error
profiles and provides uncertainty about those predictions. Q3 can then test a
small, frozen set of variants chosen for predicted complementarity, avoiding a
new outcome-driven controller search.

The clean dependency is:

1. validate a controller-held-out geometry-to-error prediction;
2. prospectively choose a small safe complementary ensemble;
3. freeze a realizable selection mechanism and utility function;
4. evaluate utility on fresh items against equal-compute baselines.

## Why Q3 should wait

The project has model-specific Q1 evidence and one cross-domain negative, but no
validated predictive geometry and no deployable selector. Starting Q3 now would
confound three unknowns: controller-bank quality, geometry prediction, and
selection quality. Q2 should first establish whether complementarity can be
predicted for unseen controllers. Until then, Q3 remains a concept only.

