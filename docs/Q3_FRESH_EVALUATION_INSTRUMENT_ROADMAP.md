# Q3 Fresh-Evaluation Instrument Roadmap

This is the current design-only supply roadmap after Q3.3. It does not allocate
a holdout, generate items, inspect future correctness, or authorize Q3. The
earlier Q3.2 minimum of 800 family-independent evaluation units is preserved as
historical planning. The Q3.3 calibrated recommendation is a **1,600-family**
instrument: 300 qualification, 1,000 confirmation, and 300 reserve families.

| Route | Task match | Exact evaluator | Family unit | Main risk | Licensing/reviewer posture |
|---|---|---|---|---|---|
| New executable CRUXEval-like traces | Closest to the closed program-execution laboratory | Sandboxed deterministic execution with typed exact output | One independently generated source program/problem | Template leakage and synthetic-family dependence | Clean only if generator, code license and release rights are frozen before creation |
| Family-disjoint public benchmark | Potentially strong if output prediction and family IDs are native | Official deterministic scorer only | Official problem/source-program family | Contamination, hidden siblings and evaluator drift | Strongest external credibility if redistribution and exact instrument are unambiguous |
| Separately generated deterministic benchmark | Tunable task and difficulty match | Frozen generator plus executable reference oracle | One generator-independent program family | Generator artifacts may dominate semantics | Credible only with diverse templates, held-out generators and public auditability |

## Required qualification sequence

1. Freeze the restricted-Python AST generator, construct quotas, family
   ontology, evaluator, deduplication, license, and split namespaces before
   generating final items.
2. Generate the 300 qualification, 1,000 confirmation, and 300 reserve families
   from prospectively disjoint namespaces; one source skeleton defines one
   independent family.
3. Qualify evaluator determinism, dual evaluator agreement, parser round-trip,
   answer-channel validity/evaluability, difficulty, family independence,
   policy opportunity, and repetition rate on the 300 qualification families.
4. Do not use routed gain as an instrument-qualification gate and do not adapt
   the frozen candidate system.
5. If every gate passes, seal the untouched 1,000-family confirmation split and
   exact seeds under a separate execution lock; preserve 300 reserve families.
6. Open the confirmation split exactly once. Keep policy-bank construction and
   router training confined to the original closed 300-family development set.

## Runtime envelope

The selected utility-only design uses one routed answer and one champion answer
per family and rollout: N=1,000, R=2, at most 4,000 semantic trajectories. The
frozen Q2 OOS scaling gives P50/P80/P95 estimates of approximately
2.03/2.30/2.59 Spark-1 hours, 79,506 expected generated tokens, and 26.7 MB of
storage. These are planning estimates, not an execution lock.

## Decision still required

The preferred source is a separately generated deterministic restricted-Python
output-prediction instrument. A future prelock must still freeze the generator,
license, evaluator, construct quotas, IDs, seeds, and split hashes. No final
item or candidate future correctness has been generated or inspected. See the
[Q3.3 review](Q3_FINAL_SYSTEM_AND_EVALUATION_SUPPLY_REVIEW.md).
