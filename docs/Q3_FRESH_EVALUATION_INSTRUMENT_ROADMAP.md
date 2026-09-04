# Q3 Fresh-Evaluation Instrument Roadmap

This is a design-only supply roadmap following Q3.2. It does not allocate a
holdout, select a source, generate items, inspect future correctness or
authorize Q3. The minimum target remains **800 family-independent evaluation
units**, pending a separate prospectively justified power review.

| Route | Task match | Exact evaluator | Family unit | Main risk | Licensing/reviewer posture |
|---|---|---|---|---|---|
| New executable CRUXEval-like traces | Closest to the closed program-execution laboratory | Sandboxed deterministic execution with typed exact output | One independently generated source program/problem | Template leakage and synthetic-family dependence | Clean only if generator, code license and release rights are frozen before creation |
| Family-disjoint public benchmark | Potentially strong if output prediction and family IDs are native | Official deterministic scorer only | Official problem/source-program family | Contamination, hidden siblings and evaluator drift | Strongest external credibility if redistribution and exact instrument are unambiguous |
| Separately generated deterministic benchmark | Tunable task and difficulty match | Frozen generator plus executable reference oracle | One generator-independent program family | Generator artifacts may dominate semantics | Credible only with diverse templates, held-out generators and public auditability |

## Required qualification sequence

1. Freeze task definition, family ontology, generator/source provenance,
   deduplication, license and evaluator before model contact.
2. Construct a development pool separate from the permanent evaluation pool.
3. Qualify commitment validity, semantic evaluability, difficulty, family
   independence and policy opportunity on development families only.
4. Run an outcome-independent power review for the surviving Q3 claim.
5. Allocate and hash at least 800 evaluation families only after all gates pass.
6. Keep policy-bank construction and router training outside the evaluation
   families; open the holdout exactly once under a new confirmatory lock.

## Runtime envelope

At the observed Q2 OOS rate (19,200 trajectories in 9.42 h), a minimal
8-policy × 2-rollout evaluation over 800 families would contain 12,800 semantic
trajectories and scale to roughly 6.3 Spark-1 hours before qualification,
prompt capture, retries or safety overhead. This is an operational extrapolation,
not a runtime lock.

## Decision still required

No source should be chosen merely for convenience. A future principal review
must select the instrument only after model-free provenance, licensing,
family-independence, exact-evaluator and contamination audits. No candidate
future correctness has been inspected here.
