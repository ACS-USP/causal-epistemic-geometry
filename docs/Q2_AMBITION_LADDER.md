# Q2 ambition ladder

Q2 should be judged by predictive and constructive generalization, not by the
most favorable retrospective correlation.

## Level 0 — no stable geometry-error relation

Candidate metrics fail controller-held-out prediction, are unstable under item
bootstrap, or do not beat controller-label permutation and constant predictors.

Evidence required: a diverse common-panel bank, valid blocked splits, and
adequate error opportunity. This is a scientifically useful negative for the
tested metric/bank, not proof that geometry is irrelevant.

## Level 1 — flat geometry predicts held-out error relations

Normalized Euclidean/cosine distance predicts D for unseen controllers and
beats constant/permuted baselines under controller-blocked evaluation.

Evidence required: source-family-held-out prediction, calibrated loss plus rank
score, item-cluster intervals, and replication across at least two controller
splits. This could anchor a focused Paper 2 result if robust and surprising.

## Level 2 — control-aware geometry outperforms flat geometry

Whitened activation or finite-secant geometry materially improves held-out
prediction over Euclidean geometry under one frozen comparison.

Evidence required: paired controller-held-out loss difference, predeclared
regularization, no layer/metric selection on test outcomes, and permutation-
aware uncertainty. This is strong Paper 2 territory and an excellent MSc
dissertation result.

## Level 3 — geometry predicts unseen behavioral movement

The metric predicts not just pairwise D among characterized controllers, but
the magnitude/direction of movement for a previously unseen controller or
source family.

Evidence required: geometry measured before behavioral reveal, a frozen mapping
from metric features to D/G/C, and a controller-family holdout. This would be a
high-impact MSc result and a strong PhD-launching platform.

## Level 4 — geometry prospectively selects a useful controller

Among frozen candidates with no semantic outcome access, geometry selects one
controller predicted to produce complementarity. A fresh evaluation confirms
the predicted ordering/effect and safety guards.

Evidence required: candidate pool and selection rule frozen first, new random
controls, fresh behavioral outcomes, competence/validity preservation, and no
fallback candidate. This is a compelling Paper 2 headline.

## Level 5 — geometry-guided constrained construction

Construct

\[
v^*=\arg\max_v \widehat{C}(v)
\quad\text{subject to}\quad
\widehat{\Delta A}(v)\ge-\epsilon_A,\
\widehat{\Delta V}(v)\ge-\epsilon_V,
\]

or an equivalent predeclared objective. Freeze v* and test once on fresh
outcomes.

Evidence required: a validated predictive model, an outcome-sealed candidate
or continuous search space, prospective safety constraints, fresh random and
optimization baselines, and a no-rescue evaluation. This would be exceptional
MSc work, a strong standalone Paper 2, and a clear PhD-launching result.

## Level 6 — cross-family/domain predictive principle

An analogous metric-selection or construction principle holds separately in
multiple model families and/or domains.

Evidence required: architecture-specific charts or a justified meta-metric,
separate prospective tests, objective evaluators, and no pooling that hides
heterogeneity. This is stretch territory: potentially a broad research program,
not a prerequisite for a successful dissertation.

## Program interpretation

| Outcome | Paper 2 | MSc dissertation | PhD-launching program |
|---|---|---|---|
| Level 0 with rigorous bank | Bounded negative/methods note | Solid if diagnosis is deep | Motivates alternative control representations |
| Level 1 | Focused empirical paper | Strong | Moderate |
| Level 2 | Strong comparative geometry paper | Excellent | Strong |
| Level 3 | High-value predictive paper | Exceptional | Very strong |
| Level 4 | Compelling causal-prediction paper | Exceptional | Excellent |
| Level 5 | Major constructive result | Beyond typical MSc scope | Central PhD program |
| Level 6 | Broad multi-system result | Stretch | Mature research agenda |

Higher levels are hypotheses, not milestones the data are expected to satisfy.
The project should stop or redirect when a well-powered lower-level test fails.
