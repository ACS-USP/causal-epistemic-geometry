# Q2 V4.1 semantic execution closeout

## Terminal result

The complete frozen campaign supports the mechanical relational classification
`Q2_V4_1_G2`, with independent radial classifications `RS+` and `RT+`.

This means the finite-response A2 geometry satisfies every frozen primary
qualification criterion. It does **not** satisfy the stronger G3 claim of
incremental superiority beyond both static geometries. Indeed, A0 and A1 have
higher observed aggregate rank correlations than A2.

## Execution and sealing

| Integrity field | Result |
|---|---:|
| Scheduled trajectories | 37,800 |
| Completed unique logical keys | 37,800 |
| Missing / unexpected / duplicates | 0 / 0 / 0 |
| Replacements / rows with retries | 0 / 0 |
| Summed Spark-1 generation time | 86.3159 GPU-hours |
| Raw journal SHA-256 | `d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99` |
| Parsed-score SHA-256 | `a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f` |
| Raw-data seal SHA-256 | `ce08ae11606b69a027393777e78f0cb40028f5bd2dcb5126eec6bfd1bfa309fc` |

Raw data were sealed before parser scoring. The primary analysis was then run
exactly once, followed by the independent forensic implementation.

## Basic semantic decomposition

Baseline accuracy was 0.4550; commitment validity and semantic evaluability
were both 0.9833. Across the 31 controllers, mean accuracy was 0.4676 at the
MEDIUM shell and 0.4496 at STRONG. Mean competence-adjusted complementarity C
was 0.01123 at MEDIUM and 0.03413 at STRONG; mean profile movement D was
0.02559 and 0.06667, respectively. The stronger shell therefore produced more
blind-spot movement and more answer-channel stress, rather than a uniform
accuracy improvement.

## Frozen relational metrics

| Metric | MEDIUM rho | STRONG rho | Aggregate rho | QAP raw p | maxT p | Item-bootstrap 95% interval | Qualifies |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 0.55482 | 0.57281 | 0.56382 | 0.00002 | 0.00002 | [0.39745, 0.54124] | PASS |
| A1 | 0.55468 | 0.55794 | 0.55631 | 0.00002 | 0.00002 | [0.39070, 0.53432] | PASS |
| A2 | 0.44257 | 0.43969 | 0.44113 | 0.00002 | 0.00002 | [0.30051, 0.43439] | PASS |

All three metrics also passed the frozen delete-one-controller sign-stability
criterion.

The bootstrap intervals above are the frozen item-cluster bootstrap summaries
of the shell-aggregated statistic; their reported point estimates therefore
need not equal the full-sample aggregate rho exactly.

## Comparative geometry and mechanical class

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| A2 full qualification | all six frozen criteria pass | all pass | PASS |
| A2 − A0 margin | -0.12269 | >= 0.10 | FAIL |
| A2 − A1 margin | -0.11518 | >= 0.10 | FAIL |
| A2 − A0 bootstrap lower bound | -0.14616 | > 0 | FAIL |
| A2 − A1 bootstrap lower bound | -0.13878 | > 0 | FAIL |
| A2 superiority maxT p values | 1.0 / 1.0 | both <= 0.05 | FAIL |

G3 therefore fails. Because A2 itself qualifies, the frozen precedence table
assigns `Q2_V4_1_G2`. The result supports finite-response relational geometry
without the stronger claim that A2 is incrementally superior to A0 and A1.

## Radial result

| Endpoint | Median STRONG−MEDIUM | Positive directions | Permutation p | Bootstrap 95% interval | Class |
|---|---:|---:|---:|---:|---|
| Blind-spot shape | 0.04406 | 31/31 | 0.00002 | [0.03001, 0.05114] | `RS+` |
| Total displacement | 0.04333 | 31/31 | 0.00014 | [0.03000, 0.05333] | `RT+` |

These radial results are secondary and independent of G0-G3.

## D2 secondary reporting boundary

The D2 finite-response-total matrices remain hash-pinned in the prediction
lock. The frozen executable semantic pipeline loaded those matrices but did
not materialize a D2 association statistic or decision rule. This closeout
does not invent one after observing outcomes. The omission has no effect on
the A0/A1/A2-based `Q2_V4_1_G2` classification. Any later D2 semantic analysis
would require an explicitly post-hoc or separately prospective ruling.

## Independent forensic audit

The independent audit returned `Q2_V4_1_SEMANTIC_FORENSIC_CLEAN`, reproduced
the primary `Q2_V4_1_G2` classification, found zero parser-field differences,
and had maximum primary/audit metric discrepancy 0.0.

## Post-hoc generation diagnostic

The separately labeled `POST_HOC / DIAGNOSTIC` report found 721/37,800 rows
(1.907%) at the frozen 4,096-token cap and 709/37,800 (1.876%) under the newly
persisted stringent token-periodicity criterion. Every capped/truncated row was
retained unchanged and scored under the frozen rule. This diagnostic cannot
alter G2 or either radial classification.

## Scientific interpretation and limits

Within this prospectively fixed, safety-conditioned 31-controller bank on one
Qwen3-8B revision and the historical 300-item CRUXEval panel, pre-outcome
intervention geometry predicts semantic blind-spot-shape geometry. Static
coordinate and covariance-whitened geometries were at least as predictive as
the finite-response A2 geometry in this experiment. Stronger intervention
amplitude also increased both total and shape displacement in every direction.

The experiment does not establish model, architecture, benchmark, or universal
geometry generality. A2 is pre-semantic-outcome but not purely
pre-intervention. G3's planning limitation remains relevant to interpretation,
but the observed superiority contrasts are negative and the frozen G3 rule was
not modified.

Historical integrity is preserved: original V4 remains
`Q2_V4_SAFE_BANK_INSUFFICIENT`; V4.1 is a distinct prospectively locked
experiment. Q3 was not run.

Final state: `Q2_V4_1_SEMANTIC_EXECUTION_COMPLETE`.
