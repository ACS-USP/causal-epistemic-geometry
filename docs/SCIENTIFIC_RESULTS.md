# Scientific results

This page summarizes current evidence for external readers. Frozen closeouts
and forensic audits remain authoritative if any narrative conflicts with this
summary. Last reconciled: 2026-08-30.

## Results at a glance

| Question | Experiment | Evidence | Result | Scope and canonical evidence |
|---|---|---|---|---|
| Q1 | Fixed Qwen confirmatory | **CONFIRMATORY — PASS** | Safe, null-specific competence-adjusted complementarity | Qwen3-8B + 57 held-out CRUXEval items; [closeout](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md), [result JSON](../review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json) |
| Q1 cross-model | Fixed Ministral confirmatory | **CONFIRMATORY — FAIL** | Complementarity components positive; mandatory validity/evaluability guards failed | Not a partial model pass; [closeout](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md) |
| Q1 transfer | Long character counting | **NEGATIVE_BOUNDARY** | Fixed Qwen controller did not transfer | Same controller, no adaptation, N=200; [Gate 10 closeout](GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md) |
| Q1 second task | LiveCodeBench Stage B | **OPEN_RUNNING** | No partial scientific result | Frozen DEVELOPMENT transfer test; branch `research/q1-second-task-spark2-design`, authorization commit `91c3db4` |
| Q2 | V4.1 relational geometry | **DEVELOPMENT — G2** | A0, A1, and A2 predict blind-spot-shape geometry; A2 does not outperform A0/A1 | 31 directions, N=300, two shells; [closeout](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.md) |
| Q2 radial | V4.1 radial tests | **DEVELOPMENT — RS+ / RT+** | STRONG exceeds MEDIUM shape and total displacement in all 31 directions | [radial artifact](../review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json) |
| Q3 | Collective utility | **NOT_RUN** | No result | [concept note](Q3_CONCEPT_NOTE.md) only |

## Q1: confirmatory controllability

The exact terminal classification is
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`.

| Quantity | Qwen3-8B | Ministral-3-8B |
|---|---:|---:|
| Competence-adjusted complementarity, C | 0.0543546 | 0.0729950 |
| 95% item-bootstrap interval | [0.0144110, 0.0968045] | [0.0217732, 0.1228090] |
| C minus matched-null mean | 0.0387101 | 0.0654174 |
| Contrast 95% interval | [0.0060111, 0.0745614] | [0.0257279, 0.1049107] |
| Commitment validity | Passed frozen guards | 0.885965 — **failed** |
| Semantic evaluability | Passed frozen guards | 0.885965 — **failed** |
| Model-level result | **PASS** | **FAIL** |

The Qwen result supports a causal change in *which* held-out CRUXEval items
fail, above what is explained by mean competence change and matched random
directions. It does not establish task-, model-, or domain-generality.

## Q1: negative and open boundaries

The fixed Qwen controller failed to produce positive/null-specific transfer on
200 fresh long character-count items (`GATE10_NO_CROSS_DOMAIN_TRANSFER`). This
is a genuine negative boundary for that fixed controller and task, not proof
that no task-adapted controller could work.

LiveCodeBench Stage A1 failed its frozen baseline answer-channel guards. A
generic parser repair was then prospectively qualified on 20 untouched Stage A2
families. Stage B is now open under a separate lock. A1 and A2 are not pooled,
and no Stage-B result is reported before complete sealing and analysis.

## Q2: relational geometry

Q2 V4.1 completed 37,800/37,800 frozen trajectories with zero missing,
unexpected, duplicate, replacement, or retried logical rows. Raw data were
sealed before scoring; the independent audit reproduced the classification
with maximum difference 0.0.

| Geometry | MEDIUM rho | STRONG rho | Aggregate rho | maxT QAP p | Qualified? |
|---|---:|---:|---:|---:|---|
| A0 flat | 0.554822 | 0.572814 | 0.563818 | 0.00002 | Yes |
| A1 whitened | 0.554681 | 0.557941 | 0.556311 | 0.00002 | Yes |
| A2 finite response | 0.442569 | 0.439688 | 0.441128 | 0.00002 | Yes |

The frozen G3 superiority contrasts were negative: A2−A0 = −0.122690 and
A2−A1 = −0.115183, with corrected permutation p=1.0 for both. Therefore A2
contains real relational signal, but the simpler static geometries were more
predictive in this experiment. The mechanical class is `Q2_V4_1_G2`.

## Q2: radial geometry

| Result | Median STRONG−MEDIUM displacement | Positive directions | Permutation p | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| RS+ — blind-spot shape | 0.0440580 | 31/31 | 0.00002 | [0.0300111, 0.0511371] |
| RT+ — total displacement | 0.0433333 | 31/31 | 0.00014 | [0.0300000, 0.0533333] |

These results establish amplitude ordering within the tested shells and fixed
subspace. They do not establish global linearity or a manifold theorem.

## Forensic and artifact identity

- Q1 forensic classification: `Q1_CONFIRMATORY_FORENSIC_CLEAN`.
- Q2 forensic classification: `Q2_V4_1_SEMANTIC_FORENSIC_CLEAN`.
- Q2 raw journal SHA-256:
  `d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99`.
- Q2 semantic-score SHA-256:
  `a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f`.
- Complete Q2 tracked/private ledger:
  [`ARTIFACT_HASHES.json`](../review/q2_v4_1_semantic_execution/ARTIFACT_HASHES.json).

