# Q1 Second-Task Stage-A2 Closeout

## 1. Authorization and pre-open seal

The reviewed Amendment-2 draft remained byte-identical at
`c8b006e039d2f826f45ad58f9356d00056f73c83d92686557e6b55d6e64f9d34`.
The principal authorization hash is
`d9d39f90ddf5cb8d9805fe47989433d133a1a083dafb51d36eb2ee9a0997d9ec`.
The approved family-manifest and schedule hashes remained respectively
`5dc6acea72968c3d8bce8b0ab7f567fd2692801602011b8f9d9a72437443d306`
and `f5cb5e300f922872959bc5d2c36e4314b4494a5d8ccd91e4323eabebed21ed9e`.

The repaired parser source is
`src/epistemic_geometry/experiments/q1_second_task_stage_a_failure.py`, frozen
at implementation commit `31f7f31c90ddbfc0fdead7fb6eb5ace9750f5638`, with source SHA-256
`0557447697a1f75caab0f9680863982fe2636393335e3fffd94223e6d92e50d9`.
The evaluator-function hash is
`9c1e1d1572e9ee51033af3810d7c75d526441cda068183e8ef4c4a59fdc42169`;
the unchanged prompt-builder hash is
`1ce7e7d4a53844de47f51d2d2477263515bd5d19f6a516e604f9468fe934da2c`.
The Spark-2 fingerprint was exactly
`306d65af9643cc1144d344ae57141ac96ffbbcf70520f67e9276a907d29660bc`.
Fresh Stage-A2 outcomes before opening were zero.

The first preflight attempt exposed a schema-only runner defect: the approved
two-field selected-item identity was compared with a richer public record. It
was repaired and documented before opening, without changing any reviewed
manifest, schedule, parser, prompt, seed, threshold, or scientific semantics.

## 2. Execution

- Scheduled/completed: **80 / 80**
- Retries: **0**
- Duplicates: **0**
- Missing/unexpected: **0 / 0**
- Runtime errors: **0**
- Collection wall time excluding checkpoint load: **3,775.20 s** (62m 55s)
- Generated tokens: mean **539.625**, median **429**, p90 **837**, p95 **961**, max **4,096**
- Frozen-cap outputs retained: **2 / 80**
- Raw journal SHA-256: `be7b711bbca2b205eca2d2cee9985ff892f734d9a22943bc38e5904e49f156ba`

Raw data were sealed before parsing. The sealed journal remains in the private
Spark-2 scientific-data namespace; benchmark text and raw outputs are not
committed to the public repository.

## 3. Baseline qualification

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Commitment validity | 0.950 | >= 0.950 (38/40) | PASS |
| Semantic evaluability | 0.950 | >= 0.950 (38/40) | PASS |
| Accuracy | 0.650 | [0.25, 0.90] | PASS |
| Repeated-baseline B00 | 0.250 | >= 0.050 | PASS |
| Families wrong in both rollouts | 5 | >= 2 | PASS |
| Families correct at least once | 15 | >= 4 | PASS |

Baseline generated-token mean was 448.975 and median was 355.5.

## 4. TEXTUAL_CAREFUL qualification

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Commitment validity | 0.975 | >= 0.950 | PASS |
| Semantic evaluability | 0.975 | >= 0.950 | PASS |
| Non-harm accuracy | 0.675 | >= 0.620 | PASS |
| Accuracy gain manifestation | +0.025 | >= +0.030 | FAIL |
| Mean-token manifestation | 1.4038x | >= 1.5x | FAIL |
| Median-token manifestation | +155.5 | >= +10 | PASS |
| At least one manifestation | true | true | PASS |

The exact descriptive label is:

`TEXTUAL_CAREFUL_NONHARMFUL_COMPUTE_MANIFESTATION`

The token-length manifestation is not an accuracy, semantic, reasoning, or
task-utility improvement claim.

## 5. Primary Stage-A2 classification

Every frozen gate passed mechanically:

`Q1_SECOND_TASK_STAGE_A2_QUALIFIED`

## 6. Independent forensic audit

The independent path reproduced schedule coverage, identities, seeds, parser
decisions, all point metrics, gate booleans, manifestations, and the terminal
classification. Maximum metric discrepancy was **0.0**.

`Q1_SECOND_TASK_STAGE_A2_FORENSIC_CLEAN`

## 7. Stage-A1 preservation

The historical result remains unchanged:

`Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED`

Stage-A1 and Stage-A2 were not pooled. The A1 diagnostic remains post-hoc
instrument-development evidence only.

## 8. Stage-B status

`NOT_AUTHORIZED_NOT_OPENED`

- Meaningful-controller LiveCodeBench trajectories: **0**
- Activation-null LiveCodeBench trajectories: **0**

## 9. Scientific boundary

Stage A2 establishes only that the repaired LiveCodeBench test-output
instrument prospectively qualified on 20 fresh question families for a
possible activation-controller transfer test. It does not establish Q1
second-task replication, activation-controller transfer, complementarity
transfer, domain/model generality, Q2 geometry, or collective utility.

## 10. Q2 firewall

- Spark 1 used: **NO**
- Q2 outputs inspected: **NO**
- Q2 process modified/interrupted: **NO**

## 11. Repository/resource state

The execution source was
`c882219e31db45998c19270de3d0f168a88ec011` on
`research/q1-second-task-spark2-design`. Spark 2 alone ran one GB10 and the
model process terminated after collection. Stage B remains closed pending a
new explicit principal review.

`Q1_SECOND_TASK_STAGE_A2_QUALIFIED_AWAITING_PRINCIPAL_REVIEW`
