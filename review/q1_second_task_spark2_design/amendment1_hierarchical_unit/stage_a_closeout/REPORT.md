# Q1 Second-Task Stage-A Closeout

## Pre-opening integrity

- Approved design HEAD: `e20c68f1cbf9276425da63d8b2128418ae9e755b`
- Execution code commit: `d0f3d1da0988fcf905d974377eb6b5180876dd17`
- Authorization SHA-256: `d89dd95dfa433d67ddac31fb855447e54a969583524f93cc7d291b7648c80213`
- Amendment SHA-256: `5cf14cb5c41f6a6547c1c5d2385511faee26a7549576caa75c9c84b9bee02594`
- Stage-A manifest SHA-256: `78e6cf9b76c8c3d3b78847af2ce39449be0fe472567c7fe047a47938ed500036`
- Stage-A schedule SHA-256: `fd1267c506827672cfe6c7e5374003821c78f512a3ed13fc745f655b5b13880a`
- Spark-2 environment fingerprint: `306d65af9643cc1144d344ae57141ac96ffbbcf70520f67e9276a907d29660bc`
- Scientific outcomes before opening: 0

The pre-opening audit verified exactly 32 question families, one selected row per
family, 128 unique logical keys and seeds, disjoint Stage-A/Stage-B/reserve
families, an empty journal, and no active hook or loaded meaningful/null vector.

## Execution

All 128 scheduled trajectories completed with zero retries, duplicates, missing
keys, unexpected keys, or runtime errors. Runtime was 4,593.39 seconds. The
append-only raw journal is 619,809 bytes with SHA-256
`5b0fec6960ac414f56995d91a43c3b41c49a06b5fb868156a8e24d037b9281b1`.
Raw data were sealed before scoring.

## Baseline opportunity

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Commitment validity | 0.828125 | >= 0.95 | FAIL |
| Semantic evaluability | 0.812500 | >= 0.95 | FAIL |
| Accuracy | 0.609375 | [0.25, 0.90] | PASS |
| Repeated-baseline B00 | 0.250000 | >= 0.05 | PASS |
| Families wrong twice | 8 | >= 4 | PASS |
| Families correct at least once | 24 | >= 7 | PASS |

The benchmark had sufficient error opportunity, but the baseline commitment
channel did not meet the prospectively frozen validity and evaluability guards.

## TEXTUAL_CAREFUL qualification

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Commitment validity | 0.968750 | >= 0.95 | PASS |
| Semantic evaluability | 0.953125 | >= 0.95 | PASS |
| Accuracy | 0.781250 | >= baseline - 0.03 | PASS |
| Accuracy gain | +0.171875 | >= 0.03 | PASS |
| Mean-token ratio | 1.249332 | >= 1.5 | FAIL |
| Median-token gain | +106.0 | >= 10 | PASS |

The predeclared manifestation booleans are: accuracy `true`, mean-token ratio
`false`, and median-token gain `true`. The descriptive source-policy result is
`TEXTUAL_CAREFUL_ACCURACY_BENEFIT_PRESENT`.

## Primary classification

`Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED`

The classification is caused specifically by baseline commitment-validity and
semantic-evaluability failures. The positive textual-careful result does not
override those mandatory substrate gates.

## Independent forensic audit

The independent parser/metric path reproduced every reported quantity and the
terminal classification exactly. Maximum primary/audit metric difference was
0.0. Forensic classification:
`Q1_SECOND_TASK_STAGE_A_FORENSIC_CLEAN`.

## Stage-B and claim boundary

Stage B is `NOT_AUTHORIZED_NOT_OPENED`. Meaningful-controller and activation-null
LiveCodeBench trajectories both remain zero. Stage A therefore establishes no
activation-controller transfer, complementarity transfer, domain-general
steering, geometry, Q2 result, or collective utility result.

Spark 1 was not used, Q2 outputs were not inspected, and the Q2 process was not
modified or interrupted.

`Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED`
