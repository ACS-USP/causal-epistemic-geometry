# Q1 confirmatory fixed-controller test

## Frozen result

`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`

The exact 57-ID holdout was materialized only after the frozen complete-design
cost gate passed. Both model journals were complete before any scientific
metric was inspected. Each journal contains 798 unique logical rows, with no
duplicates, seed collisions, or scientific retries. The independent audit is
`Q1_CONFIRMATORY_FORENSIC_CLEAN` and reproduced every point metric exactly.

## Qwen

- baseline accuracy: 0.438596
- meaningful accuracy: 0.500000
- meaningful commitment validity / semantic evaluability: 0.973684 / 0.973684
- C: 0.054355; 95% item-bootstrap interval 0.014411 to 0.096805
- C minus random-null mean: 0.038710; bootstrap interval 0.006011 to 0.074561
- maximum random C: 0.024123
- G / D: 0.087719 / 0.087719
- rescue / damage: 0.105263 / 0.043860
- frozen model decision: PASS

Qwen passed every safety, positive-C, and random-null criterion.

## Ministral

- baseline accuracy: 0.570175
- meaningful accuracy: 0.649123
- meaningful commitment validity / semantic evaluability: 0.885965 / 0.885965
- C: 0.072995; 95% item-bootstrap interval 0.021773 to 0.122809
- C minus random-null mean: 0.065417; bootstrap interval 0.025728 to 0.104911
- maximum random C: 0.015742
- G / D: 0.105263 / 0.105263
- rescue / damage: 0.149123 / 0.070175
- frozen model decision: FAIL

Ministral passed the competence and all positive-complementarity/random-null
checks, but failed both frozen commitment-validity and semantic-evaluability
guards. Those safety failures determine the model and cross-model decisions.

## Interpretation boundary

This is confirmatory support for the fixed Qwen controller. It is not a
cross-model confirmatory pass and must not be relabeled as a partial cross-model
success. Gate 9 and Gate 13.1 remain DEVELOPMENT evidence; Gate 10's negative
cross-domain result and Gate 12/12.1's geometry-engine nonqualification remain
unchanged. Q2 and Q3 were not run.

RunPod was terminated after verified artifact recovery. Active Pods and
retained network volumes are both zero.
