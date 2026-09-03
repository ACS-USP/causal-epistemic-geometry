# Q1 Second-Task Stage-B Closeout

## 1. Pre-opening authorization

Stage B opened from commit
`91c3db4ba41f8ad60f89920b605cbd09fba6dff9`. The prospective principal
authorization and parser/execution addendum remained byte-identical at:

- authorization: `bd333afac9bf7e8d366599268d06eac85e94a6c25bb539217041a5c2a712cd5b`;
- parser/execution addendum: `1caf5c0f5864cf7d0f82a9c4952f3bf47728ce49a57fc9c72fd62db88220ca11`;
- family manifest: `56ed32a378bd0a1ee4cfb81c6e3bb4755098014fef6bba5bfbc9730080900605`;
- schedule: `6c1b6437d4df8116ebfbd26ee7fcaefd7abc3066d57bec2b6a4755ffa407fd03`.

The meaningful controller was the exact fixed Qwen L27-D75 controller with
canonical vector hash
`e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838`.
The eight null-vector hashes remained exactly those sealed in
`STAGE_B_PRINCIPAL_AUTHORIZATION.json`. The qualified Spark-2 environment
fingerprint was
`306d65af9643cc1144d344ae57141ac96ffbbcf70520f67e9276a907d29660bc`.
Stage-B outcomes before opening were zero.

## 2. Execution integrity

- Scheduled/completed: **5,720 / 5,720**
- Unique logical keys: **5,720**
- Duplicates: **0**
- Missing/unexpected: **0 / 0**
- Replacements: **0**
- Runtime errors: **0**
- Collection interval: **211,457.15 s** (58h 44m 17s)
- Aggregate generated tokens: **2,442,096**
- Token mean/median: **426.940 / 357**
- Token p90/p95/max: **646 / 835 / 4,096**
- Frozen-cap trajectories retained: **61 / 5,720**
- Raw journal SHA-256:
  `14ee3691de42c609dd4be1749ab4ea7d8b1c25b3cfbd47483423a83741fdb364`

The raw journal was sealed before parsing. Raw benchmark content and model
outputs remain in the private scientific-data namespace and are represented in
Git by hashes only.

## 3. Instrument and safety

| Gate | Baseline | Meaningful | Required | Result |
|---|---:|---:|---:|---|
| Commitment validity | 0.97308 | 0.85769 | meaningful >= 0.92308 | FAIL |
| Semantic evaluability | 0.97308 | 0.83654 | meaningful >= 0.92308 | FAIL |
| Accuracy | 0.66923 | 0.59615 | meaningful >= 0.56923 | PASS |

The safety conjunction failed because the meaningful controller reduced answer
channel validity and evaluability beyond the frozen five-point margins. Here,
"safety" is the frozen instrument guard, not an alignment or real-world safety
claim.

## 4. Primary complementarity

| Quantity | Estimate | Frozen 95% family-bootstrap interval |
|---|---:|---:|
| Meaningful C | 0.0068798 | [-0.0106404, 0.0244633] |
| Mean null C | -0.0049214 | — |
| Meaningful C minus mean null C | 0.0118012 | [-0.0005908, 0.0244391] |

Individual null C values:

| Null | C |
|---|---:|
| R0 | -0.0105695 |
| R1 | -0.0037567 |
| R2 | 0.0046512 |
| R3 | -0.0008348 |
| R4 | -0.0028473 |
| R5 | -0.0024001 |
| R6 | -0.0080426 |
| R7 | -0.0155710 |

The meaningful point estimate exceeded every individual null point estimate,
but both frozen lower-bound requirements failed.

## 5. Split-half consistency

| Half | Meaningful C | Mean null C | Difference | Result |
|---|---:|---:|---:|---|
| A, rollouts {0,1} | -0.0039952 | -0.0113521 | 0.0073569 | FAIL: meaningful C was not positive |
| B, rollouts {2,3} | 0.0267144 | 0.0230024 | 0.0037120 | PASS |

The prospectively required two-half conjunction therefore failed.

## 6. Secondary estimands

For the meaningful controller, the frozen descriptive estimates were:

- accuracy difference: **-0.07308**;
- G: **-0.01763**;
- D: **0.03462**;
- rescue: **0.12981**;
- damage: **0.20288**.

TEXTUAL_CAREFUL had an accuracy difference of **+0.05962**, a mean-token ratio
of **1.4872x**, and a median-token increase of **65** tokens. Its descriptive
label is `TEXTUAL_CAREFUL_ACCURACY_BENEFIT_PRESENT`. This is secondary Stage-B
description and is not activation-controller transfer evidence.

## 7. Primary terminal classification

| Frozen gate | Observed | Required | Result |
|---|---:|---:|---|
| P1: C lower bound > 0 | -0.0106404 | > 0 | FAIL |
| P2a: C-minus-null-mean lower bound > 0 | -0.0005908 | > 0 | FAIL |
| P2b: meaningful C > every null C | true | true | PASS |
| Split-half A | false | true | FAIL |
| Split-half B | true | true | PASS |
| Scientific conjunction | false | true | FAIL |
| Instrument safety conjunction | false | true | FAIL |

The mechanical primary terminal classification is:

`Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY`

At N=130/R=4, the fixed controller did not satisfy the prospectively frozen
LiveCodeBench transfer criterion. This does not establish exactly zero transfer
or exclude smaller effects.

## 8. Independent forensic audit

The independent audit reproduced the same terminal classification and the same
P1, P2b, split-half, and safety decisions. It did **not** qualify as clean:

- parser-decision disagreements: **30 / 5,720 rows**;
- maximum primary/audit metric difference: **0.0192308**;
- primary P2a: **FAIL**, lower bound `-0.0005908`;
- audit P2a: **PASS**, lower bound `+0.0016538`;
- final terminal classification agreement: **YES**.

The primary path used the exact parser identity sealed in the prospective
Stage-B addendum. The independent low-level parser implementation was not
decision-equivalent on the realized corpus. Per protocol, this discrepancy was
surfaced and preserved; no parser, metric, threshold, output, or classification
was changed after opening.

`Q1_SECOND_TASK_STAGE_B_FORENSIC_DISAGREEMENT`

This closeout is therefore awaiting principal review of the forensic
disagreement even though both implementations return the same frozen terminal
class.

## 9. Historical-context preservation

Stage A1 remains:

`Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED`

Stage A2 remains:

`Q1_SECOND_TASK_STAGE_A2_QUALIFIED`

A1 and A2 were not pooled, and neither historical result was rewritten.

## 10. Scientific interpretation

The fixed Qwen L27-D75 controller did not meet the complete prospective
LiveCodeBench null-specific complementarity criterion. The meaningful C point
estimate was positive and exceeded every null point estimate, but its interval
included zero, the meaningful-minus-null-mean interval narrowly included zero,
and the predesignated first rollout half had negative meaningful C. In addition,
the meaningful condition failed the frozen commitment-validity and semantic-
evaluability guards.

This bounds the earlier CRUXEval result: it does not support claiming that the
fixed controller transfers safely and null-specifically to a second objective
program-execution benchmark under this design. It does not overturn the frozen
Qwen CRUXEval confirmatory result, prove that transfer is zero, establish a
mechanism, or say anything about model/domain generality.

## 11. Q2 firewall

- Spark 1 used: **NO**
- Q2 outputs inspected: **NO**
- Q2 process modified/interrupted: **NO**

## 12. Repository/resource state

Collection and analysis used Spark 2 only, with one GB10. The model runner had
terminated before analysis and Spark 2 was idle afterward. The research branch
is not merged to main. Aggregate closeout artifacts and hashes are tracked;
raw benchmark-derived outputs remain private.

`Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY_AWAITING_PRINCIPAL_REVIEW`
