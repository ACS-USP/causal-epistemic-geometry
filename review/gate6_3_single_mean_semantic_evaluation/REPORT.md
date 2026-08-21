# GATE 6.3 — SINGLE-MEAN SEMANTIC EVALUATION

## Status

`GATE6_3_SINGLE_MEAN_DESTRUCTIVE`

The frozen Gate 6.3 protocol completed its authorized 80-row matched-random
supplement and conditional 840-row evaluation. Gate 6.2 remains immutable.
This is a development result, not a confirmatory result and not a Q2 result.

## Execution integrity

- 920 total journal rows: 80 matched-random + 840 primary evaluation.
- 920 unique logical keys; no duplicates or missing scheduled keys.
- 20 matched items and 60 evaluation items; zero overlap.
- Matched random seeds were exactly coupled within item.
- Primary evaluation used two independent rollouts per item-condition.
- Parser reanalysis of every raw row produced zero status/answer/correctness mismatches.
- All rows used `external-semantic-v2`, the frozen Qwen revision, and eta0
  `12.849903937136261`.
- Independent raw-row recomputation agrees with the primary estimand file.
- Item-cluster bootstrap contains 5,000 resamples per reported interval.

The effective remote checkout commits are recorded separately from the
prospective lock. The matched commit `1a343e7…` contains only the matched
journal; evaluation commit `3a2dc3e…` adds no scientific-code changes beyond
that journal. The protocol lock remains `958b5a2…`.

## Matched-random gate

The frozen matched gate passed and authorized the evaluation:

| Controller | Validity | Semantic change | Mean random comparison |
|---|---:|---:|---:|
| BEST_SINGLE_MEAN_PLUS | 0.95 (historical reanalysis) | 0.70 | > random mean 0.30 and random max 0.40 |
| SINGLE_L27_RANDOM_R0 | 0.95 | 0.40 | — |
| SINGLE_L27_RANDOM_R1 | 1.00 | 0.30 | — |
| SINGLE_L27_RANDOM_R2 | 0.95 | 0.25 | — |
| SINGLE_L27_RANDOM_R3 | 1.00 | 0.25 | — |

The matched gate is a manipulation authorization gate, not the primary causal
estimand.

## Primary evaluation

| Condition | Valid / 120 | Correct | Wrong | Invalid | Truncated | Accuracy | Validity | Mean tokens | Median tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | 117 | 62 | 55 | 0 | 3 | 0.5167 | 0.9750 | 114.0 | 8.5 |
| TEXTUAL_CAREFUL_REFERENCE | 114 | 87 | 27 | 5 | 1 | 0.7250 | 0.9500 | 335.3 | 251.5 |
| BEST_SINGLE_MEAN_PLUS | 109 | 81 | 28 | 10 | 1 | 0.6750 | 0.9083 | 351.4 | 283.5 |
| RANDOM_R0 | 116 | 58 | 58 | 3 | 1 | 0.4833 | 0.9667 | 46.4 | 8.0 |
| RANDOM_R1 | 113 | 61 | 52 | 6 | 1 | 0.5083 | 0.9417 | 47.0 | 8.5 |
| RANDOM_R2 | 115 | 58 | 57 | 2 | 3 | 0.4833 | 0.9583 | 114.4 | 8.5 |
| RANDOM_R3 | 115 | 58 | 57 | 2 | 3 | 0.4833 | 0.9583 | 113.7 | 7.5 |

## Estimands

Repeated baseline: `B00 = 0.4667`, `O00 = 0.5333`.

| Condition | G | C | D | Rescue | Damage |
|---|---:|---:|---:|---:|---:|
| BEST_SINGLE_MEAN_PLUS | 0.1875 | 0.1130 | 0.1667 | 0.2042 | 0.0458 |
| TEXTUAL_CAREFUL_REFERENCE | 0.2250 | 0.1266 | 0.2000 | 0.2417 | 0.0333 |
| RANDOM_R0 | 0.0083 | 0.0250 | 0.0500 | 0.0250 | 0.0583 |
| RANDOM_R1 | 0.0125 | 0.0169 | 0.0167 | 0.0292 | 0.0375 |
| RANDOM_R2 | -0.0167 | -0.0004 | 0.0167 | 0.0000 | 0.0333 |
| RANDOM_R3 | -0.0125 | 0.0038 | 0.0167 | 0.0042 | 0.0375 |

The meaningful controller exceeds the random mean and maximum on the raw G/C/D
point estimates, and it preserves the competence tolerance. However, it fails
the frozen validity guard: `0.9083 < 0.9750 - 0.05 = 0.9250`. Under the
pre-registered classification rule this makes the controller mechanically
destructive/uninterpretable for useful complementarity. The result is therefore
not promoted to a movement or useful-complementarity signal.

## Interpretation boundary

This result shows that the matched single-layer controller can strongly alter
the observed output process, but under the frozen semantic-validity guard its
primary evaluation is not competence-preserving/evaluable as a useful error
profile intervention. It does not establish or refute the project-level
geometry hypothesis. No new alpha, layer, direction, benchmark, Q2 analysis,
character-count replication, or holdout access is authorized by this closeout.

## Infrastructure and cost

The A40 Pod was stopped after recovery. The restarted container temporarily
lost Python packages; the exact compatible runtime packages were restored and
the remote diff confirmed no scientific-code change. The journal SHA-256 is:

`593c89e8bf13d83d2fcfa27b2a9d7eec7d4c0b17918185f0554d37390ca601e1`

Approximate billed GPU window: 1.93 A40-hours; estimated cost: US$0.85 at
US$0.44/hour. This is within the US$1.50 hard stop. RunPod final state:
`EXITED`.

## Firewall

- Gate 6.2 historical artifacts: preserved and immutable.
- Q2: not run.
- Character count: not run.
- Confirmatory holdout: untouched.
- Next action: principal researcher review.
