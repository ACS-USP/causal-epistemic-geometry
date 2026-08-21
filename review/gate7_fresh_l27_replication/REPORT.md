# Gate 7 — Fresh Single-L27 Replication

Primary classification: `GATE7_DESTRUCTIVE`.

Source-policy classification: `CAREFUL_SOURCE_REPLICATED`.

Experiment source commit: `0dc9c3156bc86aebada93388a4e2fa28b2345f95`.

Fresh sample: 120 CRUXEval items; 1680 trajectories; seven
conditions; two independent rollouts per item-condition. Semantic evaluator:
`external-semantic-v3`.

The historical Gate 6.3 classification remains
`GATE6_3_SINGLE_MEAN_DESTRUCTIVE` and was not modified.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
| `BASELINE` | 0.9917 | 0.9917 | 0.3917 | 12.09 / 9.0 / 88 |
| `TEXTUAL_CAREFUL_REFERENCE` | 0.9917 | 0.9917 | 0.6792 | 313.43 / 241.0 / 4096 |
| `BEST_SINGLE_MEAN_PLUS` | 0.9000 | 0.9000 | 0.5375 | 413.04 / 313.0 / 4096 |
| `GATE7_RANDOM_R0` | 0.9917 | 0.9917 | 0.4417 | 46.13 / 9.0 / 4096 |
| `GATE7_RANDOM_R1` | 0.9917 | 0.9917 | 0.4125 | 45.58 / 8.5 / 4096 |
| `GATE7_RANDOM_R2` | 0.9917 | 0.9917 | 0.3750 | 12.15 / 9.0 / 87 |
| `GATE7_RANDOM_R3` | 0.9958 | 0.9958 | 0.4292 | 29.71 / 9.0 / 4096 |

## Meaningful controller

- Accuracy difference: 0.145833
- G: 0.237500
- C: 0.150070
- D: 0.216667
- Rescue: 0.245833
- Damage: 0.100000
- G minus random mean/max:
  0.205729 /
  0.185417
- C minus random mean/max:
  0.132055 /
  0.128186
- D minus random mean/max:
  0.172917 /
  0.158333

## Frozen guards

- Commitment-validity guard: False
- Semantic-evaluability guard: False
- Competence guard: True

The controller improved primary accuracy by 0.1458 and
showed large positive G/C/D beyond every new random controller, but commitment
validity and semantic evaluability were 0.9000 versus a baseline of 0.9917.
The frozen relative guard required at least 0.9417, so the exhaustive
classification is mechanically `GATE7_DESTRUCTIVE`.

## Random-controller null

| random condition | accuracy | G | C | D |
|---|---:|---:|---:|---:|
| `GATE7_RANDOM_R0` | 0.4417 | 0.052083 | 0.021884 | 0.058333 |
| `GATE7_RANDOM_R1` | 0.4125 | 0.033333 | 0.020868 | 0.050000 |
| `GATE7_RANDOM_R2` | 0.3750 | 0.000000 | 0.010259 | 0.016667 |
| `GATE7_RANDOM_R3` | 0.4292 | 0.041667 | 0.019048 | 0.050000 |


## Textual CAREFUL source

- Classification: `CAREFUL_SOURCE_REPLICATED`
- Token-gain fraction recovered by activation controller:
  1.330554
- Accuracy-gain fraction recovered by activation controller:
  0.507246
- Meaningful/textual semantic agreement:
  0.583333

## Item-cluster bootstrap (10,000 resamples)

| estimand | 2.5% | 97.5% |
|---|---:|---:|
| `meaningful:C` | 0.112710 | 0.186310 |
| `meaningful:C_minus_random_max` | 0.084873 | 0.160400 |
| `meaningful:C_minus_random_mean` | 0.094642 | 0.168252 |
| `meaningful:D` | 0.141667 | 0.291667 |
| `meaningful:D_minus_random_max` | 0.075000 | 0.233333 |
| `meaningful:D_minus_random_mean` | 0.097917 | 0.250000 |
| `meaningful:G` | 0.170833 | 0.308333 |
| `meaningful:G_minus_random_max` | 0.110417 | 0.250000 |
| `meaningful:G_minus_random_mean` | 0.142708 | 0.272917 |
| `meaningful:accuracy_change` | 0.054167 | 0.237500 |
| `meaningful:commitment_validity_change` | -0.133333 | -0.054167 |
| `meaningful:damage` | 0.058333 | 0.150000 |
| `meaningful:rescue` | 0.179167 | 0.316667 |
| `meaningful:semantic_evaluability_change` | -0.133333 | -0.054167 |


## Interpretation boundary

This is an independent DEVELOPMENT replication under a parser frozen before
collection. It is not confirmatory, Q2, character-count replication, or a
general claim beyond Qwen3-8B × CRUXEval. The controller produced a strong,
specific semantic-error-profile and accuracy signal, but it also induced a
condition-specific commitment/evaluability loss that violated the frozen
non-destructiveness guard. The exhaustive classification was applied
mechanically after all rows were collected.
