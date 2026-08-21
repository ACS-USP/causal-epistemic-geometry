# Gate 9 — Fresh D75 Selected-Dose Evaluation

Primary classification: `GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION`.

Source-policy classification: `CAREFUL_SOURCE_REPLICATED`.

Experiment source commit: `0eb5fd466ed78d2dcbc07a3f506e02c098cf2945`.

This is a 100-item independent DEVELOPMENT evaluation of the exact frozen L27
paired-mean plus controller at eta `9.637427952852196`. Seven conditions and two
independent rollouts produce 1,400 scientific trajectories. No controller,
layer, dose, item, or parser selection occurred after outcomes.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
| `BASELINE` | 0.9850 | 0.9850 | 0.4700 | 51.95 / 8.0 / 4096 |
| `TEXTUAL_CAREFUL_REFERENCE` | 0.9850 | 0.9850 | 0.8000 | 267.74 / 215.0 / 4096 |
| `MEANINGFUL_L27_D75` | 0.9700 | 0.9700 | 0.6000 | 154.49 / 12.0 / 4096 |
| `RANDOM_L27_D75_R0` | 0.9900 | 0.9900 | 0.4750 | 11.03 / 7.0 / 75 |
| `RANDOM_L27_D75_R1` | 0.9800 | 0.9800 | 0.4600 | 52.00 / 7.5 / 4096 |
| `RANDOM_L27_D75_R2` | 0.9800 | 0.9800 | 0.5000 | 51.74 / 8.0 / 4096 |
| `RANDOM_L27_D75_R3` | 0.9800 | 0.9800 | 0.4850 | 92.92 / 8.0 / 4096 |


## Meaningful D75 estimands

- Accuracy difference: 0.130000
- G: 0.132500
- C: 0.064343
- D: 0.120000
- Rescue: 0.152500
- Damage: 0.022500
- G minus random mean/max: 0.125000 / 0.115000
- C minus random mean/max: 0.062020 / 0.054596
- D minus random mean/max: 0.110000 / 0.100000

## Frozen guards and source anchor

- Commitment-validity guard: True
- Semantic-evaluability guard: True
- Competence guard: True
- CAREFUL source: `CAREFUL_SOURCE_REPLICATED`
- CAREFUL accuracy-gain fraction recovered: 0.39393939393939387
- CAREFUL token-increase fraction recovered: 0.4751830568171285

## Interpretation boundary

The classification above was applied mechanically after all rows were collected.
This is DEVELOPMENT evidence only. It is not Q2, character-count replication,
confirmatory holdout evidence, or execution of Gate 10.
