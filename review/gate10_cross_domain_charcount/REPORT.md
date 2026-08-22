# GATE 10 — CROSS-DOMAIN CHARACTER COUNTING

Primary classification: `GATE10_NO_CROSS_DOMAIN_TRANSFER`.

Opportunity classification: `CHARCOUNT_OPPORTUNITY_PASS`.

Source-policy classification: `CHARCOUNT_CAREFUL_SOURCE_NOT_REPLICATED`.

This frozen DEVELOPMENT evaluation used 200 fresh `FRESH_PSEUDOWORD_LONG`
items, seven conditions, and two independent rollouts: 2,800 trajectories.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
| `BASELINE` | 0.9975 | 0.9975 | 0.8625 | 363.05 / 356.5 / 799 |
| `TEXTUAL_CAREFUL_CHARCOUNT_REFERENCE` | 1.0000 | 1.0000 | 0.8050 | 550.32 / 490.5 / 1592 |
| `MEANINGFUL_L27_D75` | 0.9975 | 0.9975 | 0.8350 | 399.26 / 390.5 / 813 |
| `RANDOM_L27_D75_R0` | 1.0000 | 1.0000 | 0.8325 | 325.25 / 278.0 / 739 |
| `RANDOM_L27_D75_R1` | 0.9975 | 0.9975 | 0.8175 | 343.27 / 270.5 / 4096 |
| `RANDOM_L27_D75_R2` | 1.0000 | 1.0000 | 0.8475 | 356.11 / 353.5 / 932 |
| `RANDOM_L27_D75_R3` | 1.0000 | 1.0000 | 0.8075 | 333.87 / 290.5 / 1650 |


## Baseline opportunity

- B00 / O00: 0.045000 / 0.955000
- Double-wrong items: 9
- Correct in at least one rollout: 191

## Meaningful fixed L27-D75 controller

- Accuracy difference: -0.027500
- G / C / D: -0.016250 / -0.012299 / -0.025000
- G_norm: -0.361111
- Rescue / damage: 0.076250 / 0.103750
- G minus random mean/max: -0.005312 / -0.010000
- C minus random mean/max: -0.006548 / -0.012469
- D minus random mean/max: -0.023750 / -0.055000

Safety guards: commitment=True,
evaluability=True, and
competence=True.

## Random-controller null

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| G | -0.010938 | -0.008750 | -0.020000 | -0.006250 |
| C | -0.005751 | -0.005452 | -0.012268 | 0.000170 |
| D | -0.001250 | -0.002500 | -0.030000 | 0.030000 |


## Textual CAREFUL reference

- Classification: `CHARCOUNT_CAREFUL_SOURCE_NOT_REPLICATED`
- Accuracy-gain fraction recovered: N/A
- Token-increase fraction recovered: 0.1933382284226686

## Item-cluster bootstrap (10,000 resamples)

| estimand | 2.5% | 97.5% |
|---|---:|---:|
| `meaningful:accuracy_change` | -0.067500 | 0.012500 |
| `meaningful:commitment_validity_change` | -0.007500 | 0.007500 |
| `meaningful:semantic_evaluability_change` | -0.007500 | 0.007500 |
| `meaningful:G` | -0.038781 | 0.010000 |
| `meaningful:C` | -0.033216 | 0.010786 |
| `meaningful:D` | -0.060000 | 0.005000 |
| `meaningful:G_norm` | -1.700000 | 0.166667 |
| `meaningful:G_minus_random_mean` | -0.020633 | 0.010312 |
| `meaningful:C_minus_random_mean` | -0.018854 | 0.006038 |
| `meaningful:D_minus_random_mean` | -0.057500 | 0.007500 |
| `meaningful:rescue` | 0.051250 | 0.103750 |
| `meaningful:damage` | 0.077500 | 0.131250 |


## Cost and firewall

- Scientific trajectories: 2800
- Total A40 runtime: 11.798889 hours
- Incremental GPU cost: US$5.1915
- RunPod: `STOPPED`
- Q2: `NOT RUN`
- Confirmatory holdout: `UNTOUCHED`
- Gate 11: `DRAFTED, NOT RUN`

## Interpretation boundary

The classification was applied mechanically after all frozen rows were
collected. This is cross-domain DEVELOPMENT evidence only. It is not Q2,
confirmatory holdout evidence, or execution of Gate 11. Descriptive generator
bins were not used for selection or classification.
