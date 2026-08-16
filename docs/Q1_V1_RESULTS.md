# Q1 V1 Development Pilot Results

**Status: DEVELOPMENT ONLY — NO SCIENTIFIC CLAIM FROZEN**

RunPod artifact:
`/workspace/causal-epistemic-geometry/runs/q1_v1/20260816T172847Z_q1-v1-development_ac2f36265c`

## Execution and provenance

- Model: `Qwen/Qwen3-8B`, commit
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Dataset: `TIGER-Lab/MMLU-Pro`, commit
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
- Host: one NVIDIA A40, BF16, `torch==2.4.1+cu124`,
  `transformers==4.57.6`; no quantization or CPU offload.
- Prompt: fixed Qwen chat template with `enable_thinking=False`; direct-choice
  complete candidate log-likelihood; no sampling or free generation.
- Layer: block 17, last rendered-prompt token.
- Data: 512 calibration items and 512 development evaluation items. The
  11,008-item confirmatory holdout was not accessed for evaluation.
- Conditions: exactly 15 — baseline, six signed PCA conditions, and eight
  signed random-null conditions.

Artifact validation passed with 7,680 prediction rows, 15 conditions, and
prediction hash
`2c5f3a6a95692e6aab62388cfc23b2fb5d819eb8b41d7d642ee05282f2cfc2c1`.
The independent repeat audit covered baseline, `pca_pc1_minus`, and
`random_0_minus` on 32 fixed evaluation items / 96 rows. Maximum absolute
candidate-score difference was `0.0` against tolerance `1e-5`.

## Descriptive metrics

Baseline accuracy was **0.4219**. The fixed competence band was
`A_v >= A_0 - 0.02`.

| condition | accuracy | delta | error phi | rescue | damage | headroom |
|---|---:|---:|---:|---:|---:|---:|
| pca_pc1_minus | 0.4062 | -0.0156 | 0.9361 | 0.0135 | 0.0556 | 0.0078 |
| pca_pc1_plus | 0.4395 | +0.0176 | 0.9409 | 0.0405 | 0.0139 | 0.0059 |
| pca_pc2_minus | 0.4219 | +0.0000 | 0.9920 | 0.0034 | 0.0046 | 0.0020 |
| pca_pc2_plus | 0.4180 | -0.0039 | 0.9840 | 0.0034 | 0.0139 | 0.0020 |
| pca_pc3_minus | 0.4297 | +0.0078 | 0.9841 | 0.0135 | 0.0000 | 0.0000 |
| pca_pc3_plus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_0_minus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_0_plus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_1_minus | 0.4219 | +0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_1_plus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_2_minus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_2_plus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_3_minus | 0.4199 | -0.0020 | 0.9960 | 0.0000 | 0.0046 | 0.0000 |
| random_3_plus | 0.4219 | +0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

The paired 2×2 table for `pca_pc1_plus` was `216/3/12/284` in the order
baseline-correct/treatment-correct, baseline-correct/treatment-wrong,
baseline-wrong/treatment-correct, baseline-wrong/treatment-wrong. The other
conditions are in `metrics.json`.

## Interpretation boundary

These are descriptive DEVELOPMENT outputs from one frozen model, one fixed
layer, one fixed intervention scale, and one development evaluation split.
They do not establish Q1, do not justify a V2 search, and do not test whether
geometry between `v_i` and `v_j` predicts error covariance. Pair-oracle
headroom is not an implementable ensemble result. No confirmatory holdout
claim is permitted.
