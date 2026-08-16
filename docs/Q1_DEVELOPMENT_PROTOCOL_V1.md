# Q1 Development Protocol V1

**DEVELOPMENT PROTOCOL — NOT CONFIRMATORY**

Locked on 2026-08-16 before downloading experimental model/data material. This
document records the first real-model development pilot. It does not freeze a
scientific claim and it does not authorize access to the confirmatory holdout.

## Question and boundary

Q1 asks whether one controlled activation intervention can change where a
frozen language model fails while approximately preserving individual
competence. Q2 geometry, majority-vote ensembles, and collective utility are
out of scope.

## Frozen model and runtime

- Requested model: `Qwen/Qwen3-8B` post-trained dense causal LM.
- Model rationale: approximately 8B parameters, 36 transformer blocks, hidden
  size 4096, residual-stream intervention is straightforward, public
  Apache-2.0 weights, and explicit non-thinking chat mode.
- No Qwen3.5, quantization, LoRA, fine-tuning, or parameter updates.
- Precision: BF16.
- Device: one NVIDIA A40; no CPU offload expected.
- Inference: `model.eval()` and `torch.inference_mode()`.
- Requested model revision: resolved HuggingFace commit is recorded below
  before the first smoke and reused thereafter.
- Resolved model revision: `PENDING_RESOLUTION_BEFORE_SMOKE`.
- Canonical cache: `/workspace/hf-cache`.

The revision field is an operational resolution of the named model, not a
post-result scientific choice. It must never revert to a floating `main`.

## Inference and prompt

- Primary inference mode: deterministic candidate choice log-likelihood.
- No free-form generation as the primary outcome; no sampling, temperature,
  chain-of-thought generation, or few-shot examples.
- Qwen chat template with `enable_thinking=False`.
- Fixed task template:

  ```text
  Choose the correct answer to the following multiple-choice question.
  Respond with only the answer letter.

  Question:
  {question}

  A. {option_a}
  B. {option_b}
  ...
  J. {option_j}
  ```

- Candidate scores are complete continuation log-likelihoods for the labels
  present in each item. Candidate tokenization is inspected and recorded; no
  one-token assumption is made.
- Final rendered prompt hashes, candidate token IDs/counts, and all scores are
  stored with predictions.
- Prompt template hash: `PENDING_FINAL_TEMPLATE_HASH`.

## Dataset and calibration firewall

- Dataset: `TIGER-Lab/MMLU-Pro`.
- Report label: **MMLU-Pro-derived direct-choice evaluation**.
- This is not official MMLU-Pro leaderboard evaluation and must not be
  compared directly with the official leaderboard.
- Dataset revision/hash: `PENDING_RESOLUTION_BEFORE_SMOKE`.
- Official `validation` split: 70 items, protocol calibration only.
- First technical smoke: 8 validation items.
- Baseline calibration gate: all 70 validation items, baseline only.
- Gate: if accuracy is below 30% or above 90%, stop and report.
- Prompt, model revision, tokenizer revision, and scoring implementation are
  frozen after this gate.

Before steering, the official `test` split is deterministically stratified by
category using split seed `20260816` into:

- `DEV_CALIBRATION`: 512 item IDs;
- `DEV_EVALUATION`: 512 different item IDs;
- `CONFIRMATORY_HOLDOUT`: every remaining test item ID.

Only item IDs are stored in the split manifest. Development configuration is
programmatically forbidden from selecting `confirmatory_holdout`.

## Intervention

- One residual-stream location: Qwen block index 17 (18th block, zero-based).
- The resolved runtime module path is recorded in model provenance.
- Extraction and intervention target the same block output tensor.
- Token scope: `last_token`, meaning the final token of the rendered prompt.
- Previous sequence positions must remain untouched.
- Model weights remain frozen.

## Calibration directions

On `DEV_CALIBRATION`, extract the final prompt-token activation matrix
`H ∈ R^(512×4096)` at block 17, center it, and construct exactly:

- unit `PC1`, `PC2`, and `PC3`, using the first three SVD/PCA directions;
- four deterministic unit random nulls: `random_0` through `random_3`.

PCA signs are arbitrary, so both signs are predeclared. Random seeds derive
from protocol seed `20260816`; no post-result regeneration or orthogonalization
is permitted in V1. Direction hashes, singular values, explained variance,
pairwise random cosines, and source IDs are artifacts.

## Standardized intervention scale and conditions

For every unit direction `v`, calculate

`s_v = SD_x(vᵀ h_17(x))`

on `DEV_CALIBRATION`, then use `alpha(v, beta) = beta * s_v` with beta exactly
`-0.5` and `+0.5`. Store beta, raw alpha, `s_v`, and the relative shift norm.

The 15 predeclared conditions are:

- baseline;
- `PC1_-0.5`, `PC1_+0.5`, `PC2_-0.5`, `PC2_+0.5`, `PC3_-0.5`, `PC3_+0.5`;
- `random_0` through `random_3`, each at both signs.

No condition may be added because it looks interesting.

## Primary development evaluation

Run all 15 conditions on the same paired `DEV_EVALUATION` IDs. Report for each
treatment relative to baseline:

- baseline/treatment accuracy and delta accuracy;
- paired 2×2 counts, rescues, damages, double faults;
- disagreement, rescue rate, damage rate;
- error Jaccard and phi/Pearson with undefined status;
- pair-oracle accuracy and complementarity headroom;
- deterministic item bootstrap intervals for descriptive uncertainty.

The development competence band is fixed before results as
`A_v >= A_0 - 0.02`. It is a descriptive flag, not an inferential threshold.
High disagreement or low error correlation without preserved accuracy is not a
success.

## Reproducibility, stop rules, and artifacts

Repeat baseline, one PCA condition, and one random condition on a deterministic
subset. Require identical discrete predictions and numerically stable scores.

Stop immediately for model-fit failure, CPU offload, quantization, NaNs,
hook mismatch, scoring ambiguity, baseline gate failure, reproducibility
failure, split leakage, holdout access, or artifact-validator failure. Report
the exact failure; do not tune around it.

Artifacts live under `runs/q1_v1/` and include the protocol, resolved configs,
split manifest/hash, dataset/model/tokenizer revisions, prompt hashes,
activations, direction artifacts, scores, paired predictions, metrics,
bootstrap results, figures, and `summary.md`. Model weights remain in the
persistent HuggingFace cache and never enter Git.

After V1, stop. Do not scan layers, beta values, extra PCs/seeds, semantic
vectors, all-token scope, thinking mode, CoT, majority vote, or the
confirmatory holdout automatically.
