# Q1 DEVELOPMENT PROTOCOL V1.1

**DEVELOPMENT FOLLOW-UP — NOT CONFIRMATORY**

This protocol freezes the controlled V1.1 follow-up before any V1.1 outcomes
are produced. It stops after this follow-up. It does not authorize V1.2, Q2,
layer search, beta search, semantic-vector construction, or confirmatory
holdout access.

## Freeze record

- protocol timestamp: `2026-08-16T20:14:27Z`
- protocol freeze commit: `b5d97fd`
- original V1 run ID: `20260816T172847Z_q1-v1-development_ac2f36265c`
- original V1 experiment commit: `8236e9887e452dc252ef36a8da470c16ef2dd610`
- model: `Qwen/Qwen3-8B`
- model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`
- dataset: `TIGER-Lab/MMLU-Pro`
- dataset revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`
- split manifest SHA-256: `84982e4c72e230ffff78363f085d4d5c53447fd1e248e5e170ed5e8c508d343e`
- prompt template: `Q1_V1_MMLU_PRO_DIRECT_CHOICE_V1`
- layer: `17` zero-based
- token scope: `last_token`
- inference mode: deterministic `choice_loglikelihood`
- model weights: BF16
- thinking: disabled
- DEV_EVALUATION size: `512`
- CONFIRMATORY_HOLDOUT: forbidden and untouched

## Reused V1 scientific objects

V1.1 must load, not reconstruct, the original V1 artifacts:

- original PC1 vector hash: `abca43ae3b9621614562798dbfbd8c3ad9932fcf9cb0cfd2c58d28adc48897c5`
- original random_0 hash: `d6ef7d2c8146196330fb14aa2b1e1d6e7d94177b9e2033a6eeda82bc64d00a28`
- original random_1 hash: `8d440a17db54034db10fe52ed1237cd821c763df68aa4f8a9b51181a6956d853`
- original random_2 hash: `1951e428065d639ce4308da3c3ebf41c250678c12141dc1415fcd08c1db7f8ab`
- original random_3 hash: `28847139eca3c31f40253e4496eb85814bd28d3e3b232d77964272b7f255ac6b`

The exact V1 PC1 alpha is loaded from V1 metadata. It is approximately
`4.8855751862975145` at the positive sign, but the stored value is canonical.

## Fixed V1.1 conditions

### Control A: FP32 numerical audit

On the original option order and exact DEV_EVALUATION IDs:

- `baseline_fp32`
- `pca_pc1_minus_fp32`
- `pca_pc1_plus_fp32`

Only the log-softmax input is promoted to FP32:

```python
log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
```

The model, activations, candidate likelihood semantics, layer, prompt, and
token scope remain unchanged. The numerical stop rule is: if more than 1% of
discrete predictions change against the stored V1 rows solely because of this
promotion, stop before the remaining V1.1 controls.

### Original random controls under FP32

If the numerical audit passes, rerun the exact V1 native-direction-SD random
controls under the FP32 scorer:

- `random_0_native_scale_neg`, `random_0_native_scale_pos`
- `random_1_native_scale_neg`, `random_1_native_scale_pos`
- `random_2_native_scale_neg`, `random_2_native_scale_pos`
- `random_3_native_scale_neg`, `random_3_native_scale_pos`

These retain each V1 random direction's original calibration scale and are
not Euclidean norm-matched controls.

### Control B: PC1-Euclidean-norm-matched random directions

Use the exact V1 random unit vectors and no new random draws. Define
`alpha_match = abs(alpha_PC1_at_beta_+0.5)` from the V1 PC1 metadata. Run:

- `random_0_normmatched_pc1_neg`, `random_0_normmatched_pc1_pos`
- `random_1_normmatched_pc1_neg`, `random_1_normmatched_pc1_pos`
- `random_2_normmatched_pc1_neg`, `random_2_normmatched_pc1_pos`
- `random_3_normmatched_pc1_neg`, `random_3_normmatched_pc1_pos`

These conditions are defined by `alpha = ±alpha_match` and unit-vector
Euclidean displacement equal to PC1. They are not described as beta ±0.5.

### Control C: deterministic option permutations

Derive exactly four item-wise option permutations from SHA-256 of the protocol
seed, item ID, and permutation ID. The original ordering is not one of the
four. For each permutation, permute option contents, relabel A–J, and remap
the target consistently. Run only:

- `permutation_0_baseline`, `permutation_0_pc1_minus`, `permutation_0_pc1_plus`
- `permutation_1_baseline`, `permutation_1_pc1_minus`, `permutation_1_pc1_plus`
- `permutation_2_baseline`, `permutation_2_pc1_minus`, `permutation_2_pc1_plus`
- `permutation_3_baseline`, `permutation_3_pc1_minus`, `permutation_3_pc1_plus`

The original V1 PC1 vector and exact V1 alpha values are held fixed. PCA is not
recomputed for permuted prompts. Predictions retain both displayed labels and
semantic original-option identities.

The full frozen family therefore contains 31 logical condition IDs: 3 FP32
original-order conditions, 8 FP32 native random controls, 8 norm-matched random
controls, and 12 permutation conditions.

## Prespecified diagnostics

For original-order FP32 conditions and all random controls, retain V1 paired
metrics, deterministic item bootstrap intervals, intervention norm, normalized
intervention norm, calibration scale, alpha, and cosine to PC1. For every
permutation retain baseline/PC1 accuracies, deltas, rescues, damages,
prediction changes, pair-oracle headroom, displayed-letter distributions,
target-letter distributions, and semantic option identities.

Define baseline choice margin as the FP32 score of the best candidate minus the
FP32 score of the second-best candidate. Report median and quartiles for
unchanged items, changed items, rescues, and damages, plus fixed baseline-margin
quartile rates. Report PC1+ rescues/damages by MMLU-Pro category descriptively.

The prespecified V1.1 questions are separate:

- Q-A: Does PC1+ survive FP32 probability computation?
- Q-B: Do equal-Euclidean-norm random directions behave similarly to PC1?
- Q-C: Does PC1 survive deterministic option permutations?
- Q-D: Is PC1 concentrated on low-margin decisions?
- Q-E: Is the change plausibly reducible to displayed answer-letter bias?

No question is collapsed into a single score or automatic success/failure
decision.

## Firewall and stop rules

Before every real run, assert that all requested IDs are in DEV_EVALUATION and
none are in CONFIRMATORY_HOLDOUT. Every manifest must state
`confirmatory_accessed: NO`.

Stop immediately if the model revision, dataset revision, split hash, PC1 hash,
or random-vector hashes differ from V1; if permutation remapping fails; if the
FP32 numerical audit changes more than 1% of discrete predictions; if
reproducibility fails; if the artifact validator fails; or if projected GPU
cost exceeds `$2.00`.

All language remains DEVELOPMENT ONLY: development signal, robustness control,
and descriptive evidence. No confirmatory claim, significance claim, or Q2
claim is authorized.
