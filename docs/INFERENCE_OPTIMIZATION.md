# Inference optimization status

This document records engineering status only. The Q1 V1.1 scientific
protocol remains frozen. No optimized engine has been approved for a real
Qwen3 run yet.

## Implemented locally

- `serial_reference` remains the preserved candidate-wise full-prompt oracle.
- Prepared choice items render and tokenize prompts once, retaining prompt
  hashes, candidate IDs, semantic IDs, and token-count audits.
- Single-token candidates are scored from one prompt state. The optional
  `candidate_only` head gathers only allowed LM-head rows; its unnormalized
  score semantics are explicit in provenance.
- `full_prompt_batched` supports deterministic item/condition batching,
  explicit masks/positions, left or right padding, and row-wise deltas.
- `cached_decode` prefills `prompt[:-1]` once, then decodes the final prompt
  token with condition chunks and temporary vectorized steering hooks.
- Multi-token labels use a shared-prefix continuation fallback rather than
  repeating the long prompt for every candidate.
- Batched activation extraction captures selected last-token layers without
  retaining graphs; difference-of-means uses it when available.
- The planner enforces deterministic length buckets, item limits, and padded
  token budgets.
- Prediction journals append and fsync completed rows, quarantine a truncated
  tail, reject duplicates/conflicts, and support V1.1 `--resume` semantics.
- Optional `torch.compile` and CUDA-graph boundaries exist but are off by
  default. Qwen3 suffix replay is isolated behind strict model/version guards
  and fails closed until equivalence is audited.

## Local evidence

`tests/test_huggingface_tiny.py` exercises real Torch/Transformers forwards on
a randomly initialized two-layer GPT-2-style model. It covers alpha-zero and
zero-vector identity, exact hook shifts, token isolation, cleanup, repeated
contexts, padding, candidate-only ranking/margins, multi-token fallback,
batched activations, cache reuse, and forward-call accounting.

`benchmarks/serial_reference_profile.json` is a tiny CPU engineering profile.
Its timings are not transferable to Qwen3 or an A40 and must not be cited as a
scientific or deployment result.

## Still requires the restarted Pod

- Qwen3 tokenizer boundary audit for A–J under the frozen chat template.
- BF16 cached-decode equivalence against the preserved serial Qwen rows.
- Qwen3 layer-17 suffix replay implementation/equivalence.
- Attention backend audit, A40 item/condition autotuning, compile/graph
  benchmarks, GPU utilization, VRAM, and end-to-end speedup.
- Complete-512 equivalence and the clean optimized V1.1 rerun.

The local implementation must stop at `RUNPOD_REQUIRED_FOR_EQUIVALENCE` when
these are the only remaining blockers. No V1.1 scientific conclusion is made
by this engineering work.
