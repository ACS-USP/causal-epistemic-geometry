# Inference optimization status

This document records engineering status only. The Q1 V1.1 scientific
protocol remains frozen. The optimized run is DEVELOPMENT infrastructure and
does not establish a scientific claim.

## Approved Q1 V1.1 profile

The clean Q1 V1.1 run used:

```text
engine: full_prompt_batched
serial_shape_reference: true
candidate_head_mode: candidate_only
item_batch_size: 1
condition_chunk_size: 1
padding: left
attention: SDPA (requested: auto)
torch.compile: false
CUDA graphs: false
```

The important speedup is the single-token choice fast path: the prompt is
evaluated once and the ten allowed A–J LM-head rows are gathered. The
`serial_shape_reference` option deliberately preserves the old candidate-wise
sequence shape, which was necessary for exact BF16 equivalence on Qwen3.
`candidate_only` stores unnormalized candidate logits. Their rankings and
pairwise margins are valid; they must not be described as full-vocabulary
log-probabilities.

## Real-Qwen engineering evidence

The pinned Qwen3 tokenizer audit found A–J to be single-token and
context-compatible under the frozen chat template and `enable_thinking=false`.
The following gates were run on the A40 using the pinned Qwen3 revision:

- serial reference versus `full_prompt_batched` with serial-shape preservation:
  512 DEV items × baseline/PC1−/PC1+, zero discrete prediction differences;
- the same gate with `candidate_only`: zero discrete prediction differences;
- complete clean V1.1: 512 DEV items × 31 frozen conditions = 15,872 rows,
  validated with no duplicate keys and no holdout access;
- native Qwen3 suffix replay: exercised on technical smoke and matched the
  cached path there, but it was slower and is not canonical;
- ordinary BF16 cached decode and shape-changing batching: exercised, but not
  approved because they produced discrete flips against the serial oracle.

The validated clean run is recorded in the local review bundle under
`review/q1_v1_1_optimized_clean_run/`. The full prediction file remains on
RunPod because real experimental predictions are remote-only by policy.

Observed wall-time accounting uses the run timestamp and final prediction
journal mtime: approximately 17.48 minutes for 15,872 item-condition rows,
versus the serial estimate of 183.80 minutes, or approximately 10.52×. The
cost estimate is approximately $0.12 at the recorded $0.40/A40-hour planning
rate. This is an engineering runtime observation, not a scientific result.

## Preserved alternatives

- `serial_reference` remains the correctness oracle and is never deleted.
- `full_prompt_batched` remains available, but ordinary shape-changing BF16
  execution is not accepted as an exact Q1 replacement.
- `cached_decode` shares the prompt prefix and performs one-token decoding. It
  is useful infrastructure and passed local/tiny mechanics tests, but its
  Qwen3 512-item gate produced flips and it is not canonical for V1.1.
- `cached_suffix_replay` uses guarded native Qwen3 decoder/cache APIs. It fails
  closed on unsupported model/version combinations and is not canonical after
  the technical speed comparison.

## Not approved or not measured

`torch.compile`, CUDA graphs, asynchronous CPU/GPU pipelines, and A40 batch
autotuning were not enabled for the clean scientific-development run. They
must remain optional and require a fresh equivalence gate before use. Peak GPU
utilization and a formal VRAM autotuning profile were not captured as part of
the approved run, so no claim is made for those quantities.

No V1.2 or Q2 experiment was run. No confirmatory holdout was accessed.
