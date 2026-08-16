# Inference engine architecture

The scientific protocol is independent of execution mode. Every optimized
engine must reproduce the preserved `serial_reference` prediction rows before
it can be used for a real Q1 run.

| Engine | Purpose | Current status |
|---|---|---|
| `serial_reference` | Candidate-wise full-prompt correctness oracle | Preserved |
| `full_prompt_batched` | Prepared prompts, explicit padding/positions, optional row batching | Qwen3 512×3 gate PASS only with `serial_shape_reference`; canonical V1.1 profile |
| `cached_decode` | Prefix KV prefill plus one-token query decode | Implemented and Qwen3 exercised; rejected for exact Q1 equivalence after BF16 flips |
| `cached_suffix_replay` | Guarded Qwen3 suffix replay using native decoder/cache APIs | Implemented; technical equality test passed, slower; not canonical |

For the common single-token choice path, candidate scoring gathers the allowed
LM-head rows. `full_vocab_reference` retains vocabulary-normalized log
probabilities; `candidate_only` stores unnormalized candidate logits and is
valid for ranking and pairwise margins only. Score semantics are recorded in
each output row.

The `serial_shape_reference` profile appends a fixed dummy candidate token to
match the old candidate-wise sequence shape, while reading logits at the
prompt's final position. This was required because otherwise BF16 shape/kernel
changes caused discrete differences even when the mathematical computation was
equivalent in intent.

The planner and cache engines remain available for future engineering work,
but batch boundaries and cache reuse are never allowed to become scientific
keys. `serial_reference` is the permanent audit path.

`torch.compile` and CUDA graphs are optional prototypes and remain off by
default. They require warm-up, memory, and exact discrete-equivalence audits
before activation.
