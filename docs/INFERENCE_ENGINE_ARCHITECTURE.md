# Inference engine architecture

The scientific protocol is independent of execution mode. Every optimized
engine must reproduce the preserved `serial_reference` prediction rows before
it can be used for a real Q1 run.

| Engine | Purpose | Current status |
|---|---|---|
| `serial_reference` | Existing candidate-wise, full-prompt correctness oracle | Preserved |
| `full_prompt_batched` | Prepared prompts, deterministic padding, item/condition batching | Local tiny-transformer validated |
| `cached_decode` | Prefix KV prefill once, one-token query decode, row-wise steering | Local tiny-transformer validated |
| `cached_suffix_replay` | Qwen3 layer-suffix replay using native decoder/cache APIs | Strictly gated; RunPod equivalence required |

For the common single-token choice path, candidate scoring gathers the allowed
LM-head rows. `full_vocab_reference` retains vocabulary-normalized log
probabilities; `candidate_only` stores unnormalized candidate logits and is
valid for ranking and pairwise margins only. The score semantics are recorded
in every output row.

The batch planner sorts prepared prompts by `(token_count, item_id)` and enforces
both an item limit and a padded-token budget. Results are reconstructed by the
canonical item/condition key, so batch boundaries never become scientific
ordering.

`cached_decode` currently requires left padding. Right padding is supported by
the full-prompt batched path with explicit per-row target positions. This is an
intentional fail-closed constraint until a cache-position audit on the target
Transformers/Qwen3 stack is complete.

`torch.compile` and CUDA graphs are optional runtime prototypes and remain off
by default. They must be benchmarked for warm-up amortization, memory safety,
and exact discrete equivalence before being enabled.
