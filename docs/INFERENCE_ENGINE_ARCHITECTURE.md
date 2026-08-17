# Q1 V3 reasoning inference engines

This document describes execution mechanics only. It does not change the
frozen Q1 V3 scientific protocol, and no engine is approved for Stage A until
its rows have been compared with the preserved reference.

## Scientific invariant

The scientific key remains:

```text
latent item × surface view × rollout index × reasoning budget
```

Batch boundaries, cache objects, CUDA graphs, compile artifacts, and physical
generation IDs are execution provenance. They never replace the scientific
key. Every optimized result must preserve the exact generated token prefix
used by its parser and must retain enough metadata to reconstruct its physical
source trajectory.

## Engines

| Engine | Physical work | Status |
|---|---|---|
| `serial_reasoning_reference` | One independent Transformers generation per budget row | Permanent correctness oracle; preserved |
| `max_budget_prefix_reuse` | One maximum-budget generation per item/view/seed; derive shorter rows by exact token censoring | Implemented; local unit-tested; real-Qwen prefix gate pending |
| `batched_reasoning` | The same maximum-budget prefix reuse, with deterministic length buckets and per-row seeded cached decoding | Implemented; tiny random-transformer equivalence passed; real-Qwen gate pending |

The serial reference is intentionally not deleted or rewritten when an
optimized engine is changed. `max_budget_prefix_reuse` parses every censored
prefix independently. Thus a 512-token prefix remains
`THINKING_UNCLOSED` even when the 2048-token source later contains a valid
`FINAL:` field.

## Batched generation mechanics

`batched_reasoning` prepares all prompts once, sorts rows by
`(prompt_length, original_index)`, and creates batches under both:

```text
maximum rows
maximum padded prefill tokens
```

The output list is restored to caller order. Each row owns a deterministic
Torch generator seeded from the frozen rollout seed, so reordering a batch
does not change another row's sampling stream. The prefill uses
`use_cache=True`; subsequent tokens use one-token cached decode. No early
final-answer stopping or semantic parser behavior is introduced by this
engine.

Decoder-only padding is explicit. Right-padding plus gathering the final real
token is covered by the network-free GPT-2 fixture. Left-padding remains a
model-specific engineering choice and must be benchmarked against the serial
oracle on Qwen3 before use. Position IDs, attention masks, batch prompt
lengths, and padded-token cost are recorded in output provenance.

## Future optional accelerators

The repository also contains conservative boundaries for `torch.compile` and
CUDA graphs. They are disabled by default and are not silently activated by a
reasoning run. A compiled or graphed mode is eligible only after:

1. token IDs, parse status, and correctness match `serial_reasoning_reference`;
2. warm-up/compile cost is measured separately from steady-state throughput;
3. memory remains stable under the target batch shapes; and
4. the net speedup is material for the actual Q1 workload.

Continuous batching or an external serving engine may be used only as an
engineering upper bound. It cannot become the canonical scientific engine
without the same equivalence gate and complete provenance.

## Real-Qwen approval gate

Before a clean Stage A launch, compare on a bounded technical subset:

- independently generated 512 and 1024 outputs against prefixes of a 2048
  output with the same item/view/seed;
- serial versus prefix-reuse token IDs, parse status, and correctness;
- serial versus batched token IDs, parse status, and correctness;
- candidate/runtime metadata, forward-call accounting, and peak memory.

Floating-point equality is not required, but discrete token and parser
differences are not acceptable for canonical approval. If the real-Qwen gate
fails, keep the serial reference and use the safest passing engine only.
