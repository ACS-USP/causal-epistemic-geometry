# Pre-RunPod audit

## Audit scope

This audit was performed after the initial repository bootstrap and before any
large-model or scientific run. The repository was clean at the start of the
audit. The unrelated `~/dev/emergence-specialization` repository was not
modified.

## Already correct

- `src/` packaging, local Git history, lightweight core dependencies, and
  optional HF dependency separation were sound.
- The mock backend used deterministic representations and literal
  `h' = h + alpha*v` intervention arithmetic.
- Paired item identity, exact-label JSONL ground truth, vector hashing, and the
  baseline-versus-treatment metrics were present.
- Temporary mock intervention state already restored correctly in its context
  manager.
- The initial mock smoke, tests, lint, compile, and reproducibility checks were
  passing.

## Risks found and repaired

- The HF backend loaded only from a model ID/path, so its actual PyTorch hook
  path had never been exercised. It now accepts injected model/tokenizer
  objects, and `tiny_transformer` builds a two-layer random GPT-2-style model
  entirely from config.
- Automatic layer discovery assumed paths began at the backend wrapper. It now
  tries both wrapper and model roots, so GPT-2 `transformer.h` is exercised.
- Parser output had no parse-status distinction. It now records `OK`, `EMPTY`,
  `AMBIGUOUS`, or `INVALID`; ambiguous/invalid output is retained separately
  from model correctness.
- Phi correlation previously assigned conventions to zero-variance vectors. It
  now emits NaN in machine output (JSON null) plus an explicit status.
- Artifact writes were final-only. Runs now use append-only, fsynced prediction
  rows, atomic manifest/metrics writes, status transitions, provenance checks,
  deterministic resume, tail quarantine, and `ceg validate-run`.
- Same-second output collisions were already fixed in the initial repository;
  the resumable session preserves that collision-safe behavior.
- Prediction rows now carry config, model, vector, intervention, decoding,
  prompt-mode, seed, and benchmark provenance.

## What was exercised

- Actual Torch/Transformers forward passes on a CPU randomly initialized tiny
  decoder-only model.
- Tensor/tuple layer outputs, exact last-token/all-token hook arithmetic,
  alpha-zero and zero-vector identity, cleanup, repeated contexts, dimension
  and layer errors, no-grad inference, last-non-padding extraction, difference
  of means, vector serialization, end-to-end metrics, and run validation.

## Remains untested

- No pretrained model was downloaded, and no Qwen3-8B/Llama/large model was
  loaded.
- No real GPU, CUDA device map, bf16 kernel, MPS production run, chat template
  from a real instruct tokenizer, or multi-GPU sharding has been exercised.
- No scientific benchmark, frozen layer/alpha, or Q1 claim has been run.
- The optional tiny pretrained smoke was skipped intentionally; the network
  path is not needed to validate mechanics.

## Scientific boundary

The tiny random transformer and all mock outputs are software validation only.
They do not support or reject Q1, do not establish useful complementarity, and
do not make any Q2 geometry claim.

