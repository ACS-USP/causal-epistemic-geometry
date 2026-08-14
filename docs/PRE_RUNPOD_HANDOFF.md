# Pre-RunPod handoff

## What changed

The optional HuggingFace path is now exercised by a network-free two-layer
random GPT-2-style Transformer. The path supports injected test models, robust
layer discovery, explicit plain/chat prompt modes, parse statuses, detailed
model/vector/intervention provenance, deterministic bootstrap diagnostics,
append-only resumable prediction rows, atomic run status, and `ceg validate-run`.

New commands/configs include:

```bash
ceg preflight configs/tiny_transformer_smoke.yaml
ceg run configs/tiny_transformer_smoke.yaml
ceg validate-run runs/<run>
ceg preflight configs/runpod_q1_smoke.example.yaml
```

## Bugs fixed

- GPT-2 `transformer.h` layer discovery failed because the explicit path was
  resolved only on the backend wrapper.
- Zero-variance phi correlation is now undefined/null with status instead of a
  silent numeric convention.
- Ambiguous generation output is now distinct from exact parser success.
- Same-config interrupted runs resume only when identity provenance matches.
- Truncated final JSONL tails are quarantined; duplicate keys and conflicting
  provenance fail loudly.

## Validation performed

- 30 local tests pass, including actual Torch/Transformers mechanics.
- alpha-zero and zero-vector identity pass.
- exact hidden shift, last-token isolation, all-token shift, hook cleanup, and
  repeated intervention isolation pass.
- activation extraction, padding policy, difference-of-means, vector roundtrip,
  metadata, and hash checks pass.
- tiny random transformer end-to-end run and `validate-run` pass.
- interrupted/resumed predictions and metrics equal an uninterrupted run.

## Not tested

No pretrained model, GPU, CUDA device map, Qwen3-8B, large model, scientific
benchmark, or Q1 result was run. Optional tiny pretrained smoke is skipped.
The tiny random transformer report is software validation only.

## Exact next commands

Local mechanics:

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
make tiny-smoke
ceg validate-run "$(find runs -maxdepth 1 -type d -name '*tiny-random*' | sort | tail -n 1)"
```

RunPod technical preparation:

```bash
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
bash scripts/runpod_preflight.sh configs/runpod_q1_smoke.example.yaml
```

The researcher must choose and review the real model, benchmark, vector
construction, layer, token scope, alpha, and controls before any Q1 run.

