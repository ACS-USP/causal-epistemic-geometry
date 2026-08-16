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
  decoder-only model, followed by the same path on a live NVIDIA A40 CUDA
  device.
- Tensor/tuple layer outputs, exact last-token/all-token hook arithmetic,
  alpha-zero and zero-vector identity, cleanup, repeated contexts, dimension
  and layer errors, no-grad inference, last-non-padding extraction, difference
  of means, vector serialization, end-to-end metrics, and run validation.
- The checked-in `configs/tiny_transformer_cuda_smoke.yaml` completed an
  8-item, 16-prediction paired run on CUDA. Its artifact was pulled back and
  passed `ceg validate-run` locally.
- Remote bootstrap, `ceg doctor`, full pytest (37 tests), Ruff, and compileall
  passed after the compatibility repairs below.

## Remains untested

- No pretrained model was downloaded, and no Qwen3-8B/Llama/large model was
  loaded.
- No real tokenizer/chat template, pretrained-model generation, device map,
  bf16 production inference, MPS run, or multi-GPU sharding has been exercised.
- No scientific benchmark, frozen layer/alpha, Q1 claim, or Q2 geometry
  experiment has been run.
- The optional tiny pretrained smoke was skipped intentionally; the network
  path is not needed to validate mechanics.

## Scientific boundary

The tiny random transformer and all mock outputs are software validation only.
They do not support or reject Q1, do not establish useful complementarity, and
do not make any Q2 geometry claim.

## RunPod connection audit (2026-08-16)

The old `masters-project` workflow and local SSH metadata were inspected
read-only. The old project has a destructive bootstrap that removes `.venv`,
force-installs CUDA Torch, and installs unrelated Node/Gemini tooling; none of
that was reused. The existing dedicated RunPod Ed25519 identity and its public
fingerprint are documented in `LOCAL_SSH_AUDIT.md`; no private key contents
were read or committed.

The new repository provides scoped SSH alias configuration, read-only
connection diagnostics, additive rsync/tar push-pull, persistent `/workspace`
cache setup, storage checks, Pod stop checks, and a local `make predeploy` gate.
The real alias is now configured locally, non-interactive SSH succeeds, and the
repository was synchronized without touching unrelated repositories.

Observed remote facts:

- Ubuntu 22.04, Python 3.11.10, x86_64.
- NVIDIA A40, 46,068 MiB, driver 580.159.04; one CUDA device visible.
- `torch==2.4.1+cu124`, `transformers==4.57.6`, CUDA available, bf16 support
  reported by Torch.
- `/workspace/hf-cache` is the active cache path. No model files were added.
- The image did not have `rsync`; the tested additive tar-over-SSH fallback was
  used. macOS AppleDouble metadata and remote ownership preservation were
  explicitly hardened.

Two engineering issues were found and repaired during the live check:

1. Unbounded `transformers` installed 5.15.0, which disables itself with the
   image's Torch 2.4.1. The optional dependency is now constrained to
   `transformers>=4.45,<5`, without replacing Torch.
2. The SSH-helper test depended on the researcher's private local key path. It
   now uses a temporary test-only identity while production invocation still
   validates the configured identity file.

## Gate 2 pre-pilot audit (2026-08-16)

The principal researcher authorized the fixed Q1 V1 development protocol. The
following operational choices are pinned in `docs/Q1_DEVELOPMENT_PROTOCOL_V1.md`:
Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`,
TIGER-Lab/MMLU-Pro revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`,
direct-choice log-likelihood, Qwen non-thinking chat mode, block 17, last
prompt token, and the 15-condition calibration/evaluation design. The
development split manifest is `data/splits/mmlu_pro_q1_v1.json` with hash
`84982e4c72e230ffff78363f085d4d5c53447fd1e248e5e170ed5e8c508d343e`.

The live pinned smoke and validation audit now show:

- 8-item technical smoke: PASS; alpha=0 identity and zero-vector identity
  held for predictions and candidate scores within the configured tolerance.
- Validation baseline: 70 items, 0.5429 accuracy, 0 parse failures; the
  30%/90% stop gate was not triggered.
- Validation repeat: exact prediction rows and candidate scores matched.
- Qwen provenance: 36 layers, hidden size 4096, BF16, one A40, no CPU
  offload, no quantization, resolved path `model.model.layers`, block 17.

One optimization was explicitly rejected during audit. Scoring the ten labels
in one BF16 batch changed candidate log-likelihoods by as much as 1.375 versus
the serial scorer, despite identical top labels. The batched path was removed;
the canonical scorer remains serial so candidate scores are stable and
auditable. This is a correctness decision, not a scientific tuning choice.

The fixed Q1 pilot itself remains pending at the time of this audit. No
calibration direction, evaluation condition, Q1 result, or Q2 claim is being
interpreted here. After the pilot, this document must be supplemented with the
artifact validator result and exact stop/continue status.
