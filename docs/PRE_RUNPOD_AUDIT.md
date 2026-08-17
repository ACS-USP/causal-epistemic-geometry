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

- Before Gate 2, no pretrained model had been downloaded. Gate 2 subsequently
  loaded only the authorized pinned Qwen3-8B on RunPod.
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

The fixed Q1 pilot completed after this audit update. Its 7,680-row artifact
passed validation and its explicit 32-item repeat audit passed with maximum
absolute score difference 0.0 at tolerance 1e-5. The descriptive results are
recorded in `docs/Q1_V1_RESULTS.md`; they do not establish Q1 or authorize V2.

## V1.1 controlled-follow-up preflight (2026-08-16)

Before V1.1 execution, the local and remote code paths were re-audited. The
protocol is frozen in `docs/Q1_DEVELOPMENT_PROTOCOL_V1_1.md`; no V1.1 outcomes
were used to choose its controls.

- Local `pytest`: 50 passed; Ruff and `compileall` passed.
- Local preflight stayed offline: it did not instantiate the MMLU-Pro adapter,
  resolve a dataset, load a model, or download anything. It reported the
  expected local blockers for remote-only data/model availability.
- The frozen V1.1 condition table is 31 conditions: 19 original-order and 12
  option-permutation conditions, for 15,872 item-condition evaluations and
  158,720 candidate forward passes.
- The deterministic cost estimate is 183.799 minutes and US$1.2253 under the
  explicitly recorded US$0.40/A40-hour assumption, below the US$2 stop gate.
- Commit `6e8e6d1` was pushed to the authorized GitHub branch and synchronized
  to RunPod. The sync is additive and excludes `runs/`, `review/`, caches,
  model material, and secrets.
- Remote tests, Ruff, compileall, and V1.1 preflight passed on the A40 Pod.

The V1.1 real run is remote-only and must be reported separately from this
engineering audit. Its scientific status remains DEVELOPMENT; no V1.2, Q2, or
confirmatory holdout access is authorized.

## Serial V1.1 interruption and local optimization pivot (2026-08-16)

The first long serial V1.1 worker was stopped deliberately to avoid idle GPU
cost. The exact worker received SIGINT, did not exit within the grace window,
and then received SIGTERM. No SSH daemon, shell, or unrelated process was
terminated. The remote run was marked
`INTERRUPTED_FOR_INFERENCE_OPTIMIZATION`; it contained zero persisted
prediction rows because the old runner buffered rows until finalization.

Its metadata-only artifact was pulled to the ignored local directory
`review/serial_reference_v1_1_partial/` and hashed. It remains a provenance
record and is not a scientific result or a run to resume as final V1.1.

After the Pod was stopped, no remote commands were attempted. Local engineering
added the preserved `serial_reference` mode, prepared prompts, single-token
scoring, candidate-only head semantics, deterministic length planning, batched
activation extraction, cached decode, multi-token shared-prefix fallback,
crash-safe V1.1 journals, optional accelerator boundaries, and local
serial/optimized tiny-transformer tests. The local profile is explicitly
`TINY_RANDOM_TRANSFORMER_ENGINEERING_ONLY`.

Current local validation: 75 tests pass, Ruff passes, and the tiny profile
reports forward-call accounting.

## Post-restart Qwen optimization audit (2026-08-16)

The Pod was restarted later with a confirmed SSH host fingerprint. All model
operations remained remote-only under the pinned cache. The Qwen3 A–J
candidate audit found single-token, context-compatible continuations. The
serial-shape `full_prompt_batched` profile with `candidate_only` passed the
512-item DEV baseline/PC1−/PC1+ equivalence gate with zero discrete prediction
differences.

Ordinary BF16 cached decode and shape-changing batching were also exercised,
but they produced prediction flips against the serial oracle and were rejected
as canonical. Native Qwen3 suffix replay matched the cached path on technical
smoke but was slower, so it remains an optional guarded engine. The approved
profile completed a clean 31-condition V1.1 DEVELOPMENT run: 15,872 rows,
validated hashes, no duplicate scientific keys, and no confirmatory holdout
access. The observed runtime was approximately 17.48 minutes versus the old
183.80-minute estimate, about 10.52×.

The run's full predictions remain on RunPod. Only small review artifacts are
stored in `review/q1_v1_1_optimized_clean_run/`. No Q1 scientific conclusion
is frozen; no V1.2 or Q2 experiment was run.

## V1.2 local implementation audit (2026-08-17)

The authorized V1.2 label/position-bias deconfounding protocol was implemented
locally while the Pod was stopped. The implementation freezes the exact cyclic
ordering, the three main conditions plus the finite-difference probe, centered
semantic-logit symmetrization, the probability-mean secondary aggregator,
paired metrics, descriptive bootstrap, six planned figures, raw score storage,
and the development firewall. It reuses original-order V1.1 rows only after
checking prompt, model/tokenizer, scorer, vector, alpha, layer, and token-scope
identity; otherwise it recomputes them.

The local validator now recomputes symmetrized scores and paired metrics from
raw JSONL and checks the exact DEV_EVALUATION ID set, cyclic grid, target
identity, candidate-score semantics, hashes, and required artifacts. A
synthetic one-item/ten-option validator fixture passed without model or dataset
access. Full local validation reached 81 tests, Ruff, compileall, mock smoke,
and offline V1.2 cost preflight. No V1.2 real-model data, holdout data, or
scientific result existed at this local-only checkpoint; the subsequent
remote execution is documented below.

At that checkpoint, the only remaining blocker was execution against the
already-cached Qwen3-8B and MMLU-Pro artifacts on RunPod.

## V1.2 remote preflight correction (2026-08-17)

The first remote V1.2 launch was stopped before model inference by a false
implementation guard requiring ten options for every item. A read-only audit
of the cached pinned MMLU-Pro revision found the frozen 512-item evaluation
distribution `K=3:1, K=4:35, K=5:2, K=6:3, K=7:4, K=8:12, K=9:30,
K=10:425`. The protocol itself specifies per-item `K`, so the guard and local
validator were repaired to use each item's exact cyclic length. No scientific
choice, item, score, model operation, or result was changed; the worker exited
before constructing the model.

## V1.2 remote development run (2026-08-17)

After the per-item correction, the pinned cached remote run completed at
`runs/q1_v1_2/20260817T012330Z_q1-v1-2-development_05cfe01b68`. The run used
the verified A40 and `HF_HOME=/workspace/hf-cache`; no model or dataset was
downloaded during execution. The remote validator recomputed all derived
artifacts and passed with 512 DEV items, 24,075 raw rows, 1,536 symmetrized
rows, six figures, and `confirmatory_accessed=NO`. The raw score SHA-256 is
`8a8c6b1afd4d019ee4fcb18dc0b787997218f0fcb8ddda6997239b68ae8cdd60`.

The local review bundle is `review/q1_v1_2_principal_review/` and its archive
is `review/q1_v1_2_principal_review.tar.gz`. This is descriptive DEVELOPMENT
evidence only; no Q1 claim, V1.3, Q2, or confirmatory result is frozen.
