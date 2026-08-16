# Handoff

## What works now

- Modern `src/` Python package with YAML validation and a typed domain model.
- Deterministic NumPy mock benchmark and representation-space classifier.
- Paired baseline/treatment experiment with exact item identity.
- Accuracy-aware error metrics, 2×2 outcomes, and markdown summaries.
- `.npz` vectors with adjacent JSON provenance and content hashes.
- Optional, dependency-gated HuggingFace causal-LM backend with temporary
  activation hooks, last-token/all-token scope, layer discovery, inference mode,
  and actionable OOM/dependency errors.
- Network-free `tiny_transformer` backend using an actual randomly initialized
  GPT-2-style decoder, with hook integration tests and an end-to-end smoke run.
- CLI commands: `doctor`, `run`, `build-vector`, `inspect-vector`, `summarize`.
- CLI commands: `preflight`, `validate-run`, `environment`, and
  `estimate-memory`.
- Reproducibility helpers, manifests, tests, linting, compile checks, and mock
  smoke workflow.
- Append-only resumable runs with status transitions, provenance checks, tail
  quarantine, deterministic metrics recomputation, and run validation.
- Local-only RunPod preparation: SSH alias helper, read-only connection check,
  additive rsync push/pull, persistent cache environment, storage diagnostics,
  and a no-cost pre-deploy gate.

## What is mocked

The local mock uses class prototypes, deterministic item noise, and a fixed
linear readout. Mock useful/destructive/random vectors are fixtures designed to
exercise rescue and damage paths. They are not model evidence, benchmark
evidence, or a simulated scientific result.

## Current readiness

Q1 software infrastructure: READY.

Q1 real-transformer mechanics: VALIDATED ON TINY RANDOM TRANSFORMER AND ON A
PINNED QWEN3-8B/A40 TECHNICAL PATH.

Q1 real 8B model: V1.1 DEVELOPMENT RUN COMPLETE with 15,872 rows under the
approved serial-shape/candidate-only engine; NO CONFIRMATORY CLAIM.

Q1 scientific result: NONE FROZEN; DESCRIPTIVE PILOT ARTIFACT ONLY.

Q2 geometry: NOT RUN.

## What remains untested on a real deployment

The Qwen3 path was exercised on the pinned revision and one A40. The exact
512-item equivalence gate passed for the canonical profile, and the complete
V1.1 DEVELOPMENT artifact was validated remotely. Ordinary cached decode and
shape-changing batching were deliberately not approved because they produced
BF16 prediction flips against the serial oracle. `torch.compile`, CUDA graphs,
formal batch autotuning, and a full utilization/VRAM profile remain optional
engineering work. The confirmatory holdout was not evaluated.

## Exact local smoke

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
ceg run configs/mock_smoke.yaml
ceg run configs/tiny_transformer_smoke.yaml
ceg run configs/tiny_transformer_cuda_smoke.yaml  # CUDA machine only
```

Or use `make smoke`. Run artifacts appear in `runs/`, which is ignored by Git.

## Exact RunPod setup

```bash
cd ~/dev/causal-epistemic-geometry
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
pip install -e ".[hf,dev]"  # after confirming a compatible existing Torch build
ceg doctor --config configs/runpod_qwen3_8b.example.yaml
ceg storage-check
```

On the validated Pod, Torch was `2.4.1+cu124`, CUDA was visible on one NVIDIA
A40, and Transformers `4.57.6` was installed without replacing Torch. Model
download/cache, vector creation, and the real smoke run are explicit
reviewed steps documented in [RUNPOD.md](RUNPOD.md) and
[RUNPOD_Q1_CHECKLIST.md](RUNPOD_Q1_CHECKLIST.md).

## Vector and experiment commands

```bash
ceg build-vector path/to/config.yaml vectors/example.npz
ceg inspect-vector vectors/example.npz
ceg run path/to/config.yaml
ceg q1-v1 configs/q1_v1_qwen3_8b.example.yaml data/splits/mmlu_pro_q1_v1.json
ceg q1-v1-1 configs/q1_v1_1_qwen3_8b.yaml data/splits/mmlu_pro_q1_v1.json
ceg validate-run runs/q1_v1/<completed-run>
ceg repair-q1-v1-1 runs/q1_v1_1/<failed-row-complete-run>  # no inference
ceg summarize runs/<run-directory>
```

## Metric meanings

Accuracy and delta accuracy measure individual competence. Phi error
correlation and error Jaccard measure paired error similarity. Rescue rate is
the fraction of baseline errors corrected by treatment; damage rate is the
fraction of baseline successes lost. Double fault is the joint error rate.
Pair-oracle accuracy and complementarity headroom show only the potential that
at least one copy is correct; they are not a deployed ensemble.

## What not to conclude yet

Do not conclude that steering creates useful diversity, that the chosen vector
is scientifically meaningful, or that geometry between vectors predicts error
covariance. This repository currently validates software and enables the Q1
development kill-test only.

## Q1 kill criterion

An intervention should not advance merely because error similarity falls. If
accuracy collapses, or if apparent movement vanishes under controls and exact
paired reruns, Q1 is a kill or redesign outcome. A plausible survivor preserves
individual competence approximately while producing transparent, reproducible
error-profile movement.

## Q2 if Q1 survives

Add multiple reviewed interventions and small pure geometry helpers, then test
whether `geometry(v_i, v_j)` predicts held-out
`error_correlation(e_i, e_j)`. Do not add manifold or Riemannian theory before
that milestone.
