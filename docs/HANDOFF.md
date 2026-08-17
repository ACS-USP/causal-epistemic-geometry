# Handoff

> **Live-status note:** this document is a capability and command handoff, not
> the live experiment dashboard. Q1 V3 Stage A is now running baseline-only on
> RunPod. For the current state, read [CURRENT_STATUS.md](CURRENT_STATUS.md).

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

Q1 V1–V1.2 multiple-choice instrument series: CLOSED AS DEVELOPMENT. Its
artifacts remain available for audit, but no scientific result is frozen.

Q1 V2 / E3-10 direct instrument: CLOSED — NOT QUALIFIED. The direct
first-response semantic-logit calibration did not provide a stable measurement
channel on Qwen3-8B. This is an instrument ablation, not a result about
procedural reasoning or steering. See
[Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md](Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md).

Q1 V3 reasoning-agent software: READY locally for procedural generators,
exact oracles, deterministic surface twins, stable rollout seeds, raw
trajectory records, exact FINAL parsing, repeated-rollout metrics, and the
Stage A/B manifest/firewall tooling.

Q1 V3 model-free structural gate: PASS on at least 5,000 generated items per
family/cell. All current cells are eligible; two low-complexity SAT cells have
documented shortcut warnings, not failures. No model outcome was used.

Q1 V3 Stage A: RUNNING baseline-only. Q1 V3 Stage B: NOT RUN. Q1 V3 steering:
NOT READY and NOT RUN. The active run uses the frozen scientific execution
commit `4faea97`; no result is available yet. The current review bundle is
`review/q1_v3_reasoning_instrument/` (ignored local artifact). See
[CURRENT_STATUS.md](CURRENT_STATUS.md) for the remote journal and live state.

Q1 real 8B model: prior V1 development runs and the E3-10 baseline calibration
are closed; Q1 V3 Stage A is the first active V3 baseline-only calibration.

Q1 scientific result: NONE FROZEN.

Q1 V2 steering: NOT RUN. No E3-10 activation direction, PCA, or random control
has been constructed.

Q2 geometry: NOT RUN.

## What remains untested on a real deployment

The E3-10 Qwen path and baseline-only calibration are preserved at
`review/q1_v2_instrument_review/` (ignored because it contains real model
outputs). The V1/Qwen engineering history is retained separately and was not
used to qualify Q1 V3. The new Q1 V3 bundle contains no model outputs. The
active Q1 V3 Stage-A outputs remain on RunPod until the run completes and the
artifact is explicitly pulled and validated.

## Exact local smoke

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
ceg run configs/mock_smoke.yaml
ceg run configs/tiny_transformer_smoke.yaml
ceg run configs/tiny_transformer_cuda_smoke.yaml  # CUDA machine only
```

Or use `make smoke`. Run artifacts appear in `runs/`, which is ignored by Git.

## Exact E3-10 local gate

```bash
ceg validate-e3 --n-per-cell 500
MPLCONFIGDIR=/tmp/ceg-mpl python scripts/run_e3_structural_gate.py
```

This is model-free and performs no network operation. The full 5,000-item
per-cell bundle is in `review/q1_v2_instrument_design_v2/`; the prior v1
design artifact remains archival. The executed remote calibration and its
CPU-only audit are in the ignored `review/q1_v2_instrument_review/` bundle.
Because the suite failed the frozen qualification rule, the manifest builder
did not generate fresh scientific splits and the next action is principal
researcher review, not steering.

## Exact Q1 V3 local gate

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
python scripts/run_q1_v3_structural_gate.py --n-per-cell 5000
python scripts/build_q1_v3_design_artifact.py
python scripts/build_q1_v3_calibration_manifests.py stage_a \
  --gate review/q1_v3_reasoning_instrument/structural_gate_summary.json \
  --output review/q1_v3_reasoning_instrument/stage_a_manifest.json
```

This is model-free. It creates 36 Stage-A budget conditions (12 cells × 3
budgets) over 12 frozen 60-item latent sets. It does not load Qwen, construct
steering, or touch the future scientific splits. Principal review was required
before the current RunPod command and has been completed. Do not regenerate the
active manifest while Stage A is running.

After principal review and only on RunPod with the pinned cache/model, the
baseline-only calibration runner is:

```bash
python scripts/run_q1_v3_calibration.py \
  configs/q1_v3_reasoning_instrument.example.yaml \
  --manifest review/q1_v3_reasoning_instrument/stage_a_manifest.json \
  --manifest-key MODREG-R/depth_4/512 \
  --output runs/q1_v3_stage_a/modreg_depth4_512
```

Omit `--manifest-key` only when deliberately running every Stage-A manifest.
The runner loads one model, performs baseline reasoning rollouts only, stores
raw trajectories and parse records, and writes mechanical outcomes. It never
constructs a steering direction. The model-loading guard refuses to run on the
Mac.

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
ceg q1-v1-2 configs/q1_v1_2_qwen3_8b.yaml data/splits/mmlu_pro_q1_v1.json
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

Do not conclude that steering creates useful diversity, that Q1 V3 will
qualify, that any vector is scientifically meaningful, or that geometry between
vectors predicts error covariance. E3-10 is a closed ablation; Q1 V3 is still
pre-calibration infrastructure, not a scientific result.

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
