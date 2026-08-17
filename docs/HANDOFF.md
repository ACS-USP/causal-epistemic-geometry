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

Q1 V1–V1.2 multiple-choice instrument series: CLOSED AS DEVELOPMENT. Its
artifacts remain available for audit, but no scientific result is frozen.

Q1 V2 E3-10 software infrastructure: READY locally for model-free generators,
exact oracles, views, balance, split, qualification recomputation checks, and
the v2 structural gate.

Q1 V2 pre-model structural gate: PASS. MODREG effective depth is monotonic;
MODREG10, FSM10, and eligible SATCOUNT10 cells may proceed to baseline-only
calibration. REACHCOUNT10 cells and SATCOUNT10 `vars4_clauses4` were excluded
by the frozen shallow-shortcut failure rule. No model outcome was used.

Q1 V2 real-transformer mechanics: Qwen tokenization audit PASS on the cached
Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; decimal and
number-word candidates were all unique context-compatible single tokens.
Baseline-only calibration then completed remotely with 6,600 view rows.

The frozen mechanical qualification rule returned zero qualifying cells across
the 11 scheduled cells. Canonical decimal accuracy was approximately chance in
every family/cell, and the decimal/word stability criterion also failed. The
suite therefore returned `E3_10_INSTRUMENT_NOT_QUALIFIED`. No fresh geometry,
development, or confirmatory splits were generated.

Q1 real 8B model: prior V1 development runs are closed; E3-10 baseline
calibration was run on Qwen, but no steering or scientific Q1 experiment was
run.

Q1 scientific result: NONE FROZEN.

Q1 V2 steering: NOT RUN. No E3-10 activation direction, PCA, or random control
has been constructed.

Q2 geometry: NOT RUN.

## What remains untested on a real deployment

The E3-10 Qwen path has now been exercised through tokenization and
baseline-only calibration. The complete CPU-audited review bundle is at
`review/q1_v2_instrument_review/` (ignored because it contains real model
outputs), including enriched per-view semantic rows, six figures, the complete
11-cell table, independent recomputation, and validator output. The V1/Qwen
engineering history is retained separately and was not used to qualify E3-10.

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

Do not conclude that steering creates useful diversity, that E3-10 will
qualify, that any vector is scientifically meaningful, or that geometry between
vectors predicts error covariance. E3-10 is software-ready, not a scientific
result.

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
