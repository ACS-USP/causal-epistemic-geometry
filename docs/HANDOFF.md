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
- CLI commands: `doctor`, `run`, `build-vector`, `inspect-vector`, `summarize`.
- Reproducibility helpers, manifests, tests, linting, compile checks, and mock
  smoke workflow.

## What is mocked

The local mock uses class prototypes, deterministic item noise, and a fixed
linear readout. Mock useful/destructive/random vectors are fixtures designed to
exercise rescue and damage paths. They are not model evidence, benchmark
evidence, or a simulated scientific result.

## What remains untested on a real transformer

The HuggingFace code path has not been run in this local smoke workflow. A
principal researcher must verify the exact model revision, tokenizer behavior,
layer path, hidden size, device map, dtype, generation format, and vector
construction on a prepared GPU machine.

## Exact local smoke

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
ceg run configs/mock_smoke.yaml
```

Or use `make smoke`. Run artifacts appear in `runs/`, which is ignored by Git.

## Exact RunPod setup

```bash
cd ~/dev/causal-epistemic-geometry
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
pip install -e ".[hf,dev]"  # after confirming a compatible existing Torch build
ceg doctor --config configs/runpod_qwen3_8b.example.yaml
```

Model download/cache, vector creation, and the real smoke run are explicit
reviewed steps documented in [RUNPOD.md](RUNPOD.md).

## Vector and experiment commands

```bash
ceg build-vector path/to/config.yaml vectors/example.npz
ceg inspect-vector vectors/example.npz
ceg run path/to/config.yaml
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

