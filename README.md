# Causal Geometry of Epistemic Complementarity

This repository is DEVELOPMENT infrastructure for a staged research program on
whether internal interventions can create useful, competence-preserving changes
in a frozen model's semantic error profile.

<!-- PROJECT_STATUS:START -->
**Current stage:** `DEVELOPMENT` — **scientific claim:** `NONE_FROZEN`.

Gate 3 is authorized as a bounded, baseline-only incremental race between Qwen3-8B full non-thinking and Llama-3.1-8B-Instruct on two fresh common objective instruments. The purpose is development substrate selection; original activation steering, geometry, Q2, and the confirmatory holdout remain forbidden.

**Next authorized action:** Execute the frozen Gate 3 Stage 1 -> Stage 2 -> at most two-cell Stage 3 substrate race after local manifests, model access, and cost checks pass.
<!-- PROJECT_STATUS:END -->

The machine-readable source for this block and the live status page is
[`project_state.yaml`](project_state.yaml). Run `make state-check` to detect
stale generated status.

## The research program

```text
representation / intervention geometry
                  |
                  v
       controlled causal intervention
                  |
                  v
 semantic error profile and covariance
                  |
                  v
   realizable committee utility (later)
```

The minimal Q1 is deliberately narrower:

> Can one controlled activation intervention change where a frozen model fails
> beyond ordinary stochastic resampling without merely making it worse?

Q2 asks whether geometry between interventions predicts geometry between error
profiles. Q3 would ask whether a realizable committee can exploit that
complementarity. Behavioral difference, semantic error difference, useful
complementarity, and implementable ensemble gain are four different claims.

No one of them is currently established.

## Start here

For a first visit, read:

1. [Current status](docs/CURRENT_STATUS.md) — generated live state.
2. [Scientific constitution](docs/SCIENTIFIC_CONSTITUTION.md) — questions,
   claim boundaries, and stop rules.
3. [Instrument history](docs/INSTRUMENT_HISTORY.md) — what V1–V4 actually
   established and why each instrument closed or paused.
4. [Metrics and statistics](docs/METRICS_AND_STATISTICS.md) — hard errors,
   stochastic estimands, uncertainty, and seed semantics.
5. [Experiment ladder](docs/EXPERIMENT_LADDER.md) — the prospective sequence
   from cheap smoke to micro-Q1 and future Q2.
6. [Document index](docs/DOCUMENT_INDEX.md) — canonical versus archival files.

The complete machine-readable experiment history is
[`experiments/registry.yaml`](experiments/registry.yaml). Historical protocol
and review files remain preserved for audit; they are not silently rewritten
as current plans.

## Current evidence boundary

- V1–V1.2/MMLU-Pro: closed DEVELOPMENT instrument; aggregation-sensitive.
- V2/E3-10: closed non-qualified direct-readout ablation.
- V3 reasoning agent: baseline Stage A complete; frozen screen failed; no
  steering.
- External benchmark search: development diagnostics only; no qualifier.
- V4 character count: closed after semantic saturation and parser artifacts.
- V4 geometry: tiny descriptive activation diagnostic only; no causal or
  behavioral result.
- V4 dense code: objective nested outcomes look promising, but secure execution
  is not production-ready and no model pilot ran.
- Published weekday positive control: PASS under its frozen metric and endpoint
  criteria; this validates one known intervention stack, not original Q1 or Q2.
- Confirmatory holdout: untouched.

These are instrument outcomes, not a positive or negative result for the full
causal-geometry theory.

## Lightweight local workflow

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

make test
make lint
python -m compileall -q src scripts
make scientific-audit
```

Mock mode requires no GPU, model, network, or benchmark download:

```bash
ceg doctor
ceg run configs/mock_smoke.yaml
ceg preflight configs/mock_smoke.yaml
```

The mock linear fixture and the network-free tiny random transformer validate
software mechanics only. Their outputs are not scientific evidence.

## Execution policy

The local Mac is canonical for code, configuration, tests, documentation, and
Git history. Real model inference belongs on an explicitly authorized remote
GPU host. Local Hugging Face model downloads are fail-closed. RunPod, Docker,
and generated-code execution are not part of the current offline
rearchitecture.

The repository distinguishes three regimes:

- `EXPLORATION`: cheap 5–20 item screens designed to fail quickly.
- `DEVELOPMENT_LOCK`: freeze code, estimands, seeds, schemas, cost, and review
  after a real signal appears.
- `CONFIRMATORY`: sealed estimands, source commit, validator, and holdout with
  no adaptive tuning.

See [engineering policy](docs/ENGINEERING_POLICY.md) for reproducibility,
engine-equivalence, security, and artifact rules.

## Package shape

```text
src/epistemic_geometry/
    backends/       mock, Hugging Face, and tiny-transformer mechanics
    benchmarks/     exact task adapters and historical instruments
    experiments/    paired and reasoning runners
    inference/      reference and optimized execution utilities
    metrics/        paired, stochastic, and uncertainty estimands
    steering/       vector artifacts and temporary interventions

experiments/
    registry.yaml   canonical history and status
    decision_log.yaml
    specs/           prospective, not-yet-executed protocols
```

## What not to conclude

This repository does not currently show that activation steering creates useful
diversity, that a privileged geometric direction exists, that intervention
geometry predicts error covariance, or that an oracle pair gain is realizable
by an ensemble. Negative instrument screens are useful engineering evidence,
but they do not settle the underlying theory.

## License

The package metadata currently declares MIT licensing. Model and benchmark
assets retain their own licenses and are not stored in this repository.
