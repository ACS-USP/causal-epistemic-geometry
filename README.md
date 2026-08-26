# Causal Geometry of Epistemic Complementarity

This repository is DEVELOPMENT infrastructure for a staged research program on
whether internal interventions can create useful, competence-preserving changes
in a frozen model's semantic error profile.

<!-- PROJECT_STATUS:START -->
**Current stage:** `Q2_V4_1_31_SAFE_BANK_ADEQUATE` — **scientific claim:** `NONE_FROZEN`.

Q2 V4.1 performed a CPU-only, outcome-free adequacy review of the complete historical 40-candidate bank and all 31 directions that passed both frozen safety shells. The 31-safe bank retained full rank, effective rank 7.225679, condition number 2.021583, and all inherited coverage checks. Synthetic K=31/N=300 planning retained 97.17% omnibus power at rho=0.25, with 1.33 percentage points loss versus K=32 and A2 width ratio 1.022. The bank is adequate for a future V4.1 design, but no semantic experiment was run.

**Next authorized action:** PRINCIPAL_RESEARCHER_REVIEW
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
- Gate 3 substrate race: baseline-only exploration complete; Qwen full
  non-thinking × CRUXEval semantic is the provisional primary development
  substrate, with fresh long character count as backup. No steering was run.
- Gate 4 first original micro-Q1: audited bounded null; the one-shot
  careful-minus-direct direction did not exceed baseline resampling and the
  norm-matched random control.
- Gate 5 source/duration bridge: textual careful/direct source separation and
  sustained manipulation passed their frozen gates, but the 60-item primary
  evaluation was below the frozen movement threshold.
- Gate 6.3 single-mean semantic evaluation: matched architecture-specific
  random gate passed, but the frozen L27 meaningful controller failed the
  primary validity guard (0.9083 versus 0.9250) and is classified
  `GATE6_3_SINGLE_MEAN_DESTRUCTIVE`. A later condition-blind, model-free V3
  audit preserved that historical classification but found 0.9750 commitment
  validity/evaluability and a strong controller-specific G/C/D diagnostic.
  Gate 7 then tested the exact controller on 120 fresh items with semantic V3
  frozen before collection. Accuracy and G/C/D improved beyond four new random
  controls, but commitment/evaluability fell to 0.9000 and violated the frozen
  relative guard; classification: `GATE7_DESTRUCTIVE`.
- Gate 8 prospectively calibrated the same controller and selected D75 as the
  lowest safe dose with a specific behavioral first stage. Gate 9 then tested
  D75 on 100 fresh CRUXEval items and obtained a strong safe DEVELOPMENT
  replication beyond four new random controls.
- Gate 10 transported the fixed L27-D75 controller without adaptation to 200
  fresh long character-count items. Baseline opportunity and safety passed,
  but G/C/D were negative and below the random null; classification:
  `GATE10_NO_CROSS_DOMAIN_TRANSFER`. This bounds domain generality without
  overturning the fresh CRUXEval replication.
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
