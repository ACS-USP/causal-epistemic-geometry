# Causal Geometry of Epistemic Complementarity

This repository is currently **DEVELOPMENT infrastructure**. It is a small,
deterministic harness for asking whether one controlled activation intervention
can change *where a frozen language model fails* while approximately preserving
individual competence. It does not contain a scientific result.

The first question is intentionally narrower than the long-term program:

```text
one frozen model f_theta
        |
        +--> baseline: h_l -> h_l
        |
        +--> treatment: h_l -> h_l + alpha * v_i
                    |
          same held-out items with exact ground truth
                    |
          paired error vectors e_0(t), e_i(t)
                    |
          accuracy + error similarity + rescue/damage trade-off
```

Q1 CURRENT: Can one activation intervention change the error profile while
preserving individual competence?

Q2 FUTURE: Does geometry between interventions, such as `cosine(v_i, v_j)`,
predict pairwise error complementarity?

The code is deliberately shaped so Q2 can be added later without making a
large pairwise experiment the current default.

## Quick start: mock mode

The mock path needs no GPU, model, network, or benchmark download. From this
repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ceg doctor
ceg run configs/mock_smoke.yaml
ceg preflight configs/mock_smoke.yaml
```

The same commands are available through `python -m epistemic_geometry.cli`.
Run artifacts are written under `runs/` and include resolved configuration,
paired predictions, metrics, a manifest, and `summary.md`.

```bash
make test
make lint
make smoke
python -m compileall -q src
```

With the optional Torch/Transformers stack installed, exercise the real
mechanics without downloading a model:

```bash
make tiny-smoke
ceg validate-run runs/<tiny-transformer-run>
```

Completed runs are resumable only with matching provenance:
`ceg run <config> --resume <interrupted-run>`.

The mock is a miniature representation-space classifier: deterministic item
representations are passed through a fixed linear readout, and steering is
literally applied before that readout. Named useful/destructive/random mock
vectors are software fixtures only. **MOCK RESULTS ARE SOFTWARE VALIDATION
ONLY.**

## Scientific status

The MMLU-Pro multiple-choice Q1 V1–V1.2 instrument series is formally closed
as DEVELOPMENT. Its artifacts remain preserved for audit, but its
estimator-sensitive results do not freeze a scientific claim. Q1 V2 is a new
instrument, E3-10: a procedural exact semantic ten-way answer space with no
answer-position slots or cross-rendering aggregation. See
[the closeout](docs/Q1_V1_SERIES_CLOSEOUT.md) and [the E3-10 design
protocol](docs/Q1_V2_EXACT_SEMANTIC_INSTRUMENT.md).

The primary summary always shows baseline accuracy, steered accuracy, delta
accuracy, error correlation, error Jaccard, rescue rate, damage rate, double
fault, and pair-oracle complementarity headroom together. Low error similarity
with collapsed accuracy is not a useful result. Pair-oracle headroom is a
diagnostic upper bound, not an implementable ensemble or majority-vote claim.

The repository starts with exact-label JSONL ground truth and a mechanical
normalizer. It stores raw model output and normalized output separately so
parsing failures cannot silently become model claims.

The real-transformer mechanics are exercised locally with a randomly
initialized two-layer GPT-2-style model built from config. This path is labeled
`TINY_RANDOM_TRANSFORMER` and is software validation only. It does not test
language capability or support Q1. The same path has also been exercised on a
live NVIDIA A40 CUDA device using
`configs/tiny_transformer_cuda_smoke.yaml`; that run validates deployment
mechanics only and is not a scientific result.

Current readiness:

```text
Q1 V1–V1.2 INSTRUMENT SERIES: CLOSED AS DEVELOPMENT
Q1 V2 E3-10 SOFTWARE: GENERATORS/ORACLES/VALIDATOR READY
Q1 V2 STRUCTURAL GATE: PASS (MODEL-FREE; Qwen NOT RUN)
Q1 V2 BASELINE CALIBRATION: NOT RUN (RunPod-only)
Q1 V2 QWEN TOKEN AUDIT: PENDING
Q1 SCIENTIFIC RESULT: NONE FROZEN
Q2 GEOMETRY: NOT RUN
CONFIRMATORY HOLDOUT: UNTOUCHED
```

The local E3-10 structural gate is model-free and uses 5,000 balanced items
per family/cell:

```bash
ceg validate-e3 --n-per-cell 500
MPLCONFIGDIR=/tmp/ceg-mpl python scripts/run_e3_structural_gate.py
```

The versioned bundle is written to
`review/q1_v2_instrument_design_v2/`. It records effective MODREG depth,
structural validity, target support, target-conditional features, shortcut
baselines, rejection efficiency, twins, and latent namespace leakage without
model outcomes. Real baseline calibration must run on
the explicitly approved remote model and stop for principal review before
steering, development evaluation, or the confirmatory holdout.
The committed `configs/q1_v2_structural_eligibility.json` is the fail-closed
allowlist used to keep structurally failed cells out of the future Qwen
calibration manifest.

The optimized execution path includes prepared prompts, single-token candidate
scoring, deterministic batching, prefix-cache and suffix-replay prototypes,
row-wise steering, batched activations, a shared-prefix multi-token fallback,
and crash-safe resume journals. The canonical Q1 V1.1 profile passed the
512-item × 3-condition exact discrete-equivalence gate against the serial
reference and completed 15,872 DEVELOPMENT rows in approximately 17.5 minutes
on the A40. See [Inference optimization](docs/INFERENCE_OPTIMIZATION.md).
Cache/decode alternatives remain preserved but are not canonical because their
BF16 shape changes produced prediction flips. The full V1.2 score artifacts
were later recovered locally, under ignored `review/` paths, solely for the
authorized analysis-only aggregator audit; they are not Git-tracked source.

The previous V1.2 artifacts remain historical DEVELOPMENT evidence only. Their
aggregator-sensitive result does not authorize V1.3 or Q2; the instrument is
closed rather than reinterpreted.

Real model/data operations are RunPod-only. The Mac is the canonical source
for code, configs, tests, documentation, and Git history; `scripts/sync_to_runpod.sh`
copies committed local state over the scoped SSH alias without using GitHub.
Weights, dataset contents, activations, and real predictions remain on the
RunPod cache/workspace, except for the explicitly authorized V1.2 raw-score
recovery under ignored `review/` paths for analysis-only audit. See
[Q1 V1 results](docs/Q1_V1_RESULTS.md).

## Optional HuggingFace path

Torch is intentionally not forced by the generic package because CUDA builds
are machine-specific. On a prepared GPU machine, install the existing
compatible Torch build first, then:

```bash
pip install -e ".[hf,dev]"
ceg doctor --config configs/runpod_qwen3_8b.example.yaml
```

The example config is not executed locally and does not download Qwen3-8B.
It makes model ID/path, dtype, device map, layer, alpha, token scope,
generation, and benchmark path explicit. See [docs/RUNPOD.md](docs/RUNPOD.md)
for the deliberately boring setup procedure.

For a saved vector:

```bash
ceg build-vector configs/runpod_qwen3_8b.example.yaml vectors/qwen3_contrast.npz
ceg inspect-vector vectors/qwen3_contrast.npz
ceg run configs/runpod_qwen3_8b.example.yaml
```

The vector command is an explicit model-loading action. The default mock
workflow does not call it.

## What is not claimed yet

This project does not currently claim that activation steering creates useful
diversity, that one selected vector is scientifically privileged, or that
representation geometry predicts error covariance. It has not frozen a model,
benchmark split, vector construction, layer, alpha, controls, or confirmatory
hypothesis. Those decisions belong after development review. See:

- [Scientific question](docs/SCIENTIFIC_QUESTION.md)
- [Development protocol](docs/DEVELOPMENT_PROTOCOL.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Next Q2 geometry](docs/NEXT_Q2_GEOMETRY.md)
- [RunPod guide](docs/RUNPOD.md)
- [RunPod Q1 checklist](docs/RUNPOD_Q1_CHECKLIST.md)
- [Pre-RunPod audit](docs/PRE_RUNPOD_AUDIT.md)
- [RunPod cost gates](docs/RUNPOD_COST_GATES.md)
- [Codex Remote SSH workflow](docs/CODEX_REMOTE_SSH.md)
- [Legacy RunPod workflow audit](docs/OLD_RUNPOD_WORKFLOW_AUDIT.md)
- [Handoff](docs/HANDOFF.md)
