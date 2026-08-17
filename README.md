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
Q1 SOFTWARE: READY
Q1 REAL-TRANSFORMER MECHANICS: VALIDATED ON TINY MODEL AND QWEN3/A40 TECHNICAL SMOKE
Q1 REAL 8B MODEL: V1.1 DEVELOPMENT RUN COMPLETE; NO CONFIRMATORY RESULT
Q1 SCIENTIFIC RESULT: NONE FROZEN
Q1 V1.2: DEVELOPMENT RUN COMPLETE / REVIEW BUNDLE LOCAL / NO CLAIM FROZEN
Q2 GEOMETRY: NOT RUN
```

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

V1.2 is the explicitly authorized development follow-up for label/position-bias
deconfounding. It uses exact cyclic option balance, centered semantic-logit
aggregation, a secondary probability aggregator, and a pre-specified finite-
difference slot-tracking probe. See
[Q1 DEVELOPMENT PROTOCOL V1.2](docs/Q1_DEVELOPMENT_PROTOCOL_V1_2.md). It has
now completed the frozen 512-item DEV_EVALUATION run on the pinned remote
Qwen3-8B/A40 path. The remote validator recomputed the derived artifacts and
reported `COMPLETE`, 24,075 raw rows, 1,536 symmetrized rows, and no holdout
access. The small principal-review bundle is in
`review/q1_v1_2_principal_review/`; this remains DEVELOPMENT evidence only and
does not freeze a scientific claim. The complete raw-score audit bundle is in
`review/q1_v1_2_principal_review_complete/`; see
[the aggregator audit](docs/Q1_V1_2_AGGREGATOR_AUDIT.md). Its descriptive
classification is aggregator-sensitive and does not authorize V1.3 or Q2.

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
