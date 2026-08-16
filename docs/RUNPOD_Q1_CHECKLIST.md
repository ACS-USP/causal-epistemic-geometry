# RunPod Q1 checklist

This is a technical checklist for a future small DEVELOPMENT run. It does not
choose the model, benchmark, layer, vector, or alpha for the researcher.

## A. Instance and environment

```bash
cd /workspace/causal-epistemic-geometry
source scripts/runpod_environment.sh
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
ceg doctor
ceg storage-check
```

If the image already provides Torch, do not replace it blindly. Install a
machine-compatible Torch build only through the approved environment procedure.

## B. Review before model download

```bash
ceg doctor --config configs/runpod_q1_smoke.example.yaml
ceg preflight configs/runpod_q1_smoke.example.yaml
```

Resolve every `REPLACE_ME`, choose an exact model revision, and confirm the
license/cache policy. Preflight never downloads weights.

## C. Explicit model cache/download

After review, perform the approved HuggingFace cache/download step. Keep tokens
outside the repository. Do not continue if the selected model or revision is
not the one reviewed.

```bash
ceg doctor --config path/to/reviewed_q1_smoke.yaml
ceg preflight path/to/reviewed_q1_smoke.yaml
```

The cache path should be persistent. No quantization is implicit; if later
chosen, it must be recorded as a scientific model change.

## D. Technical smoke without and with steering

First run a tiny exact-label benchmark with steering disabled or alpha zero,
then review raw outputs and parse statuses. Build or copy a vector only after
the model/tokenizer/layer provenance is recorded.

```bash
ceg run path/to/reviewed_baseline_smoke.yaml
ceg build-vector path/to/reviewed_q1_smoke.yaml vectors/q1.npz
ceg inspect-vector vectors/q1.npz
ceg run path/to/reviewed_q1_smoke.yaml
```

Validate the artifacts:

```bash
ceg validate-run runs/<completed-run>
```

## E. Small Q1 development run

Only after the technical smoke is understood:

```bash
ceg preflight path/to/reviewed_q1_killtest.yaml
ceg run path/to/reviewed_q1_killtest.yaml
```

If interrupted, resume the exact same resolved config:

```bash
ceg run path/to/reviewed_q1_killtest.yaml --resume runs/<interrupted-run>
```

Changing alpha, vector, model, layer, token scope, or other scientific config
must refuse resume and requires a new run directory.

## F. Copy artifacts and stop

Copy `manifest.json`, `config_resolved.yaml`, `predictions.jsonl`,
`metrics.json`, `summary.md`, and vector metadata home. Review status,
provenance, parse counts, 2×2 outcomes, accuracy, and complementarity together.
Stop the instance when the approved technical/development work is complete.
Before termination, pull artifacts from the Mac and follow
[BEFORE_TERMINATING_POD.md](BEFORE_TERMINATING_POD.md).
