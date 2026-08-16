# RunPod / DGX Spark guide

This procedure is intentionally conservative and does not assume a particular
CUDA image. Copy or clone the repository to a persistent volume, and keep model
caches and run artifacts outside ephemeral container storage when possible.

## 1. Persistent paths

Choose persistent paths appropriate to the machine, for example:

```bash
export CEG_ROOT=/workspace/causal-epistemic-geometry
export HF_HOME=/workspace/hf-cache
mkdir -p "$HF_HOME"
cd "$CEG_ROOT"
```

The checked-in helper is equivalent and is preferred on the Pod:

```bash
source scripts/runpod_environment.sh
```

Keep `/workspace/hf-cache` as the canonical persistent HuggingFace cache. Do
not silently fall back to `/root/.cache/huggingface` on a Pod.

Do not commit caches, tokens, or model files. Never print a HuggingFace token.

## 2. Environment and Torch check

If the image already provides a compatible Torch/CUDA build, preserve it. The
bootstrap script installs this package and development dependencies but does
not install or replace Torch:

```bash
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
ceg doctor
ceg storage-check
```

If Torch is missing, install a build matched to the image, driver, and Python
version according to the machine's approved procedure. Do not guess a CUDA
wheel in this repository's generic setup script.

## 3. Optional Transformers dependencies

After confirming Torch:

```bash
pip install -e ".[hf,dev]"
ceg doctor --config configs/runpod_qwen3_8b.example.yaml
```

`doctor` does not load or download the model. The example config is a template;
verify the model revision, layer path, and hidden size for the exact model you
intend to test.

Before any inference, use the no-download preflight:

```bash
ceg preflight configs/runpod_q1_smoke.example.yaml
```

It reports item count, generation calls, layer/alpha/token scope, cache status,
missing placeholders, vector path, and expected output root. A nonzero result
for the template is expected until the researcher fills in reviewed choices.

## 4. Explicit model cache/download step

Only after the principal researcher approves the exact open-weight model,
revision, cache path, and license should the model be downloaded or loaded.
Use the machine's approved HuggingFace authentication method if required, and
keep credentials out of the repository. A first load through the Transformers
backend may populate the configured cache; this is an explicit user action,
not part of `doctor` or mock smoke.

## 5. Build/load a steering vector

For a tiny real-model development path, edit a copy of the example config with
the real hidden size, layer, benchmark, and reviewed constructor. Then run:

```bash
ceg build-vector configs/runpod_qwen3_8b.example.yaml vectors/qwen3_contrast.npz
ceg inspect-vector vectors/qwen3_contrast.npz
```

The current difference-of-means constructor uses the configured backend's
activation policy. For a transformer this is the last non-padding token; no
silent token averaging occurs. The adjacent JSON records vector hash, layer,
construction, normalization, source IDs, extraction policy, and git commit
when supplied by a caller.

## 6. Tiny real-model smoke test

Use a checked-in tiny exact-label JSONL file first, a very small `max_items`,
`do_sample: false`, and a reviewed alpha. Run:

```bash
ceg doctor --config configs/runpod_qwen3_8b.example.yaml
ceg run configs/runpod_qwen3_8b.example.yaml
```

If the model does not produce mechanically parseable labels, fix the prompt or
parser during development; do not substitute fuzzy judging silently. If CUDA
OOM occurs, reduce items/max tokens or choose a smaller model/dtype. Do not
silently quantize, because quantization changes the model and is a scientific
choice.

## 7. Baseline-versus-steering development run

Once the tiny smoke path is understood, review a small development config with
one vector, one layer, one token scope, explicit generation settings, and exact
ground truth. Then:

```bash
ceg run path/to/reviewed_development.yaml
```

The current implementation runs items one at a time. This is deliberate while
hook and paired-state correctness are being established.

Runs are append-only and resumable at item-condition granularity. If a process
is interrupted, reuse the exact same config and pass the run directory:

```bash
ceg run path/to/reviewed_development.yaml --resume runs/<interrupted-run>
ceg validate-run runs/<completed-run>
```

Changing the resolved scientific config, model/vector identity, alpha, layer,
or token scope refuses resume.

For local-to-Pod transfer, use the checked-in SSH/rsync helpers described in
[CODEX_REMOTE_SSH.md](CODEX_REMOTE_SSH.md). They preserve Git history and do
not delete remote files by default.

## 8. Copying artifacts back

Copy only the run directory and vector metadata needed for review:

```bash
rsync -a runs/<timestamped-run>/ /path/to/local/review/runs/<timestamped-run>/
```

Review `manifest.json`, `predictions.jsonl`, `metrics.json`, and `summary.md`
together. The manifest records timestamp, seed, backend/model, config hash,
vector hash, git state, Python/package versions, and device metadata.
