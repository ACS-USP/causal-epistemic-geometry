# RunPod cost gates

These gates are operational safeguards. They do not choose the model,
benchmark, vector, layer, alpha, or scientific hypothesis.

## Gate 0 — local, $0 expected

Before starting a paid Pod, run:

```bash
make predeploy
```

This covers tests, lint, compile checks, mock and tiny-transformer smoke,
hook/vector/resume tests, artifact validation, and placeholder preflight. A
failed software check means the GPU is not the right debugging environment.

## Gate 1 — Pod technical setup, short

After SSH works:

```bash
source scripts/runpod_environment.sh
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
ceg doctor
ceg storage-check
```

Do not download a model at this gate. Confirm GPU visibility, persistent cache
location, package installation, and tiny technical smoke first.

## Gate 2 — reviewed real-model technical smoke

Only after explicit researcher approval, cache the chosen open-weight model at
`/workspace/hf-cache`. Run 8–16 technical exact-label items baseline-only,
then alpha-zero, zero-vector, and one reviewed nonzero-vector checks. Inspect
raw outputs, parse statuses, provenance, and validated artifacts.

## Gate 3 — small Q1 DEVELOPMENT run

Proceed only if Gate 2 is mechanically clean. Use one frozen model, one
reviewed benchmark/configuration, one vector, one layer, one alpha, and exact
paired artifacts. This remains development infrastructure; it is not a
confirmatory result.

## Gate 4 — larger development work

Requires principal-researcher review of the frozen configuration, controls,
seeds, exclusion rules, and budget. No script crosses this gate automatically.

## Stop rule

Never use a paid GPU to debug a hook, parser, vector serializer, resume path,
or artifact validator that can be tested locally. Stop or terminate the Pod
only after [BEFORE_TERMINATING_POD.md](BEFORE_TERMINATING_POD.md) is complete.
