# Pre-RunPod handoff

## Gate 1 and Gate 2 result — 2026-08-16

Gate 1 is complete for software and CUDA mechanics. The live Pod was used only
with a randomly initialized tiny GPT-2-style Transformer built from config.
Gate 2 then ran one fixed Qwen3-8B/MMLU-Pro-derived DEVELOPMENT pilot. It did
not access the confirmatory holdout and does not establish Q1.

## What changed

The optional HuggingFace path was exercised both locally and on the Pod by a
network-free two-layer random GPT-2-style Transformer. The path supports
injected test models, robust layer discovery, explicit plain/chat prompt modes,
parse statuses, detailed model/vector/intervention provenance, deterministic
bootstrap diagnostics, append-only resumable prediction rows, atomic run
status, and `ceg validate-run`.

New commands/configs include:

```bash
ceg preflight configs/tiny_transformer_smoke.yaml
ceg run configs/tiny_transformer_smoke.yaml
ceg validate-run runs/<run>
ceg preflight configs/runpod_q1_smoke.example.yaml
```

RunPod connection and transfer helpers are explicit; the live Gate 1 run used
the configured `runpod-ceg` alias:

```bash
scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT --dry-run
scripts/check_runpod_connection.sh
scripts/sync_to_runpod.sh --dry-run
```

The alias is `runpod-ceg`; the persistent cache is `/workspace/hf-cache`.
See `RUNPOD_COST_GATES.md` before starting the reviewed pretrained-model gate.

## Local RunPod preparation

- Legacy and local SSH workflow audits are recorded in
  `OLD_RUNPOD_WORKFLOW_AUDIT.md` and `LOCAL_SSH_AUDIT.md`.
- `runpod-ceg` is configured and non-interactive SSH was confirmed.
- Push/pull helpers preserve Git history, exclude caches/models/runs/secrets,
  and avoid remote deletion by default.
- `/workspace/hf-cache` is the canonical cache; `ceg storage-check` reports
  persistent storage without deleting anything.
- `make predeploy` is the local cost gate. It passed all software checks.
- Remote bootstrap preserved the image's `torch==2.4.1+cu124`; the optional HF
  extra is constrained below Transformers 5 for that image.

## Bugs fixed

- GPT-2 `transformer.h` layer discovery failed because the explicit path was
  resolved only on the backend wrapper.
- Zero-variance phi correlation is now undefined/null with status instead of a
  silent numeric convention.
- Ambiguous generation output is now distinct from exact parser success.
- Same-config interrupted runs resume only when identity provenance matches.
- Truncated final JSONL tails are quarantined; duplicate keys and conflicting
  provenance fail loudly.
- The macOS-to-Pod transfer path now falls back safely to tar when remote
  `rsync` is absent, suppressing AppleDouble metadata and remote owner changes.
- Vector metadata exposes source IDs and model/tokenizer provenance explicitly.

## Validation performed

- 37 local tests pass, including actual Torch/Transformers mechanics and
  environment-independent SSH helper tests; the same 37 passed remotely.
- alpha-zero and zero-vector identity pass.
- exact hidden shift, last-token isolation, all-token shift, hook cleanup, and
  repeated intervention isolation pass.
- activation extraction, padding policy, difference-of-means, vector roundtrip,
  metadata, and hash checks pass.
- tiny random transformer end-to-end CPU and CUDA runs and `validate-run` pass.
- interrupted/resumed predictions and metrics equal an uninterrupted run.
- focused remote resume/hook/artifact tests pass (15 tests).
- A single 8-item CUDA technical artifact was pulled locally and validated.
- `ceg preflight` reports the expected missing placeholders for the reviewed
  Q1 template without inference or downloads.

## Gate 2 result

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` loaded on one
  A40 in BF16 with no quantization or CPU offload.
- MMLU-Pro revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` was accessed
  only on RunPod under `/workspace/hf-cache`.
- Technical 8-item smoke, alpha-zero/zero-vector identity, and exact serial
  score repeat passed.
- Validation baseline: 70 items, accuracy 0.5429, no parse failures; the
  30%/90% gate did not stop the pilot.
- Q1 V1: 512 calibration, 512 development evaluation, exactly 15 conditions,
  7,680 rows; artifact validation passed.
- Explicit repeat audit: 32 items, baseline + `pca_pc1_minus` + `random_0_minus`,
  96 rows, max score difference 0.0 at tolerance 1e-5.
- Full descriptive results: [Q1_V1_RESULTS.md](Q1_V1_RESULTS.md).

## Not tested

No confirmatory experiment, holdout analysis, Q2 geometry, committee/majority
vote, or scientific claim was run. Free generation was not the primary Q1
scorer. The tiny random transformer report is software validation only.

## Remote facts

- Ubuntu 22.04, Python 3.11.10, x86_64.
- NVIDIA A40, 46,068 MiB, driver 580.159.04; one CUDA device visible.
- `torch==2.4.1+cu124`, `transformers==4.57.6`, CUDA available, bf16 support
  reported by Torch.
- Gate 1 itself downloaded no pretrained model. Gate 2 later downloaded the
  authorized Qwen3-8B only on RunPod; the cache remains remote.
- Codex CLI was not installed on the Pod. The existing SSH path is ready for a
  desktop Remote SSH session at `/workspace/causal-epistemic-geometry`.

## Exact next commands

Local mechanics:

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
make tiny-smoke
ceg validate-run "$(find runs -maxdepth 1 -type d -name '*tiny-random*' | sort | tail -n 1)"
# On a CUDA machine, the explicit technical fixture is:
ceg run configs/tiny_transformer_cuda_smoke.yaml
```

RunPod technical preparation (completed for the current Pod):

```bash
scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT --dry-run
scripts/configure_runpod_ssh.sh --host PUBLIC_IP --port PUBLIC_PORT
scripts/check_runpod_connection.sh
scripts/sync_to_runpod.sh --dry-run
ssh runpod-ceg
cd /workspace/causal-epistemic-geometry
source scripts/runpod_environment.sh
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
bash scripts/runpod_preflight.sh configs/runpod_q1_smoke.example.yaml
```

The researcher must choose and review the real model, benchmark, vector
construction, layer, token scope, alpha, and controls before any pretrained
model download or Q1 run. The unresolved questions are intentionally
scientific, not hook/resume/artifact engineering.

## V1.1 controlled follow-up status

The V1.1 protocol is now frozen and committed locally/GitHub. It reuses the
exact Q1 V1 model, dataset revision, split, vectors, layer, token scope, and
deterministic choice-log-likelihood scorer. It adds only the pre-registered
FP32 numerical audit, equal-norm random controls, and four deterministic option
permutations. It remains DEVELOPMENT and stops after V1.1.

Local preflight:

```bash
ceg preflight-q1-v1-1 configs/q1_v1_1_qwen3_8b.yaml
ceg preflight configs/q1_v1_1_qwen3_8b.yaml
```

The first command estimates the frozen workload/cost without inference. The
second is deliberately offline on the Mac and may report that the remote-only
model/dataset are unavailable locally.

Remote launch, only after reviewing the frozen protocol and cost gate:

```bash
cd /workspace/causal-epistemic-geometry
source scripts/runpod_environment.sh
ceg q1-v1-1 configs/q1_v1_1_qwen3_8b.yaml data/splits/mmlu_pro_q1_v1.json
ceg validate-run runs/q1_v1_1/<completed-run>
```

The real run must print and satisfy `hostname`, `/workspace/causal-epistemic-geometry`,
and `HF_HOME=/workspace/hf-cache` before model/data operations. It must not
access `confirmatory_holdout`. The final principal-review bundle is assembled
on the Mac only after remote validation and a read-only artifact pull.
Before terminating the Pod, pull artifacts with `scripts/sync_from_runpod.sh`
and follow `BEFORE_TERMINATING_POD.md`.

## Current optimization pivot

The Pod is intentionally stopped. Do not retry SSH until it is restarted and
the endpoint is confirmed. The serial V1.1 attempt was safely interrupted and
preserved as `review/serial_reference_v1_1_partial/`; it had no persisted
prediction rows and is not a scientific result.

Local engineering now includes serial reference, full-prompt batching, cached
one-token decode, a shared-prefix multi-token fallback, prepared prompts,
candidate token audits, candidate-only head provenance, row-wise hooks,
deterministic length buckets, batched activation capture, append/fsync
prediction journals, duplicate/conflict protection, tail quarantine, and V1.1
`--resume` support. Optional compile/CUDA-graph prototypes are disabled, and
Qwen3 suffix replay is strictly gated.

The exact local checks are:

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
pytest -q
ruff check .
python -m compileall -q src
python scripts/profile_tiny_engines.py
ceg preflight configs/runpod_q1_killtest.example.yaml
```

The remaining blocker is real-Qwen/A40 equivalence and performance. At that
point the local agent must print `RUNPOD_REQUIRED_FOR_EQUIVALENCE` and stop
remote attempts until the researcher restarts the Pod.
