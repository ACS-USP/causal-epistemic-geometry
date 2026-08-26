# CIAAM DGX Spark bring-up report

Report date: 2026-08-25 (America/Sao_Paulo)

Source base: `2fbcc98fa6026b844074bdaef2ba311b360b7f81`

Branch: `infra/dgx-spark-bringup`

This report distinguishes administrator-supplied expectations from observed
evidence. Both SSH servers were reached under normal strict host-key checking,
the user completed first-login password rotation independently, and the user
authorized an existing Ed25519 public key on both nodes. No password was handled
by the tooling.

## A. Access status

| Node | SSH transport | Key authentication | Hostname | Architecture |
| --- | --- | --- | --- | --- |
| Spark 1 (`200.144.205.82:1232`) | PASS | PASS, Ed25519 | `spark1` | `aarch64` |
| Spark 2 (`200.144.205.83:1232`) | PASS | PASS, Ed25519 | `spark2` | `aarch64` |

Manual password rotation: confirmed complete by the user. Credentials are not
recorded. Batch-mode key authentication succeeds on both nodes.

Suggested non-destructive SSH config:

```text
Host ciaam-spark1
    HostName 200.144.205.82
    User gabriel.alexandre
    Port 1232

Host ciaam-spark2
    HostName 200.144.205.83
    User gabriel.alexandre
    Port 1232
```

No insecure host-key option is needed.

## B. Hardware census

| Property | Spark 1 | Spark 2 |
| --- | --- | --- |
| GPU | 1 x NVIDIA GB10, capability 12.1 | 1 x NVIDIA GB10, capability 12.1 |
| Architecture | `aarch64` | `aarch64` |
| CPU | 20 ARM cores: 10 Cortex-X925 + 10 Cortex-A725 | Same |
| Total / idle available memory | 121 GiB / 115 GiB | 121 GiB / 116 GiB |
| Exact total bytes | 130,663,170,048 | 130,663,165,952 |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| Kernel | `6.17.0-1026-nvidia` | `6.17.0-1026-nvidia` |
| Driver | 580.159.03, open aarch64 kernel module | Same |
| Driver-advertised CUDA | 13.0 | 13.0 |
| `nvcc` | Not installed | Not installed |
| Disk | 3.7 TB total, 3.2 TB available, 320 GB used | 3.7 TB total, 3.2 TB available, 331 GB used |
| System Python | 3.12.3 at `/usr/bin/python3` | Same |

The Python doctor collects hostname, OS, kernel, `lscpu`, `/proc/meminfo`,
`nvidia-smi`, driver text, `nvcc`, Python packages, native NVIDIA packages,
shared storage, network interfaces, Git HEAD, and canonical SHA-256 fingerprint.

## C. PyTorch status

| Node | Torch | CUDA | GB10 detected | BF16 arithmetic | CPU/GPU operation |
| --- | --- | --- | --- | --- | --- |
| Spark 1 | 2.13.0+cu130 | 13.0 | PASS | PASS | PASS |
| Spark 2 | 2.13.0+cu130 | 13.0 | PASS | PASS | PASS |

The bounded smoke used deterministic CPU/FP32, CUDA/FP32, actual CUDA/BF16,
repeat-seed checks, and a touched 512 MiB allocation. Every arithmetic check
passed on both nodes. Torch reported the full system-memory total as device
memory; the allocation reduced Linux `MemAvailable`, empirically confirming the
shared pool. Library loading/cache effects make the before/during/after delta
descriptive rather than an exact allocator accounting. A simple repeat does not
establish deterministic Transformer generation.

## D. Environment symmetry

Current classification: `SPARK_NODES_ENVIRONMENT_DIFFER`.

OS, kernel, CPU topology, driver, driver CUDA, Python, firmware-visible GPU,
compute capability, Torch/CUDA stack, Python package freeze, Git commit, and
RoCE/NCCL availability match. Exact differences:

- Spark 1 exposes 4,096 more memory bytes;
- Spark 1 alone has `nvidia-spark-avahi-conf 1.0-1`,
  `nvidia-spark-limits 1.0-1`, and `nvidia-spark-ota-check 1.0.16-1`;
- package versions differ for `nvidia-ai-workbench` (0.169.2~17-7778 vs
  0.132.25-5599), `nvidia-dgx-telemetry` (5.22 vs 4.11), `nvidia-modprobe`
  (610.43.02-1ubuntu1 vs 580.82.09-0ubuntu1), and
  `nvidia-spark-wifi-fw-ppa` (1.2-1 vs 1.1-1);
- the shared-sync oneshot/timer is installed only on Spark 1;
- root-disk usage differs by about 11 GB, as expected for independent disks.

The environment SHA-256 values therefore differ:

- Spark 1: `f059ab04bb477c50f69518caae34a9e338571d122be1513c7f794ff071ae05fc`;
- Spark 2: `309f26be943053f4558b13068707649c2de0aa8cb8a79f0e24ae419d67c76213`.

## E. Shared storage

`PASS` for presence and additive Spark 1 to Spark 2 propagation. Both
`~/shared` paths are symlinks to local `/srv/shared` directories with expected
subdirectories. A 50-byte technical marker propagated with identical timestamp
and SHA-256 after a successful manual start of the existing Spark 1 oneshot.
The ~15-minute timer exists only on Spark 1; deletion was deliberately not
tested. Treat the directories as eventually consistent local copies, not a
concurrent shared filesystem. See `DGX_SPARK_SHARED_STORAGE_REPORT.md`.

## F. ARM64 compatibility

Current result: `PASS` for the declared CEG/Hugging Face stack; optional
extensions remain explicitly unqualified.

- Both dedicated venvs resolved the same wheel set with no source build for
  runtime dependencies and `pip check` passed.
- NumPy 2.5.2, pandas 3.0.5, PyYAML 6.0.3, Matplotlib 3.11.1, Tokenizers 0.22.2,
  PyArrow 25.0.1, Transformers 4.57.6, Accelerate 1.14.0, Datasets 5.0.1,
  Torch 2.13.0+cu130, and Triton 3.7.1 import on ARM64.
- The repository's tiny Torch/Transformers hook suite passed 29/29 on both
  nodes without loading pretrained weights.
- Flash Attention, bitsandbytes, and vLLM are not required by the canonical CEG
  path. They must not be treated as working merely because an image exists.
- No custom CUDA kernels and no direct Flash Attention/bitsandbytes/vLLM/Triton
  imports were found in CEG.

See `DGX_SPARK_ARM64_COMPATIBILITY.md`.

## G. dstack

`DSTACK_OPERATIONAL`.

The dstack 0.21.2 server listens on Spark 1 port 3030. The public port timed out
from the development machine, so the local dstack 0.21.2 client uses an SSH
loopback tunnel. The active identity `gabriel-alexandre` is authorized in project
`main`. The secret remained only in dstack's private user configuration and was
not written to this repository or the evidence.

The ARM/GB10/shared-volume task completed with exit status 0 on both nodes. Spark
1 and Spark 2 each reported `aarch64`, Python 3.12.3, Torch
`2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2 available, `NVIDIA GB10`, and all three
expected `/shared` directories. An initial exit-2 packaging failure established
that virtual-repository jobs do not automatically contain the local probe; the
task now declares an explicit `files:` upload. The failed attempt and the two
successful runs were deleted after log capture.

The cached `nvcr.io/nvidia/vllm:26.05-py3` image was verified on Spark 1 with
`--pull never`: ARM64 image hash
`c5c116c732f24a640650e1e6e399112ef7c2b1e80c5fd7c13808b10a579646e6`,
vLLM `0.20.1+7124b12a.dev`, Torch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2
forward-compatibility mode, GB10 detected, and `/shared` visible. This was an
import/device smoke only; no model server or inference ran.

## H. Model readiness

The exact CEG Qwen model is `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`.

- Presence under `~/shared/modelos`: absent; the directory contains no model.
- Exact revision load: not tested.
- Weight download: none.
- Semantic inference: none.

Current classification: `EXACT_CEG_MODEL_BLOCKED_ABSENT`; no multi-GB download
was started.

## I. A40 reference availability

Historical label-free A40 reference artifacts are available. Gate 12.1 contains
exact environment/model/revision metadata and twelve synthetic fixtures with
full 151,936-element baseline logits, JVPs, finite derivatives, cotangents, and
hashes. M3 provides supplementary synthetic Gram/bridge/reproducibility data;
V2 provides non-scientific deterministic token-sequence/runtime records.

This is enough for a later **descriptive** GB10 comparison without buying an
A40 run. It is not a prospectively frozen equivalence gate.

## J. A40/GB10 descriptive comparison

`NOT_RUN`: first-phase Spark bring-up has not passed, so the scientific model
was not touched. No numerical difference or equivalence claim is made.

## K. Proposed formal backend qualification

Freeze a small synthetic probe and thresholds before viewing qualification
metrics. Compare exact model/tokenizer hashes and tokens, FP32/BF16 baseline
logits, top-k, full-softmax JS, selected hidden states, requested/implemented
controller amplitude, post-intervention teacher-forced logits, and JVP/finite
derivative quantities. Keep within-backend repeatability separate from
cross-backend tolerance. See `A40_GB10_EQUIVALENCE_PLAN.md`.

Current classification: `A40_EQUIVALENCE_REQUIRES_PROSPECTIVE_QUALIFICATION`.

## L. Parallel workload recommendation

Start with two independent full-model workers and deterministic logical-key
sharding. It avoids distributed-kernel/NCCL coupling, gives failure isolation,
and matches CEG's embarrassingly parallel `(condition,item,rollout)` structure.
The utility hashes canonical logical keys, rejects duplicate inputs, guarantees
one assignment per key, and recombines in node/order-independent order.

Do not start with multi-node model parallelism. Record whether `rocep1s0f1` and
NCCL exist, but use the direct interconnect only if a later model cannot fit or
a separate distributed protocol justifies the added backend.

Observed network evidence: both nodes expose two link-up, full-duplex
200,000 Mb/s interfaces (`enp1s0f1np1` and `enP2p1s0f1np1`), InfiniBand/RoCE
devices including the administrator-named `rocep1s0f1`, `ibv_devinfo`, and NCCL
2.29.7 in the venv. Link-local routing is configured. No NCCL or bandwidth
benchmark was rerun; the administrator-reported 20.5 GB/s result remains
historical rather than independently observed here.

## Unified-memory implications

CPU arrays and CUDA tensors compete for the same roughly 121 GiB usable pool.
Approximate lower bounds (excluding allocator/runtime overhead) are:

- Qwen3-8B parameters: about 14.9 GiB at BF16 or 29.8 GiB at FP32;
- a 32B dense model: about 59.6 GiB at BF16, leaving limited room for KV cache,
  activations, CPU datasets, and transient casts;
- a 70B dense model: about 130 GiB at BF16 and therefore does not fit as a raw
  unquantized single-node model;
- one 4,096-dimensional controller: 16 KiB FP32; one thousand are about 15.6 MiB;
- full FP32 logits at vocabulary 151,936: about 0.58 MiB per token, or about
  2.32 GiB for 4,096 stored token positions;
- all-layer BF16 residual capture for an assumed 36-layer, width-4,096 Qwen3-8B:
  about 288 KiB/token, or 1.125 GiB for 4,096 tokens per batch element;
- BF16 KV cache under the assumed 36-layer/8-KV-head/128-head-dimension layout:
  about 144 KiB/token, or 4.5 GiB at 32K context per batch element.

JVP/VJP and FP32 lifting can multiply parameter/activation residency. Use
streamed capture, CPU/GPU residency accounting, and at least 20–30 GiB practical
headroom rather than planning from raw weight size alone. These are analytical
budgets, not benchmarks.

## M. Repository state

- Worktree: `/private/tmp/causal-epistemic-geometry-dgx` (isolated).
- Branch: `infra/dgx-spark-bringup`.
- Base HEAD: `2fbcc98fa6026b844074bdaef2ba311b360b7f81`.
- Commit/push: pending final validation.
- Remote clones: clean detached checkouts of the base commit on both nodes.
- Active Q2 V3 worktree: not modified.

## N. Security audit

- passwords committed: **NO**;
- private keys committed: **NO**;
- dstack token committed: **NO**;
- API keys committed: **NO**.

A repository secrets scanner is part of the final validation gate.

## O. Scientific isolation

- Q2 V3 changed: **NO**;
- semantic scientific inference: **NONE**;
- Q2 outcomes opened: **NO**;
- Q3: **NOT RUN**;
- controller qualification/calibration/M0/M1/M2/M3: **NOT RUN**.

## Subsystem status

| Subsystem | Status |
| --- | --- |
| SSH/access | PASS |
| GB10 PyTorch | PASS |
| BF16 | PASS |
| shared storage | PASS |
| ARM dependency stack | PASS for declared stack; optional extensions untested |
| dstack | PASS: auth, project access, both-node scheduling, CUDA and shared mount |
| exact CEG model load | BLOCKED: exact weights absent; not downloaded |
| CEG intervention engine smoke | NOT TESTED |
| A40 numerical equivalence | NOT TESTED; prospective qualification required |

Infrastructure terminal state: `DGX_SPARK_TECHNICALLY_OPERATIONAL`; the dstack
smoke now passes on both nodes, while the exact model remains deliberately
absent. Scientific backend state remains
`A40_EQUIVALENCE_REQUIRES_PROSPECTIVE_QUALIFICATION`.
