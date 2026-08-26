# dstack on CIAAM DGX Spark

## Observed server/client status

- Server: dstack 0.21.2, listening on Spark 1 `0.0.0.0:3030`.
- Direct development-machine access: timed out after five seconds.
- Local client: dstack 0.21.2 in
  `/private/tmp/ceg-dstack-client-venv312` (Python 3.12).
- Required route: SSH local forward to Spark 1.
- Project: local profile `main` points to `http://127.0.0.1:3030`; the active
  dstack identity is `gabriel-alexandre`, with project membership verified by
  successful run submission.

## Secret-safe local configuration

The server is reported at `http://200.144.205.82:3030`. The dstack token must
never be copied into this repository, shell history, committed logs, or a Codex
transcript.

Obtain/configure it manually in an interactive terminal on a Spark using the
administrator-provided `sudo meu-token-dstack` workflow, then follow dstack's
local profile login prompt. Store the credential only in dstack's user-level
configuration. Use a dedicated local venv if the client is absent:

```bash
python3 -m venv ~/.venvs/dstack-ceg
source ~/.venvs/dstack-ceg/bin/activate
python -m pip install dstack
dstack --version
```

Do not paste the token into an environment file under the project.

## Smoke task

The checked-in task is [`infra/dstack/dgx-spark-smoke.dstack.yml`](../../../infra/dstack/dgx-spark-smoke.dstack.yml).
It pins the administrator-recommended ARM image and declares:

- `cpu: arm:2..`;
- `gpu: GB10:1`;
- `/srv/shared:/shared`;
- an explicit `files:` mapping that uploads the probe into the container;
- a tiny script that prints host, architecture, Python, Torch/CUDA/GPU, and
  visible shared-directory names.

It performs no model inference and no large allocation.

After local authentication and with the tunnel active:

```bash
dstack apply -f infra/dstack/dgx-spark-smoke.dstack.yml
```

Capture the run ID and logs in an untracked local evidence directory. Confirm
the reported architecture is `aarch64`, the GPU contains `GB10`, CUDA is true,
and `/shared` is visible. Submit a second clean run only if needed to observe
the other node; scheduler placement on one node does not prove both are usable.
Stop/delete the task through dstack after its terminal log is retrieved.

## Executed smoke evidence

The checked-in task completed with exit status 0 on Spark 2. A second temporary
configuration selected fleet instance `sparks-0` and completed with exit status
0 on Spark 1. Both reported:

| Field | Spark 1 | Spark 2 |
| --- | --- | --- |
| architecture | `aarch64` | `aarch64` |
| Python | `3.12.3` | `3.12.3` |
| Torch | `2.12.0a0+5aff3928d8.nv26.05` | `2.12.0a0+5aff3928d8.nv26.05` |
| Torch CUDA | `13.2` | `13.2` |
| CUDA available | `true` | `true` |
| GPU | `NVIDIA GB10` | `NVIDIA GB10` |
| `/shared` | visible | visible |
| expected directories | all present | all present |
| probe status | `PASS` | `PASS` |

The first submission exposed a packaging defect: without `files:`, a virtual
repository run did not contain the local probe script and exited 2. The checked-in
configuration now uploads that script explicitly. The failed attempt and both
successful runs were deleted after their terminal logs were collected. No fleet,
project, membership, or shared-volume configuration was changed by the smoke.

## vLLM boundary

The cached image was inspected and executed with `--pull never` on Spark 1. It
is ARM64 with image SHA-256
`c5c116c732f24a640650e1e6e399112ef7c2b1e80c5fd7c13808b10a579646e6`.
The import/device smoke observed vLLM `0.20.1+7124b12a.dev`, Torch
`2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2 forward compatibility, CUDA available,
NVIDIA GB10, and `/shared` visible.

The cached image name and import pass are not evidence that model serving works.
The executed check was equivalent to:

```bash
python -c 'import platform, torch, vllm; print(platform.machine(), vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

No model server or extension compilation was launched.

## Multi-node note

Do not use distributed inference initially. If a later non-scientific NCCL
probe is authorized, first verify that `rocep1s0f1` exists on both nodes. A
two-node dstack job would then use `nodes: 2` and
`NCCL_IB_HCA=rocep1s0f1`; it must not change network configuration.

## Current classification

`DSTACK_OPERATIONAL`: server/client compatibility, authentication, project
authorization, scheduling on both nodes, ARM64/CUDA/GB10 execution, and the
shared-volume mount all passed. This classification does not qualify model
serving or scientific equivalence.
