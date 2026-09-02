# Q2 OOS V2 post-maintenance Spark-1 qualification

Spark 1 was requalified from a cold post-maintenance state before Qwen loading.
All scientifically pinned components match the frozen profile: machine and
architecture, GB10 compute capability, Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA 13.0, Transformers 4.57.6, BF16, SDPA, exact model/tokenizer revision,
all 15 model/tokenizer file hashes, protocol artifacts, selected bank, runner,
consolidator, and the frozen A1/A2 manifests.

The reboot changed the observed kernel from 6.17.0-1026-nvidia to
6.17.0-1032-nvidia and the driver from 580.159.03 to 580.173.02. Neither field
is a frozen static scientific gate in the qualified environment check. The
pinned CUDA/Torch interface, compute capability, model bytes, dtype, attention
stack, and package versions remain exact. These differences are recorded as
operational maintenance changes, not silently ignored.

GPU utilization was 0%, temperature 33 C, no GPU compute process or stale A2
capture process was present, and 3.2 TB of disk was available. The detached
execution worktree was clean at the exact expected commit.

Classification: `Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_QUALIFIED`.
