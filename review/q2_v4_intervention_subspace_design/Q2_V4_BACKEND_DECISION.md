# Q2 V4 backend decision

Decision: `V4_NATIVE_SPARK1`.

The infrastructure branch `infra/dgx-spark-bringup` at
`b222fc8be01738f6bb0075ae8edcc7e9293102bb` was inspected read-only and was not
merged. Spark 1 is technically operational (GB10, ARM64, BF16, dstack, about
121 GiB unified memory), but the exact Qwen3-8B revision was absent at bring-up
and neither the intervention engine nor A40↔GB10 numerical equivalence was
qualified.

V4 is a new experiment, so importing A40 vectors while running response
geometry and outcomes on GB10 would silently add an A40-to-GB10 geometry-
preservation claim. Historical M3 work showed why top-1 similarity is
insufficient. A native-Spark design is cleaner: reconstruct and qualify the
source basis, intervention amplitudes, M1, M2, and semantic execution on one
fixed backend. The four source concepts remain explicitly development-selected
from A40 work; native reconstruction does not erase that provenance.

The cost is a bounded source/technical qualification phase before semantic
execution. That burden is preferable to an underidentified cross-backend
equivalence assumption. Existing A40 arrays can remain descriptive diagnostics
but are not an acceptance gate for native V4.

## Resource policy

- Spark 1 only;
- Spark 2 not used and not a synchronization dependency;
- one GB10 per job;
- no multi-node, distributed model parallelism, or two-node sharding;
- SSH direct for development/debugging;
- dstack preferred for long frozen jobs once exact project access is verified;
- unique logical-key output paths and SHA-256 verification because shared
  storage is eventually synchronized local storage;
- no secrets or tokens in repository artifacts.

The exact model remains `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, with tokenizer pinned to the same
revision. Future protocol must pin dtype, attention implementation, Torch/CUDA,
decoding stack, and model file hashes. No model download or Spark inference was
performed in this sprint.

## Scale estimate

The semantic panel is 39,000 trajectories. Linear scaling from Q2 V2's A40
reference (6,960 rows in 2.6602 billed hours) gives about 14.9 A40-equivalent
GPU-hours before qualification. GB10 throughput for the exact engine is
unknown; plan 15–30 Spark-1 GPU-hours and 18–36 wall-clock hours including
source reconstruction, calibration, M1/M2 capture, startup, and conservative
tail. A mandatory preflight benchmark must replace this range before execution.

Expected storage:

- semantic journal/raw token artifacts: approximately 3–10 GiB;
- 65-condition × 48-checkpoint full-vocabulary float32 M2 logits: about 1.9 GiB
  uncompressed, plus metadata/checksums;
- directions, covariance summaries, schedules, and reports: <1 GiB;
- operational headroom recommendation: 20 GiB.

Institutional compute cost does not authorize uncontrolled exploration.
