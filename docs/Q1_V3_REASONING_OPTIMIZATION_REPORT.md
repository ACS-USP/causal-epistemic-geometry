# Q1 V3 — reasoning inference optimization gate

Status: **NOT APPROVED — real-Qwen equivalence pending GPU availability**

This is an engineering report. No Stage-A calibration, steering experiment,
holdout evaluation, or scientific conclusion was produced.

## Frozen protocol

The Q1 V3 reasoning protocol was not changed. The implementation preserves the
serial reference and changes only physical execution:

- budgets remain 512, 1024, and 2048;
- independent seeded rollouts remain independent;
- the model, prompts, parser, item IDs, and split manifest remain unchanged;
- every shorter budget is parsed from its own exact token prefix;
- no early `FINAL:` stopping was added.

## Engines

### `serial_reasoning_reference`

The existing one-generation-per-budget path is preserved as the permanent
correctness oracle.

### `max_budget_prefix_reuse`

For each item/view/rollout seed, the 2048 trajectory is generated once. The
512 and 1024 rows are exact token prefixes of that trajectory and receive
independent parser calls. Each row records the physical generation ID, source
budget, prefix length, and natural completion length.

### `batched_reasoning`

The same maximum-budget strategy is combined with:

- immutable prompt preparation;
- deterministic length buckets;
- maximum padded-prefill-token enforcement;
- right/left padding and explicit position IDs;
- cached one-token autoregressive decode;
- one independent Torch generator per rollout seed;
- restoration of caller order after length sorting;
- batch provenance in every output.

## Local validation

The network-free randomly initialized GPT-2 fixture exercised actual
PyTorch/Transformers forward passes and cached generation. It is explicitly
software validation only.

| Mode | Scientific rows | Physical generations | Token mismatches vs serial | Local speedup |
|---|---:|---:|---:|---:|
| Serial reference | 12 | 12 | 0 | 1.00× |
| Max-budget prefix reuse | 12 | 4 | 0 | ~2.3× |
| Batched reasoning, batch size 2 | 12 | 4 | 0 | ~4.2× |

The benchmark uses four toy prompts and budgets 4/8/12. These numbers are not
an estimate of Qwen3/A40 performance and are not scientific evidence.

The local suite also verifies:

- exact per-row seed-stream preservation under batch reordering;
- exact serial-vs-batched token IDs on variable prompt lengths;
- deterministic length planning and padded-token limits;
- exact budget-prefix derivation and shared physical IDs;
- independent parser-ready outputs for each censored budget;
- no gradient graphs in the production inference path.

## Real-Qwen gate

The bounded gate is implemented in:

```text
scripts/benchmark_q1_v3_reasoning_qwen.py
```

It compares independently generated 512/1024/2048 trajectories against the
corresponding prefixes of a separately generated 2048 trajectory, then
compares serial rows with batched rows. It is constrained to already-consumed
development items and never accesses the holdout or launches Stage A.

It has not run in this turn. Starting both available A40 Pods failed with the
RunPod capacity response “There are not enough free GPUs on the host machine to
start this pod.” No Pod was terminated, deleted, reconfigured, or used for a
download.

Therefore the following remain unmeasured on the real Qwen3-8B revision:

- prefix-token equivalence;
- left/right padding choice;
- Qwen batch RNG equivalence;
- A40 throughput, utilization, and peak VRAM;
- attention backend;
- compile warm-up and steady-state performance;
- CUDA graph feasibility;
- suffix replay or any layer-specific steering reuse.

These are engineering gates, not missing scientific outcomes.

## Approval rule

The optimized engine is not approved for a clean Stage-A run until the real
Qwen gate has zero discrete token, parser, and correctness differences against
the serial reference. The clean Stage-A run must then be launched from the
beginning; this engineering work must not be mixed with the old serial partial
run.

Current recommendation: retain `serial_reasoning_reference` as the canonical
Q1 V3 engine until the bounded real-Qwen gate passes. If max-budget prefix
reuse passes but batched padding or RNG equivalence fails, use the passing
prefix engine and keep batching disabled.
