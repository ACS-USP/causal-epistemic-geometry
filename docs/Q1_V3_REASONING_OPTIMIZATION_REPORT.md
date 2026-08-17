# Q1 V3 — reasoning inference optimization gate

Status: **PREFIX-REUSE APPROVED — BATCHED SAMPLING REJECTED FOR REAL QWEN**

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

The bounded gate ran on the authorized RunPod A40 using only the existing
offline cache:

- model snapshot: `b968826d9c46dd6066d109eabc6255188de91218`;
- model class: `Qwen3ForCausalLM`, 8.19B parameters, BF16, SDPA;
- host: `effad46be16c`, `HF_HOME=/workspace/hf-cache`;
- data: already-consumed Stage-A engineering items only;
- holdout and Stage A: untouched.

For three groups and two items per group (18 budget rows), independent
512/1024/2048 generations matched the corresponding 2048 prefixes exactly.
The serial reference took 622.0 s for 18 physical generations. Prefix reuse
took 294.4 s for 6 physical generations: zero token mismatches, zero parse
mismatches, and zero correctness mismatches. This is an engineering approval,
not a scientific result.

The batched cached-decoding path took 181.1 s for 6 physical generations but
failed equivalence: 18/18 token trajectories differed, with 2 parse and 2
correctness differences. The failure persisted under left padding (6/6 token,
4 parse, 3 correctness differences), in a two-copy same-prompt test, and with
the eager attention implementation. The batch path is therefore not approved
for real-Qwen scientific data. Its local tiny-transformer result remains only
software validation.

The one-item batch-size-1 diagnostic passed after two repairs: explicit Qwen
cache positions and float32 logits before sampling, matching Transformers'
canonical `generate` path. The remaining failure is genuine batch-shape
numerical sensitivity, not a known missing cache argument.

Unmeasured or not approved: compile warm-up/steady state, CUDA graphs,
Qwen-specific suffix replay, and a full Stage-A run. These are engineering
gates, not missing scientific outcomes.

## Performance implication

The passing prefix path is materially safer but does not meet the aspirational
10× end-to-end target by itself. On the bounded representative probe it was
~2.11× faster than serial generation (622.0 s → 294.4 s). A simple linear
projection of the previously estimated 183.8-minute serial Stage-A runtime is
approximately 87 minutes before model-load and I/O overhead. Therefore no
clean Stage-A run was launched: the optimized path is eligible from a
correctness perspective, but the throughput target remains unmet and should
be reviewed before spending on the full calibration.

At the observed `$0.44/hour` A40 rate, the bounded probe's generation-only
compute was approximately $0.08 serial and $0.04 prefix-reuse. These are
engineering costs, not scientific results. The Pod was stopped immediately
after the gate.

## Approval rule

The clean Stage-A run must still be launched from the beginning; this
engineering work must not be mixed with the old serial partial run. The
approved execution choice is now `max_budget_prefix_reuse`. It is the fastest
bounded path that preserved the serial Qwen trajectory. `serial_reasoning_reference`
remains permanently available as the correctness oracle. `batched_reasoning`
must remain disabled for scientific runs unless a future implementation passes
the same gate.
