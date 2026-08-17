# Q1 V3 — reasoning inference optimization gate

Status: **FINAL GATE COMPLETE — MAX-BUDGET PREFIX REUSE CANONICAL**

> **Timing note:** this report records the engineering gate immediately before
> the clean Stage-A launch. Stage A was launched afterward in a separate run
> from commit `4faea97`; this file contains no Stage-A outcomes. See
> [CURRENT_STATUS.md](CURRENT_STATUS.md) for the live state.

This is an engineering report. No Stage-A calibration result, steering
experiment, holdout evaluation, or scientific conclusion is contained here.

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
- host: `081a89558983`, `HF_HOME=/workspace/hf-cache`;
- data: already-consumed Stage-A engineering items only;
- holdout and Stage A: untouched **at the time of this engineering gate**.

For three groups and two items per group (18 budget rows), independent
512/1024/2048 generations matched the corresponding 2048 prefixes exactly.
The serial reference took 622.0 s for 18 physical generations. Prefix reuse
took 294.4 s for 6 physical generations: zero token mismatches, zero parse
mismatches, and zero correctness mismatches. This is an engineering approval,
not a scientific result.

The earlier shape-changing batched cached-decoding path took 181.1 s for 6
physical generations but failed equivalence: 18/18 token trajectories
differed, with parse and correctness differences. The failure persisted under
left padding, in a two-copy same-prompt test, and with eager attention. It is
not approved for real-Qwen scientific data.

The one-item batch-size-1 diagnostic passed after two repairs: explicit Qwen
cache positions and float32 logits before sampling, matching Transformers'
canonical `generate` path. The remaining failure is genuine batch-shape
numerical sensitivity, not a known missing cache argument.

The final bounded B=1 gate used the same three groups and two items per group:

| Engine | Physical generations | Wall time (s) | Token mismatches | Parse mismatches | Correctness mismatches |
|---|---:|---:|---:|---:|---:|
| Serial reference | 18 | 620.397 | 0 | 0 | 0 |
| HF generate + prefix reuse | 6 | 293.840 | 0 | 0 | 0 |
| Custom cached decode, B=1 | 6 | 294.704 | 6 | 0 | 0 |

The custom B=1 decoder is therefore rejected: it does not preserve exact
Qwen trajectories and is marginally slower (0.3%). The approved
`max_budget_prefix_reuse` path remains canonical. Prefill-fork reuse was not
adopted because the available safe HF generation path did not expose a stable
exact cache-fork interface, and the custom decoder failed the primary gate.
Static-cache/`torch.compile` execution remains disabled; no unsafe compiler
integration is needed after the exact B=1 rejection. No CUDA graph or altered
attention path was introduced.

## Performance implication

The `183.8 minute` value belongs to the historical Q1 V1.1 direct-answer
campaign. It is not a Q1 V3 Stage-A estimate and is excluded from this report.

The correct previously measured Q1 V3 Stage-A workload is:

- 4,320 scientific budget-rollout outcomes;
- approximately 34.65 A40 hours for the unoptimized serial projection;
- approximately US$15.25 at US$0.44 per A40-hour.

Under the already approved max-budget prefix reuse, the frozen 4,320 rows are
derived from approximately 1,440 physical 2,048-token trajectories. The final
bounded gate measured 620.397 s for 18 serial physical generations and 293.840
s for 6 prefix-reuse physical generations. The two prospective projections
are:

- empirical bounded-gate scaling: 1,440 × (293.840 / 6) = **19.59 A40 h**,
  approximately **US$8.62**;
- workload-weighted existing telemetry: approximately **13.74 A40 h**,
  approximately **US$6.04**.

The conservative operational estimate is therefore **19.59 A40 h / US$8.62**,
below the authorized US$15.25 ceiling. It excludes model-cache download,
startup, and small artifact I/O overhead; the margin to the authorization is
approximately US$6.63. No V3 Stage-A outcome is implied by the projection.

At the observed `$0.44/hour` A40 rate, the bounded probe's generation-only
compute was approximately $0.08 serial and $0.04 prefix-reuse. These are
engineering costs, not scientific results. The historical
`review/serial_reference_v1_1_partial/` artifact is retained as an explicitly
archival V1.1 record; it is not the Q1 V3 serial reference. Q1 V3 uses
`serial_reasoning_reference` plus the bounded real-Qwen equivalence gate.

## Approval rule

The clean Stage-A run was launched afterward from the beginning; this
engineering work was not mixed with the old serial partial run. The final B=1
gate compared the approved HF-generation prefix path with the custom
batch-size-1 decoder. The candidate had six token mismatches and no material
speed gain, so it was rejected. The approved
`max_budget_prefix_reuse` path remains canonical, with
`serial_reasoning_reference` permanently available as the correctness oracle.
`batched_reasoning` remains disabled for scientific runs.
