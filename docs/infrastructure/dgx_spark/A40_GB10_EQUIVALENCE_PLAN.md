# A40 to GB10 backend-qualification plan

## Qualification boundary

The GB10 is a distinct backend until a prospective qualification is frozen and
passed. Matching environment fingerprints would establish software/hardware
identity fields only; it would not establish numerical or scientific
equivalence.

This plan is label-free. It uses no correctness, Q2 V3 panel, blind-spot
outcomes, G metrics, controller qualification, or scientific trajectory.

## Existing A40 reference inventory

The repository already contains a strong descriptive reference at
`review/gate12_1_continuous_geometry_engine/`:

- exact model: `Qwen/Qwen3-8B`;
- exact revision: `b968826d9c46dd6066d109eabc6255188de91218`;
- source commit: `0e7e1505a2456c0a78ee9a8ba63d2eaf5f1f8d43`;
- A40, Torch `2.4.1+cu124`, CUDA `12.4`;
- twelve synthetic, non-benchmark token-sequence fixtures;
- per-fixture FP32 full-vocabulary baseline logits (151,936 values), forward
  JVP, independent JVP, cotangent, seven finite-difference derivative vectors,
  epsilons, and target-token ID;
- fixed engineering directions, BF16/FP32 bridge summaries, hashes, and
  environment provenance.

The M3 provenance directory supplies additional synthetic token fixtures,
exact model revision, A40 environment metadata, Gram/bridge summaries, and
reproducibility statistics. Its checked-in raw sufficient-statistics NPZ does
not contain the full per-fixture logits, so it is supplementary rather than the
primary cross-backend reference.

The V2 secure A40 benchmark contains deterministic generated token IDs on
non-scientific fixtures and runtime measurements. It is useful as a later
sequence/throughput diagnostic, but not as the first numerical comparator.

These artifacts are sufficient for a **descriptive** GB10 comparison after
bring-up. No paid A40 run is needed merely to measure differences against them.
They are not a prospectively frozen A40-to-GB10 equivalence gate.

## Phase 0: freeze before GB10 comparison

Principal review must freeze all of the following before inspecting GB10
comparison metrics:

1. reference artifact hashes and the exact subset of synthetic fixtures;
2. model, revision, tokenizer files, source commit, dtype, layer, attention
   implementation, eager/compile policy, seed policy, and CUDA determinism
   settings;
3. quantities and aggregation rules;
4. acceptance thresholds and failure handling;
5. whether software versions must match the historical A40 or are allowed to
   differ as an explicitly qualified backend;
6. the terminal labels `EQUIVALENT_WITHIN_FROZEN_TOLERANCES` and
   `DISTINCT_BACKEND`.

Do not derive thresholds from GB10 measurements. Historical A40 repeat noise,
numerical analysis, and tolerances justified by the future scientific
estimands—not observed GB10 convenience—must drive the lock.

## Minimal technical probe set

Use four of the existing synthetic Gate 12.1 fixtures spanning short/long
prompt and continuation lengths, plus two fixed engineering directions. If the
formal lock requires all twelve, the incremental compute remains small.

For each fixture record:

| Quantity | Comparison |
| --- | --- |
| Model/tokenizer artifact hashes | Exact match |
| Input token IDs and attention mask | Exact match |
| FP32 baseline logits | max absolute/relative error, cosine, top-1 agreement, softmax JS |
| Top-k token IDs and logits | exact token order plus numeric errors |
| Selected hidden state | norm ratio, cosine, max absolute/relative error |
| BF16 baseline logits | same metrics, separately from FP32 |
| Controller application | exact requested/implemented amplitude and selected hidden delta |
| Post-intervention teacher-forced logits | same metrics and softmax JS |
| JVP/finite derivative | cosine, relative norm/error, JVP-VJP duality diagnostic |
| Repeated run | exact within-backend repeat for deterministic simple path |

Synthetic prompt bytes should be added to the prospective lock for exact
tokenizer comparison because the historical Gate 12.1 fixtures preserve token
IDs rather than source prompt bytes. Tokenization is CPU-side and must match
exactly; it is not inferred from logit similarity.

## Descriptive report before formal qualification

It is acceptable to compute the metrics above against the existing A40 arrays
and label the result `DESCRIPTIVE_ONLY`. Such a report must present raw
differences without pass/fail or equivalence language. It can inform, but must
not set, a later protocol after the same GB10 values have been seen.

## When an A40 refresh is required

Set `A40_REFERENCE_REFRESH_REQUIRED` and request separate authorization if:

- the historical exact model/source path cannot execute on GB10;
- the reference hashes or required arrays fail validation;
- a future scientific experiment uses a materially different model revision,
  controller engine, attention implementation, Torch/CUDA stack, or measured
  quantity;
- prospective reviewers require within-A40 repeat distributions or source
  prompt bytes that the historical artifacts do not contain.

The refresh should use only the frozen synthetic probes, one A40, no sampling,
and a few forward/JVP calls. At the historical `$0.44/A40-hour` accounting rate,
even a conservative 15-minute run is about `$0.11`, excluding startup/storage.
Do not provision it automatically.

## Current classification

- Existing A40 descriptive reference: `AVAILABLE`.
- GB10 descriptive comparison: `NOT_RUN`.
- Frozen numerical equivalence: `NOT_TESTED`.
- Scientific backend status: `A40_EQUIVALENCE_REQUIRES_PROSPECTIVE_QUALIFICATION`.
