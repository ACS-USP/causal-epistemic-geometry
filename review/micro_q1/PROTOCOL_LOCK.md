# FIRST ORIGINAL MICRO-Q1 — PROTOCOL LOCK

This is the pre-outcome development lock for Gate 4. It authorizes one
prospectively constructed direction, one norm-matched random control, four
conditions, and two independent rollouts per item. It is not confirmatory and
does not unlock Q2 or the confirmatory holdout.

## Frozen substrate

- Model: `Qwen/Qwen3-8B`
- Model and tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`
- BF16, no quantization, SDPA
- Full autoregressive generation, `enable_thinking=false`
- Sampling: `do_sample=true`, temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0.0`
- `max_new_tokens=4096`
- Engine: `hf_generate_serial_prefill_one_shot_hook`
- Instrument: CRUXEval semantic output prediction
- Dataset revision: `b96af0450242eb4da433032b90998f25588a5d0f`
- Evaluator: deterministic type-aware Python-literal semantics; no LLM judge

## Fresh allocation

The historical exclusion digest was reconstructed from preserved local
CRUXEval manifests/journals and the fresh rows were selected remotely using
stable official dataset ordering and the frozen namespace
`GATE4-MICRO-Q1-CRUX-FRESH-ALLOCATION`, seed `20260819`.

- `DIRECTION_CONSTRUCTION`: 64
- `DIRECTION_VALIDATION`: 16
- `MICRO_Q1_EVALUATION`: 50
- all three sets are mutually disjoint; no model outcomes were available at selection
- source IDs and task prompts are frozen in the three manifest JSON files

## Direction and intervention

The careful system prompt is:

> You are a meticulous program tracer. Carefully track every operation, mutation, intermediate value, branch, and loop. Verify the result before answering. End with exactly one line in the form FINAL: <answer>.

The direct system prompt is:

> Answer the program-output question immediately. Do not trace, deliberate, explain, or verify. End with exactly one line in the form FINAL: <answer>.

The user task is identical within each pair. Activations are extracted at
zero-based `model.model.layers[17]` block output, at the final non-padding
prompt token, during prompt prefill only. The direction is
`unit(mean(h_careful - h_direct))`, oriented toward careful tracing. The
held-out gate requires mean signed gap > 0 and at least 12/16 positive gaps.

After construction, `Delta` is the mean signed construction projection and
`alpha = 0.5 * Delta`. A Gaussian 4096-dimensional control is generated with
stable seed `20260819`, its component parallel to the meaningful direction is
removed, and it is normalized. The same alpha is used for both vectors.

The hook is one-shot: it applies once to the final prompt token during prefill
and cannot modify generated/decode tokens. Alpha-zero identity, exact additive
shift, non-target scope, and hook cleanup are checked before evaluation.

## Scientific table

`BASELINE`, `CPLUS`, `CMINUS`, and `CRANDOM`; 50 items × 2 independent
rollouts each. Seeds are derived from the experiment ID, item ID, condition,
and rollout index under `INDEPENDENT_PRIMARY`. Primary error is 1 for every
non-`VALID_CORRECT` model outcome; infrastructure errors are not outcomes.

Validity must be at least 90% and no more than five percentage points below
baseline. Accuracy may not fall more than ten percentage points below
baseline. The frozen movement thresholds are `D >= .05` and
`Delta_D(random) >= .05`; useful complementarity additionally requires
`G >= .03`, `C >= .03`, and `Delta_C(random) >= .05`.

`G`, `C`, `D`, rescue, and damage use the unbiased independent two-rollout
estimands; `C` uses the ordered cross-item U-statistic. Confidence intervals
are descriptive item-cluster percentile bootstrap intervals with 5,000
resamples and seed `20260819`.

## Firewall

Character-count replication, another alpha/layer/direction, Q2, geometry,
committee analysis, and confirmatory holdout access are forbidden in this
run. The scientific result remains development-only.

Implementation source immediately before this lock: `7b5788dfcae0f67ecfa869c25e50693980c83e81`.
