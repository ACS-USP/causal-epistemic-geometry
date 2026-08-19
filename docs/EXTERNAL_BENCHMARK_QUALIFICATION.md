# External benchmark qualification campaign

This branch opens a cheap instrument search after the Q1 V3 procedural reasoning
screen failed. The scientific question is unchanged: can a frozen reasoning model
have a useful, reproducible pattern of semantic errors? No steering, activation
extraction, PCA, geometry, or confirmatory evaluation is part of this campaign.

## Gates

1. Q0 is model-free and offline. It validates stable IDs, deterministic prompts,
   objective evaluators, and a failure taxonomy.
2. A completion diagnostic uses only 3--5 new development items and the fixed
   `8192 -> 16384 -> 32768` ladder. It estimates natural reasoning length; its
   outcomes never enter qualification tables.
3. Q1 smoke uses at most 20 new items and one independent seed per candidate,
   with one generous cap frozen from the completion diagnostic.
4. Only a non-obviously unusable candidate may advance to Q2: 50 new items and two
   independent seeds. The Q2 cap is the same prospective cap selected before Q1
   outcomes are observed.
5. The campaign stops after Q2 and reports either
   `BENCHMARK_QUALIFIED_FOR_STEERING_PILOT` or
   `NO_EXTERNAL_BENCHMARK_QUALIFIED`. It never starts a steering pilot.

## Fixed model policy

- `Qwen/Qwen3-8B`
- revision `b968826d9c46dd6066d109eabc6255188de91218`
- BF16, thinking enabled
- sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`
- deterministic explicit rollout seeds
- a generous fixed cap chosen prospectively from the completion diagnostic;
  truncation is diagnostic, not the desired source of difficulty

Real model and dataset operations are remote-only. The preparation and runner
scripts refuse to load HuggingFace content unless the process is under
`/workspace/causal-epistemic-geometry` with `HF_HOME=/workspace/hf-cache`.

## Candidates

The initial order is RE2-Bench, LiveCodeBench test-output prediction, CRUXEval
output prediction, and objective LiveBench subtasks. RE2-Bench is represented in
the Q0 schema fixtures, but its official executable artifact was not resolved by
the model-free audit and is therefore blocked rather than simulated.

CRUXEval has a deterministic output evaluator and a small official dataset. The
LiveCodeBench and LiveBench adapters accept normalized records produced by their
official loaders/evaluators; an LLM judge is never accepted by the normalized
schema. The former 2048-token CRUXEval, LiveCodeBench, and LiveBench runs are
preserved as `LOW_CAP_DIAGNOSTIC` artifacts. They are not scientific benchmark
failures and cannot qualify or disqualify a candidate.

## Status taxonomy

`VALID_CORRECT`, `VALID_WRONG`, `INVALID_FORMAT`, `TRUNCATED_THINKING`, and
`RUNTIME_ERROR` remain separate in the journal and summaries. Invalid/truncated
outcomes are not relabeled as semantic errors.

## Commands

```bash
PYTHONPATH=src python scripts/validate_external_benchmarks.py
PYTHONPATH=src pytest -q tests/test_external_benchmarks.py
```

The completion diagnostics are complete and remain diagnostic-only. The currently
authorized remote step is the corrected CRUXEval Q1 smoke: 20 new items, selected
with `--offset 20` so neither the completion-diagnostic items nor the old 2048-token
smoke items are reused, and a cap fixed prospectively at `16384`.

The historical diagnostic command was:

```bash
python scripts/run_completion_diagnostics.py \
  --candidate CRUXEval \
  --data /workspace/causal-epistemic-geometry/data/cruxeval_output.jsonl \
  --items 5 \
  --output review/external_benchmark_qualification/cruxeval_completion_diagnostic
```

The script retries the exact same item and seed only when the previous cap
truncates, and records each attempt. Diagnostic outcomes never enter qualification
tables. The CRUXEval cap correction is an explicit prospective instrument decision,
not an outcome-driven tuning step.
If any diagnostic item requires 32,768, the candidate receives an explicit
`high_cap_warning` and must be treated as operationally expensive even though it
is not scientifically rejected.

On RunPod, after explicitly resolving the official source and writing a normalized
JSONL file:

```bash
python scripts/run_external_qualification.py \
  --candidate CRUXEval \
  --data /workspace/causal-epistemic-geometry/data/cruxeval_output.jsonl \
  --stage q1_smoke \
  --offset 20 \
  --max-new-tokens 16384 \
  --output review/external_benchmark_qualification/cruxeval_q1_corrected_16384

python scripts/run_external_qualification.py \
  --candidate CRUXEval \
  --data /workspace/causal-epistemic-geometry/data/cruxeval_output.jsonl \
  --stage q2_qualification \
  --offset 20 \
  --max-new-tokens 16384 \
  --output review/external_benchmark_qualification/cruxeval_q2
```

The Q2 command is run only after the Q1 smoke gate passes. Both commands journal
each completed item/seed and are resumable only with the same candidate, source
digest, selected IDs, model revision, generation config, and source commit.

## Reclassifying the old low-cap runs

After the persistent remote artifacts are reachable, reclassify only their
manifests (never their journals or raw outputs):

```bash
python scripts/reclassify_low_cap_runs.py \
  review/external_benchmark_qualification/cruxeval_q1 \
  review/external_benchmark_qualification/livecodebench_q1 \
  review/external_benchmark_qualification/livebench_q1
```

The resulting status is `LOW_CAP_DIAGNOSTIC`. A 2048-token thinking truncation
means only that the operational cap was too small; it is not evidence that the
benchmark lacks genuine semantic errors.

## Scientific boundary

Even a Q2 qualifier is only a benchmark-instrument result. It does not support a
steering effect, geometry claim, or collective-utility claim. The principal must
review the Q2 bundle before any future steering pilot is designed. In particular,
this correction explicitly prevents using a short reasoning cap to manufacture
intermediate raw accuracy from `THINKING_UNCLOSED` outcomes.
