# Published positive-control protocol

Status: **COMPLETED DEVELOPMENT POSITIVE CONTROL — PASS**

## Selection

The preferred positive control is the weekday path-steering result from
Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior* (arXiv:2605.05115). The authors provide a public
`manifold_steering` branch of
[`goodfire-ai/causalab`](https://github.com/goodfire-ai/causalab/tree/7dcc8ec4ffd11efec8b3cf9febd6b523df7637b6)
and an end-to-end `weekdays_8b_pipeline` for Llama-3.1-8B on a GPU with at least
24 GB VRAM. The published demonstration uses exact weekday concepts and tests
whether representation-manifold interventions move output probability through
the corresponding weekday behavior manifold.

The frozen upstream revision is
`7dcc8ec4ffd11efec8b3cf9febd6b523df7637b6`; the paper is frozen at
[arXiv:2605.05115v1](https://arxiv.org/abs/2605.05115). The model is the
authors' base `meta-llama/Llama-3.1-8B`, not an instruct variant, at model and
tokenizer revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. Hugging Face
access is gated and must be confirmed remotely before any model download.

This choice is prospective and methodological. It does not endorse manifold
geometry as the metric for this repository's Q2.

## Why this control was selected

| Criterion | Weekday replication | Beaglehole et al. RFM alternative |
|---|---|---|
| Public implementation | End-to-end official weekday pipeline | Public `neural_controllers` implementation |
| Objective outcome | Exact weekday token/probability trajectory | Several demonstrations rely on style, safety, or judge-based outcomes |
| Hook relevance | Direct internal intervention on an 8B decoder | Direct internal additive intervention |
| Model fit | Llama-3.1-8B, compatible with one A40 | Llama-family support, but RFM fitting adds dependencies and multi-layer training |
| Cost/complexity | Bounded single-domain replication | Broader concept extraction and tuning stack |

Beaglehole et al., *Toward Universal Steering and Monitoring of AI Models*
(arXiv:2502.03708; later published in *Science*), remains the backup. It has
public code and clearly describes per-block concept vectors added to block
outputs, but the simplest published outcomes are less exact or require more RFM
infrastructure than the weekday control.

## What the control validates

The positive control asks only whether our environment can reproduce a
published activation-to-behavior intervention with the authors' model, task,
and method. It validates:

- model and tokenizer provenance;
- activation location and replacement/addition semantics;
- intervention lifecycle;
- exact outcome extraction;
- a qualitative/quantitative manipulation effect against the authors' stated
  baseline and linear/random controls.

It does not validate our Q1 instrument, steering direction, stochastic
estimands, or future Q2 metric.

## Freeze gate before execution

Before any model download or outcome collection, a `DEVELOPMENT_LOCK` review
must record:

- exact CausaLab commit from the `manifold_steering` branch;
- exact paper version and pipeline config;
- exact `meta-llama/Llama-3.1-8B` model revision and license/access approval;
- layer, token position, subspace/manifold construction, intervention operator,
  path points, and controls exactly as implemented upstream;
- exact weekday prompt set and candidate tokenization;
- output-probability metric and endpoint/top-1 success criteria;
- seed schedule, dtype, attention engine, source commit, hardware, cost cap;
- an outcome-independent pass/fail tolerance derived from the published result
  and numerical implementation, not from our observed run.

The complete lock is recorded in
`review/positive_control_weekdays/POSITIVE_CONTROL_LOCK.md` and
`experiments/specs/positive_control_weekdays.yaml`. The pinned runner uses
layer 28, the last prompt token, BF16, and the upstream default eager attention
implementation. The bounded weekday path-steering subset includes baseline,
subspace, activation-manifold, output-manifold, and path-steering stages. The
pullback analysis is omitted because it is not required for the primary weekday
metric and is outside this explicitly bounded reproduction; no upstream path or
metric is changed by that omission.

The frozen primary metric is the upstream
`distance_from_behavior_manifold.mean`: mean cumulative Bhattacharyya distance
along the behavior-probability path, where lower is better. The control passes
only if manifold energy is below linear energy and the relative reduction is at
least 30%, with endpoint top-1 weekday match fraction at least 90% and no gross
probability/intervention corruption. These criteria are fixed before outcomes.

Do not adapt the paper's method to Qwen during the replication. Do not simplify
to our V4 centroid plot and call that a replication.

## Required controls

- unintervened baseline;
- the upstream linear path/control;
- shuffled concept order or norm-matched off-structure control if present in
  the frozen upstream pipeline;
- alpha-zero or identity replacement check;
- exact hook cleanup and repeated-context isolation;
- tokenization audit over weekday variants.

## Execution result

The authenticated retry completed on 2026-08-19 using the exact frozen model,
model revision, CausaLab commit, A40 class, BF16 dtype, eager attention, layer,
token position, paths, prompts, and primary metric. The upstream mean cumulative
behavior distance was `0.32336652278900146` for manifold/geometric steering and
`1.3987454175949097` for linear steering, a relative reduction of
`0.7688167419736623` (76.9%). Endpoint top-1 weekday sanity was 672/672 (100%)
for each path mode. The frozen 30% reduction and 90% endpoint thresholds both
passed, so the classification is `POSITIVE_CONTROL_PASS`.

The original HTTP-401 attempt remains preserved separately. Authentication,
the exact gated-file probe, remote cache provenance, compatibility environment,
cost, endpoint calculation, and compact upstream results are recorded under
`review/positive_control_weekdays/retry_authenticated/`. The RunPod was stopped
after artifact recovery. No original Q1 steering, substrate race, Q2, or
holdout access occurred.

## Interpretation boundary

`POSITIVE_CONTROL_PASS` means the intervention stack can reproduce one known
published internal-control phenomenon. It authorizes consideration of an
original micro-Q1; it does not support Q1 itself.

`POSITIVE_CONTROL_FAIL` blocks interpreting a later original-Q1 null as evidence
against causal complementarity. Diagnose environment/method differences or stop.
Do not tune the original Q1 to compensate.

## Why this does not import manifold machinery into Q2

The replication runs in an isolated compatibility adapter or pinned upstream
environment. The main Q2 ladder still starts with cosine/normalized Euclidean
distance and only advances to richer metrics after pre-specified evidence. A
positive control can be methodologically adjacent without becoming the theory
under test.
