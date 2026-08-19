# Published positive-control protocol

Status: **PROSPECTIVE — NOT EXECUTED**

## Selection

The preferred positive control is the weekday path-steering result from
Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior* (arXiv:2605.05115). The authors provide a public
`manifold_steering` branch of
[`goodfire-ai/causalab`](https://github.com/goodfire-ai/causalab/tree/manifold_steering)
and an end-to-end `weekdays_8b_pipeline` for Llama-3.1-8B on a GPU with at least
24 GB VRAM. The published demonstration uses exact weekday concepts and tests
whether representation-manifold interventions move output probability through
the corresponding weekday behavior manifold.

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
- exact Llama-3.1-8B-Instruct model revision and license/access approval;
- layer, token position, subspace/manifold construction, intervention operator,
  path points, and controls exactly as implemented upstream;
- exact weekday prompt set and candidate tokenization;
- output-probability metric and endpoint/top-1 success criteria;
- seed schedule, dtype, attention engine, source commit, hardware, cost cap;
- an outcome-independent pass/fail tolerance derived from the published result
  and numerical implementation, not from our observed run.

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

