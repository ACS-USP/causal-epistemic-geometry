# Q3 related work and design priors

This map distinguishes established methodological precedents from the
project-specific Q3 proposal. It is not a novelty claim and does not imply that
Q3 has been run.

## Model routing and conditional computation

Sparse mixture-of-experts work established learned, input-dependent routing as
a way to activate only part of a model for each example. The original
[sparsely-gated MoE](https://arxiv.org/abs/1701.06538) and
[Mixtral](https://arxiv.org/abs/2401.04088) are architectural precedents, not
direct evidence that activation-steered policies can be routed by semantic
problem.

At the whole-model level, [RouteLLM](https://arxiv.org/abs/2406.18665),
[Zooter](https://arxiv.org/abs/2311.08692), and
[routing from benchmark datasets](https://arxiv.org/abs/2309.15789) learn a
query-to-model decision from preference, reward, or benchmark supervision.
Q3 borrows the selection problem but changes the action set: the candidate
policies are causal interventions in one frozen model, and the comparator is
the best single policy selected entirely from development data.

Design prior: include geometry-blind, fixed-prior, random and global-best
routers. A geometry-aware router must win at matched capacity before geometry
receives causal credit for selection.

## Selective prediction and confidence

[Selective classification for deep networks](https://arxiv.org/abs/1705.08500)
formalizes the risk–coverage trade-off when a predictor may reject. Language
model confidence is not automatically calibrated:
[Jiang et al.](https://arxiv.org/abs/2012.00955) report substantial
miscalibration in QA, while [Kadavath et al.](https://arxiv.org/abs/2207.05221)
show that suitably elicited self-evaluation can contain useful signal but may
generalize imperfectly.

Design prior: abstention, entropy or confidence can enter Route B only through
a frozen, training-only calibration with coverage and compute reported. They
cannot substitute for a reference-free correctness test.

## Verifiers, committees and self-consistency

[Training verifiers for mathematical reasoning](https://arxiv.org/abs/2110.14168)
demonstrates selection among generated candidates using a learned verifier.
[Self-consistency](https://arxiv.org/abs/2203.11171) improves some reasoning
tasks by sampling and aggregating multiple chains. The classic
[ambiguity decomposition for ensembles](https://proceedings.neurips.cc/paper_files/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html)
connects ensemble diversity to potential gain, but disagreement alone does not
identify the correct member.

Design prior: pair- or committee-oracle accuracy is opportunity only. Route C
must beat repeated-baseline/self-consistency and other equal-compute controls;
extra calls are not free evidence of collective utility.

## Budgeted inference and cascades

[FrugalGPT](https://arxiv.org/abs/2305.05176) studies learned LLM cascades that
trade quality against query cost. This directly motivates the baseline-first
Route B, but Q3 avoids choosing a favorable scalar cost coefficient after the
fact. Expected calls, tokens and latency are reported separately under a
frozen budget.

Design prior: prefer the one-call route. Open a cascade only if one-call
selectability is inadequate and compare it at equal expected compute.

## Contextual policy selection

Contextual bandits formalize choosing an action from observed context when
reward depends on both. Examples include
[linear-payoff contextual bandits](https://proceedings.mlr.press/v15/chu11a.html)
and [policy-class guarantees](https://proceedings.mlr.press/v15/beygelzimer11a.html).
Q3.0 is an offline full-information development problem because all closed
policy outcomes exist on the development panel; a future deployment would be
one-call and observe only the chosen policy's answer.

Design prior: split by problem, never by policy×item row; use nested
cross-fitting; compare the router with a development-selected champion; and do
not treat policy rows as independent observations.

## Project-specific proposal

The Q3-specific hypothesis is not merely that prompts can route models. It is
that frozen controller coordinates and pre-outcome geometry might help choose
among causally distinct policies whose blind spots are complementary. The
regularized bilinear score

```text
score(x, k) = policy_bias(k) + phi(x)^T W c_k
```

tests this interaction directly. A capacity-matched geometry-blind code tests
whether any low-dimensional policy identity works as well. Deterministic prompt
features are the lowest-cost item representation; a label-free prompt prefill
is a later, explicitly charged option.

The learned rank-8 subspace may reflect generic local smoothness. Q3 can test
bank diversity, geometry-aware routing and realized utility, but it cannot
establish learned-subspace specificity without the separate matched-random
rank-8 control.
