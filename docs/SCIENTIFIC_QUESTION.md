# Scientific question

## Q1: the minimal development kill-test

MMLU-Pro V1–V1.2, E3-10 V2, the V3 procedural reasoning agent, and V4
micro-instruments are closed or paused development instruments. None produced a
scientific steering result. The current task is an offline scientific
rearchitecture; no model execution is authorized. The normative current
framing is in [SCIENTIFIC_CONSTITUTION.md](SCIENTIFIC_CONSTITUTION.md), with
history in [INSTRUMENT_HISTORY.md](INSTRUMENT_HISTORY.md).

For one frozen model `f_theta`, compare:

```text
baseline   h_l -> h_l
treatment  h_l -> h_l + alpha * v_i
```

on exactly the same held-out items with mechanically known ground truth. Define
binary errors `e_0(t)` and `e_i(t)`, then ask:

> Can one activation intervention change where the model fails without merely
> making it worse?

This is a development experiment, not a confirmatory claim. The useful regime,
if one exists, must preserve individual accuracy approximately while changing
the error profile enough to create measurable complementarity.

For a future stochastic Q1, each condition is a sampled policy
`(trajectory, y_hat) ~ pi_theta,v(. | x, seed)`. Matched-seed baseline and
treatment rollouts provide the causal pair; independent seeds estimate error
propensity and repeated-agent complementarity. The exact oracle and raw
trajectory remain primary provenance, while propensity and hard error metrics
are reported separately.

## Motivation and limits

Algorithmic monoculture is a useful motivation: multiple copies of one model
may inherit correlated blind spots rather than behaving like independent
experts. Response diversity alone is not enough. Two systems can produce
different text while making the same decision error, or disagree because one
has simply become less competent.

Likewise, low error correlation after accuracy destruction is meaningless for
collective utility. This is why every summary puts baseline accuracy, treatment
accuracy, delta accuracy, error similarity, rescue rate, and damage rate side by
side. The pair oracle is labeled **COMPLEMENTARITY HEADROOM** and is not an
implementable ensemble result.

We begin with baseline versus one vector because it isolates the first causal
question. A giant `v_i × v_j` geometry study would introduce many choices before
we know whether a single intervention can move the error profile at all.

## Nearby work and novelty discipline

A literature audit found nearby work on activation steering, diversity
steering, concept or latent experts, correlated LLM errors, and
representational geometry. The target is therefore not the broad claim
“steering creates diversity.” A possible later target is the more specific
chain:

```text
representation geometry
    -> controlled activation intervention
    -> held-out error covariance
    -> useful committee diversity
    -> collective utility
```

That chain is not being tested in full here, and no novelty claim is frozen.

## Ground truth and paired units

Each item has a stable ID, prompt, exact target, and optional metadata. Baseline
and treatment use the same immutable item object. The paired unit is the item ID,
not a free-form response. Raw output and normalized answer are both retained.
The current JSONL adapter supports exact short labels; it does not use fuzzy LLM
judging.

## Development versus confirmatory work

This entire repository is DEVELOPMENT infrastructure. Debugging prompts,
parsers, layers, and tiny exploratory alpha sweeps is allowed. A future
confirmatory campaign must first freeze the choices listed in
[DEVELOPMENT_PROTOCOL.md](DEVELOPMENT_PROTOCOL.md).

## Kill criteria

The Q1 line should be treated as a kill-test, not a search for a positive:

- If treatment accuracy falls materially without a pre-specified compensating
  benefit, the intervention is not useful for this purpose.
- If error similarity does not move under competence-preserving settings, Q1
  does not support progressing to Q2 for that intervention family.
- If apparent movement disappears under parser, seed, task-split, or null-vector
  controls, treat it as a development failure or artifact.
- A negative result is a valid outcome and should remain easy to report.
