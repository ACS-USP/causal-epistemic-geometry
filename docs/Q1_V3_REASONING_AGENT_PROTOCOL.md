# Q1 V3 — Reasoning-Agent Structural Reset

Q1 V3 is a new DEVELOPMENT measurement protocol, not V1.3. It preserves the
project question while replacing the failed direct-answer instrument:

> Can local representation geometry of a frozen language model predict and
> causally control the covariance structure of its semantic errors?

The measured object is now a stochastic reasoning policy:

```text
(trajectory, y_hat) ~ pi_theta,v(. | x, seed)
```

The primary unit is a latent procedural problem and a seeded rollout. The
primary hard observable is still the exact semantic error `1[y_hat != y]`;
sampling is part of the policy, not a reason to weaken the oracle.

## Why the instrument changed

The preceding E3-10 direct instrument read first response-state candidate
logits with thinking disabled. It failed its frozen calibration rule on the
cached Qwen3-8B model. That outcome is an ablation result, not evidence that
procedural tasks cannot be used with a reasoning policy. Q1 V3 therefore keeps
the exact procedural oracle but evaluates full sampled reasoning trajectories
with `enable_thinking=true` and an exact final-answer contract.

The design is informed by nearby work, without treating any paper as evidence
for this project:

| Lesson | Q1 V3 consequence |
| --- | --- |
| [Geometry of Truth](https://arxiv.org/abs/2310.06824) separates internal representations from task-level truth structure. | Keep latent specifications, executable oracles, raw trajectories, and final semantic answers as separate provenance objects. |
| [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) and [Inference-Time Intervention](https://arxiv.org/abs/2306.03341) show that activation changes can alter behavior. | A behavioral change is not by itself epistemic complementarity; steering remains downstream of instrument qualification. |
| [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) illustrates why a low-dimensional intervention effect may be behavior- or channel-specific. | Require exact output parsing, matched seeds, independent seeds, and a secondary response-channel robustness view. |
| [There Is More to Refusal Than a Single Direction](https://arxiv.org/abs/2602.02132) is a reminder that apparently simple behavioral directions can have heterogeneous mechanisms. | Do not treat a future direction as a universal epistemic axis or collapse family-specific effects. |
| [CLRS](https://arxiv.org/abs/2205.15659) and [CLRS-Text](https://arxiv.org/abs/2406.04229) motivate executable algorithmic tasks with exact answers. | Use deterministic procedural generators and interpreters rather than a static question bank or an LLM judge. |
| [GSM-Symbolic](https://arxiv.org/abs/2410.05229) highlights sensitivity to symbolic surface variation. | Generate deterministic surface twins and audit their semantic agreement before steering. |
| [LiveBench](https://arxiv.org/abs/2406.19314) emphasizes freshness and contamination resistance. | Use fresh seed namespaces and disjoint calibration, development, and holdout manifests. |
| [Option-order sensitivity](https://arxiv.org/abs/2308.11483) and [answer-position representation/steering](https://arxiv.org/abs/2605.01846) expose the old slot confound. | Remove A/B/C/D positions entirely; decimal and word channels are semantic names, never displayed option slots. |

## Frozen policy configuration

The canonical Qwen3 reasoning policy is:

```yaml
model: Qwen/Qwen3-8B
model_revision: b968826d9c46dd6066d109eabc6255188de91218
dtype: bf16
quantization: none
enable_thinking: true
do_sample: true
temperature: 0.6
top_p: 0.95
top_k: 20
min_p: 0.0
```

The actual model revision is immutable and must be recorded in every rollout
manifest. The answer contract is deterministic:

```text
Reason through the problem. End with exactly one machine-readable line in the form FINAL: <answer>.
```

The parser takes the last valid `FINAL:` line outside a closed thinking block.
Missing, malformed, truncated, or unclosed-thinking outputs remain visible as
parse failures and count as incorrect. No fuzzy judge, explanation parser, or
post-hoc retry is permitted.

## Seed regimes and observables

For matched causal comparisons, baseline and intervention share the same seed
for each latent item and rollout. For independent-seed ensemble evaluation,
each agent receives an independent deterministic seed. The seed derivation uses
SHA-256-based stable seeds, never Python's process-randomized `hash()`.

Each rollout stores the full raw trajectory, generated token IDs, token-count
fields where available, stop reason, parse status, final answer, exact target,
intervention ID, policy configuration, model provenance, and seed.

For intervention `j`, the repeated-rollout error propensity is
`p[t,j] = P_seed(error)`. Pre-registered descriptive metrics include:

- propensity correlation and covariance;
- expected double fault and expected pair oracle;
- absolute and squared propensity distance;
- hard rollout phi/Jaccard and split-half reliability;
- excess pair oracle over a repeated baseline;
- exact plurality ensembles with baseline tie-break and canonical ordering.

These are separate observables. No one-number diversity score is introduced.

## Procedural families

The current model-free suite contains:

- `MODREG-R`: modular register execution, depths 4/8/12/16;
- `FSM-R`: composition of bijective ten-state transitions, lengths 4/8/12/16;
- `SATCOUNT-R`: exact raw satisfying-assignment count, cells with 4–6
  variables and 4–10 clauses.

All families have deterministic canonical and surface-twin renderings. There
are no displayed answer choices and no requirement that all families share the
same answer range. `SATCOUNT-R` deliberately does **not** reduce its answer
modulo ten. No number-word view is used to qualify the instrument; it is not
needed for this protocol and the old direct-channel qualification result is
kept as an E3-10 ablation.

The model-free gate generates at least 5,000 instances per family/cell and
checks determinism, exact oracles, serialization, latent IDs, twin invariance,
answer-distribution collapse, and shallow structural shortcuts. It is not a
model result. The current gate passes; two low-complexity SAT cells carry a
documented shortcut warning, not a failure.

## Calibration stages

No steering, activation extraction, PCA, random vector, DEV split, or holdout
is allowed during instrument qualification.

### Stage A — screen

For every eligible family/cell, one frozen set of 60 latent items is evaluated
under each budget in 512, 1024, and 2048. The latent IDs are identical across
the three budget conditions, while the two independent rollout seed identities
are also identical across budgets. A cell/budget
passes only if accuracy is 20–90%, parse success is at least 98%, and the
maximum seed accuracy gap is at most 15 percentage points.

If fewer than two families survive Stage A, print:

```text
REASONING_INSTRUMENT_SCREEN_FAILED
```

and stop.

### Stage B — calibration

Within each surviving family choose the Stage-A cell/budget closest to 55%
accuracy, breaking ties by lower budget and then lower structural difficulty.
Evaluate 200 new latent items, canonical and surface twin, with four
independent seeds. Qualification requires accuracy 30–80%, parse success at
least 99%, canonical/twin accuracy gap at most 7 percentage points, twin
semantic prediction agreement at least 70%, and across-seed accuracy SD at
most 7 percentage points.

If fewer than two families qualify, print:

```text
REASONING_INSTRUMENT_NOT_QUALIFIED
```

and stop. No steering is constructed.

Family selection is allowed only because it uses baseline-only instrument
stability and competence. It cannot depend on an observed intervention.

## Sampling-floor and answer-distribution audits

The suite does not force target balance. Instead, it reports the procedural
answer distribution and rejects structural answer collapse before model use:

- modal answer frequency >= 25%: warning;
- modal answer frequency >= 40%: failure;
- shallow structural predictor failure: failure under the model-free gate.

The full generated split is retained; no fresh item is filtered by model
correctness, confidence, margin, or response to steering.

## Fresh split firewall

After, and only after, at least two families qualify in Stage B, generate fresh
disjoint manifests per retained family:

```text
GEOMETRY_CALIBRATION: 400
STEERING_DEVELOPMENT: 400
CONFIRMATORY_HOLDOUT: 800
```

These are new latent seeds, disjoint from all calibration stages. Development
code cannot load the confirmatory manifest. This task authorizes instrument
qualification only; no Qwen evaluation on these fresh scientific splits is
performed now.

## Steering gate

One-shot scientific steering is **NOT READY** until the instrument qualifies.
If qualification succeeds, the future intervention will be frozen before
steering: add `alpha * v` once at the final prompt-token residual state at a
pre-registered layer, then generate the reasoning trajectory. The exact
intervention, controls, norms, and seed regime must be reviewed separately.

The present reset does not construct directions, run PCA, inspect DEV
outcomes, or access the confirmatory holdout.

## Status at protocol freeze

```text
Q1 V1–V1.2: CLOSED AS DEVELOPMENT
Q1 V2 / E3-10: NOT QUALIFIED; retained as direct-readout ablation
Q1 V3 model-free structural gate: PASS
Q1 V3 Stage A: NOT RUN AT PROTOCOL FREEZE
Q1 V3 Stage B: NOT RUN
Q1 V3 steering: NOT READY / NOT RUN
Q1 V3 fresh scientific splits: NOT GENERATED
Q1 scientific result: NONE FROZEN
Q2 geometry: NOT RUN
CONFIRMATORY HOLDOUT: UNTOUCHED
```

The next legitimate action at the time of protocol freeze was
principal-researcher review of the model-free design bundle, followed by a
cost-gated baseline-only Stage-A calibration on the remote RunPod. That review
and launch have since occurred. The live state is maintained separately in
[CURRENT_STATUS.md](CURRENT_STATUS.md); this protocol remains the normative
scientific specification and is not a progress log.
