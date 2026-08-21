# Experiment ladder

This is a prospective sequence, not authorization to execute model work.
Machine-readable drafts live under [`experiments/specs/`](../experiments/specs/).

## Gate 0 — offline repairs

Complete tied-rank correction, paired-metric semantics, stochastic estimands,
seed regimes, synthetic validation, state registry, and documentation checks.
No model, GPU, benchmark outcome, intervention, or holdout access.

## Gate 1 — full-generation non-thinking smoke

The repository has tested thinking and first-state non-thinking readout, but not
normal full non-thinking generation under the same model. Prospectively compare
Qwen3-8B native thinking with full non-thinking generation on fresh tiny
charcount/CRUX semantic samples and, only if objectively available, one
LiveBench-like task. Start with five items per arm and use successive halving.

Measure valid completion, conditional semantic accuracy, genuine wrong mass,
and token cost. The cap is an operational ceiling, not a tuned difficulty
parameter. No steering.

## Gate 2 — published positive control

Reproduce one technically adjacent published activation intervention with
public methodology and exact evaluation. Freeze model, layer, direction
construction, prompts, alpha, and manipulation check before outcomes. A failed
positive control blocks scientific interpretation of a null original-Q1 pilot.
See [the positive-control protocol](POSITIVE_CONTROL_PROTOCOL.md).

## Gate 3 — model × policy × benchmark substrate race

Race a small set of substrates rather than repeatedly repairing one:

- Qwen3-8B native thinking;
- Qwen3-8B full non-thinking generation;
- the selected positive-control model/policy (prospectively Llama-3.1-8B if
  licensing/access permits).

Use fresh charcount, semantic CRUX-like items, and dense code only after its
objective evaluator is safe. Five items per arm; add fifteen only for survivors;
use two seeds only for the best one or two arms. Optimize for completion,
genuine correct/wrong mass, evaluator clarity, token cost, and baseline
resampling structure—not closeness to a pretty 55% number.

## Gate 4 — first micro-Q1

On one qualified instrument, compare:

- baseline, independent seed bank A;
- baseline resampling, independent seed bank B;
- `+alpha * v`;
- `-alpha * v`;
- norm-matched random direction;
- `alpha=0` identity.

Include a known-positive implementation and an activation manipulation check.
Use 20–50 DEVELOPMENT problems. Freeze a competence tolerance and futility rule
before execution. Primary questions are whether error movement exceeds baseline
resampling and random-direction controls while preserving competence. Report
`C_j` and the unbiased two-rollout distance as designated estimands, with their
limitations.

### Psychometric calibration without steering leakage

Two prospective patterns are allowed:

1. **Generator calibration.** Use baseline-only data to select a generator
   parameter or structural cell by a frozen rule, then generate entirely fresh
   evaluation items.
2. **Pool calibration.** Use baseline seed bank A to choose a balanced item pool
   by a frozen psychometric rule, then evaluate baseline and intervention with a
   new seed bank B. Confirmatory work uses a fresh pool.

Neither pattern may inspect intervention outcomes during item or difficulty
selection. Item views from the same latent problem remain one statistical
cluster.

## Gate 5 — future Q2 geometry

Only after an identifiable micro-Q1 effect:

1. flat cosine and normalized Euclidean geometry;
2. held-out regularized activation-covariance/whitened metric;
3. behavioral pullback/pushforward or pre-specified finite differences.

Compare intervention and error-distance matrices by direction-label
permutation. Do not run an indiscriminate layer × vector × alpha sweep and do
not introduce manifold machinery.

## Gate 6.3 — single-mean semantic closeout

Gate 6.3 completed the conditional single-L27 semantic evaluation with an
architecture-matched random bank. Its frozen outcome was
`GATE6_3_SINGLE_MEAN_DESTRUCTIVE` because the meaningful controller failed the
validity guard despite point-estimate movement beyond the random controls.
No follow-up controller, alpha, layer, benchmark, Q2, or holdout action is
authorized without a new prospective lock and principal review.

## Hard portfolio boundary

Continue past the structured sprint only if all are true:

1. at least one instrument yields high completion and genuine intermediate
   semantic errors;
2. the published positive control reproduces;
3. the micro-Q1 identifies intervention-dependent error movement beyond
   ordinary resampling with acceptable competence.

Otherwise pause B′ pending materially better model access, instrument design,
or external evidence. Preserve the repository as an audited negative
instrument-development record.
