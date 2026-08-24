# Paper 1 skeptical gap review

## Strongest defensible paper now

The strongest paper is a methods-and-phenomenon paper about causal control of
semantic error profiles. Its empirical center is a safety-qualified Qwen
confirmatory result, a cross-model Ministral complementarity/safety
dissociation, and a cross-domain negative that bounds generality. This is
stronger than presenting a sequence of steering wins because it defines an
estimand, uses repeated baseline and random-direction controls, preserves
failures, and separates causal movement from utility and answer-channel safety.

## Likely reviewer attacks

1. **One positive task family.** Both model families are positive on CRUXEval;
   the only objective cross-domain test is negative.
2. **Controller family concentration.** Both positive controllers originate
   from careful-versus-direct prompt-boundary contrasts, even though their
   architecture-specific realization differs.
3. **Verbosity/policy mediation.** Meaningful steering lengthens generation.
   Reviewers may ask whether complementarity is merely extra computation or
   format drift.
4. **Ministral safety fragility.** The confirmatory effect coexists with an
   0.886 commitment/evaluability rate.
5. **Finite null bank.** Four random controls are much stronger than one, but
   they do not characterize the full random-direction distribution.
6. **Holdout precision.** N=57 is prospectively powered for the frozen C test,
   not for every mechanistic subgroup or cross-model interaction.
7. **No geometry prediction.** Representation and causal evidence should not
   be marketed as a geometry result.

## Ranked evidence gaps

Scores are qualitative: value and risk from low/medium/high; cost uses measured
project history rather than invented throughput.

| Rank | Gap | Scientific value | Cost | Risk | Paper 1 or Q2? | Judgment |
|---:|---|---|---|---|---|---|
| 1 | Second positive objective task with fixed controllers and frozen evaluator | High | Medium-high | High | Paper 1, if truly independent | Largest acceptance gain; distinguishes semantic control from CRUXEval-specific program tracing |
| 2 | Prospective validity/commitment robustness study for Ministral | High | Medium | Medium | Paper 1 follow-up | Could test whether complementarity and answer-channel safety can coexist, but must not rescue the closed holdout |
| 3 | Alternative verified behavioral source, not careful/direct | High | High | High | Better reserved for Q2 bank | Reduces source-family concentration and creates geometry-identifying variation |
| 4 | Verbosity/computation mediator control | Medium-high | Medium | Medium | Paper 1 if tightly designed | Compare controller with token-budget or deliberation-matched controls without post-treatment truncation |
| 5 | Larger random-control distribution | Medium | Medium | Low | Paper 1 robustness | Useful, but less valuable than another source/task because current effects already exceed all frozen randoms |
| 6 | Third model | Medium | High | High | Q2/stretch | Adds breadth but may repeat architecture-specific engineering without resolving task generality |
| 7 | More confirmatory rollouts on the same 57 items | Low-medium | Medium | Medium | Neither without new lock | Improves propensity precision but risks looking like post-result sample expansion |
| 8 | More CRUXEval items from already consumed pools | Low | Medium | High | Not recommended | Adds volume without a clean independent allocation or new scientific dimension |

## The one missing experiment with the highest marginal value

A **second positive exact task** is the highest-value missing result. It should
use a frozen controller and dose, an objective evaluator frozen before outputs,
fresh items, repeated baseline, and a new random bank. The task must have a
verified behavioral source and enough baseline double-fault opportunity.

However, the project should not rush into another benchmark search. Candidate
tasks should first pass model-free evaluator checks and a small baseline/source
qualification whose decision does not inspect steering outcomes. A positive
task would materially strengthen Paper 1; another unstructured benchmark sweep
would merely add garden-of-forking-paths risk.

## Experiments that mostly add volume

- a third model using the same source but no new task;
- more random seeds after the current random-specific effects are already
  clear, unless the goal is explicit null-distribution estimation;
- extra rollouts added after seeing the holdout;
- another layer/dose search on CRUXEval;
- a character-count redesign chosen to reverse Gate 10.

## Recommended publication decision

Draft Paper 1 now around the strongest bounded claim. Before submission, seek
one of two upgrades, not both automatically:

1. a second positive objective task, if a pre-qualified instrument emerges; or
2. a prospective Ministral commitment-safety study that preserves the closed
   confirmatory fail.

Do not delay Paper 1 for exact pullback geometry. That belongs to Q2 and has a
different technical risk profile.
