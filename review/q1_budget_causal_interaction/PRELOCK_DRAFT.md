# Q1 Budget × D75 causal interaction — prospective prelock draft

**Status:** `DRAFT_NOT_FROZEN_NO_NEW_QWEN_OUTCOMES`  
**Branch:** `research/q1-budget-causal-interaction`  
**Purpose:** define one finite causal experiment prompted by the historical Q1-V3
completion failure. This is not a coverage qualification, a router study, or a
search for a task population that makes D75 look favorable.

## Question and estimands

For a fixed Qwen3-8B controller, does applying D75 change the unconditional
probability that a new reasoning item receives a correct answer within a fixed
generation budget? Does that effect differ between budgets?

For a latent item \(i\), budget \(L\), and rollout \(r\), let \(Y\) be one if the
answer is valid and correct, and zero otherwise. Incompleteness and invalid
answers are therefore outcomes, never exclusions. The paired latent contrasts
are

\[
d_{iL} = \frac{1}{2}\sum_{r=1}^{2}
  (Y_{iLr,\mathrm{D75}}-Y_{iLr,\mathrm{BASELINE}}),
\qquad
j_i=d_{i,4096}-d_{i,2048}.
\]

The primary estimands are the equal-latent, then equal-cell, then equal-family
means \(\Delta(L)\) of \(d_{iL}\), and the corresponding interaction \(\tau\) of
\(j_i\). A shared random seed within each baseline/D75 pair is part of the
estimand. The independent unit is the whole latent item, with its eight
trajectories; the twelve generator cells are fixed strata, not independent
replications.

## Frozen candidate treatment and population

The final lock must verify every value below before any new model forward pass.

| Component | Candidate value |
| --- | --- |
| Model and tokenizer | `Qwen/Qwen3-8B`, both at revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16/SDPA |
| Treatment | existing Q1 layer-27 D75 controller; hook scope `sustained_current_token` (final prompt token and current decode token) |
| Direction | `review/gate6_2_first_stage_repair_mean_bridge/PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy` |
| Direction SHA-256 | `b1630039fcbb829028a0e8f9f521d7e87bb24e831bc81c74a1591a6c39f40772` |
| Canonical float64 direction SHA-256 | `e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838` |
| Dose | `eta = 9.637427952852196` |
| Conditions | `BASELINE`, `D75` |
| Prompt surface | `canonical` only |
| Reasoning caps | exactly `2048` and `4096` generated tokens |
| Rollouts | exactly two per condition, cap, and latent |
| Seed regime | matched baseline/D75 seed inside each latent × cap × rollout; distinct across cap and rollout |

The candidate population contains eight newly generated latents in every cell
of the already frozen three-family, twelve-cell Q1-V3 generator map: 96 latents
and 768 trajectories. The public generator namespace is
`Q1-V3-BUDGET-CAUSAL-INTERACTION-V1`. Its schedule contains 384 calls per cap
and at most 2,359,296 generated tokens. There will be no replacement of a
latent, cell, family, cap, dose, or rollout after the schedule is locked.

Before materializing the new manifest, the historical Stage-A manifest will be
read only through `scripts/materialize_budget_interaction_prelock.py`. That
command calls `extract_historical_latent_ids()`, extracts only IDs, and never
opens outcomes or journals. Its frozen source digest is
`2a0cce17262f9f33bf8820f0e234eb02e18121a4c88ffc7e9e4a41473013a66b`.
All extracted IDs are excluded from the new manifest. The command writes the
new manifest, complete schedule, and a provenance record with all three
digests; it refuses to overwrite an existing artifact set by default. Those
digests will be added to the final lock before execution.

## Outcomes and raw-data rules

Every scheduled trajectory belongs to exactly one of:

1. `correct`;
2. `valid_wrong`;
3. `incomplete` (`MISSING_FINAL`, `THINKING_UNCLOSED`, or
   `TRUNCATED_NO_FINAL`);
4. `invalid` (`INVALID_FINAL`).

The primary analysis uses `correct` unconditionally. The other three rates are
reported by cap and condition to describe the mechanism without conditioning on
completed responses. A stop caused by the cap is stored as
`max_new_tokens`, then parsed as `TRUNCATED_NO_FINAL` when no final answer is
present.

The append-only journal key is `(latent_id, cap, condition, rollout_index)`.
It fsyncs each row, validates a caller-supplied identity hash, and permits only
an incomplete final JSONL tail to be quarantined and recovered. On restart it
validates all persisted rows against the frozen schedule and raw response
before any generation. Exact existing rows are reused; only an absent logical
key may be generated. A conflict, a non-final corrupt row, or an identity
mismatch terminates collection without substitution or rerun.

## Predeclared decision rule

The practical relevance threshold is \(\delta=0.10\): a ten-percentage-point
absolute gain in correct answers at a given cap. It is a threshold for deciding
whether to promote this fixed controller, not a claim that smaller effects lack
scientific interest.

For latent weights \(q_i\), let \(N=96\), \(a_i=Nq_i\), and
\(c=1/(2\max_i a_i)\). For any latent contrast \(z_i\in[-1,1]\), define

\[
E_+(m;z)=\prod_i[1+c a_i(z_i-m)],\qquad
E_-(m;z)=\prod_i[1+c a_i(m-z_i)].
\]

The implementation fixes a threshold of 60 for each of these three rules:

| Declaration | Statistic |
| --- | --- |
| Relevant gain at at least one cap | \([E_+(\delta;d_{2048})+E_+(\delta;d_{4096})]/2\ge60\) |
| A gain of at least 10 points excluded at both caps | \(\min\{E_-(\delta;d_{2048}),E_-(\delta;d_{4096})\}\ge60\) |
| Nonzero cap interaction | \([E_+(0;j/2)+E_-(0;j/2)]/2\ge60\) |

These are finite, fixed-rule tests under independence between latent items.
They do not use a normal approximation, relabel matched treatments, or call a
bootstrap an inferential decision. The three threshold-60 declarations together
have a predeclared Markov/union-bound error budget of at most 5%. Descriptive
stratified resampling remains supplementary.

Results may trigger more than one compatible statement and will report every
triggered rule. If none triggers, the result is `INCONCLUSIVE_AT_N96`; it is
not a license to add items or modify the experiment.

The candidate closes after one collection and one sealed analysis in every
case:

- relevant gain demonstrated;
- gain of 10 points excluded in both caps; or
- inconclusive at the fixed 96-latent design.

No outcome opens a second cap, a larger sample, a different generator law, a
different dose, selective removal of SATCOUNT, or a second controller inside
this candidate.

## Gates before freezing and execution

The status can change from draft to `FROZEN_NOT_RUN` only after all of the
following have been recorded together:

1. source commit and hashes of the manifest builder, schedule builder, runner,
   journal, parser, and decision-rule code;
2. verified historical-ID exclusion set, regenerated new manifest, complete
   schedule, and their hashes;
3. verified controller vector file and float64 hashes, layer, dose, model and
   tokenizer repositories/revisions, dtype, attention backend, decoding
   configuration, and hook scope.
   The runner checks those frozen values against `backend.provenance()`,
   `backend.config`, and the live `Intervention` object before its first call;
4. a hash-pinned journal identity containing those values;
5. passing focused unit tests and a fresh independent audit of the prelock
   implementation; and
6. a Spark 1 health preflight with no scientific model generation.

No model correctness, response text, or aggregated outcome may be inspected
before the completed raw journal is sealed and its schedule completeness and
identity are mechanically audited.

## Interpretation limits

This study concerns one fixed controller, this model revision, these synthetic
reasoning families, canonical rendering, and two stated token caps. It does
not establish transfer to code, a general property of steering vectors, a
mechanism mediated specifically by completion, or a deployable routing policy.
