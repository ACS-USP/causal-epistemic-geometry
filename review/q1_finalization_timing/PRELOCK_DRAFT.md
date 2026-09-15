# Q1 D75 finalization timing — prospective prelock draft

**Status:** `OUTCOME_FREE_DRAFT_NOT_FROZEN`  
**Question:** On fresh items from the fixed Q1 generator law, does the frozen
layer-27 D75 controller shift the structural time at which Qwen3 exits its
thinking block?

This follows the closed Q1 budget study; it does not reopen, reweight, or
reinterpret that candidate. The endpoint is deliberately about the observable
generation transition, not about routing, competence, or a mediation claim for
correctness.

## Why this is the next question

The prior Q1 aggregate was compatible with a difference in delivered responses,
but it could not distinguish a change in solution quality from a change in
when a response is emitted. Qwen's published Qwen3 documentation defines
`</think>` as the transition between thinking and the regular response. The
new primary endpoint is that token event, mechanically extracted from raw token
IDs. No text, parse result, answer key, or correctness value enters it.

The documented Qwen3 thinking-budget method was considered and deliberately
not made the treatment here. It requires a second generation and an injected
early-stopping text; that would introduce a second policy intervention before
establishing whether D75 changes timing at all.

## Frozen design to materialize only after this draft passes audit

| Component | Candidate specification |
| --- | --- |
| Model/controller | Existing frozen Qwen3-8B, layer-27 D75 identity; exact candidate identity will be copied and re-hashed in the lock. |
| Population | 16 fresh latents in each of the existing 12 Q1 generator cells: 192 total. Historical latent IDs are excluded through an ID-only materializer. |
| Prompt | Canonical surface, unchanged. |
| Horizon | Exactly 4096 generated tokens for every trajectory. |
| Conditions | `BASELINE`, `D75`, using matched sampling seeds within latent × rollout. |
| Rollouts | Two per condition and latent: 768 trajectories, at most 3,145,728 sampled-token positions. |
| Event | First generated token with ID `151668` (`</think>`), validated against the pinned Qwen tokenizer before any model forward. |
| Censoring | If the event is absent, its restricted time is 4096, including an early terminal sequence without a closing token. |

For latent \(i\), rollout \(r\), and cap \(L=4096\), let \(T\) be the
one-based closing-token position, administratively censored at \(L\). The
latent contrast is

\[
Z_i=\frac{1}{2}\sum_{r=1}^2
  \frac{T_{i,r,\mathrm{BASELINE}}-T_{i,r,\mathrm{D75}}}{L}.
\]

Positive values mean D75 closes the thinking block earlier; negative values
mean it closes later. The analysis averages latents equally within cell, cells
equally within family, and families equally.

## Decisions and interpretation

A relevant timing movement is \(\delta=0.10\), or 409.6 tokens of the fixed
horizon. This is a scale for a practically visible change in response timing;
it was selected without looking at timing data.

Three fixed e-value declarations, each at threshold 60, will control the
family of directional timing claims:

1. D75 closes at least \(\delta\) earlier;
2. D75 closes at least \(\delta\) later;
3. both movements of at least \(\delta\) are excluded.

All use independent latent items as the statistical unit and bounded latent
contrasts in \([-1,1]\). If none triggers, the result is
`FINALIZATION_TIMING_INCONCLUSIVE_AT_N192`; there will be no added latents,
new cap, dose, or alternative timing marker.

A positive result establishes a causal shift in this **structural transition**
for the stated controller, model, renderer, generator law, and horizon. It
does not establish that timing caused an accuracy difference. A two-by-two
forced-finalization study would be considered only if this study establishes a
relevant timing movement. A negative exclusion result applies at the fixed
409.6-token scale; it does not rule out smaller shifts.

## Outcome-free planning

The fixed 60-cell sensitivity grid in `PLANNING_SENSITIVITY.json` uses only
synthetic bounded latent contrasts. At the deliberately adverse contrast SD of
0.30, N=192 had a 0.94 simulated support rate for a 0.20 normalized shortening
and a 0.89 simulated exclusion rate under zero shift. These are planning
operating characteristics, not a Qwen forecast. N=192 is chosen because it is
more resolving than 96 or 144 under that declared stress case while remaining
a single finite collection.

## Gates before a lock

1. Add a dedicated materializer and runner; do not reuse Q1's namespace or
   journal identity.
2. Verify token ID `151668` with the pinned tokenizer and record its tokenizer
   encoding and revision in the lock.
3. Verify the raw journal retains exact token IDs before parsing and that the
   structural timing analysis cannot read semantic outcome fields.
4. Run synthetic tests and an independent audit of manifest, schedule, runner,
   raw seal, and timing analysis.
5. Complete a Spark 1 engineering preflight without scientific generation.

No new scientific seed, manifest, prompt, model forward, response, or
correctness inspection is authorized by this draft.
