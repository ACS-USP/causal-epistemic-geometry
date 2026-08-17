# Q1 V2 — E3-10 Exact Semantic Instrument

E3-10 is a new development measurement instrument, not V1.3. It preserves the
project question while removing the answer-position and aggregation confounds
that made the earlier MMLU-Pro instrument unsuitable for a clean semantic
test.

## Measurement choice

Every latent problem has an executable oracle `y(x) ∈ {0, ..., 9}`. The model
produces ten scores for the ten semantic outcomes themselves. The primary
prediction is the argmax over those scores. There are no displayed answer
slots, answer-order permutations, answer extractors, or judges.

The canonical channel asks for exactly one decimal digit. A prospective
number-word channel asks for the same semantic answer as `zero` through `nine`.
The two channels are reported independently; they are never averaged or
aggregated. Canonical and surface-twin views are also paired views of one
latent item, not independent statistical examples.

The procedural families are deliberately structurally different:

- `MODREG10`: four-register modular arithmetic with depths 4, 8, 12, and 16.
- `FSM10`: composition of three random bijections over ten states with sequence
  lengths 4, 8, 12, and 16.
- `REACHCOUNT10`: bounded reachability counts on ten-node directed graphs with
  the four pre-registered `(H, p)` cells.
- `SATCOUNT10`: exhaustive Boolean model counts modulo ten with four fixed
  variable/clause cells.

The SATCOUNT D1 support correction is recorded below. No family is selected
because it appears likely to respond to steering; selection occurs only from
baseline calibration stability and competence.

## Why this follows from the methodological lessons

- **Geometry and causal intervention.** [Marks & Tegmark, “Geometry of Truth”](https://arxiv.org/abs/2310.06824)
  motivates treating representation directions and interventions as testable
  mechanistic objects, not as evidence by themselves. E3-10 therefore keeps
  the semantic score vector and exact oracle visible before any steering is
  attempted.
- **Steering changes behavior and can trade off capability.** [Panickssery et
  al., CAA](https://arxiv.org/abs/2312.06681) and [Li et al., ITI](https://arxiv.org/abs/2306.03341)
  show why a behavioral shift must be evaluated with competence and direct
  scores together. E3-10 records hard error, semantic NLL, Brier score,
  true-answer margin, and the full ten-way score vector without collapsing them
  into a diversity scalar.
- **Avoid universal one-direction stories.** [Arditi et al.](https://arxiv.org/abs/2406.11717)
  provide a causal direction result for refusal, while [“There Is More to
  Refusal”](https://arxiv.org/abs/2602.02132) is a useful caution that behavior
  can involve multiple distinct directions and mechanisms. E3-10 includes
  structurally different task families and explicitly reserves universal
  geometry claims for later testing.
- **Procedural exact tasks are useful when the oracle is the object of study.**
  [CLRS](https://arxiv.org/abs/2205.15659), [CLRS-Text](https://arxiv.org/abs/2406.04229),
  and [GSM-Symbolic](https://arxiv.org/abs/2410.05229) motivate deterministic
  generators, exact executors, controlled difficulty, and fresh instances.
  MODREG10, FSM10, REACHCOUNT10, and SATCOUNT10 follow that pattern without
  relying on another language model to generate or validate data.
- **Freshness and objective evaluation matter.** [LiveBench](https://arxiv.org/abs/2406.19314)
  motivates explicit generation/version metadata and a split firewall. E3-10
  stores latent specifications, seeds, hashes, and oracle outputs so a split
  can be reconstructed independently.
- **Position effects are a direct threat to the old instrument.** [Option-order
  sensitivity](https://arxiv.org/abs/2308.11483) and recent work on [answer
  position representations and steering](https://arxiv.org/abs/2605.01846)
  make arbitrary multiple-choice slots an avoidable confound. E3-10 makes the
  candidate identity itself the semantic digit and uses surface twins only as
  a prospective robustness diagnostic.

One generator invariant was repaired before calibration: a three-variable
Boolean formula has at most eight satisfying assignments, so the originally
sketched three-variable SATCOUNT cell could never emit semantic target 9.
The D1 cell is therefore `vars4_clauses4`. This is a model-free support check,
not an outcome-driven choice; a ten-way balanced answer space cannot be
implemented honestly with an unreachable digit.

These references inform the design; they do not constitute a novelty claim for
E3-10.

## Frozen qualification protocol

Calibration is baseline-only. No PCA, activation extraction, random vector,
layer search, or steering direction may be constructed while selecting the
instrument. Each family/cell receives 200 fresh calibration latents, exactly
20 for every target digit. A cell qualifies only when all frozen thresholds
hold:

| Criterion | Threshold |
| --- | --- |
| Canonical decimal accuracy | 30% through 75% inclusive |
| Decimal/number-word semantic agreement | at least 85% |
| Canonical/surface-twin agreement | at least 80% |
| Normalized predicted-digit entropy | at least 0.80 |

Within each family, the qualifying cell closest to 50% accuracy is selected,
with lower structural difficulty as the deterministic tie-break. Every family
with a qualifying cell is retained. Fewer than two qualifying families means
the suite is not qualified and steering stops.

Calibration items are spent. Only after instrument review are fresh,
disjoint, target-balanced geometry, development, and confirmatory manifests
generated. The confirmatory manifest is inaccessible to development runners.

## Primary and secondary observables

For each item and future intervention, the primary epistemic error remains
`1[prediction != target]`. The direct secondary observables are semantic NLL,
Brier score, true-answer margin, and the centered ten-way score vector. Future
pairwise error covariance will use the existing paired diagnostics: accuracy,
the 2×2 table, rescues, damages, disagreement, Jaccard, phi, pair oracle, and
complementarity headroom.

If a future effect appears in decimal but not in number words, it is
**OUTPUT-CHANNEL-SENSITIVE**. If it disappears under a surface twin, it is
**SURFACE-SENSITIVE**. Neither effect is rescued by averaging channels.

No steering result, Q1 result, or Q2 claim is frozen by this design document.
