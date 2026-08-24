# Q2 controller bank design

Status: inventory and prospective design. The first frozen K=16 candidate bank
failed its pre-panel qualification: 3/16 controllers passed the causal
manipulation gate and the shuffled-null construction failed the frozen
orthogonality requirement. The 120-item common panel and predictive geometry
analysis were not run. “Usable” means eligible for a future DEVELOPMENT bank
under a new lock, not already authorized for another Q2 experiment.

## Existing artifact inventory

| Model | Controller/source | Layer/dose/operator | Construction and readout | Causal/safety evidence | Q2 use boundary |
|---|---|---|---|---|---|
| Qwen3-8B | Gate 4 careful-minus-direct | L17, alpha 8.399, one-shot final-prompt-token | Paired mean; direction validation/readout passed | No detectable movement beyond baseline/random | Useful inert anchor; historical evaluation sets differ from later gates |
| Qwen3-8B | Gate 5 careful/direct signs | L17, same alpha, one-shot and sustained | Same source, duration variants | Source behavior passed; sustained manipulation passed; primary movement below threshold | Useful timing/sign anchors, not positive controllers |
| Qwen3-8B | Gate 6 RFM/AGOP atlas | Multiple prompt/execution-boundary layers | Label-free RFM/AGOP and held-out source readout | No RFM controller passed frozen first-stage gate | Valuable readout-without-control bank; needs common-panel behavioral characterization |
| Qwen3-8B | Gate 6.2 paired means | Prompt-boundary L22/L27/L32 and multilayer | Paired mean; all three source-only passes | Plus arms changed behavior but harmed validity; minus did not beat random mean | Destructive/null anchors; do not select by historical outcome magnitude |
| Qwen3-8B | Full-dose paired mean plus | L27, eta 12.8499, sustained current-token | Exact frozen vector hash `e7bf…1838` | Gate 7 large G/C/D and accuracy gain, validity/evaluability 0.90; destructive by frozen guard | High-gain/unsafe anchor on a new common Q2 panel |
| Qwen3-8B | Calibrated paired mean plus | L27-D75, eta 9.63743, sustained | Same exact vector, lower dose | Gate 9 strong safe DEVELOPMENT; Q1 Qwen full confirmatory pass | Positive anchor; confirmatory outcomes must not be reused as Q2 test outcomes |
| Qwen3-8B | L27-D75 cross-domain transport | Same controller on character count | No adaptation | Gate 10 safe no-transfer negative | Domain-boundary evidence, not a second positive bank observation |
| Ministral-3-8B | Careful/direct source atlas | All 34 language layers | Paired mean; all layers source-eligible | Gate 13 four readout-shortlisted D50 cells had no safe specific first stage | Strong readout/no-control candidates; common-panel characterization needed |
| Ministral-3-8B | All-layer/D50 sweep candidates | 34 layers; disjoint dose qualification on L16/L18/L27 | Paired mean, architecture-specific scale | Multiple safe and destructive cells; L27-D25 selected without accuracy ranking | Rich within-model layer/dose bank, but Stage A/B sample sizes and items differ |
| Ministral-3-8B | Selected L27-D25 | eta 4.46991, sustained current-token, hash `0c46…2b94` | Paired careful-minus-direct mean | Gate 13.1 strong DEVELOPMENT; Q1 positive C but safety fail | Positive/high-fragility anchor; requires new developmental outcomes for Q2 prediction |
| Both | Architecture-matched random banks | Matched layer/dose/timing | Orthogonal Gaussian directions | Finite null distributions at each gate | Include new frozen randoms in Q2; do not pool random vectors across hidden spaces |

## What the inventory does and does not provide

The repository contains many directions but not a valid common Q2 matrix. They
were evaluated on different item sets, policies, stages, doses, and seed
regimes. Joining historical C/G/D values into a geometry correlation would be
post-hoc, confounded, and dyadically incomplete.

Existing vectors can seed a prospective bank if they are frozen before new
behavioral outcomes and all bank members are evaluated on the same panel. The
bank must include inert, destructive, random, and positive controls; excluding
failures would bias the map.

## Minimum viable bank

Minimum viable size: **K=16 intervention conditions within one model**, plus an
unsteered baseline and textual source anchors outside K.

Recommended composition:

- 3 independently verified behavioral source axes;
- 2 prospectively fixed layers per source;
- both signs at one energy-matched dose: 12 meaningful controllers;
- 4 new architecture-matched random directions: total K=16.

The source axes must not be renamed variants of careful/direct. Candidate axes
need an outcome-independent behavioral first stage, for example verification
policy, representation/format discipline, or decomposition policy. Exact
prompts and qualification thresholds require principal review.

At N=120 items and two independent rollouts, K=16 plus baseline implies

\[
120\times17\times2=4{,}080
\]

scientific trajectories. Adding two textual anchors would raise this to 4,560.
These are design counts, not execution authorization.

## Ambitious bank

Ambitious size: **K=40 per model**.

One balanced construction is:

- 5 verified source axes;
- 3 layers per source;
- both signs at the primary dose: 30 controllers;
- 6 predeclared second-dose anchors distributed across source families;
- 4 new random directions: total K=40.

At N=200 and two rollouts, K=40 plus baseline implies 16,400 trajectories;
two textual anchors imply 17,200. A two-model version doubles trajectory count
but must be analyzed as two separate control spaces before meta-analysis.

## Storage and compute planning

Historical confirmatory journals give an empirical raw-row storage range:

- Qwen: 5,995,795 bytes / 798 rows, approximately 7.5 KB/row;
- Ministral: 32,170,782 bytes / 798 rows, approximately 40.3 KB/row.

Using that observed range only:

- 4,080 rows imply roughly 31–164 MB of raw journal;
- 4,560 rows imply roughly 34–184 MB;
- 16,400 rows imply roughly 123–661 MB;
- 17,200 rows imply roughly 129–693 MB.

Full-vocabulary per-token logits can exceed journals by orders of magnitude.
Their storage must be estimated from the frozen vocabulary size, selected token
checkpoints, dtype, and compression policy before collection:

\[
\text{bytes}=N_{rows}\times N_{checkpoints}\times |V|\times
\text{bytes per value}.
\]

Do not promise full-logit persistence without this calculation and a recovery
test. Prefer sufficient statistics or explicitly selected checkpoints when
they answer the frozen metric.

GPU time and cost are not extrapolated as one universal rows/second number.
Historical measured runs vary materially with model, task, output length, and
condition: Gate 9 used 1,400 Qwen trajectories in 1.74 billed A40 hours; Gate 10
used 2,800 long character-count trajectories in 11.80 hours; Gate 13/13.1 used
a different model and operational history. A future protocol must run a
non-scientific preflight and project each frozen condition mix with a safety
margin.

The minimum bank plausibly fits one 8B model on an A40-class GPU under the
existing inference engine, but DGX Spark suitability must be measured rather
than assumed. Exact local derivative or large full-logit work may need a
different memory plan.

## Bank construction rules

1. Freeze source prompts and verify source behavior before controller outcomes.
2. Freeze all vectors, signs, layers, doses, normalization, and random seeds.
3. Use one common item panel and independent rollout schedule across K.
4. Measure geometry before semantic outcome reveal.
5. Split held-out controllers by source family, not random dyads.
6. Preserve invalid and destructive controllers.
7. Do not let accuracy rank bank members.
8. Do not mix model families in Euclidean coordinates.
9. Keep baseline and textual anchors explicit but outside the K geometry bank
   unless their embedding in H is formally defined.
10. Reserve a fresh controller/source family for the final prospective test.
