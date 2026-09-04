# Q3.4 External-Review Integrity Audit

## 1. State at the safe pause

The qualification runner was stopped at a journal boundary after 35 complete
scientific rows. A 36th attempt was interrupted in flight and was not
persisted. No row was discarded or replaced, and no scientific output was
scored or inspected.

- executed HEAD: `dda4f6b40d371eaa93cde575838451d98b953fc6`
- schedule SHA-256: `edba56fc8435cdc34b6f7551fc2d1b4a6d4cc3d87fc34127a5096526d670a635`
- execution-lock SHA-256: `3a51a8d6d9fe57722f9ca740e1c5281e0645031f1641cb824c469ab4dc36635f`
- dataset-seal SHA-256: `1b889c3c1de9b6d20d93fca96e866322d4a75f672577dcffdc3e49571ea0da72`
- paused journal: 35 rows, 778,228 bytes, SHA-256
  `ec838d88494adc856deefca2ab35e2db230bdfdb84b5b3975108bcca4d56c154`
- engineering activity: 44 Qwen forwards and 4 excluded-fixture generations
- qualification activity: 36 attempts started, 35 complete rows, one
  interrupted pre-persistence attempt
- correctness inspected: **NO**
- confirmation/reserve Qwen access: **0 / 0**

## 2. Structural implementation audit

The original detector is inadequate for the stated family definition.
`canonical_skeleton()`, `normalized_token_sha256`, and `operation_tokens()` all
encode `kind:variant`; `structural_near_duplicate()` compares those same
tokens. Yet `render_program()` ignores `variant` for at least `AFFINE` and
`RECURSE`. The required excluded synthetic test therefore produces identical
source and identical normalized AST while the original family identity changes.
Conversely, a change from `AFFINE` to `BRANCH` changes both the original
identity and the normalized AST.

This is a confirmed **detector defect**, but not an observed material collision
in the sealed dataset. The independent source-only auditor decoded source and
allow-listed structural metadata, skipped prompts/references opaquely, executed
no program, and emitted no private examples. Its scope-aware `ast.parse`
canonicalizer normalizes identifiers and literal values while retaining AST
productions, operators, call targets, control flow, mutation, subscripts, and
use-definition relations.

Across the actual 1,600 records it found:

- 1,600 strict normalized-AST skeletons;
- 0 strict duplicate groups within a split;
- 0 strict duplicate groups across splits;
- 0 rendered-source duplicate groups;
- 0 relaxed candidates at the frozen 0.95 sequence/Jaccard thresholds;
- 0 original identities merging different strict ASTs;
- 0 strict AST identities split across multiple original identities in this
  realized sample.

Thus the sampled dataset satisfies the intended strict structural-separation
contract even though the original detector could have failed to enforce it.
This does not prove 1,600 probabilistically IID families. The defensible claim
is narrower: the deterministic, quota-structured generator stream produced
1,600 distinct normalized AST skeletons under the prospectively frozen
independent audit, with no cross-split structural collision detected.

Canonical aggregate artifact: `Q3_EXTERNAL_REVIEW_STRUCTURAL_AUDIT.json`,
SHA-256 `583f277fdd7256d9c83f5a9faabb5ddc6b59276f567a18d303c2b9901a4a73ea`.

## 3. Oracle-gate audit

The frozen gate is the family mean of the maximum across eight policies'
two-rollout empirical accuracies, minus champion mean accuracy. It remains
unchanged.

Under eight equally competent independent policies with `R=2` and `p=0.5`,
the analytic expected empirical maximum is `0.9499359130859375`, yielding
apparent headroom `0.4499359130859375` despite zero true specialization. In
200,000 frozen simulations, the corresponding values were `0.9497525` and
`0.4493975`.

Accordingly, this gate is now explicitly described only as an
**opportunity/upper-bound diagnostic**. Passing it alone does not establish
repeatable complementarity, policy selectability, or realized routing utility.
The historical gate and threshold are not changed. A bidirectional
rollout-transfer diagnostic is frozen for exploratory use only after a complete
sealed qualification; it is not run on the partial journal and cannot alter a
gate.

## 4. Power-mechanism audit

The historical `INDEPENDENT` branch did not draw router and champion
correctness independently. It drew a ternary paired difference under a chosen
discordance parameter. The audited alternative draws each correctness outcome
from its own Bernoulli stream conditional on family-level probabilities.

Across 50,000 panels per frozen cell, the independent-Bernoulli mechanism
matched the declared variance identity to a maximum absolute discrepancy of
`9.44e-16`. Null FPR was at most `0.05154`, and minimum two-sided 95% interval
coverage was `0.94854`.

For `N=1,000`, `R=2`, and a true +3 percentage-point gain:

- frozen historical reported power: `0.8233`;
- byte-equivalent-mechanism reproduction: `0.82244`;
- matched conditionally independent Bernoulli mechanism: `0.77456`;
- simple independent laws: `0.59978` to `0.66534`.

At `N=800`, the matched historical mechanism gave `0.74344`, while the matched
independent-Bernoulli mechanism gave `0.69032`.

Therefore 82.3% remains defensible only under the historical ternary
conservative-combined planning model; it is not a mechanism-free promise for
the new generator. This does not block instrument qualification, but the
confirmation power rationale requires explicit review before confirmation may
open. No `N`, `R`, alpha, test, interval, or criterion was changed here.

The external champion is not among the eight bank policies. No exact-policy
execution sharing is planned, and the independent frozen seed contract remains
in force.

Canonical aggregate artifact: `Q3_EXTERNAL_REVIEW_ORACLE_POWER_AUDIT.json`,
SHA-256 `bd5b0a24193741119d2e853b63301a15846250bf783bc23e83d0a17b025b437e`.

## 5. Qualification disposition

The independent audit found no material structural violation in the realized
dataset and requires no change to families, schedule, seeds, system, or gates.
The oracle interpretation has been corrected additively, and the power caveat
is recorded. The same qualification may therefore resume from the original
journal, executing only missing logical keys. The 35 complete rows remain
immutable; the interrupted attempt uses its original key and seed under the
existing operational retry policy.

Confirmation and reserve remain closed to Qwen inference.

## 6. Future positioning: what geometry may buy with less experimentation

The current candidate belongs to a broader family of prompt-conditioned
activation-steering portfolios. MoSV learns several contrastive steering
directions by clustering activation differences, then uses a sparse prompt
router to select and compose vectors. Steer2Adapt instead constructs a reusable
semantic prior subspace and uses few-shot Bayesian optimization to find a
task-level linear combination. Q3's narrower question is whether a
causally/relationally motivated fixed portfolio plus a one-forward prompt router
can deliver fresh-family utility. Merely combining a frozen model, vector bank,
and router is not itself a novelty claim.

Any future efficiency comparison must count subspace construction, controller
safety, calibration, policy evaluations, and router training as upfront cost,
then separately report amortized deployment cost. Primary sources:
[MoSV](https://openreview.net/pdf/cc1ec1cff4e96c3df39eda0a1d523bc7bdbb1037.pdf)
and [Steer2Adapt](https://arxiv.org/abs/2602.07276).

## 7. Firewall and remaining decision

- Qwen inference during audit: 0
- qualification rows scored: 0
- confirmation Qwen inference: 0
- reserve Qwen inference: 0
- Spark 1 use: CPU-only model-free simulation
- Spark 2 / RunPod: NO
- private program examples emitted: 0

No principal decision is required to resume qualification. Before any future
confirmation authorization, the principal must decide whether to accept the
revised power characterization or prospectively amend the confirmation design.

**Ruling:** `Q3_FRESH_INSTRUMENT_STRUCTURALLY_CLEAN_QUALIFICATION_RESUMED`
