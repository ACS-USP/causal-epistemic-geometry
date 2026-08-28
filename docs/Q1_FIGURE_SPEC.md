# Q1 figure specification

Status: `FROZEN_BEFORE_Q1_FIGURE_IMPLEMENTATION`

This specification governs publication figures derived exclusively from frozen
Q1 artifacts. It introduces no scientific test and has no import path to Q2
semantic outcomes. Item ordering, inclusion, visual scales, and panel contents
are fixed here before plotting code is written.

## Global rules

- All 57 confirmatory items are retained in the exact order stored in
  `review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json`.
- Every terminal invalid or semantically unevaluable row retains the frozen
  primary error value `e=1`; no complete-case filtering is allowed.
- Confirmatory item profiles use the empirical two-rollout error proportion
  `q_hat in {0, 0.5, 1}` and are explicitly labeled low-resolution observations,
  not precise latent propensities.
- The rescue/damage decomposition uses all four baseline-condition rollout
  cross-products. It never pairs rollout 0 with rollout 0 as a deterministic
  item transition.
- All four prospective random controls are shown wherever a null bank is used.
- DEVELOPMENT, CALIBRATION, CONFIRMATORY, NEGATIVE_BOUNDARY, and
  POST_HOC_DESCRIPTIVE_ONLY evidence are visually and textually separated.
- Effect panels include zero and use common scales within a metric. Validity
  panels use an honest 0--1 scale or a clearly marked inset with thresholds.
- No item, controller, stage, or domain is ordered by observed effect size.
- No Q2 artifact, module, directory, status, or semantic outcome is a source.

The machine-readable companion is
`manuscript/figures/paper1/FIGURE_SPEC.json`.

## Main figures

### Figure 1 — Causal control of where a model fails

- **Status:** `CONCEPTUAL`.
- **Sources:** Q1 analysis lock, controller identity lock, Qwen null-bank lock,
  and `docs/METRICS_AND_STATISTICS.md`.
- **Panels:** causal design schematic; illustrative equal-accuracy/different-
  error-profile toy grid; compact estimand legend.
- **Inclusion:** schematic values only, unmistakably labeled illustrative.
- **Supports:** the distinction between aggregate competence and error identity.
- **Does not support:** geometry prediction, routing utility, deterministic item
  outcomes, or domain-general steering.

### Figure 2 — The qualified instrument emerged through falsifiable gates

- **Status:** `DEVELOPMENT / CALIBRATION` genealogy.
- **Sources:** frozen Gate 4, Gate 5, Gate 6.2, Gate 6.3/V3, Gate 7, Gate 8, and
  Gate 9 reports.
- **Ordering:** immutable chronological gate order.
- **Panels:** one timeline with separate methodological-decision and observed-
  result lanes.
- **Supports:** the final instrument followed explicit failures, duration
  isolation, full-dose overshoot, prospective calibration, and fresh evaluation.
- **Does not support:** inevitability, Gate-5 success, or accuracy/G/C/D-based
  Gate-8 dose selection.

### Figure 3 — Qwen confirmatory itemwise blind-spot reorganization

- **Status:** `CONFIRMATORY`.
- **Sources:** exact holdout manifest, Qwen journal, confirmatory result,
  analysis lock, and Qwen null-bank lock.
- **Panel A:** baseline and meaningful two-rollout empirical error profiles for
  all 57 items in manifest order. Invalid/unevaluable errors receive a hatch
  overlay without changing `q_hat`.
- **Panel B:** four-way cross-rollout mass: shared correct, rescue, damage, and
  shared error. Values are means of all four rollout cross-products per item.
- **Panel C:** meaningful C with frozen 95% interval, all four random C values,
  random mean, and frozen meaningful-minus-random-mean interval annotation.
- **Supports:** safe, prospective-null-specific Qwen complementarity.
- **Does not support:** precise latent propensities, domain generality,
  geometric predictability, or deployable oracle selection.

### Figure 4 — Complementarity replicates more robustly than safe realization

- **Status:** `CONFIRMATORY`.
- **Sources:** shared confirmatory result, analysis lock, both controller locks,
  both null-bank locks, and forensic audit.
- **Panels:** aligned Qwen/Ministral C-plus-null facets; accuracy change; baseline
  and meaningful commitment/evaluability against frozen floors.
- **Ordering:** Qwen then Ministral; meaningful then random R0--R3.
- **Supports:** Qwen complete pass and Ministral positive/null-specific
  complementarity with an immutable safety fail.
- **Does not support:** a Ministral “partial pass,” cross-model safe confirmation,
  or use of post-hoc recovered answers.

### Figure 5 — Fixed Qwen controller does not transfer to long character counting

- **Status:** `DEVELOPMENT_POSITIVE / NEGATIVE_BOUNDARY`.
- **Sources:** Gate 9 and Gate 10 estimands plus controller identity lock.
- **Panels:** zero-referenced delta accuracy, C, and D; meaningful and all four
  corresponding random controls; fixed-controller identity strip.
- **Ordering:** CRUXEval then long character count; meaningful then R0--R3.
- **Supports:** a task-conditioned boundary for the exact Qwen L27-D75 controller.
- **Does not support:** equal task difficulty or impossibility of any transferable
  activation controller.

## Supplementary figures

### S1 — One-shot versus sustained duration history

`DEVELOPMENT`; Gate 4 and Gate 5 estimands; chronological and condition-fixed;
shows D and validity without calling Gate 5 a movement-gate pass.

### S2 — Full-dose overshoot and prospective D75 calibration

`DEVELOPMENT / CALIBRATION`; Gate 7 estimands and Gate 8 dose table; dose order
is D25, D50, D75, D100; accuracy is not a selection axis.

### S3 — Development-to-confirmation control banks

`DEVELOPMENT / CONFIRMATORY`; Gate 9, Gate 13.1, and Q1 confirmatory estimands;
stages remain separate and every four-vector null bank is shown.

### S4 — Shared-holdout dual-model itemwise profiles

`CONFIRMATORY`; both confirmatory journals; all 57 items in manifest order;
baseline and meaningful rows only; invalid/unevaluable rows remain errors.

### S5 — Ministral invalidity taxonomy

`POST_HOC_DESCRIPTIVE_ONLY`; uses only the frozen aggregate post-hoc artifact;
human-recovered answers do not change or enter confirmatory outcomes.

### S6 — Readout versus causal qualification

`OMITTED`: available stages use heterogeneous samples and non-comparable score
fields. A combined regression would be a new and potentially misleading
analysis.

### S7 — Leave-one-item-out robustness

`CONFIRMATORY_SENSITIVITY`; plots every frozen leave-one-item-out C and
delta-C value without selection or relabeling.

### S8 — Token-regime distributions

`CONFIRMATORY_DESCRIPTIVE`; uses frozen summary statistics and labels token
length as `CORRELATE / POSSIBLE MEDIATOR — NOT ESTABLISHED CAUSE`.

### S9 — Historical plus/minus sign results

`OMITTED`: Gate 4 and Gate 5 use a historical L17 controller while later stages
use L27. A compact sign plot would invite an unsupported cross-controller
comparison; the information remains in Figure 2 and S1.

## Output and provenance contract

Every implemented figure is emitted as SVG, PDF, and PNG. Derived scientific
tables are CSV with deterministic row order and stable float serialization.
The figure-data manifest records source hashes, derivation function, ordering,
inclusion, and table hash. The source manifest preserves the historical package
under `historical_package` and separately records this implementation version.
