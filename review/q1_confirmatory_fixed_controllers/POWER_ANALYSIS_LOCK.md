# Q1 Confirmatory Power-Analysis Lock

This lock freezes the offline sample-size qualification before the power
simulation is run. The exact 57-ID confirmatory set remains
`SEALED_ASSIGNED_UNACCESSED`; its prompts, references, and outcomes are outside
the analysis path.

- Primary target: `C_meaningful > 0`.
- Target sample size: `N = 57`.
- Planning seed: `2026082301`.
- Outer item-level pseudoexperiments: `20,000`.
- Per-pseudoexperiment two-sided 95% percentile interval: `1,999` nested item
  bootstrap resamples.
- Adequacy: estimated primary-C power must be at least `0.80` for both Qwen and
  Ministral.
- Null specificity is descriptive and cannot replace the primary endpoint.
- Estimated safety-pass probability is not a stopping criterion.

Only the immutable Gate-9 and Gate-13.1 final DEVELOPMENT journals, bound by
their SHA-256 digests in the JSON lock, may be read.
