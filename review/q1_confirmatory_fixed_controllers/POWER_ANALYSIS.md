# Q1 Confirmatory Offline Power Qualification

Classification: `Q1_CONFIRMATORY_N57_POWER_QUALIFIED`.

The exact 57-ID confirmatory set remained sealed. This analysis used only binary
item-level outcomes from the immutable Gate-9 and Gate-13.1 DEVELOPMENT final
evaluations. No prompt, reference, model weight, or holdout outcome was accessed.

## Frozen planning design

- Target N: 57
- Outer pseudoexperiments: 20000
- Inner item-bootstrap resamples per pseudoexperiment: 1999
- Planning seed: 2026082301
- Interval: two-sided 95% item-percentile bootstrap
- Primary endpoint: C_meaningful > 0
- Adequacy threshold: estimated power >= 0.80 for both models

## Results

| model | C power | expected C CI width | specificity power | expected specificity CI width |
|---|---:|---:|---:|---:|
| Ministral | 0.9441 | 0.106107 | 0.9638 | 0.079596 |
| Qwen | 0.8790 | 0.085096 | 0.9426 | 0.077590 |

Null-specificity power is descriptive only and did not alter the frozen primary
endpoint or adequacy decision. Safety-pass probability was not used as a reason
to avoid the confirmatory test.

## Decision

Both models meet the prospectively frozen N=57 primary-C power threshold; dress-rehearsal and lock preparation may proceed without holdout access.
