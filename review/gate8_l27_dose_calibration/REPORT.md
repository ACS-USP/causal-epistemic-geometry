# GATE 8 — L27 DOSE CALIBRATION

## Premortem and lock

- Premortem: `PREMORTEM_PASS`.
- Class A recovery: the preserved CA-MTL-1 A40 was unavailable, so execution
  migrated prospectively to an equivalent NVIDIA A40 in EU-SE-1 with exact
  CORE_QWEN versions and exact model revision.
- Class B amendments: none.
- Experiment source commit:
  `17e7ae043c60df211ebce6c4d5893d01f2fd5816`.
- Scientific design changes after outcomes: none.

Gate 8 is calibration only. It does not compute or use G/C/D as primary
evidence, and it did not execute a later selected-dose evaluation.

## Freshness

- Historically used/reserved CRUXEval IDs excluded: 593.
- Eligible before allocation: 207.
- Calibration N: 50.
- Fresh IDs left unallocated: 157.
- Manifest logical hash:
  `42df77e79cb9111b2d6115c1a794a6d709c71238ee222a78815283421437ab17`.
- Frozen schedule: 2,200 unique rows, 22 conditions, two matched rollout
  blocks; zero missing, unauthorized, duplicate, or retried rows.

## Controller and engineering

The exact Gate-7 L27 plus paired-mean direction was reused, with canonical hash
`e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838`.
Full eta was `12.849903937136261`. Engineering classification was
`GATE8_ENGINEERING_PASS`: alpha-zero identity, dose scaling, random energy
matching, current-token scope, forward count, cache safety, hook cleanup, and
condition metadata all passed.

## Textual source anchor

- Classification: `CAREFUL_SOURCE_REPLICATED`.
- Baseline validity/evaluability/accuracy: 1.0000 / 1.0000 / 0.4500.
- Baseline mean/median tokens: 11.33 / 8.0.
- CAREFUL validity/evaluability/accuracy: 1.0000 / 1.0000 / 0.6800.
- CAREFUL mean/median tokens: 419.00 / 284.0.

## Frozen dose curve

| dose | eta | commitment | evaluability | accuracy | Q | random mean/max Q | rho tokens | rescue/damage | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D25 | 3.2124759842840653 | 1.000 | 1.000 | 0.450 | 0.050 | 0.045/0.070 | 0.000 | 0.000/0.000 | No |
| D50 | 6.4249519685681305 | 1.000 | 1.000 | 0.420 | 0.230 | 0.115/0.140 | 0.028 | 0.000/0.030 | No |
| D75 | 9.637427952852196 | 1.000 | 1.000 | 0.490 | 0.430 | 0.203/0.220 | 0.346 | 0.090/0.050 | Yes |
| D100 | 12.849903937136261 | 0.970 | 0.970 | 0.660 | 0.580 | 0.280/0.330 | 1.011 | 0.270/0.060 | Diagnostic only |

D25 was safe but inert. D50 was safe and changed semantic outcomes, but did not
reach the frozen 25% CAREFUL token-regime recovery. D75 passed all safety and
specificity gates. D100 also passed the calibration gates on this sample, but
was prospectively ineligible for selection.

No monotonicity violations were observed. The item-cluster 95% descriptive
intervals for D75 were Q `[0.31, 0.55]`, Q minus random mean
`[0.1275, 0.3325]`, accuracy difference `[-0.05, 0.13]`, and token recovery
`[0.1974, 0.5469]`. Intervals do not alter the deterministic selection rule.

## Dose selection

The frozen rule selects the lowest eligible lower dose in the order D25, D50,
D75. The selected dose is **D75**, eta `9.637427952852196`.

Primary classification: `GATE8_SAFE_LOWER_DOSE_SELECTED`.

This says only that D75 met the prospectively frozen safety and behavioral
first-stage calibration criteria. It is not independent G/C/D replication
evidence and is not a scientific success claim for the selected dose.

## Forensic audit and cost

Independent classification: `GATE8_FORENSIC_CLEAN`. The audit reparsed all raw
rows condition-symmetrically and independently recomputed Q, the random null,
token recovery, safety gates, eligibility and selection. Maximum primary/audit
metric difference was zero; classification and selected dose agreed exactly.

The A40 ran for 5,614 seconds (1.5594 h), costing approximately US$0.686 at
US$0.44/h. The Pod was stopped before analysis. Gate 9, Q2, character count,
and the confirmatory holdout were not run.
