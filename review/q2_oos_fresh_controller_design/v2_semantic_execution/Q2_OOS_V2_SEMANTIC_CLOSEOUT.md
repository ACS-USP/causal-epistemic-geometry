# Q2 OOS V2 Fresh-Controller Semantic Closeout

## 1. Raw-data seal and collection integrity

The sealed raw collection was independently revalidated before scoring.

| Field | Result |
|---|---:|
| Raw-data status | `COLLECTION_COMPLETE_RAW_UNSCORED` |
| Journal rows | 19,200 / 19,200 |
| Journal bytes | 127,968,010 |
| Journal SHA-256 | `24fdd1c818c6e507f2e1999ce6e5da380405bc533af60723da01c1ec2bd66a40` |
| Missing / unexpected | 0 / 0 |
| Conflicting duplicates | 0 |
| Replacements / retries | 0 / 0 |
| Runtime errors | 0 |
| Repetition stops / hard caps | 359 / 2 |
| Collection wall time | 33,924.2938926 s (9h25m24.2939s) |

The raw seal predates scoring. Correctness was first inspected only after the
raw seal: YES.

## 2. Frozen scoring integrity

Scoring used the frozen `external-semantic-v3` parser/scorer. The scored
dataset contains 19,200 rows, uses no complete-case filtering, and preserves
the frozen convention `e=1` for invalid, unevaluable, repetition-stop,
hard-cap, and other incorrect terminal rows. No manual adjudication, parser
tuning, or outcome-based row exclusion was performed.

The private scored-dataset SHA-256 is
`9f03d96d40839e228d6cfb55408ea056e262fbf7e9aef2e863080e035e4b721b`.

## 3. Basic semantic descriptives

These summaries are descriptive and are not the primary OOS test.

| Shell | Rows | Commitment validity | Semantic evaluability | Accuracy | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| MEDIUM | 9,600 | 0.9859375 | 0.9859375 | 0.4637500 | 13.7399 |
| STRONG | 9,600 | 0.9691667 | 0.9691667 | 0.4458333 | 26.0132 |
| Overall | 19,200 | 0.9775521 | 0.9775521 | 0.4547917 | 19.8766 |

There was no baseline rerun in OOS V2; the frozen design uses the sealed
historical reference atlas. The descriptive terminal counts were:

| Shell | Valid correct | Valid wrong | Invalid format | Repetition stop | Hard cap |
|---|---:|---:|---:|---:|---:|
| MEDIUM | 4,452 | 5,013 | 25 | 109 | 1 |
| STRONG | 4,280 | 5,024 | 45 | 250 | 1 |

## 4. Blind-spot construction

The analysis reconstructed the frozen two-rollout binary error profiles,
`Dtotal`, and `Dshape` for fresh-to-reference and fresh-to-fresh blocks. The
31-controller historical atlas and its frozen itemwise profiles were reused;
negative unbiased estimates were retained and no clipping was applied.

Private derived artifact hashes:

- error arrays: `6c0e555f7cccba0c41415c7605d161e84f99ba705b6430eaac53d2527b05086c`
- `Dshape`: `a6a6b4889e2c86df04ce42c4415281dde82af0d2deb1347b8083015e95089ea5`
- `Dtotal`: `354db2363a845c654eafa00a3865b28ee04158978dc035a8128d95cf58a05ed9`

## 5. All 16 fresh-controller r_i

Values are in the frozen selected-controller order.

| Fresh controller | r_i |
|---|---:|
| `Q2_OOS_V2_DIRECTION_01` | 0.764755916 |
| `Q2_OOS_V2_DIRECTION_02` | 0.513508065 |
| `Q2_OOS_V2_DIRECTION_03` | 0.471922575 |
| `Q2_OOS_V2_DIRECTION_04` | 0.834475806 |
| `Q2_OOS_V2_DIRECTION_05` | 0.614752420 |
| `Q2_OOS_V2_DIRECTION_06` | 0.763850315 |
| `Q2_OOS_V2_DIRECTION_07` | 0.740862927 |
| `Q2_OOS_V2_DIRECTION_08` | 0.721141248 |
| `Q2_OOS_V2_DIRECTION_09` | 0.644153226 |
| `Q2_OOS_V2_DIRECTION_11` | 0.735015197 |
| `Q2_OOS_V2_DIRECTION_13` | 0.703427419 |
| `Q2_OOS_V2_DIRECTION_14` | 0.818548387 |
| `Q2_OOS_V2_DIRECTION_15` | 0.459274194 |
| `Q2_OOS_V2_DIRECTION_16` | 0.822117968 |
| `Q2_OOS_V2_DIRECTION_17` | 0.729206577 |
| `Q2_OOS_V2_DIRECTION_18` | 0.721107708 |

## 6. Primary exact sign test

The primary statistic is the frozen average of the MEDIUM and STRONG
controller-level Spearman associations over the 31 fixed historical
references. All 16 values were finite.

- Positive: 16
- Zero: 0
- Negative: 0
- Mean r_i: 0.691132497
- Median r_i: 0.725173912
- Exact one-sided Binomial p-value: `1.52587890625e-05`
- Frozen positive-count requirement: 12 / 16 — PASS
- Frozen alpha=.05 criterion — PASS

## 7. Global fresh×old descriptive results

For A0, the global equal-shell mean is `0.6430547122`; shell-specific values
are MEDIUM `0.6560618417` and STRONG `0.6300475827`. This global rho is an
effect-size/descriptive summary, not the primary inferential unit.

## 8. A1/A2/D2 secondary results

All rows below are secondary and cannot rescue or replace the A0 primary.

| Geometry | Global equal-shell mean | MEDIUM | STRONG | Fresh-controller mean | Fresh-controller median |
|---|---:|---:|---:|---:|---:|
| A1 | 0.637812195 | 0.652877464 | 0.622746925 | 0.685203660 | 0.719358104 |
| A2 | 0.536587891 | 0.567306343 | 0.505869438 | 0.575754133 | 0.608128848 |
| D2 | 0.477605720 | 0.501197590 | 0.454013849 | 0.545020183 | 0.547959410 |

The exact positive-sign p-value was `1.52587890625e-05` for each of A1, A2,
and D2; the frozen Holm-adjusted value was `4.57763671875e-05` for each.

## 9. Studentized/bootstrap/LOFO sensitivities

- Studentized controller-level mean: mean `0.691132497`, SE `0.029850245`,
  t `23.153327364`, p `1.8728658555474e-13`.
- Controller-cluster bootstrap (10,000 resamples): estimate `0.643054712`;
  percentile 95% interval `[0.577699464, 0.706390518]`; basic 95% interval
  `[0.579718906, 0.708409961]`.
- Frozen item bootstrap (50,000 resamples): global equal-shell quantiles
  q025/q50/q975 = `0.476502275 / 0.540055626 / 0.596536019`; median row
  association q025/q50/q975 = `0.535398003 / 0.609612792 / 0.671642080`.
- All 16 leave-one-fresh-controller-out checks retained 15 positive
  controllers; positive-sign p was `3.0517578125e-05` in every fold. The
  omitted-controller fold means ranged from `0.681576276` to `0.706589717`.
- Original row-QAP: p `2e-05`, explicitly diagnostic-only.

## 10. Fresh×fresh secondary

The frozen `NODE_JACKKNIFE_PSEUDOVALUE_T` secondary gave full association
`0.646591196`, jackknife SE `0.061688118`, t `10.536655113`, and p
`1.2505954637198264e-08`. It is secondary-only and cannot rescue the
fresh×old primary.

## 11. Efficient-termination/runtime audit

The run recorded 359 `EXTREME_MECHANICAL_REPETITION_V1` stops and 2 hard-cap
stops. Total generated tokens were 381,630: mean `19.8766`, P50 `7`, P90
`22`, P95 `35`, P99 `256`, maximum `4096`. Summed per-row generation time was
33,064.6219 s; summed elapsed row time was 33,077.9313 s.

The longest 1%, 5%, and 10% of rows accounted for respectively 25.76%,
53.83%, and 60.83% of summed row elapsed time. Repetition and hard-cap rows
accounted for 8,286.5455 s and 743.2556 s respectively. A counterfactual
compute saving is not estimated from terminal outputs alone; the terminal
policy was applied exactly as frozen. No valid/evaluable edge case inconsistent
with the termination contract was found.

Observed wall time was 9.4234 h, versus the frozen forecast of approximately
9.76 h at P50, 9.90 h normal mean, 11.05 h P80, and 11.81 h P90. The observed
run was therefore within the pre-run forecast and below its central P50
estimate.

## 12. Independent forensic audit

Status: `Q2_OOS_V2_FORENSIC_CLEAN`.

The independent path reopened the sealed journal and reconstructed all 19,200
rows. Maximum differences were zero for error arrays, `Dshape`, and all 16
primary r_i values. Primary and forensic classifications agreed exactly:
`Q2_OOS_V2_A0_PASS`.

## 13. Final mechanical classification

`Q2_OOS_V2_A0_PASS`

## 14. Scientific interpretation and claim boundary

The strongest supported statement is narrow: within the same frozen
Qwen3-8B / CRUXEval / rank-8 intervention laboratory, positive A0 relational
alignment generalized across the 16 prospectively sampled,
safety-conditioned fresh controller identities under the frozen primary
sign-test criterion.

This does not establish cross-task, cross-model, random-subspace, universal,
architecture-general, or manifold-level generalization, and it does not
establish Q3 utility. Historical Q2 V4.1 remains `Q2_V4_1_G2` with `RS+` and
`RT+`; it was not changed or pooled with this result.

## 15. Reviewer/fragility audit

The main remaining fragilities are controller identity generalization beyond
this 16-controller safety-conditioned sample, the same-model/same-task/
same-panel design, dependence on the fixed 31-controller historical atlas, and
the possibility that generic local smoothness explains part of the observed
alignment. The result strengthens the case for the predeclared matched random
rank-8 subspace control as the next discriminative experiment, but that
control was not launched automatically.

## 16. Repository/resource state

- Branch: `research/q2-fresh-controller-oos-design`
- Ratified prediction-lock parent: `170dd50925c35e32a2439576f901bab1cf31eb7d`
- Selected-bank SHA-256: `9a544b4ec6d43ec1c3530feb963cd0340db516e82f91a40c2624300483e2e0fd`
- Schedule SHA-256: `dac5c284b90c726016968f31d25200a362c42d96f63b63d730665f3f47e85ec5`
- Matrix archive SHA-256: `b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Environment fingerprint: `8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386`
- Controller stream changed: NO
- Selected 16 changed: NO
- Redraws / replacements: 0 / 0
- Spark 1 used: YES, as the authorized execution and analysis host
- Spark 2 used: NO
- RunPod used: NO
- Q3 run: NO
- Historical Q2 classifications changed: NO
- Public Git contains only release-safe derived artifacts; raw journal,
  scored rows, error arrays, `Dshape`, and `Dtotal` remain private/hash-pinned.

Final classification: `Q2_OOS_V2_A0_PASS`

Forensic status: `Q2_OOS_V2_FORENSIC_CLEAN`
