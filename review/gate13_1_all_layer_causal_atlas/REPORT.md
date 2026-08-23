# Gate 13.1 — all-layer causal atlas and joint layer–dose qualification

Historical Gate 13 remains `GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. Gate 13.1 is a distinct, prospectively frozen DEVELOPMENT procedure that tested all 34 Ministral language layers at D50, qualified three candidate layers on a disjoint 28-item layer-dose split, and evaluated one mechanically selected cell on 100 untouched final-evaluation items.

## All-layer sweep

- Items: 12.
- Layers tested: 34/34.
- Highest causal-Q layer group: L18, L20, L27, L28, and L33, each with Q = 0.666667.
- Frozen quartile candidates: L16, L18, L27.
- Source-effect versus causal-Q Spearman: -0.085839.
- Source-AUROC versus causal-Q Spearman: 0.221116.
- Classification: `GATE13_1_ALL_LAYER_SWEEP_PASS`.

The weak rank associations are descriptive DEVELOPMENT evidence that source readability did not reliably rank causal sensitivity in this run. They were not used for candidate selection.

## Layer × dose qualification

The frozen grid tested D25/D50/D75/D100 at L16, L18, and L27 with isotropic and shuffled architecture-matched nulls. Eligible cells were L16-D25, L16-D50, L16-D75, L18-D25, L18-D50, and L27-D25. The frozen rule selected the lowest eligible dose within each layer, then maximized meaningful Q minus null-mean Q without using accuracy for ranking. It selected:

- layer: 27;
- dose: D25;
- alpha: 4.469907677389362;
- vector hash: `0c467b7a452619d058afb07c96fd0cd8e20abb19a58d89674ab0a42e00ef2b94`;
- Stage-B Q: 0.357143;
- Q minus null mean: 0.232143;
- Q minus null maximum: 0.214286;
- commitment validity / semantic evaluability: 1.000 / 1.000.

## Fresh final evaluation

All 1,400 frozen rows were collected: 100 items × seven conditions × two independent rollouts. The journal contains no missing or duplicate logical keys.

| Condition | Commitment validity | Semantic evaluability | Accuracy | Mean tokens |
|---|---:|---:|---:|---:|
| Baseline | 0.965 | 0.965 | 0.445 | 167.655 |
| Textual careful | 0.965 | 0.965 | 0.730 | 490.825 |
| Meaningful L27-D25 | 0.915 | 0.915 | 0.575 | 453.175 |

The meaningful controller improved accuracy by 0.130. Its validity and evaluability were exactly 0.050 below baseline, so the frozen point-estimate relative guard passed at its boundary. This cost remains scientifically important and is not hidden by the positive classification.

Primary meaningful estimands were:

- G = 0.165000 (95% item-cluster bootstrap interval 0.097500 to 0.237500);
- C = 0.094015 (0.053687 to 0.133512);
- D = 0.140000 (0.060000 to 0.220000);
- rescue = 0.210000;
- damage = 0.080000.

Across four fresh final random controllers, mean/max G were 0.034375/0.080000, mean/max C were 0.020234/0.036187, and mean/max D were 0.035000/0.050000. Meaningful-minus-random-mean contrasts were 0.130625 for G, 0.073782 for C, and 0.105000 for D. Every frozen strong-replication criterion passed, including positive bootstrap lower bounds and leave-one-item-out sign stability.

Primary classification: `GATE13_1_STRONG_CROSS_MODEL_REPLICATION`.

## Operational incident and forensic audit

The mandatory Stage-C transition rule was frozen before Stage-B inspection. The projected cumulative cost was US$9.6474, above the then-current US$9.50 ceiling. A supervisor/Git incident nevertheless started Stage C and produced 121 rows. No Stage-C metric, condition comparison, or interim scientific outcome was inspected. Collection was stopped; the principal then authorized a US$11.00 cumulative ceiling; collection resumed from the immutable journal without regenerating completed keys.

The independent audit verified the exact 121-row prefix hash, pause chronology, unchanged inclusion in the final 1,400 rows, schedule and seed completeness, parser symmetry, final random-bank freshness, all primary metrics, bootstrap outputs, and classification. Maximum primary/audit metric difference was 0.0.

Forensic classification: `GATE13_1_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES`.

## Interpretation boundary

Gate 13 is not erased: its source-decodability shortlist plus fixed-D50 procedure remains a clean bounded null. Gate 13.1 shows that a broader prospectively frozen all-layer and layer-dose procedure found a Ministral L27-D25 actuator that strongly replicated useful semantic error-profile control on fresh CRUXEval items.

This substantially strengthens Program A across Qwen and Ministral within CRUXEval. It does not establish domain-general control: Gate 10's character-count transfer null remains intact. This is DEVELOPMENT evidence, not confirmatory evidence. Q2, Q3, the 57 reserved CRUXEval IDs, and the confirmatory holdout were untouched.
