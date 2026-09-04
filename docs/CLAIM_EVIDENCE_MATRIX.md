# Claim–evidence matrix

This matrix prevents narrative drift. “Allowed” language is deliberately
narrower than the motivating theory.

| Claim | Evidence level | Supporting experiment | Allowed wording | Forbidden overclaim |
|---|---|---|---|---|
| Fixed Qwen steering reorganizes semantic blind spots on CRUXEval | CONFIRMATORY | Q1 fixed-controller holdout | Safe, null-specific competence-adjusted complementarity on Qwen3-8B + CRUXEval | Universal steering; task-general or model-general control |
| The fixed Ministral controller passed confirmatory Q1 | NEGATIVE | Q1 fixed-controller holdout | Complementarity components were positive, but the model-level result failed validity/evaluability guards | Partial pass; parser failure proves the science would pass |
| The Qwen controller transfers to long character counting | NEGATIVE_BOUNDARY | Gate 10 | The exact fixed controller did not transfer under the frozen character-count design | No controller could ever transfer; all domains differ |
| The fixed Qwen controller transfers to LiveCodeBench | NEGATIVE_BOUNDARY — DEVELOPMENT | Q1 second-task Stage B | The meaningful C point estimate exceeded all nulls, but the frozen C/contrast intervals, split-half consistency, and safety conjunction did not pass | A second-task transfer pass; proof that transfer is exactly zero; pooling Stage A1/A2 with Stage B |
| Intervention geometry predicts blind-spot geometry | DEVELOPMENT | Q2 V4.1 | Within the fixed 31-direction Qwen subspace and frozen CRUXEval panel, A0/A1/A2 predict pairwise blind-spot-shape distance | Universal geometry; cross-model law; geometric prediction outside the sampled subspace |
| A2 is the best geometry | NEGATIVE | Q2 V4.1 G3 contrasts | A2 has relational signal, but A0 and A1 were numerically more predictive | A2 nearly passed G3; power alone explains G3 failure |
| Stronger interventions move blind spots farther | DEVELOPMENT | Q2 V4.1 radial tests | STRONG exceeded MEDIUM shape and total displacement in all 31 tested directions | Global monotonicity, smoothness, linearity, or a Riemannian manifold |
| Positive A0 alignment generalizes across controller identities | DEVELOPMENT — PROSPECTIVE VALIDATION | Q2 OOS V2 | All 16 prospectively sampled safety-conditioned fresh-controller row associations were positive within the same Qwen3-8B/CRUXEval/learned-rank-8 laboratory | Cross-task, cross-model, or arbitrary-subspace generalization; treating 496 dyads as IID |
| The OOS item bootstrap supplies a conventional 95% confidence interval | POST_HOC DIAGNOSTIC — NOT SUPPORTED | Q2 OOS item-bootstrap audit | The archived distribution is an item-panel perturbation sensitivity object; the implementation reproduced exactly but percentile-CI calibration was not established | Using its quantiles as ordinary CI bounds; selecting a favorable replacement method post hoc |
| The learned rank-8 subspace is more specific than matched random rank-8 subspaces | NOT_RUN | None | A matched-random-subspace control is a prospective design question | Learned-subspace specificity has been demonstrated; one random subspace represents the population of orientations |
| Complementarity yields deployable collective utility | NOT_RUN | None | Q3 remains an open question | Oracle headroom is deployable utility; Q3 has been demonstrated |
| Label-free prompt representations make closed-panel policy diversity selectable | DEVELOPMENT_ONLY | Q3.1 prompt-representation tournament | On the closed 300-family development panel, prompt representations enabled stable routing over the A0 K=8 bank | Confirmatory Q3 utility; fresh-family generalization; deployable gain |
| True controller geometry is necessary for Q3.1 routing | DEVELOPMENT — NOT SUPPORTED | Q3.1 geometry attribution controls | True geometry exceeded learned policy identity by only +0.0033 and failed the frozen +0.01 incremental criterion | Geometry-specific routing; causal geometry is the mechanism of selectability |

## Canonical evidence

- Q1: [`Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md`](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md)
  and [`CONFIRMATORY_RESULTS.json`](../review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json).
- Character-count boundary: [`GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md`](GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md).
- LiveCodeBench boundary: [Stage-B closeout](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_b_closeout/REPORT.md)
  and [resolved forensic audit](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_b_forensic_resolution/REPORT.md).
- Q2: [`Q2_V4_1_SEMANTIC_CLOSEOUT.md`](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.md)
  and [`FORENSIC_AUDIT.json`](../review/q2_v4_1_semantic_execution/FORENSIC_AUDIT.json).
- Q2 fresh-controller validation:
  [`Q2_OOS_V2_SEMANTIC_CLOSEOUT.md`](../review/q2_oos_fresh_controller_design/v2_semantic_execution/Q2_OOS_V2_SEMANTIC_CLOSEOUT.md)
  and [`Q2_OOS_V2_FORENSIC_AUDIT.json`](../review/q2_oos_fresh_controller_design/v2_semantic_execution/Q2_OOS_V2_FORENSIC_AUDIT.json).
- Q2 OOS item-bootstrap diagnostic:
  [`Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_REPORT.md`](../review/q2_oos_fresh_controller_design/v2_semantic_execution/item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_REPORT.md).
- Q3.1 prompt-representation development:
  [`Q3_ROUTE_A_PROMPT_REPRESENTATION_REVIEW.md`](Q3_ROUTE_A_PROMPT_REPRESENTATION_REVIEW.md)
  and [`Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json`](../review/q3_route_a_prompt_representation/Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json).
- Current state: [`project_state.yaml`](../project_state.yaml).
