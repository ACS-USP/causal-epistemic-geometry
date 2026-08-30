# Claim–evidence matrix

This matrix prevents narrative drift. “Allowed” language is deliberately
narrower than the motivating theory.

| Claim | Evidence level | Supporting experiment | Allowed wording | Forbidden overclaim |
|---|---|---|---|---|
| Fixed Qwen steering reorganizes semantic blind spots on CRUXEval | CONFIRMATORY | Q1 fixed-controller holdout | Safe, null-specific competence-adjusted complementarity on Qwen3-8B + CRUXEval | Universal steering; task-general or model-general control |
| The fixed Ministral controller passed confirmatory Q1 | NEGATIVE | Q1 fixed-controller holdout | Complementarity components were positive, but the model-level result failed validity/evaluability guards | Partial pass; parser failure proves the science would pass |
| The Qwen controller transfers to long character counting | NEGATIVE_BOUNDARY | Gate 10 | The exact fixed controller did not transfer under the frozen character-count design | No controller could ever transfer; all domains differ |
| LiveCodeBench establishes second-task transfer | OPEN_RUNNING | Q1 second-task Stage B | A frozen DEVELOPMENT transfer experiment is collecting; no result yet | Any partial scientific conclusion or inferred trend |
| Intervention geometry predicts blind-spot geometry | DEVELOPMENT | Q2 V4.1 | Within the fixed 31-direction Qwen subspace and frozen CRUXEval panel, A0/A1/A2 predict pairwise blind-spot-shape distance | Universal geometry; cross-model law; geometric prediction outside the sampled subspace |
| A2 is the best geometry | NEGATIVE | Q2 V4.1 G3 contrasts | A2 has relational signal, but A0 and A1 were numerically more predictive | A2 nearly passed G3; power alone explains G3 failure |
| Stronger interventions move blind spots farther | DEVELOPMENT | Q2 V4.1 radial tests | STRONG exceeded MEDIUM shape and total displacement in all 31 tested directions | Global monotonicity, smoothness, linearity, or a Riemannian manifold |
| Complementarity yields deployable collective utility | NOT_RUN | None | Q3 remains an open question | Oracle headroom is deployable utility; Q3 has been demonstrated |

## Canonical evidence

- Q1: [`Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md`](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md)
  and [`CONFIRMATORY_RESULTS.json`](../review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json).
- Character-count boundary: [`GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md`](GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md).
- Q2: [`Q2_V4_1_SEMANTIC_CLOSEOUT.md`](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.md)
  and [`FORENSIC_AUDIT.json`](../review/q2_v4_1_semantic_execution/FORENSIC_AUDIT.json).
- Current state: [`project_state.yaml`](../project_state.yaml).

