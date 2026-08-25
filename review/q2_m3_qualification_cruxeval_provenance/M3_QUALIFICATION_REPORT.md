# M3 real-Qwen engineering qualification

Classification: `M3_DERIVATIVE_IDENTITIES_FAILED`. M3 status: `NOT_QUALIFIED`.

## Exact object

`Gamma_ij` is the uniform mean, over 16 frozen synthetic teacher-forced
fixtures and their prescribed checkpoints, of
`(J_z v_i)^T (diag(p)-pp^T) (J_z v_j)`. The derivative is evaluated in the
FP32 computational lift of the exact BF16-valued Qwen3-8B checkpoint at block
27. This is not semantic-error geometry and not the Fisher geometry of the
full free-running trajectory distribution.

## Frozen gates

- alpha-zero / FP32 sequence semantics: `True`
- repeated/order/chunked reproducibility: `True`
- independent JVP and JVP/VJP: `True`
- PSD without clipping: `True`
- direct versus polarization: `False`
- three-scale finite local window: `False` — `None`
- historical BF16 bridge: `False`

## Prespecified numerical results

- alpha-zero identity: top-1 1.0; maximum vocabulary JS 0;
- FP32 full/sequential: top-1 1.0, median JS 3.72e-12, p99 JS
  4.64e-11, maximum target-log-probability difference 2.68e-5, median logit
  cosine 0.9999999999995;
- repeat/order/chunked Gram relative errors: 0, 7.21e-16, 1.96e-16;
- forward/independent JVP cosine range: 0.9999999999994–0.9999999999999;
- JVP/VJP relative-error maximum: 4.01e-6;
- exact Gram minimum eigenvalue: 3.88e-6, without clipping;
- direct/polarization relative Frobenius error: 0.252615 (limit 0.01);
- finite ladder: no three consecutive scales passed; Gram error plateaued near
  0.2526 and radius error near 0.1181;
- BF16 baseline bridge: top-1 0.973684 (required 0.99), median JS 5.79e-5;
- BF16 geometry bridge: radius Spearman 0.714286, distance Spearman 0.014286,
  median curvature relative error 10.1082.

The offline failure localization in `M3_FAILURE_DIAGNOSIS.md` shows that the
polarization and finite-Gram plateau arose from unequal fixture/checkpoint
aggregation weights. The frozen failure is preserved. The BF16 bridge failed
independently and remains constitutive for M3 inclusion.

The BF16 bridge is constitutive for inclusion in Q2 V3. Exact FP32 coherence
alone is insufficient. No semantic correctness, CRUXEval scientific item, free
generation, or Q2 V3 behavioral trajectory was used.
