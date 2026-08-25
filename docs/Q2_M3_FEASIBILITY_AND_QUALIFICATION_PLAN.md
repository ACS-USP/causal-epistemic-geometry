# Q2 M3 feasibility and qualification plan

Status: `DESIGN ONLY — M3 NOT QUALIFIED — NO EXECUTION AUTHORIZED`

## Recommendation

M3 should enter Q2 V3 only if a separate numerical qualification passes. The
candidate is the multi-checkpoint teacher-forced categorical-Fisher Gram on the
low-dimensional prospective controller span. It is not a full activation-space
tensor and is not semantic-error geometry.

Gate 12.1 established two useful but incomplete facts:

- exact forward/independent JVP, JVP/VJP, Fisher/Hessian, and utility derivative
  identities passed in the FP32 computational lift;
- the historical BF16 bridge and the complete finite-difference stability
  window failed.

Therefore Gate 12.1 cannot be silently reused as M3 qualification.

## Candidate behavior maps

| Candidate | Scientific proximity | Feasibility | Decision |
|---|---|---|---|
| Single next-token full-vocabulary Fisher | Low | High | Diagnostic only |
| Multi-position teacher-forced Fisher | Intermediate | Moderate | Recommended M3 |
| Expected trajectory-distribution Fisher | Highest in principle | Low; sampling/path dependence | Future work |
| Small finite KL/Hellinger curvature | Same local object asymptotically | Moderate; step-size sensitive | Qualification crosscheck |

The teacher-forced continuation must be label-free and frozen. Reference
answers and correctness labels are forbidden. Baseline and per-checkpoint logits,
tangents, token indices, masks, and averaging weights must be preserved.

## Efficient estimator

For (K) base directions, compute (K) directional logit JVPs at each frozen
probe/checkpoint. If the rows of (Rin\mathbb R^{K\times V}) are those JVPs,

\[
\Gamma=R(\operatorname{diag}(p)-pp^T)R^T.
\]

Equivalently, center each row under (p) and compute a weighted Gram. This is
(O(KV+K^2V)) per checkpoint, requires no (d\times d) tensor, and is invariant
to constant logit shifts. If only directional energies (q(v)) are available,
cross terms may be recovered by

\[
\langle v,w\rangle_G=\frac{q(v+w)-q(v)-q(w)}2.
\]

Direct batched JVP rows are preferred because polarization roughly doubles the
number of directional evaluations and compounds numerical error.

## Qualification stages

### Q0: CPU toy mathematics

Use linear-softmax fixtures with known Jacobians. Require:

- Gram agreement with explicit (J^TFJ): max absolute error `<=1e-12`;
- PSD up to relative eigenvalue tolerance `1e-10`;
- polarization error `<=1e-12`;
- KL, Hellinger, and JS curvature relative errors `<=0.2%` in a verified local
  float64 window.

This stage is implemented and passes. It does not qualify Qwen.

### Q1: non-scientific Qwen fixtures

Use at least 16 synthetic/non-benchmark prompts, multiple prompt and
continuation lengths, and engineering-only random directions. Reuse no Q2
scientific item or semantic outcome.

### Q2: exact derivative identities in the FP32 lift

Require:

- independent JVP cosine `>=0.99999`;
- relative JVP norm difference `<=0.005`;
- JVP/VJP duality relative error `<=1e-4`;
- Fisher moment/Hessian relative difference `<=0.01` away from zero;
- utility directional derivative relative difference `<=0.01`;
- Gram minimum eigenvalue `>=-1e-8 * max_eigenvalue`;
- direct versus polarization Gram relative Frobenius error `<=0.01`.

### Q3: finite local curvature

Use a prospectively frozen epsilon ladder independent of deployed doses. Require
at least three consecutive epsilons where pooled medians satisfy:

- JVP cosine `>=0.999`;
- KL Fisher relative error `<=0.05`;
- Hellinger Fisher relative error `<=0.05`;
- JS Fisher relative error `<=0.05`;
- Gram/radius/angle stability under adjacent epsilons `<=0.05` relative.

The expected convergence then degradation pattern must be present. Passing at
only the two largest scales, as in Gate 12.1, is insufficient.

### Q4: sequence semantics

Under FP32, full-sequence and KV-cache teacher forcing must agree at the frozen
checkpoints using the Gate 12.1 sequence thresholds. Token, position, cache,
mask, and intervention indices must be persisted. No silent kernel mixing.

### Q5: historical BF16 bridge

The FP32 lift is not the original unquantized model. On the engineering
fixtures, compare FP32 local geometry with small finite BF16 output movement.
Require prospectively:

- top-1 baseline agreement `>=0.99`;
- median vocabulary JS `<=1e-4`;
- Spearman of direction radii `>=0.95`;
- Spearman of pairwise distances/angles `>=0.95`;
- median relative finite-curvature discrepancy `<=0.15` for non-negligible
  directions;
- no direction changes from upper to lower radius quartile solely by dtype.

If the bridge fails, report `M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED` and
exclude M3 from Q2 V3. Q2 V3 may still compare M0/M1/M2.

## Qualification classifications

- `M3_DIRECTIONAL_ENGINE_QUALIFIED`
- `M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED`
- `M3_DERIVATIVE_IDENTITIES_FAILED`
- `M3_FINITE_LOCAL_WINDOW_FAILED`
- `M3_SEQUENCE_SEMANTICS_FAILED`
- `M3_ENGINE_FAILURE`

No scientific claim follows from qualification.

## Cost estimate

Gate 12.1 used approximately 0.25 A40 hours and US$0.11. The broader M3 suite is
estimated at:

- expected: `0.5–1.5 A40 GPU-h`, approximately `US$0.25–0.75`;
- conservative: `4 GPU-h`, `US$2` hard planning envelope;
- if FP32 memory requires 80 GB: migrate prospectively, do not reduce precision.

Scientific M3 capture for a 10-direction bank is separate and estimated at
`1–3 GPU-h` including persisted checkpoint arrays. Neither activity is
authorized by this document.
