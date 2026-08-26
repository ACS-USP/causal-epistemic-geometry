# Q2 V4 — intervention-subspace construction plan

Status: preliminary design; no final bank is frozen.

## Historical A40 numerical audit

The eight persisted Amendment-1 vectors are already Euclidean unit vectors in
ambient dimension 4,096. Columns are ordered by exact controller ID and
renormalized in float64 before SVD so historical construction norms cannot
weight the span.

| quantity | value |
|---|---:|
| exact rank | 8 |
| retained rank | 8 |
| singular values | 1.586876, 1.149407, 1.060146, 1.025535, 0.824119, 0.706604, 0.685159, 0.580647 |
| relative singular values | 1.000000, 0.724321, 0.668071, 0.646260, 0.519334, 0.445280, 0.431766, 0.365906 |
| condition number | 2.732943 |
| entropy effective rank | 6.593573 |
| stable rank | 3.176903 |

The prospective numerical rule is to retain singular component i iff
`sigma_i / sigma_1 >= 1e-6`. All eight pass by more than five orders of
magnitude. This threshold is numerical, not behavioral. Exact arrays, hashes,
the Gram matrix, and orthonormality error are in
`SUBSPACE_NUMERICAL_AUDIT.json`.

The source directions are:

- CONTROL_FLOW_PATH_COVERAGE × {PROMPT_BOUNDARY, EXECUTION_BOUNDARY};
- MUTATION_ALIAS_CAUSALITY × {PROMPT_BOUNDARY, EXECUTION_BOUNDARY};
- LOOP_BOUNDARY_ACCOUNTING × {PROMPT_BOUNDARY, EXECUTION_BOUNDARY};
- HYPOTHESIS_BRANCH_ELIMINATION × {PROMPT_BOUNDARY, EXECUTION_BOUNDARY}.

## Native-Spark qualification consequence

The exact rank above describes the historical A40 vectors. The recommended V4
backend is native Spark 1, so these arrays do not become the final deployed
basis by fiat. Spark 1 must reconstruct the eight directions under the same
source concepts and pass source qualification. The same SVD rule is then
applied to the Spark-native unit columns. If its retained rank is below 6,
condition number exceeds 10, or any concept contributes less than 1% leverage
to the retained subspace, stop for principal review. No semantic panel is
opened.

## Prospective direction generator

After the Spark-native basis Q and rank r are locked, derive one 128-bit seed:

```text
first_128_bits(SHA256(
  "Q2-V4-INTERVENTION-SUBSPACE-DIRECTIONS-V1|<prospective-lock-commit>"
))
```

Use NumPy `Generator(PCG64DXSM(seed))`. For k=1,...,K draw

\[
g_k\sim N(0,I_r),\qquad c_k=g_k/\lVert g_k\rVert_2,\qquad v_k=Qc_k.
\]

No M0/M1/M2 value, model behavior, correctness, or alternate seed participates.
The actual bank is not generated in this sprint because Q is backend-dependent
and no prospective execution lock exists.

## Symmetric gross-degeneracy gate

The criteria are defined before the final bank exists:

- all values finite;
- maximum coefficient-norm error <= 1e-12;
- coefficient matrix rank = r;
- entropy effective rank >= 0.75r;
- coefficient-matrix condition number <= 3.0;
- maximum absolute pair cosine <= 0.98.

These checks are basis-free inside coefficient space and invariant to controller
labels. If the single bank fails, stop; do not redraw. Preliminary K=32
simulation-bank checks passed (rank 8, effective rank 7.2368, condition 2.4524,
maximum absolute pair cosine 0.8640), but that simulation bank is not the final
scientific bank.

Antipodes are not declared redundant: plus and minus mixtures can induce
different policies. The absolute-cosine gate merely excludes numerically near-
duplicate or near-antipodal points that would waste independent vertices.

## Functional qualification

Subspace algebra does not guarantee safe causal movement. On disjoint,
label-free calibration prompts, solve each direction's alpha independently for
implemented-radius targets 0.25 and 0.50 using the historical BF16-aware
definition and deterministic bisection. Retain the existing target pair because
it spans a twofold amplitude contrast within prior label-free safe/development
scale and below the historical high-dose anchor; it is not frozen until native
Spark qualification.

Every one of the 64 direction-shell conditions must pass validity/evaluability,
truncation, amplitude, and raw-sequence-movement gates under one symmetric rule.
Any failure stops V4 before M0/M1/M2 and semantic outcomes. There is no
behavior-dependent controller replacement.
