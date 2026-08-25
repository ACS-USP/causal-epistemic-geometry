# Q2 geometry foundations and Q2 V3 radial/angular redesign

Status: `COMPLETE DESIGN-ONLY SPRINT`

No model was loaded, no scientific trajectory was generated, and no Q2 V3
protocol was frozen.

## Findings

1. M0 is a Euclidean angular chord after unit normalization; it discards dose.
2. M1 is a regularized covariance-whitened angular chord; it also discards dose
   and is not a density/manifold metric.
3. M2 is a finite output-response Jensen-Shannon pseudometric. It includes
   finite displacement effects but is not a local pullback. The frozen archive
   has no baseline response, so it has no identified radius/angle origin.
4. A categorical-Fisher M3 is mathematically defensible on the low-dimensional
   controller span. It describes local teacher-forced token-distribution
   sensitivity, not semantic correctness.
5. Gate 12.1 means M3 remains numerically unqualified. It may enter Q2 V3 only
   after a separate non-scientific engine gate.
6. Q2 V3 should match physical intervention radius experimentally, use two
   shells over ten genuinely new oriented directions, and make within-shell
   angular prediction the primary claim.
7. CRUXEval freshness is exhausted. Q2 V3 must not freeze until a genuinely
   fresh same-distribution source is established; LiveCodeBench remains an
   optional transfer panel rather than a silent substitute.

## CPU-only validation

Twenty-four synthetic linear-softmax fixtures verified:

- explicit Fisher Gram and weighted-centered implementation;
- PSD and logit-shift invariance;
- KL coefficient (1/2);
- Hellinger and Jensen-Shannon coefficient (1/8);
- polarization and radial/angular distance identities.

Maximum algebraic discrepancy was approximately `1.33e-15`; maximum local
curvature relative discrepancy was approximately `2.17e-4` at epsilon `3e-4`.

## Decision

Retain Q2 V3 as `DRAFT / AWAITING PRINCIPAL_RESEARCHER_FREEZE`. Before any
freeze, resolve same-domain item provenance and separately qualify M3. If M3
does not qualify, run no substitute under the M3 name; a future Q2 V3 may still
test M0/M1/M2 under the radial-shell design.
