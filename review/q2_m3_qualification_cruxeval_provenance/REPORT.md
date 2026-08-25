# Q2 M3 qualification and CRUXEval provenance closeout

Status: `Q2_V3_FREEZE_READY / AWAITING_PRINCIPAL_RESEARCHER_REVIEW`.

Q2 V3 behavioral execution: **NOT RUN**. Q3: **NOT RUN**.

## M3 exact object

For six frozen engineering-only unit directions at zero-based block 27,
`Gamma_ij` is the uniform mean over 16 synthetic teacher-forced fixtures and
their prescribed continuation checkpoints of

`(J_z v_i)^T (diag(p) - p p^T) (J_z v_j)`.

The sequence is fixed; no free generation occurs. The derivative is taken in
the FP32 computational lift of the exact BF16-valued Qwen3-8B checkpoint. This
is a local teacher-forced output-Fisher controller-span Gram, not semantic-error
geometry and not Fisher geometry of the complete free-running distribution.

## M3 result

Classification: `M3_DERIVATIVE_IDENTITIES_FAILED`; status: `NOT_QUALIFIED`.

Alpha-zero identity, FP32 full/sequential equivalence, repeat/order/chunked
reproducibility, exact forward/independent JVP agreement, JVP/VJP duality, and
PSD without clipping passed. Direct/polarization relative Frobenius error was
0.252615 versus a 0.01 threshold. No three-scale finite local window passed.
The BF16 bridge also failed independently: baseline top-1 agreement 0.973684,
radius Spearman 0.714286, distance Spearman 0.014286, and median curvature
relative error 10.1082.

The array-only diagnosis localized the 0.2526 polarization/finite-Gram plateau
to unequal fixture-versus-checkpoint aggregation weights. That does not alter
the frozen classification or qualify M3. A corrected M3 requires a new
prospective authorization and still must solve the BF16 bridge.

Forensic classification: `M3_FORENSIC_CLEAN`; primary and audit classifications
agree. No scientific item or semantic outcome was read.

## CRUXEval provenance census

- Class A pristine: 0;
- Class B representation-only / label-free / allocation exposure: 25;
- Class C historical behavioral exposure outside Q2 geometry discovery: 655;
- Class D directly implicated in Q2 V2/V3 discovery: 120;
- unresolved: 0.

The 120 Q2 V2 common-panel items are Class D and excluded from primary Q2 V3.
The complete ledger covers all 800 official output-prediction IDs at dataset
revision `b96af0450242eb4da433032b90998f25588a5d0f` and uses exposure roles, not
outcome values, for classification.

## Proposed Q2 V3 primary panel

The draft proposes 200 deterministic Class-C items under namespace
`Q2-V3-HISTORICAL-C-PROSPECTIVE-CONTROLLER-V1`, ordered-ID SHA-256
`969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf`.

Exact claim: **historical-item/prospective-controller same-domain validation**.
This is prospective generalization along the controller axis on a fixed
historical same-domain distribution. It is not fresh-item confirmation.

## Q2 V3 implication

Same-domain primary V3 is feasible under the exact historical-item claim. M3
is excluded; M0/M1/M2 remain the candidate geometries. The radial/angular draft,
panel, and cost plan are freeze-ready, but nothing is frozen or execution-
authorized by this closeout. Q2 V3 semantic outcomes remain unopened.

## Resources

The technical runner used 484.01 A40 seconds (0.1344 GPU-hours). The pod was
terminated after verified artifact recovery; RunPod reports zero pods and zero
network volumes. No DGX was used.
