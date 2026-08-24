# Q2 controller-bank qualification closeout

Status: `Q2_CONTROLLER_BANK_NOT_QUALIFIED` — DEVELOPMENT.

The first Q2 controller-held-out predictive pilot stopped before its common
120-item panel. This is a bank-qualification failure, not evidence that flat,
whitened, or finite-secant geometry lacks predictive signal.

## Frozen-gate result

- Source qualification: 3/3 conceptual axes passed.
- Engineering: `Q2_ENGINEERING_PASS`.
- Controller movement: 3/16 candidates passed every frozen first-stage rule.
- Representation geometry: failed null-to-meaningful orthogonality.
- Source trajectories: 144.
- Matched manipulation trajectories: 204.
- Common-panel trajectories: 0.
- Predictive M0/M1/M2 outcomes: not run.

All source conditions had commitment validity and semantic evaluability of
1.0. Their activation gaps generalized across the separate validation items,
and each textual source met the frozen behavioral/source-style gate. The bank
therefore failed after source construction, not because the proposed concepts
were absent from the model.

Only these three controllers reached the complete movement rule:

- `MEAN_INDEPENDENT_VERIFICATION_PROMPT_BOUNDARY_MINUS`
- `MEAN_TYPE_REPRESENTATION_DISCIPLINE_PROMPT_BOUNDARY_MINUS`
- `NULL_ISOTROPIC_R0`

The other 13 candidates missed the frozen raw-token-sequence change threshold
of 0.25 on the 12-item matched panel. Since one isotropic null passed while most
meaningful directions did not, the qualification did not establish a broad,
meaningfully structured causal bank.

The second independent blocker was numerical. Unit norms, sign-pair identity,
and meaningful-base diversity passed, but null orthogonality did not. The
construction projected null candidates sequentially against correlated source
directions rather than against an orthonormal basis of their span. Maximum
absolute null-to-meaningful cosine was 0.175380 against the frozen tolerance of
1e-6. This defect was preserved as a failed qualification rather than repaired
to continue the same experiment.

## Integrity boundary

Accuracy, G, C, D, rescue, damage, and downstream complementarity were not used
for source or controller qualification. The closed Q1 confirmatory outcomes
were not reused. The independent audit reconstructed all 348 rows, source
metrics, manipulation metrics, vectors, and the terminal rule with maximum
numeric difference 8.53e-14. It classified four pre-common-panel serialization,
canonicalization, and provenance incidents as non-scientific issues.

RunPod was deleted after a 61-file recovered bundle was verified. Closeout
state was 0 active Pods and 0 retained network volumes. Q1 remains
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`; Q3 was not run.

The next protocol is draft-only and requires principal review. No second Q2
experiment is authorized by this closeout.
