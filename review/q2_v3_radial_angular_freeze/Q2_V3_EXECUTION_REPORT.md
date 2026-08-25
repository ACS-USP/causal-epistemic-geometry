# Q2 V3 execution closeout

Terminal state: `Q2_V3_PANEL_PROVENANCE_MISMATCH`

The authorized execution stopped at the first frozen pre-panel gate. No model
inference, source qualification, shell calibration, geometry construction, or
semantic-panel collection occurred.

## Provenance result

The exact public CRUXEval revision contained all 336 manifest records and all
reference-answer hashes matched. The deterministic prompt reconstruction
matched 327 records. Nine frozen `prompt_sha256` fields did not equal the raw
SHA-256 of the frozen task prompt reconstructed from the official code/input:

- M1 covariance: `sample_300`, `sample_74`;
- primary semantic panel: `sample_659`, `sample_777`, `sample_145`,
  `sample_698`, `sample_21`;
- shell calibration: `sample_745`, `sample_700`.

All nine frozen values instead equal the namespaced historical
`EXTERNAL-PROMPT` hash of the older external-qualification prompt template.
The freeze fallback copied that historical digest into a field whose other 327
records contain raw prompt SHA-256 values. The mismatch therefore concerns both
the hash convention and the prompt template, not item identity or reference
answer.

The frozen execution authorization required manifest equality, forbade panel
repair, and prescribed a stop on any mismatch. The runner stopped before the
model was loaded. The historical freeze artifacts were not rewritten.

## Scientific boundary

- scientific trajectories: 0;
- semantic outcomes opened: no;
- prediction matrices: not constructed;
- Q2 V3 G/R classification: not reached;
- Q2 V2: unchanged;
- M3: unchanged and not qualified;
- Q3: not run.

This is a pre-panel provenance terminal state, not evidence for or against any
of M0, M1, or M2.

## Infrastructure

The A40 environment and frozen package profile passed remote preflight. The Pod
was used only for package/model-cache preparation and public-manifest
verification. No persistent network volume was attached. GPU execution was
stopped immediately after the terminal state was confirmed. The Pod was then
terminated after the mismatch record was preserved locally; final RunPod checks
showed zero Pods and zero network volumes.

Next action: `PRINCIPAL_RESEARCHER_REVIEW`.
