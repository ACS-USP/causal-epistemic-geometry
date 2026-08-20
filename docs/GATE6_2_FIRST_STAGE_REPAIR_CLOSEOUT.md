# Gate 6.2 Closeout — Paired-Mean First-Stage Repair

Status: `COMPLETE_DEVELOPMENT_NO_FIRST_STAGE`.

Gate 6.2 reused the immutable Gate 6.1 source artifacts and corrected the
execution-boundary teacher-forced scoring window. RFM fitting used deterministic
cross-validation inside `SOURCE_TRAIN`; no benchmark correctness or generated
answer was used to construct or select a controller.

The source-only phase selected the paired-mean prompt-boundary bridge:

- single controller: layer 27;
- multilayer controller: layers 22, 27, and 32;
- no RFM source group met the requirement of at least two passing layers.

The complete 20-item matched-coupling manipulation phase contained 200 rows.
The frozen gate required validity at least 0.85, semantic change rate at least
0.15, and semantic change at least 0.05 above the mean of four random mean
controls.

The two positive-sign controllers changed behavior substantially but failed the
validity guard:

- `BEST_SINGLE_MEAN_PLUS`: validity 7/20, semantic change 0.90;
- `MULTILAYER_MEAN_PLUS`: validity 2/20, semantic change 1.00.

The negative-sign controller preserved validity but did not exceed the random
mean semantic-change null:

- `MULTILAYER_MEAN_MINUS`: validity 20/20, semantic change 0.25;
- random mean semantic-change average: 0.275.

Therefore the frozen first-stage gate failed as:

`GATE6_2_NO_BEHAVIORAL_FIRST_STAGE`

The 60-item, two-rollout evaluation phase was not run. Character count, Q2,
new controller search, and the confirmatory holdout remain untouched. The
RunPod was stopped immediately after artifact recovery.

The complete ignored run artifacts and deterministic analysis are under
`review/gate6_2_first_stage_repair_mean_bridge/`, including `REPORT.md`,
`MANIPULATION_RESULTS.csv`, `MANIPULATION_ESTIMANDS.json`, and the raw
`journal.jsonl`. The review bundle is
`review/gate6_2_first_stage_repair_mean_bridge.tar.gz` with SHA-256
`a87f7de66e8e73bb65d94a3373c8574503c7fce23ac4555a55d3ea91598ef351`.
