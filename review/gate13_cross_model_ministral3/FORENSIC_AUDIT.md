# Gate 13 independent forensic audit

Classification: `GATE13_FORENSIC_CLEAN`.

The audit independently reparsed all `612` raw trajectories, verified the frozen schedules and seeds, reconstructed all 34 paired-mean directions from the persisted activation archive, reproduced the source shortlist, recomputed every first-stage semantic-change and null-specificity metric, and mechanically reproduced `GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. The maximum primary/audit metric difference was `1.9184653865522705e-13`. No dose-calibration or final-evaluation row exists, as required by the frozen stage stop. The 57 untouched CRUXEval IDs and confirmatory holdout remain untouched.
