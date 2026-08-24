# Q2 first experiment options

Status: principal-review design menu. None is authorized for execution.

## Option 1 — offline historical dyad audit

**Question.** Do existing representation distances correlate with historical
error-profile relations?

**Data.** Existing vectors and historical Gate 6–13 aggregate outcomes.

**Bank/models/tasks.** Qwen and Ministral historical controllers; CRUXEval and
character count.

**Metrics.** Cosine, source readout, finite-displacement KL/JS where available;
D/C historical values.

**Primary estimand.** Descriptive rank association only.

**Controls.** Controller-label permutations and sensitivity to gate inclusion.

**Role.** `POST_HOC_DESCRIPTIVE`, not Q2 evidence.

**Compute.** CPU/offline; no new trajectories.

**Failure interpretation.** Historical stages are not commensurate; a null is
uninformative and a positive is hypothesis-generating.

**Next step.** Proceed to a common-panel prospective bank regardless.

**Assessment.** Cheap but low information gain. It risks anchoring metric
choices on outcome-contaminated, incomplete dyads. Do not make this the primary
Q2 experiment.

## Option 2 — controller-held-out finite-secant prediction (recommended)

**Question.** On a common CRUXEval development panel, do Euclidean, whitened,
or finite behavioral secant distances predict D for unseen controllers, and
does a control-aware metric outperform flat geometry?

**Minimum data.** A frozen K=16 Qwen bank, baseline, 120-item common panel, and
two independent rollouts per condition (4,080 trajectories before optional
textual anchors). Geometry fixtures and token checkpoints are frozen before
semantic outcome reveal.

**Controller bank.** Three independently verified source axes × two layers ×
two signs at one matched dose, plus four new random directions. Existing Qwen
controllers may enter only as predeclared anchors; at least two source families
must be new to the bank.

**Model/task.** Qwen3-8B and CRUXEval semantic output prediction. Previously
development-consumed item IDs may be reused under a newly frozen Q2 DEVELOPMENT
allocation; the closed 57-item confirmatory outcomes are never used as Q2 test
labels.

**Metrics.** Normalized cosine, covariance-whitened distance, and teacher-forced
finite-secant JS/KL on fixed baseline sequences. No exact JVP is required.

**Primary estimand.** Difference in controller-held-out mean squared prediction
error for D between the best prospectively selected control-aware metric and
Euclidean distance. Secondary: out-of-sample R², rank correlation, C prediction,
and calibration.

**Controls.** Constant predictor, controller-label permutation, new random
directions, item-cluster bootstrap, source-family holdout, invalid-as-error,
and a geometry-only lock before outcome reveal.

**Role.** Q2 DEVELOPMENT. It can support Level 1 or 2, not confirmation.

**Compute.** 4,080 generation trajectories plus teacher-forced geometry passes.
Historical row throughput is too condition-dependent for a single projection;
a non-scientific DGX/A40 preflight must measure the frozen workload and storage
formula before authorization.

**Failure interpretation.** If no metric predicts unseen controllers, flat and
finite-secant geometries are inadequate for this bank; examine bank diversity,
measurement reliability, and whether error profiles are too nonlinear before
opening exact derivatives. If Euclidean works, replicate before adding
complexity. If secant/whitened wins, freeze it for a prospective selection test.

**Next experiment.** Under success, Level-3 source-family holdout or Level-4
controller selection. Under null, a targeted measurement-reliability audit,
not an automatic Gate-12-style engine project.

## Option 3 — exact local Fisher/Jacobian retry

**Question.** Does exact local directional Fisher energy predict behavioral D
or sensitivity better than Euclidean distance?

**Data.** Synthetic qualification plus a scientific controller bank and fixed
teacher-forced sequences.

**Bank/model/task.** Qwen minimum bank; CRUXEval.

**Metrics.** Exact JVP/VJP pullback energy in the FP32 computational lift,
small-epsilon quadratic validation, and historical BF16 bridge diagnostics.

**Primary estimand.** Held-out D-prediction improvement over Euclidean geometry.

**Controls.** Independent AD implementations, local quadratic identity,
finite-secant comparator, controller-held-out evaluation.

**Role.** Q2 DEVELOPMENT only after a new engine lock.

**Compute.** Unknown until a memory/runtime preflight; full-vocabulary
derivatives may dominate generation.

**Failure interpretation.** Could be scientific or another numerical bridge
failure. The distinction requires an engineering gate before outcomes.

**Next step.** If engine fails, stop; if it qualifies and prediction fails,
report a bounded local-geometry null.

**Assessment.** High theoretical value, high rabbit-hole risk. Gate 12/12.1
showed exact identities in FP32 but did not qualify the complete bridge. This is
not the best first Q2 experiment.

## Option 4 — cross-domain geometry first

**Question.** Does a metric predict why the fixed controller helps CRUXEval but
not character count?

**Data.** A multi-controller common panel in both domains.

**Bank/models/tasks.** Qwen bank; CRUXEval plus a verified second objective
task.

**Metrics.** Separate within-domain secant and error geometry; no naive pooling.

**Primary estimand.** Domain-by-metric interaction in held-out controller
prediction.

**Controls.** Domain-specific baseline opportunity, source anchors, randoms,
item-cluster inference.

**Role.** Ambitious Q2 DEVELOPMENT.

**Compute.** Approximately twice the generation bank plus domain-specific
geometry passes; exact projection awaits the task and preflight.

**Failure interpretation.** Could reflect source-policy mismatch rather than
geometry. Gate 10 already warns that source utility differs by task.

**Next step.** Return to one-domain identification if source policy fails.

**Assessment.** Scientifically rich but compounds bank, source, instrument, and
domain risks too early.

## Recommendation

Choose **Option 2: controller-held-out finite-secant prediction**.

It directly tests the grand thesis, uses existing controller artifacts as
anchors, creates the missing common outcome matrix, and avoids requiring an
exact derivative engine. The finite-secant object matches the deployed
intervention scale while remaining explicitly distinct from local
pullback/Fisher geometry. Controller/source-family holdout makes the result
predictive rather than retrospective. DGX Spark access can reduce operational
friction after a read-only doctor and tiny smoke, but hardware availability must
not alter the bank, estimand, or test split.

Open principal decisions before a protocol lock:

1. approve the three source axes and verify that at least two are independent
   of careful/direct;
2. select the previously development-consumed common item pool without looking
   at new controller outcomes;
3. freeze K, layers, energy normalization, token checkpoints, covariance source,
   and controller-held-out split;
4. choose the single primary predictive loss and metric-selection rule;
5. approve the compute/storage envelope after DGX/A40 preflight.
