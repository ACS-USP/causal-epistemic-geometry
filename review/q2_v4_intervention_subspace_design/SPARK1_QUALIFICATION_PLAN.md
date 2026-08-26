# Spark-1 native V4 qualification plan

Status: `DRAFTED ONLY — REQUIRES PRINCIPAL AUTHORIZATION`.

No step below was executed in the design sprint.

## Phase 0 — prospective infrastructure/model lock

1. Freeze Spark-1 environment fingerprint, dstack image digest, repository
   source commit, exact model/tokenizer revision, package versions, dtype,
   attention path, deterministic settings, and artifact paths.
2. Place the exact model under the approved shared model directory; verify every
   model/tokenizer file SHA-256. Never download weights to the Mac.
3. Verify one GB10, unified-memory headroom >=30 GiB after model load, disk
   headroom >=100 GiB, and no Spark-2 allocation.

## Phase 1 — engine qualification

4. Run non-benchmark deterministic fixtures for tokenization, baseline logits,
   full-vocabulary JS, selected hidden states, alpha-zero identity, sustained
   current-token scope, forward count, cache safety, hook cleanup, and resume.
5. Require within-Spark repeatability before any source reconstruction. Treat
   the backend as native; A40 comparisons are descriptive only.
6. Benchmark a small non-scientific trajectory schedule to freeze throughput,
   wall time, storage, and abort thresholds.

## Phase 2 — source-basis reconstruction

7. Reuse the exact four development-selected concept prompt pairs and two
   source locations on disjoint source-construction prompts. Construct eight
   L27 paired-mean directions without correctness labels.
8. Qualify textual behavior, validity/evaluability, held-out activation
   projection, direction orientation, vector norm/hash, and representation
   separation on disjoint source-validation prompts.
9. Apply the SVD rank/conditioning/leverage gate in
   `Q2_V4_SUBSPACE_CONSTRUCTION_PLAN.md`. Stop if it fails.

## Phase 3 — bank and shell qualification

10. Derive the final K=32 coefficient bank exactly once from the future lock
    commit and run coefficient-space gross-degeneracy checks.
11. Solve each direction's medium/strong alpha at implemented radii 0.25/0.50
    with BF16-aware deterministic bisection on disjoint calibration prompts.
12. Run symmetric label-free safety/manipulation checks for all 64 controllers.
    Any failure stops; no controller replacement or seed redraw.

## Phase 4 — pre-outcome geometry and prediction lock

13. Refit M1 from 64 disjoint baseline prompt-boundary activations using the
    exact historical lambda=0.10 implementation.
14. Capture baseline plus 64 controllers on 12 disjoint M2 probes and four
    fixed checkpoints. Verify repeatability, Hilbert consistency, positive
    baseline radii, and compute A0/A1/A2 plus secondary D2.
15. Freeze all vector/alpha/matrix hashes, 300-item V4 manifest, 65-condition
    schedule, independent seed banks, estimator, QAP, bootstrap, radial test,
    classifications, throughput envelope, and source commit. Commit and push
    this prediction lock before semantic outcomes.

Only a separate principal authorization after these gates may open the
39,000-row semantic panel. M3, Q3, Spark 2, and semantic null controllers remain
excluded.
