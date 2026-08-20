# Gate 6.2 — First-Stage Repair and Paired-Mean Controller Bridge

Gate 6.2 is a development-only continuation of the reviewed Gate 6.1 source
screen. It does not rewrite Gate 6.1 and does not touch the confirmatory
holdout.

## What is frozen

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, SDPA,
  full non-thinking generation, and the existing CRUXEval semantic evaluator.
- The exact 104 source-training and 32 source-validation items from Gate 6.1.
- Layers 8, 12, 17, 22, 27, and 32 at both prompt and execution boundaries.
- The exact Gate-6 20-item manipulation and 60-item evaluation manifests.
- Paired-mean prompt-boundary L22/L27/L32 as the bridge candidates: L27 is the
  single candidate and all three form the multilayer candidate.

## Method repairs

Execution-boundary teacher-forced scores begin at the first token of the final
`FINAL:` marker. Tokens whose logits were produced before the intervention are
excluded. Prompt-boundary scores retain the complete continuation.

RFM iteration/parameter selection is performed with deterministic four-fold
cross-validation entirely inside `SOURCE_TRAIN`. `SOURCE_VALIDATION` is used
only for held-out source readout, corrected first-stage likelihoods, and
random-null comparison. No benchmark correctness or semantic answer is used
to construct or select a controller.

The Gate 6.1 archive compacted ordinary prompt activations to one vector per
split/location/layer. Gate 6.2 deterministically re-extracts the missing
ordinary prompt states only; it does not regenerate source trajectories.

## Phase boundary

The source-only phase must select a controller before the exact Gate-6
manipulation set can run. The manipulation phase is a 20-item matched-seed
first-stage gate. Only if it passes may the 60-item, two-rollout evaluation
run. Character count, Q2, new layers/alphas, semantic-label controllers, and
the holdout are not part of this protocol.

The machine-readable lock and audit are stored under
`review/gate6_2_first_stage_repair_mean_bridge/`; that directory is ignored
from Git because it contains large/raw run material. The tracked specification
is `experiments/specs/gate6_2_first_stage_repair.yaml`.

This document is a protocol reference, not a scientific outcome report.
