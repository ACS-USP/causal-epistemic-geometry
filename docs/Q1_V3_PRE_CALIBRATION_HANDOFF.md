# Q1 V3 — Pre-Calibration Handoff

## Current state

- Q1 V1–V1.2 multiple-choice instruments: formally closed as DEVELOPMENT.
- Q1 V2 / E3-10 direct instrument: not qualified; preserved as an ablation.
- Q1 V3 procedural reasoning suite: implemented and model-free structural gate
  passed at 5,000 instances per family/cell.
- Stage A/B Qwen calibration: not run.
- Steering, DEV, geometry, and confirmatory holdout: not run.
- RunPod: may remain stopped during local review.

## Local validation

```bash
cd ~/dev/causal-epistemic-geometry
source .venv/bin/activate
make test
make lint
python -m compileall -q src scripts
make q1-v3-gate
make q1-v3-design
```

The design artifact is under `review/q1_v3_reasoning_instrument/`. It contains
no model weights or model outputs. Stage A manifest construction is also local
and model-free:

```bash
python scripts/build_q1_v3_calibration_manifests.py stage_a \
  --gate review/q1_v3_reasoning_instrument/structural_gate_summary.json \
  --output review/q1_v3_reasoning_instrument/stage_a_manifest.json
```

The corrected manifest uses schema `q1-v3-stage-a-paired-budget-v1`: each
family/cell has exactly one 60-item latent set shared by budgets 512/1024/2048,
with a recorded manifest content hash. The regenerated manifest hash is stored
in the review bundle.

## After principal review

1. Start the existing RunPod only when the researcher authorizes Stage A.
2. Set `HF_HOME=/workspace/hf-cache` and verify host/cache/GPU invariants.
3. Sync this committed branch to `/workspace/causal-epistemic-geometry`.
4. Verify the pinned Qwen3-8B revision is already cached or explicitly download
   it on RunPod only.
5. Run `ceg doctor` and `ceg preflight`; neither downloads anything.
6. Run one Stage-A manifest key first as a technical baseline-only smoke.
7. If the cost gate and output audit pass, run the complete Stage A.
8. Apply the frozen mechanical Stage-A rule. Do not choose by intuition.
9. Build Stage B only from stored Stage-A outcomes.

The remote baseline runner is:

```bash
python scripts/run_q1_v3_calibration.py \
  configs/q1_v3_reasoning_instrument.example.yaml \
  --manifest review/q1_v3_reasoning_instrument/stage_a_manifest.json \
  --manifest-key MODREG-R/depth_4/512 \
  --output runs/q1_v3_stage_a/modreg_depth4_512
```

This runner is baseline-only and refuses steering. It stores full raw
trajectories, exact parse status, seeds, token IDs where available, policy
configuration, and model provenance. Omit `--manifest-key` only for an
intentional full Stage-A launch.

## Stop rules

- If primary decimal/token or policy configuration is technically invalid,
  stop and invalidate the calibration before interpreting outcomes.
- If fewer than two families pass Stage A, print
  `REASONING_INSTRUMENT_SCREEN_FAILED` and stop.
- If fewer than two families pass Stage B, print
  `REASONING_INSTRUMENT_NOT_QUALIFIED` and stop.
- Do not construct PCA, random controls, activation directions, steering, DEV,
  geometry, or holdout artifacts before Stage B qualification and review.

No Q1 scientific conclusion is available at this handoff.

When remote execution becomes the only remaining step, the explicit handoff
sentinel is:

```text
RUNPOD_REQUIRED_FOR_Q1_V3_REASONING_STAGE_A
```
