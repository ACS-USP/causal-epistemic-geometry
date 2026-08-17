# E3-10 pre-model structural validity gate

This gate was run before any Qwen inference, activation extraction, steering,
or RunPod use. It is a model-free audit of the procedural instrument only.
The previous artifact at `review/q1_v2_instrument_design/` is archival. The
current, versioned artifact is `review/q1_v2_instrument_design_v2/`.

## Scope and versioning

The generator changed before any model outcome existed, so the generator was
versioned from `e3-generators-v1` to `e3-generators-v2`. The v2 change repairs
effective MODREG depth, rejects FSM identity/duplicate transition maps, and
rejects SAT tautologies, duplicate clauses/literals, and unused variables.
All current structural artifacts were regenerated under v2. No stale v1
latent IDs are used as current evidence.
The accepted 5,000-item-per-cell pools are preserved in the deterministic
`structural_pool_manifest.jsonl`; rejected candidates are not retained.

The original SATCOUNT D1 sketch used three variables. That cell could never
produce target 9 because there are only eight Boolean assignments. D1 was
corrected model-free to `vars4_clauses4` before calibration. This is a support
requirement, not a model-driven redesign.

## MODREG effective depth

MODREG dependency cones are computed by exact backward register dataflow. The
v2 generator constructs operations around a live dependency set, so each
nominal depth cell realizes the intended computational chain. In the 5,000
item-per-cell audit, the effective operation counts are separated and
monotonic for depths 4, 8, 12, and 16; the detailed quantiles and figure are
in `modreg_effective_depth_audit.json` and
`modreg_effective_depth_audit.png`.

## Structural validity and target support

All generated items passed family-specific structural checks. Every audited
cell showed all ten targets in a raw 5,000-seed support sample, and every
balanced audit pool contains exactly 500 items for each digit. The target
balance decision is oracle-only; no model prediction, confidence, activation,
or steering result is consulted.

The audit also records target-conditional structural features. These are
descriptive diagnostics, not reasons to discard a target or item.

## Shortcut audit

The frozen shortcut rules are unchanged:

- target-frequency baseline: 10% on a balanced ten-way pool;
- multinomial logistic regression on shallow structural features;
- a deterministic decision tree with maximum depth 4;
- warning at held-out accuracy at least 25%; failure at least 40%.

The structural gate excludes cells with a shortcut failure from future Qwen
calibration. In the final 5,000-item audit, failures were the four
REACHCOUNT10 cells and SATCOUNT10 `vars4_clauses4`. SATCOUNT10
`vars4_clauses6` was a warning. MODREG10, FSM10, and SATCOUNT10 cells
`vars4_clauses6`, `vars5_clauses8`, and `vars6_clauses10` remain structurally
eligible. The complete per-cell logistic, tree, and simple heuristic values
are in `shortcut_baseline_audit.json`.

The reachability and raw SAT counts remain available as structural audit
features, but are excluded from the shallow shortcut classifier because they
would directly expose the semantic result. This keeps the shortcut test about
coarse generator artifacts rather than a disguised oracle.

## Rejection efficiency and leakage

`rejection_efficiency_audit.json` reports attempts per accepted target and
acceptance by target, including the worst cell and target operational
assessment. The audit checks that exact target balancing remains operationally
capable of producing future 1,000-item splits; it does not alter the answer
space to improve acceptance.

The train/test/calibration namespaces are checked for collisions in latent
IDs, latent seeds, and rendered canonical prompt hashes. Surface-twin audits
verify deterministic oracle equality and non-trivial textual change. These
results are recorded in `split_leakage_audit.json` and the structural audit
bundle.

## Frozen Qwen qualification remains unchanged

The pre-model structural gate adds prerequisites; it does not change the
frozen baseline-only Qwen qualification thresholds:

- decimal accuracy 30%–75%;
- decimal/number-word agreement at least 85%;
- canonical/surface-twin agreement at least 80%;
- normalized predicted-digit entropy at least 0.80.

No family/difficulty cell is selected by this gate for scientific use. The
structurally eligible cells are merely the cells allowed to reach the later
baseline-only calibration. That calibration must still select, within each
family, the qualifying cell closest to 50% accuracy. Fewer than two
model-qualified families stops the instrument.

## Status

The model-free structural gate passes because MODREG10, FSM10, and SATCOUNT10
retain eligible cells and MODREG effective depth is monotonic. The following
have not been run:

- Qwen tokenization audit;
- Qwen baseline calibration;
- activation extraction or vector construction;
- steering or development evaluation;
- confirmatory holdout evaluation;
- Q2 geometry.

The next permitted action is principal review followed by a separately
authorized RunPod-only, baseline-only E3-10 instrument calibration.

The committed `configs/q1_v2_structural_eligibility.json` is the small
model-free handoff used by the future calibration-manifest builder. It records
the v2 generator version and the eligible-cell allowlist, so structurally
failed cells cannot be scheduled for Qwen by accident.
