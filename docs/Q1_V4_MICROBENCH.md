# Q1 V4 — Bench E / Bench G micro-screen

V4 is an instrument-reconnaissance reset after the V3 reasoning screen and the
CRUXEval smoke failed to provide a clean error regime. It separates two jobs:

- Bench E: cheap, exact tasks that should produce finished semantic errors.
- Bench G: simple domains with known cyclic/sequential structure for activation
  geometry diagnostics.

The current branch contains no steering, vector learning, layer search, alpha
sweep, causal geometry test, code-generation pilot, or holdout access.

## Authorized sequence

1. Recompute the frozen CRUXEval smoke with a deterministic type-aware
   postmortem. This is diagnostic only and cannot change the original result.
2. Freeze 30 procedural character-count items: 10 each in short, medium, and
   long predefined strata. Run one Qwen3-8B sampled trajectory per item with
   the fixed 8192-token cap.
3. Collect forward-only activations for 49 weekday-cycle and 45 letter-sequence
   prompts at preselected block 31. Use both thinking prompt-boundary and
   `enable_thinking=False` direct positive-control views.
4. Audit whether the local code-benchmark boundary can expose nested
   per-test-case failure vectors. Do not run code generation.

The GPU budget target is US$0.20 and hard stop US$0.35. The Pod must be stopped
after the bounded work. Results are development diagnostics, not scientific
claims.

See the generated review bundle under `review/q1_v4_microbench/`.
