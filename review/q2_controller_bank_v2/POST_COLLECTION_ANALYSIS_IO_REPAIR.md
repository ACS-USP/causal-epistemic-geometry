# Q2 V2 post-collection analysis I/O repair

Date: 2026-08-25

Scientific collection: complete before this record (6,960/6,960 rows)
Scientific metrics inspected before failure: none

The first invocation of `scripts/analyze_q2_controller_heldout_v2.py` stopped
with `KeyError: item_id` before constructing any condition array, metric, or
scientific summary. The collector correctly persisted every trajectory in the
project's frozen `research-os-jsonl-v1` crash-safe envelope, where the
scientific payload is stored under `row`. The primary, bootstrap, and forensic
scripts incorrectly treated the outer envelope as the payload.

The deterministic repair unwraps `wrapper["row"]` and validates the frozen
wrapper version and logical key. It changes no trajectory, evaluator output,
controller, item, seed, geometry definition, estimator, permutation,
bootstrap, threshold, or classification rule. The independent forensic script
uses a separate envelope reader and additionally audits a single journal
identity hash. A unit test fixes the expected envelope behavior.

This is a post-collection Class-C mechanical I/O repair. It is not a scientific
design amendment and does not replace or filter any output.
