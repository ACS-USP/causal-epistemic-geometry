# Q2 V4 protocol amendment 02 — future schedule binding

Classification: `CLASS_B_IMPLEMENTATION_REPAIR_BEFORE_PRELOCK_AND_BEFORE_SEMANTIC_OUTCOMES`.

During implementation review, before any candidate bank existed and without inspecting
source qualification metrics, the future 39,000-row schedule helper was found to enumerate
directions `00..31` rather than accept the actual first 32 safety-qualified IDs. This would
have been wrong whenever reserve selection skipped an early candidate.

The helper now requires the exact ordered selected-ID list, verifies 32 unique members of
the one frozen 40-candidate sequence, and constructs all 64 shell conditions from that list.
Future scientific seeds use collision-checked 63-bit hashes. The QAP and radial seeds are
the prospectively specified big-endian first 128 bits of their SHA-256 namespaces, and the
radial shell-swap schedule is materialized independently.

No candidate direction, safety result, source metric, A1/A2 value, primary semantic output,
or Q3 output was inspected or used. The model, source qualification, candidate stream,
safety gates, predictor definitions, endpoint, thresholds, and classifications are
unchanged. This amendment is included in the unique PRELOCK commit; no bank predates it.

The same pre-PRELOCK implementation review also materialized the previously specified A1
numerical qualification gate before any covariance activations existed: exact 64x4096
shape, finite activations/matrix, positive lambda=0.10 ridge, finite condition number at
most 1e6, effective rank at least 2, deterministic fit hash, symmetry/diagonal error at
most 1e-10, and cosine-distance range within floating tolerance of [0,2]. These are
numerical integrity checks, not outcome-based controller selection.
