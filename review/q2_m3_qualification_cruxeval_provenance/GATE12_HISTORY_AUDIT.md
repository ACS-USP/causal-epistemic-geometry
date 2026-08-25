# Gate 12 / 12.1 derivative-engine audit

Gate 12 stopped before scientific geometry and before historical outcomes. Gate 12.1 used the FP32 computational lift of the exact BF16-valued checkpoint. FP32 full-sequence versus KV semantics, forward/independent JVP, JVP/VJP duality, Fisher/Hessian, and utility derivative identities passed. The historical BF16 bridge missed top-1 (0.977444 versus 0.99), and only epsilon 0.03 and 0.1 passed consecutively where three scales were required. The mismatch was attributed to mixed BF16 kernel/cache/reduction-order and dtype effects, not an off-by-one sequence bug.

M3 therefore receives a new prospective qualification tailored to a teacher-forced controller-span Gram; no historical failure is reinterpreted as scientific evidence.
