# Prospective M3 numerical qualification

Status: `DRAFT — MUST BE COMMITTED BEFORE REAL-MODEL MEASUREMENT`.

M3 is the teacher-forced, multi-checkpoint categorical-Fisher Gram restricted to six engineering-only directions at Qwen block 27. Each row conditions on a frozen arbitrary token prefix and averages uniformly over final-prompt and prescribed continuation checkpoints. It is the local output-information geometry of the FP32 computational lift of the frozen BF16-valued parameters, not a semantic-error metric and not the Fisher geometry of the full free-running trajectory distribution.

The exact thresholds, epsilon ladder, BF16 bridge, sequence checks, PSD rule, direct/polarization comparison, cost ceiling, and classification vocabulary are machine-frozen in `M3_QUALIFICATION_PROTOCOL.json`. No clipping of an indefinite Gram is allowed. Failure of the BF16 bridge excludes M3 from Q2 V3 even when exact FP32 identities pass.
