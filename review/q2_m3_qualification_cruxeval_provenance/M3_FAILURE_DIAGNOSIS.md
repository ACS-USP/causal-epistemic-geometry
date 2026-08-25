# M3 failure diagnosis

The frozen real-Qwen classification remains
`M3_DERIVATIVE_IDENTITIES_FAILED`. No rerun or threshold change was performed.

The exact FP32 engine was coherent: sequence equivalence, repeat/order/chunked
reproducibility, independent JVP agreement, JVP/VJP duality, and PSD all passed.
The frozen direct-versus-polarization comparison nevertheless had relative
Frobenius error **0.2526146991**, above the **0.01** limit.

An offline array-only localization found that the four prescribed fixtures had
2, 6, 6, and 7 checkpoints. The direct comparator weighted all 21 checkpoints
uniformly, while the polarization branch first averaged within each fixture and
then weighted the four fixtures uniformly. The distance between those two
weightings was **0.2526147131**. Under the same fixture weighting, polarization
agreed with the direct Gram to **1.82e-7**. Thus the frozen polarization failure
is best localized to an aggregation-weight mismatch, not to the exact JVP
identity itself.

This diagnosis does not qualify M3. The finite-difference Gram/radius errors
plateaued at the same weighting discrepancy, so no frozen three-scale window
passed. Independently, the mandatory BF16 bridge failed materially: baseline
top-1 agreement was **0.973684** (required 0.99), radius Spearman **0.714286**,
distance Spearman **0.014286**, and median curvature relative error **10.1082**.
Even a corrected aggregation implementation would therefore require a new,
separately authorized prospective qualification before M3 could enter Q2 V3.

No scientific item, correctness outcome, free generation, or Q2 V3 behavioral
trajectory entered this diagnosis.
