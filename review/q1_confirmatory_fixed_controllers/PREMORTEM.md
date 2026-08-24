# Q1 Confirmatory Premortem

Dress rehearsal: `DRESS_REHEARSAL_PASS`.

Classification: `PREMORTEM_PASS`.

The main risks are holdout leakage, controller drift, null selection, seed collision, parser drift, model-output attrition, cross-model peeking, resume duplication, and cost overrun. The locks prohibit content access before the cost gate, bind byte/canonical controller hashes, freeze two isotropic plus two source-pair sign-shuffled nulls per model, use globally distinct seeds, retain all model-level invalid outcomes, require both journals before analysis, resume by immutable logical key, and require the 25% cost margin.
