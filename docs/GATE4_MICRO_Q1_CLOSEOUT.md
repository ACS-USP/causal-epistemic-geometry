# Gate 4 — First Original Micro-Q1 Closeout

**Status:** `MICRO_Q1_NO_DETECTABLE_SIGNAL`  
**Stage:** development kill-test; not confirmatory

Gate 4 tested one frozen Qwen3-8B full-non-thinking CRUXEval substrate with
one paired careful-minus-direct direction at layer 17, its negative sign, and
one norm-matched orthogonal random direction. The fresh evaluation contained
50 CRUXEval items, four conditions, and two independent rollouts per
condition/item: 400 scientific trajectories.

The direction was constructed without correctness labels or generated outputs
from 64 paired prompt-boundary activation contrasts. Its held-out validation
passed with 16/16 positive signed gaps and positive mean gap. The alpha-zero
identity, BF16-aware additive shift, final-token scope, and hook cleanup checks
all passed. The direction and alpha were therefore not the reason for an early
stop.

| Condition | Valid/100 | Correct | Wrong | Invalid | Truncated | Accuracy | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 99 | 40 | 59 | 1 | 0 | 0.400 | 0.990 |
| + deliberation | 98 | 38 | 60 | 2 | 0 | 0.380 | 0.980 |
| - deliberation | 98 | 38 | 60 | 2 | 0 | 0.380 | 0.980 |
| Random orthogonal | 98 | 40 | 58 | 2 | 0 | 0.400 | 0.980 |

Both meaningful signs remained within the frozen competence and validity
guards. However, neither sign reached the frozen propensity movement threshold
(`D >= .05` and `Delta_D >= .05`), so neither could qualify as movement beyond
the random control. The primary classification is therefore
`MICRO_Q1_NO_DETECTABLE_SIGNAL`.

This is not evidence that activation steering or the broader Q1 hypothesis is
impossible. It is a bounded development result for this one direction, alpha,
layer, model policy, and CRUXEval substrate. Character-count replication,
another direction/layer/alpha, Q2, geometry, and the confirmatory holdout were
not run.

Artifacts and the complete journal are in `review/micro_q1/`; the A40 was
stopped before local analysis. The exact execution checkout was corrected in
`review/micro_q1/PROVENANCE_CORRECTION.json`.
