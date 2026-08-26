# Q2 V4.1 — Frozen 31-safe-bank adequacy review

Historical V4 classification remains Q2_V4_SAFE_BANK_INSUFFICIENT and Q2_V4_PRESEMANTIC_FORENSIC_CLEAN. The V4 relational hypothesis remains untested.

## Decision

Q2_V4_1_31_SAFE_BANK_ADEQUATE

The decision was applied mechanically after the design precheck was frozen. All 31 safe directions were retained in original candidate order; no direction was redrawn, added, removed, or optimized.

## Safety attrition

40 total, 31 safe at both shells, 9 unsafe: V4_DIRECTION_05, V4_DIRECTION_12, V4_DIRECTION_14, V4_DIRECTION_16, V4_DIRECTION_21, V4_DIRECTION_25, V4_DIRECTION_27, V4_DIRECTION_36, V4_DIRECTION_38.

| Metric | 40 candidates | 31 safe | V4.1 gate | Result |
|---|---:|---:|---:|---|
| Rank | 8 | 8 | full rank 8 | PASS |
| Effective rank | 7.291781 | 7.225679 | >= 6.0 | PASS |
| Stable rank | 5.143487 | 4.735927 | descriptive | — |
| Condition number | 1.906976 | 2.021583 | <= 3 | PASS |
| Max abs pair cosine | 0.844870 | 0.844870 | < 0.98 | PASS |
| A0 q90-q10 | 0.967417 | 0.923603 | >= 0.20 | PASS |
| Shell amplitude CV max | — | 0.00000112 | <= 0.03 | PASS |

The 31-safe bank is conditioned on the frozen safety gate and is not claimed to be unconditionally isotropic. Safety-label separation is descriptive: centroid distance=0.448559, permutation p=0.178482, with high uncertainty at n=40.

## Reserve fragility

| p_safe | P(#safe >= 32 | 40,p) | candidates for >=95% |
|---:|---:|---:|
| 0.700 | 0.111009 | 53 |
| 0.750 | 0.299832 | 49 |
| 0.775 | 0.438549 | 47 |
| 0.800 | 0.593127 | 46 |
| 0.850 | 0.864598 | 42 |
| 0.900 | 0.984505 | 39 |

## Scientific firewall

New GPU inference: NONE. New model inference: NONE. Correctness inspected: NO. A1/A2 new computation: NONE. Semantic outcomes: 0. Q3: NOT RUN. The original V4 classification is immutable.
