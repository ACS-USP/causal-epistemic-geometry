# Q1 budget × D75 causal interaction — closeout

**Classification:** `Q1_BUDGET_CAUSAL_INTERACTION_INCONCLUSIVE_AT_N96`  
**Candidate state:** `CLOSED_NO_FURTHER_COLLECTION`

This was one prospective, fixed-controller experiment: Qwen3-8B with the
existing layer-27 D75 intervention versus matched baseline, across 96 fresh
latent items, two token caps and two matched rollouts. The experiment tested
whether D75 changes the unconditional probability of a correct delivered
answer, and whether that change differs by cap.

## Integrity chain

- Frozen lock SHA-256: `4940c5ab434becb6d5f06e58f3199735c5582b20411ad81f4d11b1747cf0ec7b`
- Complete journal: 768/768 logical keys
- Journal SHA-256: `36324cb330b9ae7c7d1bbb6f44ee3e021a6aaca146572044151897a1ec2f74a5`
- Raw seal: `RAW_JOURNAL_SEALED`
- Independent raw audit: `RAW_SEAL_INDEPENDENT_AUDIT_PASS`
- Independent numerical audit: `Q1_BUDGET_INTERACTION_ANALYSIS_AUDIT_PASS`
- Maximum primary/audit numeric difference: `1.39e-17`

The original collection was stopped only by the executor duration limit after
765 persisted rows. A resumed job generated the three prespecified missing
keys; it reused the 765 exact journal entries and did not replace any row.
Both raw audits subsequently established exact 768-key coverage before outcome
analysis.

## Prespecified results

Equal-latent, then equal-cell, then equal-family estimates:

| Cap | Baseline correct | D75 correct | D75 − baseline | Baseline incomplete | D75 incomplete |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 55.73% | 59.90% | **+4.17 pp** | 43.75% | 39.58% |
| 4,096 | 66.15% | 63.02% | **−3.13 pp** | 33.85% | 35.42% |

The cap interaction is **−7.29 pp**. The predeclared stratified latent
bootstrap is descriptive: its 95% percentile intervals are `[0.00, 8.33] pp`
at 2,048, `[-5.21, -1.04] pp` at 4,096, and `[-11.98, -3.13] pp` for the
interaction.

## Fixed decision rules

None of the three threshold-60 rules triggered:

| Rule | Statistic | Result |
|---|---:|---|
| Relevant 10 pp gain at either cap | 0.0168 | `NOT_TRIGGERED` |
| 10 pp gain excluded at both caps | 7.9203 | `NOT_TRIGGERED` |
| Nonzero cap interaction | 2.4333 | `NOT_TRIGGERED` |

Accordingly, the fixed candidate is **inconclusive at N=96**. The descriptive
pattern is not a license to add items, alter caps, change dose, remove items,
or run another controller within this candidate. It also does not demonstrate
that D75 is beneficial at the shorter cap, harmful at the longer cap, or that
token budget mediates the intervention effect. Those would require a new,
separately predeclared study.

## Scope

This result applies only to the frozen Qwen3-8B controller, layer-27 D75,
canonical surface, synthetic reasoning families, and the two stated budgets.
It does not modify the earlier geometry findings or reopen the closed Q3
routing candidate.
