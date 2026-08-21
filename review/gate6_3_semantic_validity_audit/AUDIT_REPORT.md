# Gate 6.3 Semantic-Validity Audit

Historical frozen result: `GATE6_3_SINGLE_MEAN_DESTRUCTIVE` (unchanged).
Offline V3 diagnostic: `GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL`.

No model inference, RunPod access, Q2, character count, or holdout access occurred.

## Condition-symmetric V3 summary

| Condition | V2 validity | Commitment validity | Semantic evaluability | Accuracy |
|---|---:|---:|---:|---:|
| `BASELINE` | 0.975 | 0.975 | 0.975 | 0.517 |
| `TEXTUAL_CAREFUL_REFERENCE` | 0.950 | 0.992 | 0.992 | 0.750 |
| `BEST_SINGLE_MEAN_PLUS` | 0.908 | 0.975 | 0.975 | 0.700 |
| `SINGLE_L27_RANDOM_R0` | 0.967 | 0.992 | 0.992 | 0.483 |
| `SINGLE_L27_RANDOM_R1` | 0.942 | 0.992 | 0.992 | 0.508 |
| `SINGLE_L27_RANDOM_R2` | 0.958 | 0.975 | 0.975 | 0.483 |
| `SINGLE_L27_RANDOM_R3` | 0.958 | 0.975 | 0.975 | 0.483 |

## Primary audit estimands

- Meaningful `G`: 0.204167
- Meaningful `C`: 0.117655
- Meaningful `D`: 0.166667
- Random mean/max `G`: -0.002083 / 0.012500
- Random mean/max `C`: 0.011335 / 0.025000
- Random mean/max `D`: 0.025000 / 0.050000

## Reclassification audit

- All rows reclassified or re-extracted: 29 / 920
- Primary rows affected: 27 / 840
- Matched consistency rows affected: 2 / 80
- V2 historical crosscheck failures: 0

Every condition was processed by the same frozen parser hash. The historical V2 classification remains the registered result; V3 is an additive offline diagnostic.

## Next protocol

Drafted `GATE7_FRESH_SINGLE_L27_REPLICATION_PROTOCOL` only. It was not executed.
