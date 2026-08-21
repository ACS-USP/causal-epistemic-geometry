# Gate 6.3 — Semantic V2 forensic reanalysis

Parser: `external-semantic-v2`.

The historical Gate 6.2 journal and report are unchanged.  This report
recomputes all 200 rows offline from raw outputs with one parser applied
uniformly to every condition.

Rows reclassified or otherwise changed: **30 / 200**.

## Condition summary

| condition | V2 valid/20 | V2 correct | V2 wrong | V2 invalid | V2 truncated | V2 Q | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| BASELINE | 20/20 | 12 | 8 | 0 | 0 | 0.000 | reference/random |
| BEST_SINGLE_MEAN_PLUS | 18/20 | 10 | 8 | 2 | 0 | 0.700 | True |
| MULTILAYER_MEAN_MINUS | 20/20 | 9 | 11 | 0 | 0 | 0.250 | False |
| MULTILAYER_MEAN_PLUS | 13/20 | 8 | 5 | 4 | 3 | 0.700 | False |
| MULTILAYER_RANDOM_MEAN_R0 | 20/20 | 11 | 9 | 0 | 0 | 0.150 | reference/random |
| MULTILAYER_RANDOM_MEAN_R1 | 19/20 | 11 | 8 | 1 | 0 | 0.200 | reference/random |
| MULTILAYER_RANDOM_MEAN_R2 | 19/20 | 10 | 9 | 1 | 0 | 0.300 | reference/random |
| MULTILAYER_RANDOM_MEAN_R3 | 19/20 | 9 | 10 | 1 | 0 | 0.450 | reference/random |
| TEXTUAL_CAREFUL_REFERENCE | 20/20 | 15 | 5 | 0 | 0 | 0.500 | reference/random |
| TEXTUAL_DIRECT_REFERENCE | 20/20 | 11 | 9 | 0 | 0 | 0.250 | reference/random |

Random mean Q: `0.275000`; random maximum Q: `0.450000`.
BEST_SINGLE_MEAN_PLUS gate: `True`.

## Offline classification: `GATE6_2A_PARSER_REANALYSIS_SINGLE_MEAN_PASS`

Only BEST_SINGLE_MEAN_PLUS is promotable by the authorized continuation.
If it does not pass, no RunPod phase is authorized by this reanalysis.
The original Gate 6.2 classification remains GATE6_2_NO_BEHAVIORAL_FIRST_STAGE.
