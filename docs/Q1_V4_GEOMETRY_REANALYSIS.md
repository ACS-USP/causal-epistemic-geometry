# Q1 V4 geometry reanalysis

The original V4 analysis assigned sequential ranks to tied conceptual
distances. That is not Spearman correlation: tied observations must receive
their average rank, otherwise the statistic depends on arbitrary input order.

The historical files under `review/q1_v4_microbench/` were not modified. The
offline script [`scripts/reanalyze_q1_v4_geometry.py`](../scripts/reanalyze_q1_v4_geometry.py)
reads their checksummed rows and activations and writes a separate ignored
artifact under `review/q1_v4_microbench_reanalysis/`. No model was loaded.

| Domain | Original ρ | Corrected ρ | Original p | Corrected p | Corrected method |
|---|---:|---:|---:|---:|---|
| WEEKDAYS | 0.675325 | 0.703094 | 0.002700 | 0.005556 | exact 7! concept-label permutation |
| LETTERS | 0.357002 | 0.353588 | 0.000100 | 0.000100 | 10,000 frozen Monte Carlo label permutations, plus-one correction |

The qualitative interpretation is unchanged. These are descriptive
associations in a tiny known-structure fixture at one preselected layer. They
do not show behavioral transfer, causal control, semantic error
complementarity, intervention-geometry prediction, or collective utility.

The corrected implementation has tests for no ties, multiple ties, all-equal
inputs, a known reference, joint-permutation invariance, and the exact weekday
7! conceptual fixture.

