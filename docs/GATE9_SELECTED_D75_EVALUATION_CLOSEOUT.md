# Gate 9 selected-D75 evaluation closeout

Gate 9 is closed as
`GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION` on a fresh 100-item CRUXEval
DEVELOPMENT sample. The exact Gate-8-selected L27 plus controller at D75 raised
accuracy from 0.47 to 0.60 while preserving commitment validity and semantic
evaluability at 0.97 versus 0.985 at baseline.

The independent two-rollout estimands were G=0.1325, C=0.064343, and D=0.1200.
The corresponding random-bank maxima were 0.0175, 0.009747, and 0.0200. Rescue
was 0.1525 and damage was 0.0225. All frozen strong-safe criteria passed,
including positive item-cluster bootstrap lower bounds and leave-one-item-out
sign stability.

One recoverable instrumentation incident occurred after 1,233 rows: a nested
unhashable set literal escaped the semantic-v3 raw-string fallback. No
scientific row had been written for the affected logical key. A
condition-symmetric totality repair broadened only the exception handling,
was tested and committed, and the exact key and frozen seed were resumed. The
original 1,233 rows were untouched and the final journal contains exactly 1,400
unique scheduled rows. Full records are under
`review/gate9_selected_d75_evaluation/`.

The independent forensic recomputation agreed with the primary classification
and every audited metric (maximum absolute difference 0). Because the parser
totality repair occurred after some raw outputs already existed, the audit
honestly records `GATE9_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES`, not a clean label;
it found no scientific-integrity concern and no change to the result.

This is not confirmatory evidence, Q2, character-count replication, or a
general controller search. Gate 10 is drafted but not authorized; the
confirmatory holdout remains untouched.
