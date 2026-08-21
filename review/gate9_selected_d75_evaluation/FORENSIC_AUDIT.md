# Gate 9 independent forensic audit

Classification: `GATE9_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES`.

- Frozen/observed rows: 1400/1400
- Unique logical keys: True
- Exact independent seed schedule: True
- Condition-symmetric semantic-V3 reparse: True
- Maximum primary/audit metric difference: 0
- Classification agreement: True

One documented non-scientific issue occurred: semantic-v3's literal fallback was made total after an unhashable nested set literal raised before its row was journaled. The 1,233 preserved rows were unchanged; all 1,400 raw outputs reparse condition-symmetrically, and the exact failed logical key was resumed with its frozen seed.

All causal estimands were independently recomputed from raw binary outcome arrays without calling the primary Gate-9 analysis path.
