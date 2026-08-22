GATE 11.1 — FORENSIC AUDIT
======================================================================

classification:
    GATE11_1_FORENSIC_REPLICATION_CLEAN_AGREEMENT

The independent path read the 48 persisted raw NPZ shards and recomputed
full-vocabulary log-probabilities, KL/JS, norms, top-1 flips, careful-logit
alignment, item aggregation, random summaries, and the synthesis flags. It did
not call the primary high-level metric functions.

Integrity checks:

- 336 unique journal rows;
- 48 item groups, each with 7 conditions;
- 48 manifest entries and 48 verified shard hashes;
- all seven conditions present symmetrically;
- float32 logits and hidden-difference arrays present;
- fixed sequence metadata present;
- no free generation and no new semantic evaluation;
- historical Gate-11 result preserved.

Primary classification:
    GATE11_POLICY_UTILITY_DOMAIN_MISMATCH

Independent classification:
    GATE11_POLICY_UTILITY_DOMAIN_MISMATCH

Maximum primary/independent metric difference:
    0.0

The historical Gate-11 integrity concern is repaired for the propagation
diagnostics because the artifact-complete rerun preserves the raw vocabulary
logits and hidden-difference vectors required for recomputation. This does not
rewrite the historical Gate-11 report or classification.

The audit explicitly does not claim an exact local pullback/Fisher metric.
