# Character-count semantic V3 specification

The evaluator uses external-semantic-v3's condition-blind extraction of exactly
one explicit final commitment and its typed canonical representation. The
reference is always a canonical integer. No reasoning-text number is read and
no numeric string is coerced into an integer. Commitment validity, semantic
evaluability, and correctness are separate axes. Truncation, no commitment,
ambiguity, and model runtime failure remain primary errors.

