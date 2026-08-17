# Q1 V1 Series Closeout

## Status

Q1 V1, V1.1, and V1.2 are formally closed as a **DEVELOPMENT instrument
series**. They are not confirmatory analyses and no scientific result is
frozen from them. The confirmatory holdout remains untouched and Q2 has not
been run.

The residual V1.2 evidence showed that activation steering can perturb an
answer distribution and an observed error set. It did not establish robust
semantic complementarity. In particular, the primary centered-logit summary
was positive while the pre-specified secondary summary was negative, and the
item-level flip overlap was low. The conclusion is therefore
estimator-sensitive, not a robust Q1 finding.

## Instrument decision

The MMLU-Pro multiple-choice instantiation is closed as the primary Q1
measurement instrument. Arbitrary displayed answer slots and the need to
aggregate across answer renderings made it impossible to cleanly distinguish
semantic error movement from an output-format effect.

Old code, manifests, and run artifacts are preserved for auditability. They
must not be deleted or silently relabeled as confirmatory evidence.

## What remains open

The project question remains:

> Can the local representation geometry of a frozen language model predict and
> causally control the covariance structure of its errors?

The next instrument, E3-10, replaces answer slots with an exact semantic
answer space `{0, 1, ..., 9}`. It is a new development instrument, not V1.3.
Its baseline-only qualification must be completed and reviewed before any
activation direction or steering result is constructed.
