# Gate 6 — Layer–Source–RFM Control Atlas

Gate 6 is a development bridge after the audited Gate-5 duration result. It
does not reopen the benchmark, model, alpha, or Gate-4 controller. It asks a
narrower question: whether a source that is behaviorally validated without
semantic correctness labels yields a causal controller when its label-free
signal is learned separately at six prospectively fixed layers.

## Frozen scope

The model remains Qwen3-8B, BF16, full non-thinking generation, CRUXEval
semantic evaluation, and the Gate-5 sampling policy. The only layer indices are
8, 12, 17, 22, 27, and 32. The source-training pool is the 64 Gate-4
construction items plus the 40 Gate-5 source-check items. Fresh source
validation, manipulation, and evaluation items are disjoint from every
historical CRUXEval item.

The two source locations are deliberately distinct:

* `PROMPT_BOUNDARY` captures the final prompt-token residual stream before
  generation.
* `EXECUTION_BOUNDARY` teacher-forces the completed careful/direct continuation
  and captures the state immediately before the first token of the final
  `FINAL` marker.

No correctness, parsed answer, semantic outcome, or evaluation result enters a
controller. A controller is eligible only through held-out careful/direct
readout and teacher-forced source sensitivity. The standardized budget is
selected from those source-only gates, not from generated semantic outcomes.

## RFM provenance

The adapter follows the public `neural_controllers` RFMToolkit pattern at
`dmbeaglehole/neural_controllers` commit
`5f655a5f26daeb984dfe6c622b8fe537e1aed966`, using `xRFM` commit
`773fae81097ab000e6e7292a295e1d24adacca55`. The RFM is trained on binary
careful/direct source labels, its best AGOP is extracted, and its leading
eigenvector is oriented on held-out source pairs. The pinned upstream README
specifies the Python/Torch/Transformers/Datasets requirements; those are
recorded in the machine-readable protocol. The multi-layer current-token hook
and distributed energy accounting are repository execution adapters, tested
independently.

## Interpretation boundary

Gate 6 is not Q2. It does not build a direction bank from semantic outcomes,
search 36 layers, access the holdout, run character count, or claim a geometry
law. A positive result would justify a later replication protocol; a null would
justify a source/layer atlas before any broader geometry claim.
