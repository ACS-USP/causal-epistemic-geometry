# Gate 11 domain-conditioned control postmortem closeout

Gate 11 completed 320 prompt-activation logical rows and 336 fixed-sequence
teacher-forcing rows. It sampled no token, generated no free response, scored
no new semantic answer, and did not access Q2 or the holdout.

The careful/direct source representation transferred to character count. At
L27, the frozen-axis character-count mean gap was 112.17, all 40 gaps were
positive, its paired-bootstrap lower bound was 111.04, and the descriptive
character-count direction had cosine 0.659 with the frozen controller.

The prospective finite-displacement control-gain and policy-realization domain
shift rules did not pass. Historical Gate-9/Gate-10 outcomes did support a
policy-utility difference: CRUXEval-minus-character-count accuracy effects were
positive for both the activation controller and textual CAREFUL policy. The
frozen primary synthesis is `GATE11_POLICY_UTILITY_DOMAIN_MISMATCH`.

These measurements must remain distinct:

- source-axis metrics measure representation transfer;
- D75 KL/JS and hidden displacement are finite-displacement control-gain
  diagnostics;
- Gate 11 did not measure an exact local pullback/Fisher metric;
- historical accuracy and G/C/D measure task utility, not control energy.

The independent audit reproduced checkpoint aggregation to numerical precision
and the same synthesis. Complete per-checkpoint vocabulary logits and hidden-
difference vectors were not persisted, however, so primitive propagation
metrics cannot be independently recomputed. The forensic classification is
`GATE11_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN`. No replacement collection or
Gate 12 was run. Principal review is required.
