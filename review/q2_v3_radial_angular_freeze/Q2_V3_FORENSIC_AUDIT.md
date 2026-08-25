# Q2 V3 forensic audit

Classification: `Q2_V3_FORENSIC_CLEAN`

An independent, model-free reconstruction fetched the exact public CRUXEval
revision and compared every record in the six frozen allocation manifests.
It reproduced the primary gate result exactly: 336 records checked, 327 prompt
hashes matching, nine prompt hashes mismatching, and zero reference-hash
mismatches.

For all nine mismatches, the frozen value equals the historical namespaced
`EXTERNAL-PROMPT` digest of the older external-qualification prompt template.
The affected IDs are `sample_300`, `sample_74`, `sample_659`, `sample_777`,
`sample_145`, `sample_698`, `sample_21`, `sample_745`, and `sample_700`.

The audit verified that the runner stopped before model inference, did not
create prediction matrices or a prediction lock, did not open semantic
outcomes, did not substitute items, and did not rewrite any frozen protocol
artifact. The terminal state is therefore the mechanically required
`Q2_V3_PANEL_PROVENANCE_MISMATCH`.

This audit concerns execution integrity only. It supplies no evidence about
M0, M1, M2, semantic error geometry, or radial behavior.
