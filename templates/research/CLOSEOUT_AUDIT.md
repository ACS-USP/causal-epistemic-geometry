# Closeout audit — `<GATE_ID>`

- Gate state: `FORENSIC_AUDIT`
- Source commit: `<FULL_COMMIT>`
- Frozen protocol: `<PATH_AND_SHA256>`
- Artifact manifest: `<PATH_AND_SHA256>`
- Auditor: `<NAME_OR_ROLE>`
- Date (UTC): `<YYYY-MM-DD>`

Answer **yes**, **no**, or **unknown**, then provide evidence. Any unknown
integrity answer blocks `CLOSED`.

## Required audit questions

1. Did code implement the frozen protocol?
   - Answer: `<YES|NO|UNKNOWN>`
   - Evidence:
2. Did any condition-dependent formatting affect validity?
   - Answer: `<YES|NO|UNKNOWN>`
   - Evidence:
3. Any missing/duplicate logical rows?
   - Answer: `<YES|NO|UNKNOWN>`
   - Expected/observed/duplicate/missing counts:
4. Any retry selection?
   - Answer: `<YES|NO|UNKNOWN>`
   - Retry provenance:
5. Any mismatch between gate name/classification and actual observed mechanism?
   - Answer: `<YES|NO|UNKNOWN>`
   - Reconciliation:

## Preservation and provenance

- Original results preserved unchanged: `<YES|NO>`
- Offline amendments are additive and hash-linked: `<YES|NO|N/A>`
- Source commit and dirty status:
- Environment preflight artifact:
- Model/benchmark revisions:
- Logical identity fields:
- Seed/resume verification:
- Holdout status:

## Classification

- Stored classification:
- Independently recomputed classification:
- Mechanism actually observed:
- Auxiliary-arm results reported separately:
- Claim impact: `<NONE|PRINCIPAL_RESEARCHER_REVIEW>`

## Decision

- `READY_TO_CLOSE | BLOCKED_RECOVERABLE | BLOCKED_SCIENTIFIC_REVIEW`
- Typed incident reason (if blocked):
- Follow-up authorization already present in frozen decision tree: `<YES|NO>`
- Principal-researcher items:
