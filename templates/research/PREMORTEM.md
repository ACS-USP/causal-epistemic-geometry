# Premortem — `<GATE_ID>`

- Gate state: `PREMORTEM`
- Source commit: `<FULL_COMMIT>`
- Frozen protocol: `<PATH_AND_SHA256>`
- Reviewer: `<NAME_OR_ROLE>`
- Date (UTC): `<YYYY-MM-DD>`

Answer **yes**, **no**, or **unknown**, then explain. An unknown that can affect
validity blocks `PROSPECTIVE_LOCK`.

## Causal and measurement threats

1. Can treatment alter output formatting and therefore parser validity?
   - Answer: `<YES|NO|UNKNOWN>`
   - Mitigation/evidence:
2. Is any missingness post-treatment?
   - Answer: `<YES|NO|UNKNOWN>`
   - Mitigation/evidence:
3. Can one mechanically bad item abort a whole phase unnecessarily?
   - Answer: `<YES|NO|UNKNOWN>`
   - Mitigation/evidence:
4. Are attrition and reserve rules frozen?
   - Answer: `<YES|NO|UNKNOWN>`
   - Lock reference:
5. Is the random null architecture/duration/energy matched?
   - Answer: `<YES|NO|UNKNOWN>`
   - Evidence:
6. Are measured outputs causally downstream of the intervention?
   - Answer: `<YES|NO|UNKNOWN>`
   - Evidence:
7. Is there a silent fallback?
   - Answer: `<YES|NO|UNKNOWN>`
   - Fail-closed behavior:
8. Are seed/resume semantics explicit?
   - Answer: `<YES|NO|UNKNOWN>`
   - Logical key and retry rule:
9. Can auxiliary arms contain a scientifically important result that the primary gate classification would hide?
   - Answer: `<YES|NO|UNKNOWN>`
   - Reporting plan:
10. Can any condition change token count enough to alter instrument behavior?
    - Answer: `<YES|NO|UNKNOWN>`
    - Mitigation/evidence:

## Frozen invariants

- Hypothesis:
- Estimand:
- Model/revision:
- Benchmark/revision:
- Conditions:
- Scientific thresholds:
- Parser/evaluator:
- Attrition/reserve rule:
- Seed/resume rule:
- Random-null matching:
- Holdout status:
- Cost ceiling:

## Adversarial failure injection

- Required fault suite passed: `<YES|NO>`
- Environment preflight contract selected: `<PROFILE>`
- Unresolved risks:

## Decision

- `READY_FOR_PROSPECTIVE_LOCK | BLOCKED_RECOVERABLE | BLOCKED_SCIENTIFIC_REVIEW`
- Typed incident reason (if blocked):
- Principal-researcher items:
