# Q2 V3 prompt-provenance reconciliation

Final state: `Q2_V3_REFREEZE_REQUIRES_PRINCIPAL_RESEARCHER_DECISION`

This is a CPU-only, provenance-only closeout. It preserves the original
`Q2_V3_PANEL_PROVENANCE_MISMATCH` clean abort at
`9b1bc16ea6893ed798575a6850f6db602532ef69`. No model inference, prediction
matrix, controller qualification, semantic trajectory, or Q3 work occurred.

## Root cause

The freeze fallback copied nine legacy
`stable_digest("EXTERNAL-PROMPT", old_template_prompt)` values into a
`prompt_sha256` field whose other 327 records contain raw SHA-256 values of a
different Gate-7 prompt template; thus both the hash schema and the exact
model-visible user-message wording were mixed.

## Exact reconstruction

The official CRUXEval test JSONL at revision
`b96af0450242eb4da433032b90998f25588a5d0f` contains 800 unique records. The
336 Q2 V3 allocation IDs were persisted, outcome-free, in
`OFFICIAL_SOURCE_RECORDS.jsonl`. For every affected item:

- `code`, `input`, `output`, item identity, and reference SHA reproduce;
- the old raw prompt is reproduced by
  `scripts/prepare_external_benchmark.py::_cruxeval_prompt`;
- the old digest is exactly
  `SHA256(UTF8("EXTERNAL-PROMPT") || 0x1f || old_prompt_utf8)`;
- the new raw prompt is reproduced by
  `src/epistemic_geometry/experiments/gate7.py::task_prompt`;
- the executor comparison is exactly `SHA256(new_prompt_utf8)`;
- exact UTF-8 bytes, hex, base64, line endings, Unicode-normalization status,
  prefixes/suffixes, unified diffs, and historical artifact paths are retained
  in `NINE_ITEM_BYTE_COMPARISON.json`.

The source `code`/`input` payloads and code fences are the same. The template
wording is not: “Solve the following…” becomes “Solve this Python…”, “Python
function” becomes “Function”, and the final-answer instruction changes from
“Reason carefully, then end…” to a three-line exact-return instruction. Those
are plausible behavioral manipulations, not inert serialization.

## Nine-item classification

| Item | Purpose | Class | Exact difference |
| --- | --- | --- | --- |
| `sample_300` | M1 covariance | P2 | Legacy wording/instruction + namespaced hash versus Gate-7 wording/instruction + raw SHA |
| `sample_74` | M1 covariance | P2 | Same global template and hash-schema difference |
| `sample_745` | Shell calibration | P2 | Same global template and hash-schema difference |
| `sample_700` | Shell calibration | P2 | Same global template and hash-schema difference |
| `sample_659` | Primary panel | P2 | Same global template and hash-schema difference |
| `sample_777` | Primary panel | P2 | Same global template and hash-schema difference |
| `sample_145` | Primary panel | P2 | Same global template and hash-schema difference |
| `sample_698` | Primary panel | P2 | Same global template and hash-schema difference |
| `sample_21` | Primary panel | P2 | Same global template and hash-schema difference |

Classification totals: P0=0, P1=0, P2=9, P3=0.

## Wider prompt-schema audit

All six Q2 V3 item manifests were audited:

| Purpose | Gate-7 template + raw SHA | Legacy template + namespaced digest |
| --- | ---: | ---: |
| Source construction | 24 | 0 |
| Source validation/qualification | 24 | 0 |
| Shell calibration | 10 | 2 |
| M1 covariance | 62 | 2 |
| M2 probes | 12 | 0 |
| Primary panel | 195 | 5 |
| **Total** | **327** | **9** |

No unknown third schema, raw-legacy hash, namespaced-current hash, accidental
collision, duplicate allocation ID, reference mismatch, or hidden item-manifest
inconsistency was found.

Source construction/qualification additionally carries ten exact system
instructions (five families × two polarities). They are defined once in
`SOURCE_FAMILIES` and are separately inventoried with raw byte hashes. M2 and
execution-boundary capture use one exact teacher-forced continuation, also
inventoried. Engineering and cost preflight reuse the first two shell items and
introduce no new task prompt. Activation conditions do not alter user prompt
bytes.

The original manifests bound user-message content, not the tokenizer-rendered
chat bytes. Future provenance should bind both exact role/content bytes and the
rendered chat hash under the frozen tokenizer revision.

## Why Amendment 1 was not created

The original freeze source commit
`9a748de3706a788f8c6c5a1d12c09489808006e8` did not define/import one Q2 V3
task template, did not assign a prompt-template version, and did not embed exact
prompt bytes. The frozen manifests therefore contain conflicting implicit
bindings: 327 values identify Gate-7 prompt bytes, while nine values identify a
legacy namespaced digest and old prompt bytes.

The mechanical executor's explicit import of `gate7.task_prompt` was introduced
later in commit `ba7c91e03295c664788e6ede0dd643f6d446404b`, after the original
freeze. It cannot retroactively establish a prospective scientific choice.
Choosing the Gate-7 template globally may be sensible, but choosing it rather
than the legacy template is still a prompt-design decision. Under the frozen
decision tree, any P2 discrepancy without a previously determined global
representation requires principal-researcher review.

Accordingly:

- original freeze: preserved;
- clean abort: preserved;
- Amendment 1: **not created**;
- amended manifests: **not created**;
- experiment execution: **not authorized**.

## Candidate future contract

`PROMPT_PROVENANCE_CONTRACT.md/json` defines a non-authorized migration
candidate that separates:

- item/source/reference identity;
- purpose;
- task namespace;
- prompt-template version;
- exact UTF-8 user/system bytes;
- encoding, line-ending, and Unicode policy;
- raw-byte hashes;
- tokenizer/chat-rendering metadata;
- rendered prompt hash;
- legacy provenance fields.

The contract cannot be frozen until the principal selects the one global
scientific prompt representation.

## Invariants

- scientific question: unchanged;
- controller families/directions/layer: unchanged;
- shell targets and all qualification/identifiability gates: unchanged;
- M0/M1/M2 definitions: unchanged;
- statistical thresholds, QAP, bootstrap, radial gate, and taxonomy: unchanged;
- six allocation ID lists and schedules: unchanged;
- primary panel N: 200;
- primary ordered-ID SHA-256:
  `969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf`;
- rollouts and seeds: unchanged;
- Q2 V2/M3/Q1: unchanged;
- Q3: not run.

## Independent audit

`INDEPENDENT_AUDIT.json` recomputes the two templates and both hash procedures
without importing the primary reconciliation implementation. It reproduces
327/9, all nine P2 IDs, 336/336 references, the primary panel identity, and the
immutable zero-outcome clean abort.

Classification: `Q2_V3_PROVENANCE_RECONCILIATION_AUDIT_PASS`.

Next action: `PRINCIPAL_RESEARCHER_DECISION_ON_CANONICAL_Q2_V3_PROMPT`.
