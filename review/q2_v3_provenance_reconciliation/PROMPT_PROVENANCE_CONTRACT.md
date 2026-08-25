# Candidate Q2 prompt provenance contract

Status: `NOT_FROZEN_REQUIRES_PRINCIPAL_TEMPLATE_DECISION`

This document specifies a migration-safe provenance schema. It does **not**
select the legacy or Gate-7 prompt template and is not Amendment 1.

## Contract object

Each prompt-bearing purpose must persist:

1. schema version;
2. task namespace;
3. purpose/role;
4. item ID;
5. dataset repository and immutable revision;
6. raw reference-answer SHA-256;
7. prompt-template version;
8. exact UTF-8 user-message bytes, encoded losslessly as base64;
9. exact UTF-8 system-message bytes when present;
10. explicit `NONE` Unicode normalization and exact-byte line-ending policy;
11. raw SHA-256 for each role's exact bytes;
12. tokenizer revision, chat mode, and `enable_thinking`;
13. exact rendered-chat-prompt SHA-256 before inference;
14. a SHA-256 over the canonical JSON envelope.

The canonical JSON serialization is UTF-8, `sort_keys=True`, compact
separators, with no implicit newline in the hashed payload. Exact prompt bytes
are carried in base64, so JSON escaping cannot change them.

## Legacy migration

Legacy values must be retained under explicit fields such as:

- `legacy_hash_schema`;
- `legacy_hash_value`;
- `legacy_template_version`;
- `legacy_source_artifact`.

They must never be copied into `user_prompt_bytes_sha256` or compared directly
with a raw hash. `stable-digest-v1:EXTERNAL-PROMPT` and
`sha256-utf8-raw-v1` are different types even if both are 64 hexadecimal
characters.

## Local and remote preflight

Before any Pod is provisioned, local CPU preflight must reconstruct every
source, shell, M1, M2, primary, and technical-probe role/content envelope and
reproduce its contract. After the exact tokenizer is available but before model
inference, remote preflight must additionally reproduce the rendered-chat hash
for every unique role/content envelope.

Any missing purpose, template-version mismatch, raw-byte mismatch, ambiguous
legacy field, or rendered-chat mismatch is terminal. It must not be repaired
after scientific outcomes.

## Decision still required

The principal researcher must prospectively choose exactly one global Q2 V3
CRUXEval user-message template (or explicitly authorize another globally
defined representation). Only then may this candidate be instantiated into a
Freeze Amendment 1 and all six manifests regenerated uniformly.
