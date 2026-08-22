# Gate 11 adversarial premortem

Classification: `PREMORTEM_PASS`.

Gate 11 is restricted to prompt-only activations and sequential teacher forcing
of preserved Gate-9/Gate-10 baseline token sequences. It generates no free
continuations and never scores a new semantic answer.

- Existing data: both manifests and journals are locally available and match
  their historical SHA-256 records. The Gate-9 journal was restored from its
  already-verified local bundle without changing bytes.
- Selection: IDs are ranked only by the exact domain-tagged SHA-256 rule. No
  correctness, validity, length, rescue, or contribution field enters selection.
- Prompts: ordinary task contents and exact historical generic CAREFUL/DIRECT
  instructions are reused. Domain textual CAREFUL is preserved separately.
- Search firewall: newly described domain/layer directions are analysis-only;
  only the byte-identical historical L27 controller receives an intervention.
- Fixed sequences: rollout 0 is mandatory when token IDs exist, rollout 1 is
  the sole mechanical fallback, and absent sequences are retained as missing.
- Identification: every condition receives the same continuation for an item;
  sampling is disabled and intervention timing matches sustained current-token
  semantics with untouched KV history.
- Random null: one new prospective orthogonal four-vector bank is shared across
  both domains and matched in layer, energy, scope, duration, and engine.
- Length: the first 256 preserved tokens are used regardless of outcome or
  duration; later checkpoints are structurally missing when unavailable.
- Metrics: token/checkpoint values are aggregated within item before bootstrap.
  KL magnitude is interpreted as movement, never utility.
- Resume: logical source and propagation keys are append-only, unique, flushed,
  and fsynced. Deterministic completion never adds a replacement item.
- Q2: no pairwise error/intervention geometry, controller bank search, semantic
  outcomes, new benchmark, or holdout access is permitted.
- Cost: the frozen workload is projected below US$1.00 on A40; collection must
  stop before diagnostics if a pre-run projection exceeds US$2.50.

No unresolved scientific-design ambiguity remains.
