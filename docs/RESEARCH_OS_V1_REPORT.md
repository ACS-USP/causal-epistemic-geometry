# Research OS v1 report

## Scope and outcome

Research OS v1 adds a small, permanent governance and reliability layer for
future gates. It is additive: no historical experiment state, parser, result,
threshold, allocation, holdout record, or scientific classification was
changed. No model inference, dataset allocation, RunPod access, or external
service call was performed.

Base commit: `f4e5ffe0911cf003c48e08012d81f2df0ccaab3e`.

## Architecture

`research_policy.yaml` is the machine-readable authority for autonomy classes,
states, transitions, incident reasons, and firewalls. The human policy explains
the same boundaries. `epistemic_geometry.research.governance` provides typed,
pure validators; it does not persist or mutate an experiment ledger.

The premortem and closeout templates each have Markdown and JSON forms plus a
Draft 2020-12 JSON Schema. `scripts/validate_research_os.py` checks that policy,
Python enums, transition tables, incident timing, template fields, and schemas
remain synchronized. It is included in `make scientific-audit`.

`epistemic_geometry.research.reliability` provides reusable formatting-integrity
inspection, logical-row validation, a provenance-bound crash-safe JSONL
journal, deterministic pending-condition resume, scoped cleanup, and random
vector norm validation. Existing historical parsers and journals are not
modified.

`remote_environment.yaml` records a repository-supported core CUDA profile and
the separate pinned optional xRFM profile. `remote-preflight` reads only local
runtime/package/GPU/disk/cache/Git facts and optional expected model-revision
metadata. It does not load weights, infer, download, contact RunPod, or access a
dataset.

## Files added

- `research_policy.yaml`
- `remote_environment.yaml`
- `docs/RESEARCH_AUTONOMY_POLICY.md`
- `docs/RESEARCH_AGENT_WORKFLOW.md`
- `docs/RESEARCH_OS_V1_REPORT.md`
- `templates/research/PREMORTEM.md`
- `templates/research/PREMORTEM.json`
- `templates/research/CLOSEOUT_AUDIT.md`
- `templates/research/CLOSEOUT_AUDIT.json`
- `schemas/research/premortem.schema.json`
- `schemas/research/closeout_audit.schema.json`
- `src/epistemic_geometry/research/__init__.py`
- `src/epistemic_geometry/research/governance.py`
- `src/epistemic_geometry/research/reliability.py`
- `src/epistemic_geometry/research/preflight.py`
- `scripts/remote_preflight.py`
- `scripts/validate_research_os.py`
- `tests/test_research_governance.py`
- `tests/test_research_fault_injection.py`
- `tests/test_remote_preflight.py`

## Files modified

- `Makefile`: adds `remote-preflight` and `research-os-check`; preserves and
  extends the existing scientific audit.
- `pyproject.toml`: adds the `packaging` runtime dependency and
  `remote-preflight` console entry point.
- `docs/DOCUMENT_INDEX.md`: classifies all new Research OS documentation.

No experiment spec, registry record, `project_state.yaml`, historical parser,
runner, result, or review artifact was modified.

## Tests and validation

Focused new suite:

- 21 passed.
- Governance coverage: typed state/reason synchronization, transition guards,
  pre/post-outcome incident routing, and Class A–D invariants.
- Fault injection: `FINAL` inside a Markdown fence, fence closure after
  `FINAL`, multiple/no `FINAL`, truncation, one mechanically ineligible item,
  frozen reserve continuation, crash after the first source condition, partial
  journal line before fsync, duplicate and missing logical rows, duplicate-free
  resume, hook leakage, wrong random-vector norm, absent execution marker, and
  treatment-induced formatting differences.
- Remote preflight: fully mocked compatible stack, missing dependency, absent
  cache, dirty worktree, and wrong model revision. Tests use no network and do
  not contact RunPod.

Repository validation:

- `ruff check --no-cache .`: passed.
- `python -m compileall -q src` with an isolated bytecode cache: passed.
- State check: passed.
- Experiment registry check: passed (17 entries).
- Documentation check: passed (57 Markdown files).
- Scientific metrics check: passed.
- Research OS check: passed.
- Full pytest attempt: 219 passed, 6 skipped for unavailable Torch, and 1
  failed because the Git-ignored historical fixture
  `review/full_nonthinking_smoke/journal.jsonl` is absent from this isolated
  worktree.
- Isolated-worktree suite excluding only that artifact-dependent historical
  test file: 219 passed, 6 skipped.

The missing historical fixture was not copied from or inspected in the active
scientific worktree. The test was not changed or weakened.

## Known limitations

- V1 validates future lifecycle records but does not migrate the historical
  ledger or wire every existing runner into a global state store.
- The checked-in schemas are complete JSON Schema contracts; the dependency-free
  repository validator checks synchronization and required fields, not every
  Draft 2020-12 keyword. Standard JSON Schema tooling can perform full instance
  validation.
- Repository evidence supports exact Python/Torch/CUDA/Transformers versions
  for the validated core remote stack, an `accelerate` lower bound, and presence
  (but not an exact version) for `huggingface_hub`. The preflight reports the
  installed version rather than inventing an unsupported pin.
- GPU model is recorded and CUDA is required, but v1 does not impose a global
  GPU allowlist.
- The optional xRFM profile records exact upstream commits and the frozen
  upstream package stack, but v1 does not clone, install, or contact either
  upstream repository.
- Preflight emits JSON on request but does not choose an artifact location or
  mutate a gate record automatically.

## Merge-risk assessment

Risk is **low to moderate**. Changes are additive and isolated under the new
research package, templates, schemas, policy, tests, and command. The two
integration points are intentional: `packaging>=23` becomes a small runtime
dependency, and `make scientific-audit` now fails if Research OS policy and
typed code diverge. There are no scientific-state migrations and no existing
runner behavior changes.

The principal merge review should focus on whether the Class B/Class C timing
boundary, blocked-state return paths, exact core environment profile, and
scientific-audit integration reflect the desired governance authority.

## Research OS v2 recommendations

1. Add an append-only, signed global gate/incident ledger with explicit
   historical import rather than silent migration.
2. Integrate selected runners with typed transition attestations and durable
   preflight/premortem/closeout artifact hashes.
3. Add a provider-neutral remote control plane only after authorization,
   credential, cost, and destructive-action boundaries are reviewed.
4. Add full JSON Schema validation to the installed development toolchain.
5. Add portable journal corruption/property tests and a generic metric audit
   interface without reimplementing historical estimators.
6. Add explicit environment lock artifacts per frozen gate, including upstream
   xRFM checkout verification where that gate requires it.
