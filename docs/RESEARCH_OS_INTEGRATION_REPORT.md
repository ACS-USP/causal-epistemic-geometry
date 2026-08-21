# Research OS integration report

## Outcome

Research OS v1 commit `1b06fc6daac25764df25f9016140d69721231b40`
was integrated into scientific base `32a3cd2f0ec303fcc7951fbaf694db46265cc321`
on the isolated branch `agent/research-os-gate6-3-audit`. The cherry-pick was
clean; `docs/DOCUMENT_INDEX.md` auto-merged without a content conflict. Neither
source worktree was modified.

No historical Gate 6.3 artifact, result, parser, controller, eta, layer, item,
or journal was changed. No RunPod or model was accessed.

## Policy review and amendments

The Class A–D boundary is approved with one required tightening:
`BLOCKED_SCIENTIFIC_REVIEW` may transition only to `PREMORTEM` or
`PROSPECTIVE_LOCK`. Direct collection, offline-analysis, forensic-audit, and
closeout shortcuts are rejected by both typed code and machine policy.

Class C remains additive, offline, condition-symmetric work after outcomes.
It cannot authorize new model outputs unless an already-frozen decision tree
does so. Class D remains owned by the principal researcher.

## Isolated fixture repair

The default Gate-1 reanalysis test no longer depends on the ignored local file
`review/full_nonthinking_smoke/journal.jsonl`. A deterministic 40-row fixture at
`tests/fixtures/gate1_reanalysis_synthetic.jsonl` preserves the required schema,
15/5 character-count result, 8/12 CRUXEval result, and parser-repair cases. The
historical artifact test remains as an optional integration test and skips when
that local artifact is absent. Clean clones and isolated worktrees therefore do
not need copied scientific data.

## Named remote environment profiles

`CORE_QWEN` freezes the environment actually recorded for Gate 6.3:

- Python 3.11.x;
- torch 2.4.1+cu124;
- CUDA 12.4;
- transformers 4.57.1;
- accelerate 1.14.0;
- huggingface_hub 0.36.0;
- SDPA.

`RFM_COMPAT` remains separate and preserves the pinned Gate-6 xRFM-compatible
Python 3.10.15 / torch 2.4.0+cu118 / CUDA 11.8 / transformers 4.47.0 stack and
the frozen upstream commit identities. Normal Qwen runners do not require xRFM.
`remote-preflight --profile <name>` reports exact observed/expected package
mismatches without loading a model or using the network.

## Audit integration

`make scientific-audit` includes `research-os-check` and retains all prior
state, documentation, registry, and metrics checks. The semantic-validity audit
also adds a fail-closed artifact validator, but does not alter historical audit
commands or files.

## Validation

The isolated worktree passed the complete required suite on 2026-08-21:

- pytest: 266 passed, 7 skipped (Torch-dependent tests and one optional local
  historical-artifact integration test);
- Ruff: all checks passed;
- compileall: passed for `src` and `scripts`;
- mock smoke and doctor: passed without requiring Torch or model access;
- Research OS validation: policy, lifecycle, contracts, and named environment
  profiles valid;
- scientific audit: state, documentation, registry, metrics, and Research OS
  checks passed;
- Gate 6.3 V3 artifact validator: 920 rows, 54 bootstrap intervals, and frozen
  diagnostic classification verified.

No ignored historical fixture was required by the default test suite.

## Remaining Research OS v2 work

- append-only signed global incident/gate ledger;
- full JSON Schema instance validation in the development toolchain;
- runner-level transition attestations and durable environment reports;
- provider-neutral remote lifecycle integration under explicit cost/security
  policy;
- portable property tests for journal corruption and resume;
- gate-specific environment locks for optional upstream stacks.
