# Document index

This index classifies every tracked Markdown document. “Archival” means
preserved evidence, not deletion or irrelevance; it means the file is not the
current instruction source.

## Canonical and normative

| Document | Role |
|---|---|
| `docs/CURRENT_STATUS.md` | Generated live status; edit `project_state.yaml` |
| `docs/SCIENTIFIC_CONSTITUTION.md` | Claim hierarchy and non-negotiable evidence rules |
| `docs/METRICS_AND_STATISTICS.md` | Current estimands, seed semantics, and uncertainty |
| `docs/EXPERIMENT_LADDER.md` | Prospective gated research sequence |
| `docs/ENGINEERING_POLICY.md` | Reproducibility, engine, security, and artifact policy |
| `docs/INSTRUMENT_HISTORY.md` | Current interpretation of V1–V4 evidence |
| `docs/POSITIVE_CONTROL_PROTOCOL.md` | Prospective published-method replication |
| `docs/PORTFOLIO_STRATEGY.md` | Hard continuation boundary and parallel-program logic |
| `docs/REARCHITECTURE_REPORT.md` | 2026-08 offline reset report |
| `docs/DOCUMENT_INDEX.md` | This classification and navigation page |
| `docs/SCRIPT_INDEX.md` | Current/historical/wrapper/mergeable script inventory |

## Current supporting references

| Document | Role |
|---|---|
| `docs/SCIENTIFIC_QUESTION.md` | Original paired Q1 motivation; constitution takes precedence |
| `docs/DEVELOPMENT_PROTOCOL.md` | General development freeze checklist |
| `docs/RANDOM_STEERING_CONTROL.md` | Random-direction control rationale |
| `docs/NEXT_Q2_GEOMETRY.md` | Small historical Q2 scaffold; ladder/spec now govern |
| `docs/ARCHITECTURE.md` | Package architecture |
| `docs/INFERENCE_ENGINE_ARCHITECTURE.md` | Reference/optimized engine semantics |
| `docs/INFERENCE_OPTIMIZATION.md` | V1-era optimization details, still useful engineering history |
| `docs/RUNPOD.md` | General remote setup; no execution authorization |
| `docs/RUNPOD_Q1_CHECKLIST.md` | Historical operational checklist; verify protocol before reuse |
| `docs/RUNPOD_COST_GATES.md` | Cost-control conventions |
| `docs/CODEX_REMOTE_SSH.md` | SSH helper reference |
| `docs/BEFORE_TERMINATING_POD.md` | Artifact-recovery checklist |
| `docs/Q1_V4_GEOMETRY_REANALYSIS.md` | Corrected offline tied-rank analysis |
| `docs/Q1_V4_DENSE_CODE_PILOT.md` | Current dense-code infrastructure pause |
| `docs/GATE3_SUBSTRATE_RACE_CLOSEOUT.md` | Completed baseline-only substrate selection |
| `docs/GATE4_MICRO_Q1_CLOSEOUT.md` | First original development micro-Q1 closeout |

## Archival scientific protocols and closeouts

| Document | Historical scope |
|---|---|
| `docs/Q1_DEVELOPMENT_PROTOCOL_V1.md` | Frozen V1 protocol |
| `docs/Q1_DEVELOPMENT_PROTOCOL_V1_1.md` | Frozen V1.1 protocol |
| `docs/Q1_DEVELOPMENT_PROTOCOL_V1_2.md` | Frozen V1.2 protocol |
| `docs/Q1_V1_RESULTS.md` | V1 development report |
| `docs/Q1_V1_SERIES_CLOSEOUT.md` | Multiple-choice series closeout |
| `docs/Q1_V1_2_AGGREGATOR_AUDIT.md` | Estimator-sensitivity audit |
| `docs/Q1_V2_EXACT_SEMANTIC_INSTRUMENT.md` | E3-10 design snapshot |
| `docs/Q1_V2_PRE_MODEL_STRUCTURAL_GATE.md` | E3-10 structural gate |
| `docs/Q1_V2_CALIBRATION_REVIEW.md` | E3-10 baseline calibration review |
| `docs/Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md` | Direct-readout closeout |
| `docs/Q1_V3_REASONING_AGENT_PROTOCOL.md` | Frozen V3 protocol |
| `docs/Q1_V3_PRE_CALIBRATION_HANDOFF.md` | Pre-run V3 snapshot |
| `docs/Q1_V3_REASONING_OPTIMIZATION_REPORT.md` | V3 engine gate, not current science |
| `docs/Q1_V4_MICROBENCH.md` | Frozen V4 authorization snapshot |
| `docs/EXTERNAL_BENCHMARK_QUALIFICATION.md` | Closed development search protocol |

## Archival engineering and handoff reports

| Document | Historical scope |
|---|---|
| `docs/PRE_RUNPOD_AUDIT.md` | Pre-RunPod engineering audit |
| `docs/PRE_RUNPOD_HANDOFF.md` | Earlier pre-RunPod handoff |
| `docs/OLD_RUNPOD_WORKFLOW_AUDIT.md` | Obsolete workflow audit |
| `docs/LOCAL_SSH_AUDIT.md` | Local SSH snapshot |
| `docs/MAXIMUM_INFERENCE_OPTIMIZATION_REPORT.md` | V1 optimization record |
| `docs/HANDOFF.md` | Historical broad handoff; use current status first |

## Source-of-truth precedence

When documents disagree, use this order:

1. `project_state.yaml` and `experiments/registry.yaml` for live state/history;
2. the scientific constitution and metrics specification;
3. a separately frozen prospective experiment spec;
4. historical protocol for reconstructing that historical run;
5. old handoff/optimization reports only for context.
