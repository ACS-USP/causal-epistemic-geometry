# Current Repository Status

> This is the canonical orientation page for the repository. It is deliberately
> separate from frozen protocols, historical closeouts, and long engineering
> reports. Update this page when the operational state changes.

**Last verified:** 2026-08-18
**Documentation branch:** `agent/q1-v3-reasoning-agent`  
**Scientific execution commit:** `4faea97`  
**GitHub repository:** `ACS-USP/causal-epistemic-geometry`

## One-sentence status

Q1 V3 Stage A completed as a baseline-only calibration, and the frozen screen
failed with zero surviving families. No steering direction was constructed,
Stage B was not run, and the confirmatory holdout remains untouched.

## State at a glance

| Area | Current state | Canonical reference |
| --- | --- | --- |
| Q1 V1–V1.2 | Closed as DEVELOPMENT; no claim frozen | [V1 closeout](Q1_V1_SERIES_CLOSEOUT.md) |
| Q1 V2 / E3-10 | Closed; instrument not qualified | [V2 closeout](Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md) |
| Q1 V3 protocol | Frozen | [V3 protocol](Q1_V3_REASONING_AGENT_PROTOCOL.md) |
| Q1 V3 structural gate | PASS, model-free | Local-only design bundle: `review/q1_v3_reasoning_instrument/` |
| Q1 V3 Stage A | **COMPLETE — SCREEN FAILED** | Local artifact: `review/q1_v3_stage_a/q1_v3_stage_a_final_4faea97/` |
| Q1 V3 Stage B | NOT RUN | Requires Stage-A review |
| Scientific steering | NOT RUN | Not authorized before Stage-B qualification and review |
| Geometry calibration | NOT RUN | Q2 remains closed |
| Confirmatory holdout | UNTOUCHED | Firewall remains active |
| Canonical real-Qwen engine | `max_budget_prefix_reuse` | [Optimization report](Q1_V3_REASONING_OPTIMIZATION_REPORT.md) |
| Scientific execution source | Commit `4faea97` | Frozen source for the completed run |
| External benchmark qualification | Q0 passed; old 2048 smokes are `LOW_CAP_DIAGNOSTIC` only | [Qualification protocol](EXTERNAL_BENCHMARK_QUALIFICATION.md) |

The completed artifact contains exactly 1,440 physical generations and 4,320
derived scientific rows. Remote and local validators both returned `valid:
true`. The local descriptive analysis produced zero surviving families, so the
pre-registered stop rule applies.

## What a new reader should read

1. [README](../README.md) — project purpose, quick start, and scientific
   boundaries.
2. This page — current operational state.
3. [Scientific question](SCIENTIFIC_QUESTION.md) — the original Q1 question
   and why competence must remain visible beside complementarity.
4. [Q1 V3 protocol](Q1_V3_REASONING_AGENT_PROTOCOL.md) — the frozen reasoning
   instrument and its stop rules.
5. [Inference-engine architecture](INFERENCE_ENGINE_ARCHITECTURE.md) — the
   permanent serial oracle and the approved real-Qwen execution engine.
6. [RunPod checklist](RUNPOD_Q1_CHECKLIST.md) — operational deployment steps.
7. [Handoff](HANDOFF.md) — repository capabilities and reproducible commands.

## What is current versus historical

### Current and normative

- `docs/Q1_V3_REASONING_AGENT_PROTOCOL.md` defines the frozen scientific
  protocol.
- `docs/INFERENCE_ENGINE_ARCHITECTURE.md` defines the execution-engine
  semantics and correctness oracle.
- `configs/q1_v3_reasoning_instrument.example.yaml` and the frozen Stage-A
  manifest define the authorized baseline-only workload.
- This page reports the live operational state.

### Historical but preserved

- `docs/Q1_V3_PRE_CALIBRATION_HANDOFF.md` describes the state and procedure
  before Stage A was authorized. It is retained as a pre-run snapshot.
- `docs/Q1_V3_REASONING_OPTIMIZATION_REPORT.md` records the final engineering
  gate before the clean Stage-A launch; it contains no Stage-A outcomes.
- V1/V1.2, V2/E3-10, and earlier RunPod documents remain available for audit
  and must not be read as the current scientific state.

Historical documents are not deleted. Their labels and links are kept explicit
so a first-time reader can distinguish a frozen protocol from a past handoff,
an engineering report, and a live result.

## Current execution boundary

The completed run was strictly baseline-only. It must not be interpreted as a
steering result. The physical journal key was `latent_id × rollout_index`; a
valid completed trajectory was never regenerated merely because its answer was
wrong, truncated, or difficult to parse.

The authorized post-run sequence was:

1. validate the remote artifact;
2. pull the raw trajectories and derived rows to the local Mac;
3. independently validate and analyze Stage A locally;
4. stop the Pod after artifacts are safe locally;
5. apply the frozen stop rule: because zero families survived, do not produce
   or execute Stage B and return control to the principal researcher.

The separate external-benchmark search is also DEVELOPMENT-only. Its first
2048-token thinking smokes are preserved but cannot qualify or disqualify a
benchmark. The next remote action is the bounded completion-cap diagnostic;
there is no active Pod inference while the protocol and code are being updated.

## Scientific boundary

The repository currently has **no frozen Q1 scientific result**. The project
does not claim that activation steering creates useful diversity, that any
direction is privileged, or that representation geometry predicts error
covariance. Q2, steering, geometry calibration, and the confirmatory holdout
remain closed.
