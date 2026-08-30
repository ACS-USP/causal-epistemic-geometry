# Experiment index

This index is a scientific map, not a replacement for
[`experiments/registry.yaml`](../experiments/registry.yaml). Each gate isolated a
methodological uncertainty. Instrument failures are not silently promoted to
scientific nulls, and DEVELOPMENT results are not upgraded to CONFIRMATORY.

## Evidence classes used here

- `CONFIRMATORY`: frozen held-out hypothesis test.
- `DEVELOPMENT`: instrument construction, prospective scientific test, or
  planning evidence that remains development-level.
- `NEGATIVE_BOUNDARY`: completed negative result that limits generalization.
- `ENGINEERING_ONLY`: numerical/execution qualification without target science.
- `HISTORICAL`: superseded or earlier design retained for provenance.
- `OPEN_RUNNING`: active blind collection with no partial scientific result.
- `CLOSED_NOT_PRIMARY`: completed supporting qualification or diagnostic.
- `NOT_RUN`: proposed question without a scientific execution.

## Q1 genealogy: controllability

| Stage | Question isolated | Class | Terminal meaning | Canonical evidence |
|---|---|---|---|---|
| Q1 V1–V1.2 | Can early multiple-choice instruments measure stable profile movement? | HISTORICAL | Closed development series; aggregation-sensitive and not the final Q1 instrument | [V1 series closeout](Q1_V1_SERIES_CLOSEOUT.md) |
| Q1 V2 / E3-10 | Does an exact direct-readout task support the intended intervention? | HISTORICAL | Direct-readout instrument did not qualify | [V2 closeout](Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md) |
| Q1 V3–V4 | Which baseline task/model substrate and parser are viable? | DEVELOPMENT | Established baseline opportunity and exposed saturation/parser limits | [Instrument history](INSTRUMENT_HISTORY.md) |
| Gate 3 | Which model/policy/task cell has usable baseline opportunity? | DEVELOPMENT | Selected Qwen non-thinking + CRUXEval for development; no steering result | [Gate 3 closeout](GATE3_SUBSTRATE_RACE_CLOSEOUT.md) |
| Gate 4 | Does one-shot careful-minus-direct steering exceed baseline and a matched null? | DEVELOPMENT | Audited bounded null; motivated temporal-persistence isolation | [Gate 4 closeout](GATE4_MICRO_Q1_CLOSEOUT.md) |
| Gate 5 | Was insufficient intervention duration the missing causal component? | DEVELOPMENT | Textual source and sustained manipulation qualified, but primary movement stayed below threshold | [Gate 5 closeout](GATE5_SOURCE_DURATION_CLOSEOUT.md) |
| Gates 6–7 | Can a fixed sustained L27 controller move profiles on fresh semantic items? | DEVELOPMENT | Specific movement appeared, but frozen validity guards failed | [Gate 7 closeout](GATE7_FRESH_L27_REPLICATION_CLOSEOUT.md) |
| Gate 8 | Can dose be selected using safety and a label-free/behavioral first stage? | DEVELOPMENT | Prospectively selected D75 as the lowest qualified safe dose | [Gate 8 protocol/registry](../experiments/registry.yaml) |
| Gate 9 | Does fixed D75 replicate on fresh CRUXEval items? | DEVELOPMENT | Strong safe selected-dose replication beyond four new random controls | [Gate 9 closeout](GATE9_SELECTED_D75_EVALUATION_CLOSEOUT.md) |
| Gate 10 | Does the exact Qwen controller transfer without adaptation to long character counting? | NEGATIVE_BOUNDARY | `GATE10_NO_CROSS_DOMAIN_TRANSFER` | [Gate 10 closeout](GATE10_CROSS_DOMAIN_CHARCOUNT_CLOSEOUT.md) |
| Gates 11–12.1 | Can representation transfer, task utility, and local derivative engines be separated? | ENGINEERING_ONLY / DEVELOPMENT | Representation and utility dissociated; local JVP/Fisher engine did not qualify for target science | [Gate 11 closeout](GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM_CLOSEOUT.md), [Gate 12.1 closeout](GATE12_1_CONTINUOUS_GEOMETRY_ENGINE_CLOSEOUT.md) |
| Gates 13–13.1 | Can an architecture-specific Ministral controller be developed? | DEVELOPMENT | Strong DEVELOPMENT replication with a narrow validity margin; not the confirmatory result | [Gate 13.1 closeout](GATE13_1_ALL_LAYER_CAUSAL_ATLAS_CLOSEOUT.md) |
| Q1 fixed-controller holdout | Do frozen Qwen and Ministral controllers pass the complete confirmatory rule? | CONFIRMATORY | `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`; Qwen passes, Ministral model-level decision fails safety guards | [Confirmatory closeout](Q1_CONFIRMATORY_FIXED_CONTROLLERS_CLOSEOUT.md) |
| Q1 visual evidence | Can the frozen Q1 evidence be communicated reproducibly? | CLOSED_NOT_PRIMARY | Deterministic publication figures and provenance; no new scientific test | [Visual evidence](Q1_VISUAL_EVIDENCE.md) |

## Q1 second-task transfer

| Stage | Question isolated | Class | Terminal/current meaning | Canonical evidence |
|---|---|---|---|---|
| LiveCodeBench design | Is test-output prediction a deterministic, non-LLM-judged second task? | DEVELOPMENT | Model-free instrument and family-level sampling design prepared | Open branch `research/q1-second-task-spark2-design` |
| Stage A1 | Does the original parser/prompt instrument qualify on 32 families? | CLOSED_NOT_PRIMARY | `Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED`; baseline answer-channel guards failed | Open branch closeout; historical result immutable |
| Stage A2 | Does a generic parser repair qualify prospectively on untouched families? | CLOSED_NOT_PRIMARY | `Q1_SECOND_TASK_STAGE_A2_QUALIFIED`; forensic clean; not controller-transfer evidence | Open branch closeout at authorization ancestry |
| Stage B | Does the exact fixed Qwen L27-D75 controller transfer beyond eight frozen nulls? | OPEN_RUNNING | Frozen 130-family, 5,720-trajectory blind collection; no partial result | Branch `research/q1-second-task-spark2-design`, authorization commit `91c3db4` |

The open branch is intentionally not merged as completed evidence. Its
scientific closeout can enter the canonical result history only after complete
collection, raw sealing, frozen analysis, and independent forensic audit.

## Q2 genealogy: geometry

| Stage | Question isolated | Class | Terminal meaning | Canonical evidence |
|---|---|---|---|---|
| Q2 V1 | Does the first controller bank qualify before opening its panel? | HISTORICAL | `Q2_CONTROLLER_BANK_NOT_QUALIFIED`; no common-panel outcome and no geometry null | [Bank qualification closeout](Q2_CONTROLLER_BANK_QUALIFICATION_CLOSEOUT.md) |
| Q2 V2 | Can geometry predict an unseen controller family in a calibrated multi-family bank? | DEVELOPMENT | Composite gate failed; finite-secant rank signal did not meet calibrated RMSE criterion | [Principal review](Q2_V2_PRINCIPAL_REVIEW_Q2_V3_DRAFT.md) |
| Q2 V3 | Can matched shells isolate angular prediction from intervention magnitude? | HISTORICAL | Provenance/source-family/power gates stopped execution before the semantic panel; not a geometry null | [V3 redesign](../review/q2_v3_four_family_statistical_redesign/REPORT.md) |
| Q2 V4 | Can a native rank-8 subspace yield 32 safe directions? | HISTORICAL | `Q2_V4_SAFE_BANK_INSUFFICIENT`: 31/40 safe, one below the frozen requirement; semantic hypothesis untested | [V4 report](../review/q2_v4_spark1_presemantic/REPORT.md) |
| Q2 V4.1 adequacy | Is the complete realized 31-safe bank sufficiently covered and powered? | DEVELOPMENT | `Q2_V4_1_31_SAFE_BANK_ADEQUATE`; all 31 retained, no selection | [Adequacy review](../review/q2_v4_1_31_safe_bank_review/Q2_V4_1_31_SAFE_BANK_REVIEW.md) |
| Q2 V4.1 label-free lock | Do A0/A1/A2 instruments and prediction schedules qualify before semantics? | CLOSED_NOT_PRIMARY | A1/A2 qualified; independent presemantic forensic clean | [Prediction-lock closeout](../review/q2_v4_1_prediction_lock/Q2_V4_1_PRESEMANTIC_CLOSEOUT.md) |
| Q2 V4.1 semantic execution | Does intervention geometry predict blind-spot-shape geometry? | DEVELOPMENT | `Q2_V4_1_G2`, `RS+`, `RT+`; A0/A1/A2 qualify, A2 does not outperform A0/A1 | [Semantic closeout](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.md) |

## What G2 means

The V4.1 decision table was hierarchical. `G2` means the finite-response A2
geometry passed the frozen relational requirements, but did not satisfy the G3
requirement of incremental superiority over **both** A0 and A1. Here the A2
contrasts were negative, so G2 is not a near-G3 result.

## Q3: utility

| Stage | Class | Status | Evidence |
|---|---|---|---|
| Realizable collective utility | NOT_RUN | No selector, router, committee, or geometry-guided controller has been executed as Q3 | [Q3 concept note](Q3_CONCEPT_NOTE.md) |

Oracle pair headroom, rescue/damage decomposition, and complementarity are
inputs to Q3 design; none is itself a Q3 result.

