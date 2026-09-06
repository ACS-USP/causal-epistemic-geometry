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
| LiveCodeBench design | Is test-output prediction a deterministic, non-LLM-judged second task? | DEVELOPMENT | Model-free instrument and family-level sampling design prepared | [Design review](../review/q1_second_task_spark2_design/Q1_SECOND_TASK_SPARK2_DESIGN_REVIEW.md) |
| Stage A1 | Does the original parser/prompt instrument qualify on 32 families? | CLOSED_NOT_PRIMARY | `Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED`; baseline answer-channel guards failed | [A1 closeout](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_a_closeout/REPORT.md); historical result immutable |
| Stage A2 | Does a generic parser repair qualify prospectively on untouched families? | CLOSED_NOT_PRIMARY | `Q1_SECOND_TASK_STAGE_A2_QUALIFIED`; forensic clean; not controller-transfer evidence | [A2 closeout](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_a2_closeout/REPORT.md) |
| Stage B | Does the exact fixed Qwen L27-D75 controller transfer beyond eight frozen nulls? | NEGATIVE_BOUNDARY | `Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY`; point estimate led all nulls, but frozen inference, split-half, and safety conjunction failed | [Stage-B closeout](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_b_closeout/REPORT.md), [resolved audit](../review/q1_second_task_spark2_design/amendment1_hierarchical_unit/stage_b_forensic_resolution/REPORT.md) |

Stage A1 remains a historical instrument failure and is not reclassified by
the repaired-parser diagnosis or Stage A2. Stage A2 is instrument qualification,
not activation-transfer evidence. Stage B is the closed transfer test.

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
| Q2 OOS V1 | Can one fresh-controller candidate stream pass the frozen geometry gate? | HISTORICAL | `Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED`; presemantic stream gate failed, no redraw or outcome | [V1 report](../review/q2_oos_fresh_controller_design/REPORT.md) |
| Q2 OOS V2 initial inference | Is row-QAP calibrated under plausible fresh-controller heterogeneity? | HISTORICAL | `Q2_OOS_V2_NULL_CALIBRATION_BLOCKED`; strict exchangeable null calibrated, stress null anti-conservative | [Calibration closeout](../review/q2_oos_fresh_controller_design/v2_presemantic_qualification/Q2_OOS_V2_NULL_CALIBRATION_CLOSEOUT.md) |
| Q2 OOS robust redesign | Can controller-level inference support the fresh-identity claim? | CLOSED_NOT_PRIMARY | Exact controller sign test selected prospectively; global dyad rho retained as descriptive | [Inference review](../review/q2_oos_fresh_controller_design/heterogeneity_robust_inference/Q2_HETEROGENEITY_ROBUST_INFERENCE_REVIEW.md) |
| Q2 OOS V2 semantic execution | Does positive A0 alignment generalize to 16 prospectively sampled fresh controllers? | DEVELOPMENT | `Q2_OOS_V2_A0_PASS`; 16/16 positive, exact p=1.52587890625e-05, forensic clean | [OOS closeout](../review/q2_oos_fresh_controller_design/v2_semantic_execution/Q2_OOS_V2_SEMANTIC_CLOSEOUT.md) |
| OOS item-bootstrap diagnostic | Can the archived item-resampling percentiles be read as a conventional CI? | CLOSED_NOT_PRIMARY | `Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED`; implementation exact, conventional percentile-CI interpretation not validated | [Diagnostic report](../review/q2_oos_fresh_controller_design/v2_semantic_execution/item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_REPORT.md) |

## What G2 means

The V4.1 decision table was hierarchical. `G2` means the finite-response A2
geometry passed the frozen relational requirements, but did not satisfy the G3
requirement of incremental superiority over **both** A0 and A1. Here the A2
contrasts were negative, so G2 is not a near-G3 result.

## What the OOS pass means

The OOS test changed the independent unit from a fixed-bank dyad to one
prospectively sampled fresh controller. For each fresh controller, the frozen
statistic correlates A0 with blind-spot-shape distance over the 31 historical
references and averages MEDIUM/STRONG shells. Sixteen of 16 statistics were
positive. The result extends the historical relational finding across
controller identities inside the same learned subspace; it does not compare
that subspace with arbitrary matched rank-8 orientations.

## Q3: utility

| Stage | Class | Status | Evidence |
|---|---|---|---|
| Realizable collective utility | NOT_RUN | No selector, router, committee, or geometry-guided controller has been executed as Q3 | [Q3 concept note](Q3_CONCEPT_NOTE.md) |
| Q3.0 realizable-utility design | DESIGN_ONLY / DEVELOPMENT_PLANNING | Closed-data nested cross-fitting found oracle opportunity, but no mechanism met all feasibility gates and available fresh CRUXEval families were underpowered | [design review](Q3_REALIZABLE_UTILITY_DESIGN_REVIEW.md) |
| Q3.1 label-free prompt representations | DEVELOPMENT_ONLY / CLOSED | Prompt representations produced stable routing on 300 closed families, but true geometry added only +0.0033 over learned policy identity and failed the frozen incremental-attribution gate | [Q3.1 review](Q3_ROUTE_A_PROMPT_REPRESENTATION_REVIEW.md), [release summary](../review/q3_route_a_prompt_representation/Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json) |
| Q3.2 geometry-role decomposition | DEVELOPMENT_ONLY / CLOSED | A0-maximin K=8 passed the matched-random bank-construction gates; true coordinates did not pass historical-to-fresh controller routing-transfer gates | [Q3.2 review](Q3_GEOMETRY_ROLE_DECOMPOSITION_REVIEW.md), [release summary](../review/q3_geometry_role_decomposition/Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json) |
| Q3.3 final system and evaluation supply | DESIGN_ONLY / CLOSED | One candidate system was frozen from closed development evidence; Tier B was rejected; a fresh 1,600-family instrument was designed but not generated | [Q3.3 review](Q3_FINAL_SYSTEM_AND_EVALUATION_SUPPLY_REVIEW.md), [release summary](../review/q3_final_system_and_evaluation_supply/Q3_FINAL_SYSTEM_AND_SUPPLY_RELEASE_SUMMARY.json) |
| Q3.4 fresh-instrument qualification | CLOSED_NOT_PRIMARY / DEVELOPMENT | The 1,600-family instrument was generated; the 300-family qualification failed router answer-channel, champion-difficulty, and bank-opportunity gates. Confirmation and reserve remained unopened to Qwen | [Q3.4 closeout](../review/q3_fresh_instrument_qualification_closeout/Q3_FRESH_INSTRUMENT_QUALIFICATION_CLOSEOUT.md), [result](../review/q3_fresh_instrument_qualification_closeout/Q3_FRESH_QUALIFICATION_RESULT.json) |

Oracle pair headroom, rescue/damage decomposition, and complementarity are
inputs to Q3 design; none is itself a Q3 result.
