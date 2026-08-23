# GATE 13 — CROSS-MODEL MINISTRAL-3 REPLICATION

## Historical status

The frozen classification is `GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. The
independent forensic classification is `GATE13_FORENSIC_CLEAN`; the maximum
primary/audit metric difference is `1.9184653865522705e-13`.

## Model and engineering

- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Revision/tokenizer revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Engine: serial Transformers, BF16, sustained current-token intervention
- Evaluator: `external-semantic-v3`
- Engineering gate: `GATE13_ENGINEERING_PASS`
- Vision tower invocations: `0`

The primary 8B substrate passed. The frozen 14B competence-floor fallback was
not authorized or run.

## Substrate screen

| Condition | Accuracy | Commitment validity | Semantic evaluability | Mean tokens |
|---|---:|---:|---:|---:|
| Baseline | 0.4500 | 1.0000 | 1.0000 | 87.15 |
| Source direct | 0.3500 | 1.0000 | 1.0000 | 18.55 |
| Source careful | 0.7333 | 0.9833 | 0.9833 | 406.37 |
| Careful concise | 0.5833 | 1.0000 | 1.0000 | 226.85 |
| Verbose direct | 0.5167 | 1.0000 | 1.0000 | 139.77 |

Careful/direct cross-rollout semantic disagreement was `0.675`. The source
policy therefore transferred behaviorally to Ministral-3 8B.

## Source atlas

All `34/34` language layers passed the frozen held-out source eligibility rule.
The label-free depth-quartile shortlist was `L8`, `L12`, `L22`, and `L26`.
Source activations were extracted without generated answers, and directions
were paired careful-minus-direct means.

## Causal first stage

The frozen dose was D50. No shortlisted layer passed every safety and
random-specificity gate:

| Layer | Q | Null mean | Null max | Accuracy | Validity/evaluability | Result |
|---|---:|---:|---:|---:|---:|---|
| L8 | 0.0833 | 0.0625 | 0.0833 | 0.6250 | 1.0000 | FAIL |
| L12 | 0.0417 | 0.1042 | 0.1250 | 0.6667 | 1.0000 | FAIL |
| L22 | 0.5000 | 0.4375 | 0.5000 | 0.5417 | 0.7917 | FAIL |
| L26 | 0.4167 | 0.3958 | 0.4583 | 0.6667 | 0.9167 | FAIL |

The matched baseline accuracy was `0.6667`. L22 failed mechanical safety,
competence safety, and strict null-maximum specificity. L26 remained safe but
did not exceed the null-mean margin or null maximum. L8 and L12 failed the
semantic-change and specificity requirements.

## Frozen stage stop

No layer was selected. Consequently:

- final random bank: `NOT CONSTRUCTED`;
- dose calibration: `NOT RUN`;
- final 100-item evaluation: `NOT RUN`;
- final G/C/D, rescue, damage, and bootstrap: `NOT ESTIMATED`.

The 40-item dose-calibration allocation and 100-item final-evaluation allocation
were untouched by activation steering. The 57 historically untouched CRUXEval
IDs remain untouched.

## Interpretation

Gate 13 shows that the careful/direct source is strongly readable throughout
Ministral-3 8B, but the prospectively frozen source-decodability shortlist and
D50 intervention did not produce a safe, specific causal first stage. This is
a bounded cross-model DEVELOPMENT negative for that procedure, not evidence
that no causal layer or dose exists anywhere in Ministral. It does not modify
the Qwen results and does not test Q2 or Q3.

## Cost and infrastructure

The journal contains `612` complete scientific trajectories. Incremental cost
is estimated at `US$0.90`, including startup/cache preparation and the retained
volume through closeout. GPU compute is stopped. The volume is temporarily
retained solely under the principal-authorized Gate-13-to-13.1 infrastructure
handoff.

## Next action

`PRINCIPAL_RESEARCHER_REVIEW`. The separately authorized Gate 13.1 may begin
only after this clean closeout is accepted into `main` and prospectively locked.
