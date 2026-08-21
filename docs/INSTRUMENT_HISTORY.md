# Instrument history

This page is the narrative companion to
[`experiments/registry.yaml`](../experiments/registry.yaml). Historical reports
remain immutable; this document states their current interpretation.

## V1–V1.2 — MMLU-Pro multiple choice

The first instrument used direct multiple-choice candidate scoring and later
introduced option-position permutations and two reasonable pre-registered
aggregation estimators. Activation steering demonstrably perturbed answer
distributions and item-level errors. However, the residual V1.2 conclusion was
estimator-sensitive: the two aggregators gave qualitatively different answers
and item-level flip overlap was low.

**Decision:** closed as DEVELOPMENT. No robust semantic-complementarity claim,
positive or negative, was frozen. See
[the closeout](Q1_V1_SERIES_CLOSEOUT.md).

## V2 — E3-10 exact-semantic direct readout

E3-10 removed arbitrary A/B/C/D answer slots. It used exact procedural tasks,
semantic candidates `0..9`, thinking disabled, and a first-response-state
candidate-logit readout. The channel was chance-like or unstable across output
surfaces and no scheduled cell passed the frozen qualification rule.

**Decision:** closed as a non-qualified direct-readout ablation. This does not
show that the same model cannot reason about those tasks or that activation
interventions cannot alter semantic errors. See
[the V2 closeout](Q1_V2_DIRECT_INSTRUMENT_CLOSEOUT.md).

## V3 — stochastic reasoning agent

V3 measured full sampled thinking trajectories on exact MODREG-R, FSM-R, and
SATCOUNT-R oracles. Stage A completed baseline-only. Solvable MODREG/FSM cells
were near ceiling conditional on valid `FINAL` completion; lower raw accuracy
was largely unfinished thinking or truncation. SATCOUNT was mostly too long or
incomplete. Zero families passed the frozen screen.

**Decision:** `REASONING_INSTRUMENT_SCREEN_FAILED`; Stage B and steering were
not run. The result rejects the instrument regime, not Q1. The preserved review
bundle is `review/q1_v3_stage_a/` with the immutable hash recorded in
`project_state.yaml`.

## External benchmark search

Objective adapters for external code/reasoning benchmarks passed varying
model-free gates. Early 2048-token records were correctly reclassified as
`LOW_CAP_DIAGNOSTIC`, because token truncation was not semantic difficulty.
Generous-cap CRUXEval diagnostics then exposed saturation and answer-format
sensitivity rather than a clean, intermediate genuine-error regime.

**Decision:** closed as a cheap DEVELOPMENT search. No benchmark qualified for
steering. See [the qualification record](EXTERNAL_BENCHMARK_QUALIFICATION.md).

## V4 Bench E — character counting

Procedural short, medium, and long character counting was designed as a cheap
exact instrument. Completed semantic answers saturated; residual failures
included parser-format artifacts. Extending strings or changing prompts after
seeing outcomes would manufacture difficulty rather than qualify a clean
instrument.

**Decision:** closed, no steering.

## V4 Bench G — tiny descriptive geometry

Forward-only block-31 activations were collected for known weekday-cycle and
letter-sequence concepts. The original analysis had a tied-rank bug in
Spearman correlation. The preserved data have been reanalyzed with average
ranks, exact 7! label permutation for weekdays, and a frozen plus-one Monte
Carlo label permutation for letters.

**Decision:** the corrected associations remain tiny descriptive diagnostics.
There is no behavioral replication, intervention, error covariance, or causal
geometry result. See [the corrected reanalysis](Q1_V4_GEOMETRY_REANALYSIS.md).

## V4 dense code

Nested executable tests could provide a dense objective error vector, but tests
are nested under programs and problems rather than 500 independent examples.
The current EvalPlus-based template is not a production security sandbox: its
base image digest remains unresolved, its host timeout is not stock-macOS
portable, and no disposable credential-free Linux evaluator has passed the
gate.

**Decision:** scientifically untested and operationally paused. Dense code is
not the automatic next action. See [the pilot gate](Q1_V4_DENSE_CODE_PILOT.md).

## Gate 6.3 — single-mean semantic evaluation

Gate 6.3 reanalyzed the immutable Gate 6.2 rows with the strict
`external-semantic-v2` parser, then tested the frozen paired-mean L27 controller
against four architecture-matched single-layer random controls. The matched
80-row manipulation gate passed, and the conditional 840-row evaluation was
completed with two independent rollouts per item-condition.

The meaningful controller exceeded the random mean and maximum on point
estimates of `G`, `C`, and `D`, while remaining within the accuracy tolerance.
It failed the pre-registered validity guard, however: validity was 0.9083,
below the required 0.9250 relative to the 0.9750 baseline.

**Decision:** `GATE6_3_SINGLE_MEAN_DESTRUCTIVE`. This is a bounded development
result, not a useful-complementarity claim, Q2 result, or test of the
confirmatory holdout. See [the Gate 6.3 closeout](GATE6_3_SINGLE_MEAN_CLOSEOUT.md)
and the raw [review artifacts](../review/gate6_3_single_mean_semantic_evaluation/).

### Additive semantic-validity audit

A later local-only, condition-blind `external-semantic-v3` audit separated the
existence of a final commitment from its deterministic evaluability and
correctness. It did not alter the historical files or classification. Under V3,
the controller reached 0.9750 commitment validity/evaluability and 0.7000
accuracy; its `G=0.2042`, `C=0.1177`, and `D=0.1667` exceeded the full
four-direction random bank. The offline diagnostic classification is
`GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL`.

This is DEVELOPMENT evidence warranting a prospectively locked fresh
replication, not a rescued historical result or a confirmatory claim. See the
[audit report](../review/gate6_3_semantic_validity_audit/AUDIT_REPORT.md).

## Cross-series conclusion

The repository has learned a great deal about measurement failure and execution
semantics, but it has not produced a scientific Q1 result. None of the closed
instruments settles whether representation geometry can causally control error
covariance. The next work is prospective and staged in
[the experiment ladder](EXPERIMENT_LADDER.md).
