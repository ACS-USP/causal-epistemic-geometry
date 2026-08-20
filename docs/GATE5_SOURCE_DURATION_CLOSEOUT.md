# Gate 5 — Source Validity and Temporal Persistence Closeout

Gate 5 was a bounded DEVELOPMENT bridge for the Gate 4 null. It did not reopen
the model, benchmark, layer, alpha, controller construction, Q2 geometry, or
the confirmatory holdout.

## Frozen substrate

- Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- BF16, SDPA, full non-thinking generation, sampled with the Gate 4 policy.
- CRUXEval semantic evaluator, layer 17, Gate 4 direction and alpha
  `8.39900588973121`.
- Four orthogonal random controllers, including the exact Gate 4 random
  controller R0.

## Gate 4 forensic status

The independent audit classified Gate 4 as
`GATE4_AUDIT_MINOR_NONSCIENTIFIC_ISSUES`. It found 400 unique scientific
rows, no missing or duplicate logical rows, no split leakage, and matching
estimands. Three provenance corrections were documented without rewriting the
historical result: a protocol-lock SHA transcription, recorded-versus-
effective source commit, and stale backend stop metadata. Gate 4 remains the
historical `MICRO_Q1_NO_DETECTABLE_SIGNAL` result.

## Results

The source check used 40 fresh items, three textual conditions, and two
independent rollouts per condition. The careful/direct source passed:

- careful validity: 96.25%; accuracy: 76.25%;
- direct validity: 100%; accuracy: 60.00%;
- cross disagreement `X`: 0.5375;
- within disagreement `W`: 0.1500;
- excess source disagreement `S`: 0.3875;
- classification: `SOURCE_SEMANTIC_BEHAVIOR_PASS`.

The sustained-current-token engineering gate passed, including alpha-zero
identity, per-forward shift, cache safety, current-token scope, forward count,
and hook cleanup. The 20-item matched manipulation gate also passed:
`SUSTAINED_PLUS` changed semantic outcomes on 3/20 items, exceeding the
one-shot plus condition and the mean random-control change rate at the exact
frozen boundary.

The 60-item primary evaluation used 1,080 independent trajectories. Baseline
accuracy was 45.83% with 97.50% validity. The sustained negative sign showed a
duration contrast in propensity distance (`D_sustained - D_one_shot = 0.0500`),
but its sustained `D=0.0333` did not reach the frozen movement threshold of
0.05. Neither meaningful sign exceeded the random-control distribution and
neither reached the useful-complementarity criteria.

The frozen classification is:

`GATE5_DURATION_EFFECT_BELOW_MOVEMENT_THRESHOLD`

This is not evidence against the broader Q1 hypothesis. It says only that this
exact Gate 4 controller, at this site and alpha, did not pass the pre-registered
Gate 5 movement gate in the fresh development evaluation.

## Cost and firewall

The temporary A40 Pod `u78gx0pc1yvbqp` ran for 4,994 seconds. At US$0.44/A40
hour, the conservative Pod wall-clock estimate is US$0.6104. The journal
contains 1,500 rows and 46,533 generated tokens. The Pod was stopped before
local analysis.

- Gate 6: drafted, not run.
- Character-count replication: not run.
- Q2: not run.
- Confirmatory holdout: untouched.

Canonical artifacts are under
`review/gate5_source_duration/`, with `REPORT.md`, raw journal, manifests,
engineering checks, estimands, bootstrap intervals, and the draft
`GATE6_LAYER_SOURCE_ATLAS_PROTOCOL.md`.
