# Publication roadmap

This is a branching decision map, not a claim that the project has already
produced three papers. The current evidence is development-only and now
includes Gate 10's fixed-controller cross-domain test.

## Program A — Distributed steering of blind spots

Potential claim: coordinated temporal and/or multi-layer interventions can
causally reorganize stable semantic blind spots while preserving competence.

Evidence gate before this can become a paper:

- a behavioral first-stage effect;
- movement beyond repeated baseline resampling and a distribution of random
  controllers;
- competence preservation and mechanical-validity control;
- replication on a second benchmark or model;
- later, independently held-out geometry prediction.

## Program B — Readout versus causal control

Potential claim: representation directions can robustly read out or separate
cognitive/error states while failing to provide causal control handles.

Evidence gate:

- several independently validated, label-free directions or sources;
- strong held-out readout;
- the published positive steering control (already passed);
- several adequately powered null interventions;
- temporal, layer, and operator alternatives;
- complete reporting of nulls rather than selective steering results.

## Program C — Geometry of controllability

Potential claim: Euclidean representation geometry is not the geometry most
relevant for causal control, while Jacobian/Fisher/pullback or manifold-aware
geometry better predicts intervention effects.

Evidence gate:

- multiple controllers and independently measured behavioral sensitivities;
- a prespecified Euclidean-versus-control-geometry comparison;
- fresh held-out outcomes;
- dyadic/permutation-aware inference;
- no post-outcome selection of layers, directions, or metrics.

## Current branch point

Gate 7 found controller-specific semantic movement at full dose but failed its
relative validity guard. Gate 8 selected D75 prospectively using safety and
first-stage criteria only. Gate 9 evaluated that fixed choice on 100 fresh
CRUXEval items: accuracy rose from 0.47 to 0.60, commitment validity and
semantic evaluability were 0.97, and G/C/D were 0.1325/0.0643/0.1200. All three
exceeded four new architecture-matched random controls, with rescue 0.1525
versus damage 0.0225. Its frozen classification is
`GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION`.

Gate 10 then transported the exact vector, D75 dose, layer, timing, model, and
policy without adaptation to 200 fresh long character-count items. Baseline
opportunity passed and the controller remained safe, but G/C/D were
-0.01625/-0.01230/-0.025, below the random-controller mean; rescue was below
damage. The textual careful source also failed its frozen replication rule.
The frozen classification is `GATE10_NO_CROSS_DOMAIN_TRANSFER`, independently
audited clean.

Program A therefore retains strong within-CRUXEval DEVELOPMENT support but is
downgraded as a domain-general publication program: the present controller is
better described as domain-conditioned, plausibly tied to program tracing.
Programs B and C now receive higher priority for explaining why a readable and
causally effective controller has domain-dependent control gain. The next
protocol is a domain-conditioned control postmortem draft only. Q2 remains
closed and the confirmatory holdout remains untouched.
