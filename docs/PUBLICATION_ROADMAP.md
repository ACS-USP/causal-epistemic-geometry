# Publication roadmap

This is a branching decision map, not a claim that the project has already
produced three papers. The current evidence is development-only and now
includes the Gate-11 domain-conditioned postmortem.

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
Gate 11 then found strong representation transfer: the frozen L27 axis had a
positive character-count careful-minus-direct gap on every selected item and
the descriptive character-count L27 direction had cosine 0.659 with the frozen
controller. Finite-displacement diagnostics did not support a downstream-gain
or policy-realization domain shift under the prospective rules. Historical
outcomes did support a policy-utility shift: careful-like computation helped
CRUXEval and harmed character count. Gate 11 did not measure an exact local
pullback/Fisher metric; KL/JS and hidden displacement are finite-shift control
diagnostics, while accuracy and G/C/D are task-utility measures.

The primary synthesis is `GATE11_POLICY_UTILITY_DOMAIN_MISMATCH`. Gate 11.1
repaired the forensic artifact boundary by persisting complete per-checkpoint
vocabulary logits and hidden-difference vectors and independently reproducing
the synthesis with exact metric agreement. This repairs the artifact concern;
it does not change the historical Gate-11 result. Program A remains strong
within CRUXEval but domain-conditioned. Program B gains evidence from the
separation between representation, finite-displacement control diagnostics,
and task utility. Gate 12 then attempted to qualify exact local directional
JVP/Fisher diagnostics prospectively. Exact autograd implementations agreed,
but the frozen BF16 finite-difference, local-KL, and full-sequence/KV engineering
gates failed before scientific geometry collection. Program C is therefore not
promoted: no predictive pullback result, positive or negative, was obtained.
At that historical point Q2 remained closed pending principal review; the later
Q2 V2 authorization and result are recorded below.

Gate 12.1 subsequently showed that the FP32 computational lift has coherent
full-sequence/KV semantics and mutually consistent exact JVP/VJP,
Fisher/Hessian, and utility-derivative identities on synthetic fixtures. The
complete engine nevertheless did not qualify because the historical BF16
bridge and the frozen three-consecutive-scale finite-difference rule failed.
Program C is not promoted: no scientific geometry or utility-prediction values
were collected, and historical outcomes remained sealed.

Gate 13 then tested the careful/direct paired-mean procedure in the Ministral-3
8B family. The substrate and behavioral source passed, and all 34 language
layers showed held-out source eligibility. However, none of the four
prospectively source-selected D50 interventions passed the complete safety and
random-specificity gate. The frozen result is
`GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`, independently audited clean. This does
not show that Ministral has no causally effective layer or dose; it is a clean
negative for the source-decodability shortlist plus fixed-D50 procedure.

Gate 13.1 preserved that historical bounded null and tested a broader,
prospectively frozen causal identification procedure. The all-layer sweep and
disjoint layer-dose qualification selected Ministral L27-D25 without using
accuracy for ranking. On 100 untouched final-evaluation items, accuracy rose
from 0.445 to 0.575; G/C/D were 0.1650/0.0940/0.1400 and exceeded all four new
random controls. The frozen classification is
`GATE13_1_STRONG_CROSS_MODEL_REPLICATION`. Validity and evaluability passed at
the exact five-percentage-point relative guard boundary, which remains an
important cost of the intervention.

Program A is therefore substantially strengthened within CRUXEval: useful
semantic blind-spot control now has strong DEVELOPMENT replications in Qwen and
Ministral under independently selected, architecture-specific controllers.
This does not overturn Gate 10's character-count null and does not establish a
domain-general careful-computation controller. The next drafted priority is a
second task where textual CAREFUL is prospectively useful, with fixed
controllers and no task-outcome controller search. Program B remains important
because source readability weakly ranked causal-Q across Ministral layers.
Program C remained paused at Gate 13.1; the later Q2 V2 DEVELOPMENT result is
recorded below. Q3 remains closed.

## Q1 confirmatory fixed-controller result

The prospectively assigned 57-item CRUXEval holdout has now been consumed by
the frozen two-model confirmatory test. Qwen passed every frozen safety,
positive-complementarity, and fresh-random-null criterion: C was 0.05435 with a
95% item-bootstrap interval of 0.01441 to 0.09680, and the meaningful-minus-null
mean C contrast also had a positive lower bound. Ministral likewise showed
positive, null-specific complementarity, but its commitment validity and
semantic evaluability were both 0.88596 and therefore failed the frozen safety
guards.

The exact terminal classification is
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. It is not a partial cross-model
pass. Program A gains confirmatory support for the fixed Qwen controller, but
the strict cross-model confirmatory claim is not established. The Ministral
result localizes the failure to output safety rather than absence of measured
complementarity; that distinction is descriptive and does not override the
failed confirmatory decision rule. Programs B and C remain mechanistic research
directions, but Q2 and Q3 are not authorized by this closeout.

## Q2 V2 controller-family-held-out DEVELOPMENT result

A later, separately authorized Q2 V2 built a prospectively qualified bank of 24
meaningful L27 controllers across six conceptual source families, with
per-direction label-free dose calibration and four fresh span-orthogonal nulls.
The full 120-item common panel completed with 6,960 trajectories and a clean
independent audit.

No prespecified geometry passed the complete composite prediction gate. M0 flat
and M1 covariance-whitened geometry had mean family-held-out correlations of
0.2013 and 0.1902 with RMSE ratios of 0.9867 and 0.9882. M2 finite behavioral
secant geometry was materially stronger: mean held-out rho 0.4279, all six
family folds positive, and one-sided QAP p=0.00220. Its RMSE ratio was 0.9067,
narrowly above the frozen maximum of 0.90. The exact classification is
`Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`.

For Program C, this is a composite DEVELOPMENT null with a meaningful
dissociation: static flat/whitened representation geometry did not qualify,
whereas finite behavioral secants showed reproducible rank structure but fell
short of the calibrated-prediction requirement. Program C is therefore not a
paper claim. Its next evidence gate is a fresh, prospectively frozen replication
of the M2 association and calibration performance across unseen controller
families or models, without post-outcome metric selection. Q3 remains not run.
