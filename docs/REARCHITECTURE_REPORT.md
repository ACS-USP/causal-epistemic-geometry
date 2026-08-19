# Scientific rearchitecture report

Status: **COMPLETE — OFFLINE SCIENTIFIC/STATISTICAL RESET**

This branch started from `690cc75`. It performed no model inference, GPU work,
new benchmark outcome collection, activation extraction, steering, holdout
access, generated-code execution, Docker/Colima debugging, or work in another
repository.

## 1. Scientific question now

The program asks whether internal representation and intervention geometry can
predict and causally control which blind spots competence-preserving variants
exhibit. The nearest executable question is whether one intervention changes
where a frozen stochastic policy fails beyond ordinary resampling without just
worsening mean competence.

## 2. Q1, Q2, and collective utility

Q1 is causal, competence-aware error-profile movement. Q2 asks whether a
pre-specified intervention-space metric predicts held-out error-profile
geometry across multiple interventions. Collective utility/Q3 asks whether an
outcome-independent committee can realize value. Behavioral difference,
semantic error difference, oracle complementarity, and realizable gain are not
interchangeable.

## 3. Closed historical instruments

- V1–V1.2 MMLU-Pro closed because answer-slot symmetrization remained
  estimator-sensitive.
- V2 E3-10 closed because the non-thinking first-response readout was
  chance-like and output-channel unstable.
- V3 Stage A closed because zero families passed; apparent difficulty was
  mostly completion/truncation or near-ceiling conditional accuracy.
- External CRUX diagnostics closed because generous-cap outcomes exposed
  saturation/format sensitivity.
- V4 character count closed because semantic performance saturated and parser
  artifacts remained.
- V4 dense code is paused before inference because secure objective execution
  is not production-ready.

None is a failed test of the full theory.

## 4. Scientific versus operational learning

Scientifically, the work established strict requirements for genuine semantic
errors, baseline-resampling nulls, and competence adjustment. Operationally, it
validated hooks, deterministic provenance, crash-safe journals, exact engine
gates, and revealed failure modes from answer slots, parser semantics,
truncation, batch-shaped stochastic trajectories, and unsafe evaluators.

## 5. Documentation drift found

The README and `CURRENT_STATUS.md` still presented V3 and the external
completion diagnostic as current after V4 had closed/paused later instruments.
Old handoffs mixed pre-run instructions with live status. The reset replaces
that ambiguity with `project_state.yaml`, `experiments/registry.yaml`, generated
status, a precedence rule, and explicit canonical/archival indexing.

## 6. Statistical bugs and ambiguities

The V4 Spearman implementation assigned sequential ranks to ties. The paired
bootstrap named a subtraction of conditional rescue/damage rates as if it were
a net effect. `excess_pair_oracle` mixed competence and dependence without
showing the decomposition. Two-rollout propensity correlations lacked a strong
low-resolution warning, and matched seeds were too easily described as the
primary causal estimand.

## 7. Spearman correction

All tied observations now receive their average rank. Tests cover untied,
multi-tie, all-equal, known-reference, order-invariance, and weekday fixtures.
The corrected weekday test enumerates all 7! concept-label permutations;
letters use a frozen 10,000-draw label permutation with plus-one correction.
Original artifacts remain untouched.

## 8. Corrected weekday geometry

Yes, descriptively. WEEKDAYS changed from `rho=0.675325, p=0.002700` to
`rho=0.703094, exact p=0.005556`. LETTERS changed from `rho=0.357002` to
`rho=0.353588` with corrected Monte Carlo `p=0.000100`. These remain tiny
single-layer descriptive associations, not behavioral, causal, Q1, or Q2
evidence.

## 9. Why conditional rescue minus damage is not net

`rescue_rate` divides by baseline errors; `damage_rate` divides by baseline
successes. Their subtraction combines different denominators. The population
net is `(rescues-damages)/N`, exactly equal to delta accuracy. New reports expose
both conditional rates and population fractions; the old interval key is
deprecated and labeled.

## 10. Oracle decomposition

For per-item error propensities, `G_j = O_0j-O_00` decomposes as
`mu_0(mu_0-mu_j) + Var(p_0)-Cov(p_0,p_j)`. The first term is mean competence
change. The competence-adjusted `C_j` is the second term. Both must be shown
with raw competence.

## 11. New estimators and validation

Pure functions now compute `G_j`, `C_j`, decomposition residuals, and the
unbiased two-rollout squared propensity distance. Cluster bootstrap reduces
nested observations to equal-weight scientific clusters before resampling.
Deterministic Bernoulli simulations validate decomposition, two-rollout bias,
matched-seed rejection, and clustered weighting.

## 12. Independent versus matched seeds

Independent seed banks estimate operational repeated-agent complementarity and
support the propensity product estimands. Same-seed common-random-number
coupling is a useful secondary sensitivity view, but the realized pair depends
on that coupling. Intervention assignment is controlled in both cases; the
estimand is different.

## 13. Role of dense test vectors

Dense test vectors can reveal structured program failures, but tests are nested
under problem/program/rollout/intervention. Hundreds of tests do not create
hundreds of independent problems. Future analysis must preserve nested IDs,
cluster by problem, audit redundancy/effective rank, and report per-test and
problem-level outcomes separately.

## 14. Why sandbox engineering is paused

The objective evaluator is scientifically promising, but the current image has
an unresolved base digest, the wrapper needs GNU timeout on macOS, and no
credential-free disposable Linux executor has passed security validation.
Spending more local infrastructure effort before portfolio prioritization would
optimize an unqualified substrate. The template now fails closed.

## 15. Why a known-positive control is mandatory

A null original-Q1 result is uninterpretable if the stack cannot reproduce any
published adjacent activation effect. The prospective Wurgaft weekday pipeline
was selected because it has public end-to-end code, an exact concept/output
space, Llama-3.1-8B hardware fit, and direct hook relevance. Failure blocks a
negative interpretation; success validates machinery, not Q1.

## 16. Missing full non-thinking generation cell

V2 tested a first-response-state candidate readout with thinking disabled; it
did not test ordinary full non-thinking generation. V3 tested native thinking.
The cheap missing cell may change completion/error structure without conflating
the result with a first-token readout, so it is the first prospective smoke.

## 17. Model × policy × benchmark race

Rather than repeatedly rescue one substrate, compare Qwen native thinking,
Qwen full non-thinking, and the positive-control model/policy over fresh exact
tasks. Start with five items per arm, add fifteen only for survivors, and use
two seeds only for the best one or two. Selection values evaluator clarity,
completion, genuine errors, natural stochasticity, and cost.

## 18. First actual micro-Q1

On one qualified instrument, run baseline seed banks A/B, `+v`, `-v`, a
norm-matched random direction, and alpha-zero identity, plus a manipulation
check and known-positive implementation. Use only 20–50 development problems.
Freeze layer, alpha, direction, competence tolerance, and futility before
outcomes. Compare intervention movement with repeated-baseline and random
controls using hard tables, `C_j`, and the two-rollout distance.

## 19. Hard GO / PAUSE boundary

GO toward Q2 only with a qualified genuine-error instrument, a passed published
positive control, and an identifiable micro-Q1 effect beyond resampling at
acceptable competence. Otherwise pause B′ pending better model access,
instrumentation, or evidence. Do not add another ad hoc instrument in the same
sprint.

## 20. Recycled Evidence in the portfolio

Recycled Evidence is a separate, lower-compute program with exact Bayesian
oracles, provenance DAGs, paired counterfactuals, and retrieval relevance. It
shares the broad concern of correlated epistemic failures but must remain in a
separate repository. It reduces portfolio risk while B′ faces high
instrumentation and open-weight costs.

## 21. Cheaper, less ceremonial workflow

Exploration now defaults to 5–20 item cheap failures with minimal exact
artifacts. Full schema/provenance/statistical simulation is deferred until a
signal earns `DEVELOPMENT_LOCK`. Confirmatory ceremony appears only after a
sealed design. Script classification and generated state reduce repeated
handoff archaeology.

## 22. Exact next action after principal review

Review this offline reset and the prospective specs. If accepted, separately
authorize the five-item-per-arm **full non-thinking generation smoke**. Do not
start RunPod, the positive-control replication, the substrate race, micro-Q1,
Q2, dense-code execution, or holdout access from this branch alone.

## Validation record

Completion requires green pytest, Ruff, compileall, project-state rendering,
document links, registry checks, synthetic metric validation, exact weekday
reanalysis, clean Git status, and branch push. Final command results and commit
hash are reported in the principal handoff rather than hard-coded here.

