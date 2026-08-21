# Research autonomy policy

This policy governs future research operations. It supplements the
[scientific constitution](SCIENTIFIC_CONSTITUTION.md) and
[engineering policy](ENGINEERING_POLICY.md); it does not rewrite any historical
experiment state, result, protocol, or classification. The machine-readable
source is [`research_policy.yaml`](../research_policy.yaml).

The operating principle is **high scientific conservatism plus high operational
autonomy**. Codex may recover engineering systems aggressively while treating
scientific choices conservatively.

## Class A — autonomous engineering recovery

Class A covers SSH/proxy failures, RunPod restart or migration, cache/path
problems, missing dependencies, journal recovery, Git transport, transient
infrastructure, deterministic code bugs, and crash-safe resume.

The allowed sequence is:

```text
diagnose -> repair -> test -> document -> continue
```

This authority ends where scientific semantics begin. Model, benchmark,
hypothesis, estimand, condition, seed regime, parser meaning, thresholds,
allocation, and outcome-selection rules must remain unchanged. Infrastructure
retries retain the same logical key and seed plus retry provenance.

## Class B — prospective instrument amendment

Class B is autonomous only before scientific outcomes of the affected phase
have been observed. It covers mechanical attrition rules, deterministic reserve
handling, a parser-contract implementation bug, or a clearly incorrect
code/specification mismatch.

Every Class B amendment must satisfy all of these conditions:

- no outcome-based selection;
- hypothesis and estimand unchanged;
- model and benchmark unchanged;
- scientific conditions and thresholds unchanged;
- the amendment is documented, tested, and locked before any new affected outcomes.

If any affected outcome is already visible, use Class C or Class D.

## Class C — offline post-outcome forensic repair

After outcomes exist, Codex may autonomously preserve the original result,
diagnose parser/measurement/estimator problems, perform a condition-symmetric
offline reanalysis, implement and test a corrected instrument, and prepare a
prospective amendment.

Class C does not authorize collecting additional model outputs unless that
collection was already authorized by a frozen decision tree. Corrected work
must be additive, provenance-linked, and explicit that the historical result
remains unchanged.

## Class D — principal researcher required

The following require explicit principal-researcher review:

- a new scientific hypothesis, direction, or condition;
- outcome-dependent item, layer, alpha, vector, or analysis choice;
- model or benchmark changes;
- item replacement based on outcomes;
- scientific-threshold modification;
- Q2, holdout, or claim changes;
- material budget expansion.

The typed `HOLDOUT_FIREWALL`, `SCIENTIFIC_GATE_FAIL`, and
`SCIENTIFIC_DESIGN_DECISION_REQUIRED` reasons route here. Class D is not a
license for an agent to approve its own proposal.

## Typed gate lifecycle

Future gates use the following states:

```text
PREPARE -> PREMORTEM -> PROSPECTIVE_LOCK -> ENGINEERING -> COLLECTION
    -> OFFLINE_ANALYSIS -> FORENSIC_AUDIT -> CLOSED
```

`BLOCKED_RECOVERABLE` records Class A interruptions. It may return to the exact
prior non-scientific state after repair. `BLOCKED_SCIENTIFIC_REVIEW` records a
Class D boundary and may transition only to `PREMORTEM` or `PROSPECTIVE_LOCK`.
The transition `BLOCKED_SCIENTIFIC_REVIEW -> COLLECTION` is explicitly
forbidden, as are shortcuts into post-outcome analysis or closeout. New
collection requires a prospective lifecycle to be re-established first.

Incident reasons are typed as:

- `INFRASTRUCTURE_RECOVERABLE`
- `ENVIRONMENT_RECOVERABLE`
- `JOURNAL_RESUME`
- `MECHANICAL_ATTRITION`
- `INSTRUMENTATION_BUG`
- `SPEC_IMPLEMENTATION_MISMATCH`
- `MEASUREMENT_INTEGRITY_CONCERN`
- `SCIENTIFIC_GATE_FAIL`
- `SCIENTIFIC_DESIGN_DECISION_REQUIRED`
- `HOLDOUT_FIREWALL`

The same instrumentation incident is Class B before affected outcomes and
Class C after them. Scientific-design and firewall incidents are always Class
D. Pure validators live in `epistemic_geometry.research.governance`; they
return or reject proposed transitions and never mutate a ledger.

## Required record

Each future incident record should contain the gate ID, current state, typed
reason, whether affected outcomes were observed, classified action class,
preserved logical keys, repair commit, tests, provenance impact, and next
authorized state. A transition is invalid until both the incident and its
resolution are durable.
