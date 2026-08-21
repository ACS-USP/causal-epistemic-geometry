# Research agent workflow

This workflow reduces coordination overhead without delegating scientific
authority to a model product. Role labels describe working modes, not evidence
and not scientific claims.

## Preferred role split

**Sol High** is preferred for experimental design, adversarial premortem,
methodological ambiguity, protocol review, and forensic postmortem.

**Luna Extra High** is preferred for implementation, tests, infrastructure,
execution, resume/recovery, deterministic analysis, and packaging.

For complex scientific gates, prefer:

```text
Sol premortem
    -> prospective lock
    -> Luna implementation/execution
    -> independent deterministic audit
    -> Sol forensic interpretation
```

The principal researcher owns Class D decisions throughout. Agent identity,
model family, or reasoning setting must not appear as support for a scientific
claim. Claims derive only from the frozen protocol, preserved evidence, and
audited analysis.

## Handoff contract

Each handoff should identify the gate state, source commit, frozen protocol and
schema hashes, authorized action class, typed incident (if any), logical resume
keys, environment-preflight result, tests, and the exact next authorized action.
The receiver must stop on provenance mismatch, dirty source, missing lock,
holdout ambiguity, or a Class D choice.

Operational recovery may proceed under Class A. A pre-outcome mechanical
instrument correction follows Class B and must be relocked. Post-outcome work
is offline Class C unless a frozen decision tree already authorizes more
collection. See the [research autonomy policy](RESEARCH_AUTONOMY_POLICY.md).

A gate in `BLOCKED_SCIENTIFIC_REVIEW` cannot return directly to collection or
analysis. Principal review must route it through `PREMORTEM` or a new
`PROSPECTIVE_LOCK`; collection remains impossible until the prospective
lifecycle has been restored.
