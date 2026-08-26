# Q2 V4.1 — final presemantic freeze premortem

Status: `PREMORTEM_PASS`.

This review uses the complete realized set of 31 directions that passed the
original V4 two-shell, label-free safety gate. It does not generate, redraw,
rank, or remove a controller. The original V4 result remains
`Q2_V4_SAFE_BANK_INSUFFICIENT`, and the Q2 relational hypothesis remains
untested.

## D38 history

The immutable safety JSON and journal are authoritative. `V4_DIRECTION_38`
fails MEDIUM because raw sequence movement is `0.08333333333333333`, below the
frozen `0.10` threshold. It fails STRONG because validity/evaluability are
`0.9166666666666666`, the relative drop exceeds `0.05`, and the frozen
`0.90` floor is still satisfied. It has no truncation. The source artifact has
no normalized reason-code field; this review records normalized reason labels
without changing the raw artifact.

## Leverage

V4/V4.1 contains no prospective controller-level leverage gate. The V3
family-level leverage concept is not applicable to this controller bank.
Leverage is therefore `LEVERAGE_DESCRIPTIVE_ONLY`; the maximum observed value
is reported descriptively and cannot block the bank.

## G3 power

The K=31/N=300 CPU-only simulation preserves the V4 endpoint, controller
dependence, two-shell structure, controller-label QAP, maxT logic, and frozen
`0.10` superiority margin. It injects an explicit A2 advantage over the fixed
grid `Delta = {0, .05, .10, .15, .20}` and records the achieved synthetic
relational strengths. This characterizes the existing design; it does not
change K, N, the endpoint, the superiority margin, or the semantic protocol.

## Instrument and firewall risks

The A1 covariance set and A2 label-free probes are disjoint from the semantic
panel and contain no correctness use. A1/A2 matrices are materialized only
after this lock is committed. The 300-item semantic panel is frozen but not
executed. The future 37,800-row schedule is created as a manifest only.

The main engineering risks are numerical reproducibility, accidental use of
the old 32-controller assumptions, parser/semantic imports, and Spark-1
environment drift. The lock and audit must reject any of these conditions.

## Decision

No Class-A or Class-B scientific-design defect is present. The protocol can
proceed to a pre-outcome lock and label-free Spark-1 A1/A2 materialization.

`PREMORTEM_PASS`
