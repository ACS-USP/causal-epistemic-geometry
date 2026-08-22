# GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM — draft only

Status: `DRAFTED_NOT_AUTHORIZED`. This document does not allocate items,
authorize GPU work, open Q2, or execute Gate 11.

## Motivation

The fixed L27-D75 controller produced strong safe semantic error control on
fresh CRUXEval in Gate 9 but no specific transfer on fresh long character
counting in Gate 10. Gate 10 had sufficient baseline opportunity and the
controller was mechanically safe, so the discrepancy is domain-conditioned
rather than an instrument ceiling or destructive intervention result.

## Prospective question

Why does the frozen controller reorganize CRUXEval errors but not exact long
character-count errors? Candidate explanations to separate prospectively are:

- the careful/direct source encodes program-tracing operations rather than a
  domain-general deliberation policy;
- L27 has different local control gain for the two task representations;
- textual source-policy transfer is weak in character counting;
- the same Euclidean direction has domain-dependent downstream sensitivity.

## Required future lock

A separately authorized protocol should freeze condition-symmetric,
label-free diagnostics of textual-source transfer and local control geometry.
It may compare preserved CRUXEval and character-count artifacts but must not
search new semantic controllers, doses, signs, or layers using correctness
outcomes. Any new model outputs, Jacobian/Fisher work, or Q2 transition require
principal authorization and a new premortem/prospective lock.

Executed: `NO`.
