# Evidence interpretation policy

This document is normative for interpretation and reporting. It supplements the
[scientific constitution](SCIENTIFIC_CONSTITUTION.md) without changing any
historical protocol, threshold, classification, or outcome.

## Core rule

**Decision rules adjudicate claims; they do not erase evidence.**

Let an experiment produce an observed evidence vector

\[
E=(\text{effect estimates},\ \text{uncertainty},\ \text{safety},\
\text{validity},\ \text{controls},\ \text{provenance},\ldots)
\]

and let a prospectively frozen rule return a terminal decision

\[
f(E)\in\{\text{PASS},\text{FAIL},\text{BLOCKED},\ldots\}.
\]

The terminal decision is a function of the evidence, not the evidence itself:

\[
f(E) \ne E.
\]

A failed conjunctive rule can coexist with supported sub-results, useful
mechanistic dissociations, negative results, or unexpected structure. Those
components must be reported with their own evidence status rather than deleted
or promoted to a claim that the frozen rule did not authorize.

## Three interpretation layers

### 1. Confirmatory decision

The prospectively frozen decision is immutable after outcome access. Report the
exact rule, all conjuncts, and which conjuncts passed or failed. Never rename a
failed composite decision to a partial pass, rescue it with a new parser or
subset, or replace it with a more favorable post-hoc criterion.

### 2. Scientific evidence

Every experiment yields an evidence vector, not merely a Boolean. Preserve and
report, as applicable:

- point estimates and frozen uncertainty intervals;
- competence, commitment validity, and semantic evaluability;
- meaningful-versus-baseline and meaningful-versus-null contrasts;
- rescue, damage, and mechanical outcome composition;
- negative and null results;
- provenance, retries, exclusions, and integrity checks.

Each supported component receives the strongest status it actually earned.
Passing one component does not imply that the full composite claim passed;
failing another component does not make the supported component disappear.

### 3. Post-hoc discovery

Closed experiments may be inspected quantitatively and qualitatively to
understand mechanisms and generate new DEVELOPMENT hypotheses when all of the
following hold:

- the historical result and source artifacts remain immutable;
- the analysis is labeled `POST_HOC_DESCRIPTIVE`;
- no cleaned or reclassified result is substituted for the frozen result;
- complete-case filtering is not presented as primary evidence;
- any recovered answer, subgroup, mechanism, or taxonomy is separated from
  confirmatory estimands;
- a future test is prospectively specified on new data before it can support a
  new claim.

Post-hoc work can explain a failure mode; it cannot retroactively pass a gate.

## Motivating example: Q1 Ministral

The frozen cross-model classification is
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. Ministral had positive
competence-adjusted complementarity, a positive meaningful-minus-random-mean
contrast, and C above every frozen random control, while improving accuracy.
It nevertheless failed the prospectively frozen commitment-validity and
semantic-evaluability guards. The confirmatory model decision is therefore a
fail and remains so.

The subsequent descriptive invalidity audit found a mixture of token-cap
runaway generation and commitment-format instability; invalid rows contributed
damage and no rescues. This supports a post-hoc mechanism hypothesis that
complementarity transferred more robustly than safe policy realization. It
does not change the frozen decision, parser, or metrics.

## Reporting checklist

For any result with a composite rule, reports must answer:

1. What was the exact terminal decision `f(E)`?
2. What components of `E` were supported, unsupported, or not measured?
3. Which statements are confirmatory, development, negative, post-hoc, or not
   established?
4. Did any post-hoc analysis alter the historical result? The expected answer
   is no.
5. What prospective evidence would be required to promote a post-hoc mechanism
   or bounded result?

This policy is deliberately conservative about claims and deliberately complete
about evidence.
