# Gate 8 adversarial premortem

Classification: `PREMORTEM_PASS`.

This review was completed before any Gate-8 model output. Gate 8 is a matched,
50-item dose calibration of one byte-identical L27 controller. It cannot become
the later independent dose evaluation.

## A. Dose-selection leakage

The only selectable doses are D25, D50, and D75. Selection is the lowest dose
passing all frozen source, commitment, evaluability, competence-safety, semantic
change, matched-random specificity, and CAREFUL token-regime gates. Accuracy is
only a lower safety bound. G/C/D are neither computed nor used for selection.
D100 is diagnostic and cannot be selected.

## B. Freshness

Every `sample_N` identifier in preserved manifests, journals, reserve pools,
draft allocations, diagnostics, and holdout allocations is consumed. Allocation
requires at least 150 eligible unseen IDs before selecting exactly 50, and at
least 100 unseen IDs must remain afterward. Gate 8 records only the remaining
count and does not allocate or inspect future evaluation IDs.

## C. Controller identity

The meaningful artifact is loaded byte-for-byte from the accepted Gate-6.2
paired-mean L27 file and must match canonical float-vector hash
`e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838`.
Only the frozen scalar fractions 0.25, 0.50, 0.75, and 1.00 vary.

## D. Random matching

Four new deterministic random vectors are generated once, orthogonalized to the
meaningful vector and one another, and reused across every dose. At each dose,
meaningful and random arms share layer 27, standardized energy, sustained
current-token duration, token scope, engine, and cap. Earlier random vectors are
not reused.

## E. Parser invariance

`external-semantic-v3`, its specification, and blinded corpus are hashed before
collection. Commitment validity, semantic evaluability, and correctness remain
separate. Any later measurement concern is Class C offline work only.

## F. Matched calibration design

Two rollout blocks are independent. Within an item-rollout block, all 22
conditions share one deterministic seed. The schedule labels this explicitly as
`MATCHED_COUPLING_CALIBRATION`; it is not an independent ensemble estimand.

## G. Output length

The 4096-token cap is unchanged. Truncation, absent commitment, ambiguity,
unevaluability, wrong answers, and model runtime failures are retained as model
outcomes. They are never behaviorally retried.

## H. Monotonicity

Dose curves are reported without assuming monotonicity. Every violation of the
descriptive expected direction is preserved and cannot change the frozen
selection rule.

## I. Source anchor

The exact textual CAREFUL policy is collected contemporaneously. Token recovery
uses CAREFUL minus baseline; a nonpositive denominator fails source replication
and blocks dose selection.

## J. Resume and schedule

Logical keys are `item_id × condition × rollout_index`. The full interleaved
schedule is frozen and hashed. Journal writes are append/flush/fsync. Resume
skips completed keys and rejects duplicates or provenance mixtures.

## K. Future evaluation firewall

All 50 calibration IDs become permanently consumed. No Gate-9 ID is allocated,
rendered, or inspected. Gate 8 ends after calibration, selection, and forensic
audit. Q2, character count, and confirmatory holdout remain closed.

No unresolved Class A, B, or scientific-design ambiguity remains before lock.
