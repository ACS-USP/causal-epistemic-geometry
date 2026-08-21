# Gate 8 draft — Prospective dose calibration

Status: `DRAFT_NOT_AUTHORIZED`; no items are allocated and no model inference
has been run.

## Purpose

Determine prospectively whether the exact frozen Gate-7 L27 paired-mean
controller has a lower sustained dose that preserves commitment validity and
semantic evaluability while retaining a measurable causal first stage. This is
a calibration protocol, not a replication, Q2, or confirmatory experiment.

## Frozen scientific components

- model, revision, tokenizer, BF16/SDPA, non-thinking sampling policy, and
  CRUXEval evaluator remain identical to Gate 7;
- meaningful vector, layer 27, plus orientation, and sustained current-token
  timing remain byte-for-byte fixed;
- no controller, layer, sign, benchmark, or parser search;
- all calibration IDs must be fresh and disjoint from every historical,
  reserved, evaluation, and holdout allocation;
- random controls must be architecture-, energy-, duration-, and scope-matched;
- invalid and truncated model outcomes remain primary errors.

## Prospective design required before authorization

Use a separately frozen calibration split and a small, monotone dose set stated
as fractions of the Gate-7 eta, including zero and the original eta. Freeze the
dose grid, sample size, random bank, seeds, schedule, competence/validity guards,
selection rule, and cost ceiling before any output. Select at most one dose by
a deterministic rule that first requires both Gate-7 validity guards and then
requires a pre-specified manipulation minimum. Accuracy or complementarity may
not rescue a dose that fails validity.

Any selected dose must then be evaluated later on a second fresh split under a
separate prospective lock. Calibration outcomes may not be reused as evaluation
evidence. If no dose passes, retire this fixed controller/duration family rather
than adding layers, signs, or controllers adaptively.

## Firewall

No Q2, character-count run, confirmatory holdout access, controller
reconstruction, or new model output is authorized by this draft.
