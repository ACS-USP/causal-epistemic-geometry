# Q1 V2 — E3-10 Calibration Review

The authorized baseline-only Qwen calibration has completed and is **not
qualified** under the frozen E3-10 instrument rule.

## Scope actually executed

- Model: `Qwen/Qwen3-8B`, immutable revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Host: RunPod NVIDIA A40; model snapshot was already cached remotely.
- Cells: 11 structurally eligible family/cell combinations.
- Latents: 2,200 total, exactly 20 per target digit per cell.
- Views: 6,600 total — canonical decimal, surface-twin decimal, and canonical
  number-word.
- Engine: approved `full_prompt_batched` serial-shape reference profile.
- Steering, activation extraction, PCA, DEV evaluation, and holdout access:
  **none**.

The tokenization audit passed before inference. Decimal candidates `0`–`9` and
number-word candidates `zero`–`nine` were unique context-compatible single
tokens.

## Mechanical outcome

Zero of 11 cells qualified. Canonical decimal accuracy was approximately
chance in every family/cell, and decimal/number-word agreement failed the
frozen threshold in every cell. Surface-twin agreement and normalized
prediction entropy also failed in most or all cells. No cell was selected and
the suite-level rule therefore emitted:

```text
E3_10_INSTRUMENT_NOT_QUALIFIED
```

This is an instrument qualification outcome, not evidence for or against the
project-level causal geometry question. It does not support a steering claim.

## Firewall result

Fresh `GEOMETRY_CALIBRATION`, `DEV_EVALUATION`, and
`CONFIRMATORY_HOLDOUT` manifests were **not generated** after the failed
qualification. No steering direction was constructed. The earlier V1–V1.2
multiple-choice series remains closed as development infrastructure and is not
being combined with this result.

## Audit bundle

The ignored local review bundle is:

```text
review/q1_v2_instrument_review/
```

It contains the remote raw score vectors and hashes, enriched per-view rows
with logits/probabilities/margins/NLL/Brier, the full 11-cell review table,
figures 1–6, independent CPU recomputation, and validator output. Model
weights, HuggingFace cache contents, and dataset caches were not pulled to the
Mac.

Principal review is required before any instrument redesign or new calibration
protocol. Do not run steering, DEV, holdout, V1.3, or Q2 from this bundle.
