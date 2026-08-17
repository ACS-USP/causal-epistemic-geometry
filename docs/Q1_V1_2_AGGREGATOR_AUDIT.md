# Q1 V1.2 — Analysis-Only Aggregator Audit

This document records an independent audit of the completed Q1 V1.2
DEVELOPMENT artifact. It uses existing raw candidate scores only. No model was
loaded, no dataset was downloaded, no new inference was generated, and the
confirmatory holdout was not accessed.

## Reproducibility

```bash
source .venv/bin/activate
python scripts/recompute_v1_v2_from_raw.py
ceg validate-v1-2-audit review/q1_v1_2_principal_review_complete
```

The complete ignored review bundle is
`review/q1_v1_2_principal_review_complete/`. It contains the recovered raw
cyclic scores, the stored symmetrized scores, independently recomputed S/Q
rows, paired metrics, item-level flip audit, margin and scale diagnostics,
provenance, and a human-readable summary.

Source artifact checks:

- raw rows: 24,075; SHA-256
  `8a8c6b1afd4d019ee4fcb18dc0b787997218f0fcb8ddda6997239b68ae8cdd60`;
- stored symmetrized rows: 1,536; SHA-256
  `c4d0ebd57c3b6e920e65784d74161919f211e8768ff7adc8216602dc03bcf4d1`;
- stored-primary S prediction mismatches after independent recomputation: 0;
- previously generated primary paired metrics: exact match.

## Primary versus secondary aggregation

The frozen primary S estimator is the mean of per-order centered candidate
logits. The frozen secondary Q estimator is the mean of per-order softmaxes
over the allowed candidates. No third estimator was introduced.

| condition | S accuracy | Q accuracy | S delta | Q delta |
|---|---:|---:|---:|---:|
| baseline | 47.27% | 47.66% | — | — |
| PC1+ | 48.05% | 47.07% | +0.78 pp | −0.59 pp |
| PC1− | 47.46% | 45.90% | +0.20 pp | −1.76 pp |

For PC1+, S has 6 rescues and 2 damages; Q has 6 rescues and 9 damages.
Thus rescue exceeds damage under S but not under Q. Q complementarity
headroom is 1.17 percentage points.

## Item-level robustness

For PC1+, semantic prediction flips are:

- S: 22;
- Q: 39;
- intersection: 8;
- union: 53;
- Jaccard: 0.1509.

Rescue-set overlap is 1 item (Jaccard 0.0909). Damage-set overlap is 1 item
(Jaccard 0.1000). The complete item-level table is
`primary_pc1_plus_flip_audit.csv`.

S/Q prediction agreement is 433/512 for baseline (84.57%), 441/512 for PC1+
(86.13%), and 423/512 for PC1− (82.62%). Disagreement items are generally
near the decision boundary: for baseline, median S margin is 0.325 on
disagreement items versus 4.10 on agreement items; median Q margin is 0.0311
versus 0.4002.

Aggregator disagreement changes descriptively under steering: baseline 15.43%,
PC1+ 13.87%, and PC1− 17.38%. Candidate-logit spread diagnostics are stored
separately and are explanatory only.

## Descriptive classification

Using the pre-specified audit rule that a sign reversal in the accuracy delta
or rescue-minus-damage balance is non-robust, the V1.2 residual signal is

```text
NON-ROBUST — aggregator-sensitive descriptive result
```

This is not a confirmatory scientific conclusion. It is a reason to avoid
promoting the V1.2 residual effect to a frozen claim or moving to Q2 without
principal-researcher review.

## Split provenance

The two similar-looking hashes refer to different objects:

- logical split-manifest digest (`manifest_sha256`):
  `84982e4c72e230ffff78363f085d4d5c53447fd1e248e5e170ed5e8c508d343e`;
- SHA-256 of the split JSON file bytes:
  `57435aab3df4bf0c345097b72e38d806b6849e9fedf89ecf269c3528b22a98dc`;
- digest of the 512 DEV_EVALUATION IDs:
  `cd3ae17e6e35231e8318da5464bdd8795d4bb4764c68b3c9bf8cf4ab31ffcd8c`.

The V1.2 artifact's `split_manifest_sha256` records the split-file byte SHA,
while the split file's own `manifest_sha256` records the logical manifest
payload. The exact DEV_EVALUATION IDs match. This is not a changed scientific
split.

## Scientific boundary

This audit does not establish Q1, does not touch the holdout, does not run
V1.3, and does not run Q2. It only establishes that the stored primary
aggregation is reproducible and that the pre-registered secondary aggregation
produces materially different item-level and effect-level descriptions.
