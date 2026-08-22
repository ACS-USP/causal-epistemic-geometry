GATE 11.1 — ARTIFACT-COMPLETE FORENSIC REPLICATION
======================================================================

PURPOSE
----------------------------------------------------------------------

This run repaired the Gate-11 raw-artifact boundary. It reran only the frozen
teacher-forced propagation diagnostics for the original Gate-11 items and
conditions. Historical Gate-11 source and policy-utility results remain
immutable. No free generation, new semantic evaluation, controller search,
new item, or holdout access occurred.

COLLECTION
----------------------------------------------------------------------

model:
    Qwen/Qwen3-8B

revision:
    b968826d9c46dd6066d109eabc6255188de91218

controller:
    frozen Gate-11 L27 paired-mean plus, D75

eta:
    9.637427952852196

conditions:
    baseline, textual careful, meaningful L27 D75, four Gate-11 random controls

items:
    24 CRUXEval + 24 CHARCOUNT

logical rows:
    336

checkpoints:
    PREFILL, 0, 1, 3, 7, 15, 31, 63, 127

captured layers:
    27, 28, 30, 32, 35

raw persistence:
    complete float32 vocabulary logits and hidden-difference vectors in 48
    losslessly compressed NPZ shards

raw archive:
    814597682 bytes

raw archive SHA-256:
    3fe24aab5ba5a60f39facb0f79f77a266988905bcb1d4f23a267a6421d929b86

ENGINEERING / PROVENANCE
----------------------------------------------------------------------

remote preflight:
    PASS

engineering gate:
    GATE11_1_ENGINEERING_PASS

free generation:
    0

new semantic evaluation:
    0

source commit:
    c4c6c77132bdd44907497af3d4239f812bf6d9cf

Pod:
    pdmiiro0rd63om, EU-SE-1, NVIDIA A40, stopped after recovery

SOURCE-AXIS RESULT
----------------------------------------------------------------------

The source-axis metrics measure representation transfer, not control energy.
The immutable source activations support transfer in both domains:

| domain | mean frozen gap | positive-gap fraction | bootstrap lower | cosine to frozen |
|---|---:|---:|---:|---:|
| CRUXEval | 124.4309 | 1.00 | 121.6008 | 0.9983 |
| CHARCOUNT | 112.1702 | 1.00 | 111.0310 | 0.6594 |

source-axis transfer:
    SUPPORTED

PROPAGATION CONTROL DIAGNOSTICS
----------------------------------------------------------------------

These are finite-displacement control-gain diagnostics from persisted logits
and downstream hidden differences. They are not an exact local pullback/Fisher
metric.

| domain | metric | meaningful | random mean | random max | meaningful - random mean | meaningful - random max |
|---|---|---:|---:|---:|---:|---:|
| CRUXEval | next-token KL | 0.212327 | 0.029154 | 0.046555 | 0.183173 | 0.165772 |
| CRUXEval | A35 hidden displacement | 300.7451 | 198.7775 | 203.4964 | 101.9676 | 97.2487 |
| CRUXEval | top-1 flip | 0.118849 | 0.030729 | 0.048611 | 0.088120 | 0.070238 |
| CHARCOUNT | next-token KL | 0.067664 | 0.014349 | 0.020220 | 0.053315 | 0.047444 |
| CHARCOUNT | A35 hidden displacement | 355.7536 | 184.6166 | 188.1814 | 171.1370 | 167.5723 |
| CHARCOUNT | top-1 flip | 0.065278 | 0.030324 | 0.048148 | 0.034954 | 0.017130 |

The cross-domain contrasts used by the frozen synthesis were:

- KL: CRUXEval minus CHARCOUNT = 0.129858, bootstrap 95% interval
  [0.035764, 0.237242];
- A35: -69.1694, interval [-86.1045, -51.8002];
- top-1 flip: 0.053166, interval [-0.0042328, 0.116187].

Therefore the frozen control-gain domain-shift rule is not supported: only one
of the three primary contrast families is positive with the required evidence.

POLICY-REALIZATION DIAGNOSTIC
----------------------------------------------------------------------

Meaningful-logit alignment with the textual careful reference was:

| domain | mean alignment |
|---|---:|
| CRUXEval | 0.244982 |
| CHARCOUNT | 0.555507 |

The CRUXEval minus CHARCOUNT contrast was -0.310525 with bootstrap 95%
interval [-0.394293, -0.229736]. The frozen policy-realization shift rule is
not supported.

POLICY-UTILITY RESULT
----------------------------------------------------------------------

Historical accuracy/G/C/D are task-utility estimands, not control-energy
metrics. The immutable Gate-11 utility reanalysis supports a domain shift:

- meaningful accuracy change: CRUXEval +0.1300, CHARCOUNT -0.0275;
- cross-domain meaningful accuracy contrast: +0.1575, bootstrap 95% interval
  [+0.0775, +0.2350];
- textual careful contrast: +0.3875, interval [+0.2825, +0.4925].

FINAL SYNTHESIS
----------------------------------------------------------------------

GATE11_POLICY_UTILITY_DOMAIN_MISMATCH

The artifact-complete propagation rerun confirms that the historical synthesis
is not an artifact of missing primitive propagation data. Representation
transfer is supported, but the tested finite-displacement downstream-control
diagnostics and careful-logit alignment do not establish the frozen
domain-shift rules. The robust domain-conditioned difference remains in task
utility: the same controller improved CRUXEval utility while the historical
CHARCOUNT utility effect was negative.

METRIC BOUNDARIES
----------------------------------------------------------------------

1. Source-axis metrics measure representation transfer.
2. D75 KL/JS and downstream hidden displacement are finite-displacement
   control-gain diagnostics.
3. They are not an exact local pullback/Fisher metric; Gate 11.1 did not
   establish Fisher geometry.
4. Historical accuracy/G/C/D measure task utility, not control energy.

The raw per-checkpoint baseline and condition logits, hidden-state differences,
token indices, and normalization/provenance metadata are preserved for a future
offline or separately authorized exact JVP/Fisher audit.

FORENSIC CLOSEOUT
----------------------------------------------------------------------

primary result:
    preserved

independent audit:
    GATE11_1_FORENSIC_REPLICATION_CLEAN_AGREEMENT

primary/independent maximum difference:
    0.0

Q2:
    NOT RUN

confirmatory holdout:
    UNTOUCHED

Gate 12:
    NOT drafted or executed; principal review required
