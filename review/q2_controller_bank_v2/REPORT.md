# Q2 V2 — Calibrated controller-family-held-out geometry

## Outcome

The frozen 120-item DEVELOPMENT common panel completed with all 6,960 expected
trajectories. The independent audit reproduced the primary analysis to floating-
point tolerance and returned `Q2_V2_FORENSIC_CLEAN`.

The mechanically frozen classification is:

`Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL`

No metric passed all three frozen criteria simultaneously: mean family-held-out
Spearman at least 0.30, one-sided family-aware QAP p-value at most 0.05, and
held-out RMSE ratio to the constant predictor at most 0.90.

That composite classification does not erase a supported sub-result. M2 finite
secant showed a substantially stronger and fold-consistent association than M0
or M1: mean held-out rho was 0.4279, every family fold was positive, and
`p_QAP=0.00220`. It missed the composite gate only because its RMSE ratio was
0.9067 rather than at most 0.90. This is DEVELOPMENT evidence of association,
not a qualifying predictive-geometry claim.

## Frozen design and collection integrity

- source axes proposed / qualified: 6 / 6;
- meaningful controllers: 24 across six conceptual families;
- fresh random controls: 4;
- common panel: 120 items, 29 conditions, two independent rollouts;
- scientific trajectories: 6,960 / 6,960;
- logical-key duplicates, missing rows, and unexpected rows: 0 / 0 / 0;
- unique seeds: 6,960;
- scientific retries or regenerated keys: 0;
- execution source commit: `080dfc83a6c51711f782fa889c6ffff4fcec13e1`;
- journal SHA-256: `9a635787561d5bc6e56cf2c7ffae9e391bebc817488f38369ffc8c0fea14a5b7`;
- model/revision, manifest, schedule, protocol, and identity hashes: exact;
- scientific peeking during collection: none.

The corrected pre-collection RunPod wallet value was US$35.23. The frozen
scientific design, execution engine, cost ceiling, and all estimators remained
unchanged.

## Primary family-held-out prediction

| Metric | Mean rho | Median rho | One-sided QAP p | Mean RMSE | Constant RMSE | RMSE ratio | Mean-rho bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 flat | 0.2013 | 0.2218 | 0.000700 | 0.016088 | 0.016305 | 0.9867 | [0.0541, 0.2513] |
| M1 whitened | 0.1902 | 0.1970 | 0.000900 | 0.016112 | 0.016305 | 0.9882 | [0.0543, 0.2376] |
| M2 finite secant | 0.4279 | 0.4025 | 0.002200 | 0.014785 | 0.016305 | 0.9067 | [0.1205, 0.4826] |

The bootstrap RMSE-ratio intervals were respectively `[0.9709, 1.0048]`,
`[0.9743, 1.0045]`, and `[0.8537, 1.0074]`. Bootstrap intervals are descriptive;
the frozen classification uses the prespecified point-estimate decision rule.

### M2 fold stability

All six held-out source-family correlations were positive:

| Held-out family | M2 rho |
|---|---:|
| Counterfactual checking | 0.3785 |
| Decompose then solve | 0.4185 |
| Explicit state tracking | 0.5243 |
| Independent verification | 0.3321 |
| Invariant checking | 0.3864 |
| Type/representation discipline | 0.5277 |

M0 and M1 were both statistically associated under the QAP diagnostic but had
small held-out correlations and essentially no calibrated RMSE improvement.
Whitening did not improve over flat geometry. M2 improved both rank association
and RMSE, but not enough to cross the frozen 0.90 RMSE-ratio threshold.

## Controller-bank evidence vector

The final bank retained the prospectively qualified dynamic range. At selected
doses, semantic movement ranged from 0 to 0.25 (mean 0.0903; median 0.0833), and
12 of 24 directions met the frozen causal rule. All 24 met the calibration
safety rule. Selected displacement norms spanned 7.819 to 66.800 across all
four dose bins.

On the common panel, the 24 meaningful controllers remained mechanically safe:
commitment validity and semantic evaluability ranged from 0.9833 to 1.0000.
Their accuracy changes ranged from -0.0167 to +0.0292. The bank was not selected
on accuracy, G, C, D, rescue, damage, or common-panel outcomes.

Descriptively:

- calibration movement versus mean common-panel error distance: rho 0.4760;
- displacement norm versus mean error distance: rho 0.4086;
- all-edge geometry/error-distance rho: M0 0.1521, M1 0.1476, M2 0.4326;
- the four random controls had near-zero or negative mean G/C/D and remained
  secondary controls rather than the primary prediction population.

These descriptive results do not modify the frozen classification and are not
controller-selection evidence.

## Analysis and forensic incidents

Before outcomes, M1 was corrected to match the already frozen lock: shrinkage
uses `(1-lambda) Sigma + lambda * mean_variance * I`, and distances use normalized
whitened vectors. The amendment was prospective and changed no scientific
object or threshold.

After collection, the first analysis invocation stopped before producing any
metric because the crash-safe journal stores each scientific row inside a
versioned envelope. The three analysis paths were independently repaired to
unwrap and validate the immutable envelope and logical key. No output was
regenerated, no parser value changed, and no metric had been produced or
inspected before the repair. This is a deterministic non-scientific I/O repair.

The independent audit then verified schedule completeness, unique keys and
seeds, exact source/model/provenance, zero retries, estimands, geometries,
family folds, QAP values, and the final classification. Maximum primary/audit
numeric difference was `1.5543122344752192e-15`.

## Cost and infrastructure

The selected A40 serial reference engine accumulated 2.6602 billed GPU-hours.
The final post-deletion RunPod billing snapshot attributes US$1.1853 to the
common-panel Pod, bringing cumulative Q2 V2 cost to US$5.1335. Summed
per-trajectory generation duration was 2.8978 hours. The complete accounting is
persisted in `V2_COST_CLOSEOUT.json`; actual cost was far below the US$32.1623
panel projection and the US$45 cumulative ceiling.

After journal recovery and SHA verification, the Pod was deleted. The dedicated
RunPod control plane reports zero Pods and zero network volumes.

## Scientific interpretation

Q2 V2 is a DEVELOPMENT composite null under its frozen decision rule. It is not
evidence that all intervention geometry is unrelated to error geometry. The
evidence vector instead separates three observations:

1. flat and covariance-whitened representation geometry showed weak held-out
   rank association and negligible RMSE improvement;
2. finite behavioral secant geometry showed a stronger, family-consistent,
   permutation-supported association;
3. that M2 association narrowly missed the frozen calibrated-prediction gate.

The next scientific question is whether the M2 association is reproducible and
can clear a prospectively justified calibration criterion on fresh families or
models, without selecting a geometry after outcomes. No next experiment is
authorized or executed here.

## Firewall

- Q1: `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`, unchanged;
- Q2 role: DEVELOPMENT;
- Q3: not run;
- confirmatory holdout: already consumed by Q1 and not reused;
- new inference after Q2 V2: none;
- next action: `PRINCIPAL_RESEARCHER_REVIEW`.
