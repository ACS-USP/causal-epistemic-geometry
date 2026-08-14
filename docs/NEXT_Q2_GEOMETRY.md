# Next Q2: geometry scaffold

Only if Q1 survives development review should the project move from
baseline-versus-one-vector to multiple interventions:

```text
v_i, v_j  ->  per-intervention predictions
          ->  e_i(t), e_j(t)
          ->  error correlation(e_i, e_j)
```

Candidate relationships include `cosine(v_i, v_j)` or Euclidean distance for
normalized vectors versus pairwise error similarity. The causal question would
be whether representation geometry predicts held-out error covariance, not
whether arbitrary steering produces varied strings.

This repository intentionally does not implement a large pairwise scientific
pipeline, Riemannian APIs, a manifold metric, geodesics, Jacobian-induced
geometry, or differential geometry. Those choices require a successful Q1,
literature/theory work, frozen controls, and a separate development protocol.

A small future helper can remain pure and reusable:

```python
cosine_distance(v_i, v_j)
euclidean_distance(normalized_v_i, normalized_v_j)
```

The current vector storage and paired prediction records are the intended seam
for adding that work later.

