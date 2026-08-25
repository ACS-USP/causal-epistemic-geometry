# Q2 V2 pre-outcome analysis implementation amendment

Status: `PROSPECTIVE_BEFORE_COMMON_PANEL_OUTPUTS`

Date: 2026-08-25 (America/Sao_Paulo)

The frozen Q2 V2 protocol and final lock define M1 as activation-covariance
whitened geometry with:

`Sigma_lambda = (1 - lambda) Sigma + lambda mean_variance I`

and normalized Euclidean/cosine geometry in that whitened inner product. Before
any common-panel row existed, inspection found that the primary analysis helper
instead used `Sigma + lambda mean_variance I` and an unnormalized Mahalanobis
distance.

The implementation now calls the repository's already-tested canonical
`fit_whitening` and `whitened_geometry` functions with the frozen `lambda=0.10`,
returning `normalized_euclidean`. This is a prospective Class-B implementation
repair to conform code to the already-frozen mathematical object. It changes no
controller, panel row, family split, outcome, threshold, prediction score,
permutation, or classification rule.

The frozen item-cluster bootstrap (`10,000` resamples, seed `2026082401`) and an
independent forensic recomputation are also materialized before collection.
Bootstrap intervals remain descriptive and do not enter the frozen composite
classification.
