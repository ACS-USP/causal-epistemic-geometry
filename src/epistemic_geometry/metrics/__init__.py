"""Interpretable paired-error metrics."""

from .complementarity import compute_paired_metrics
from .errors import accuracy, double_fault, error_jaccard, phi_correlation
from .reasoning import (
    SeedRegime,
    propensity_correlation_from_rollouts,
    stochastic_complementarity_estimands,
    unbiased_two_rollout_propensity_distance,
)
from .uncertainty import bootstrap_paired_metrics, cluster_bootstrap_mean

__all__ = [
    "accuracy",
    "double_fault",
    "error_jaccard",
    "phi_correlation",
    "compute_paired_metrics",
    "bootstrap_paired_metrics",
    "cluster_bootstrap_mean",
    "SeedRegime",
    "stochastic_complementarity_estimands",
    "unbiased_two_rollout_propensity_distance",
    "propensity_correlation_from_rollouts",
]
