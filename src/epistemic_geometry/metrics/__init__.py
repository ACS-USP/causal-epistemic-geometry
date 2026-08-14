"""Interpretable paired-error metrics."""

from .complementarity import compute_paired_metrics
from .errors import accuracy, double_fault, error_jaccard, phi_correlation
from .uncertainty import bootstrap_paired_metrics

__all__ = [
    "accuracy",
    "double_fault",
    "error_jaccard",
    "phi_correlation",
    "compute_paired_metrics",
    "bootstrap_paired_metrics",
]
