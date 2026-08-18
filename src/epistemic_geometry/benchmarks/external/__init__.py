"""External-benchmark qualification interfaces.

This package deliberately separates benchmark loading, response parsing, and
objective evaluation.  It does not download datasets; real loaders receive a
local path prepared on the execution host.
"""

from .adapters import (
    CandidateSpec,
    ExternalBenchmarkAdapter,
    JsonlExternalAdapter,
    candidate_specs,
)
from .base import ExternalItem, ExternalResult, ExternalStatus
from .metrics import QualificationSummary, summarize_qualification

__all__ = [
    "CandidateSpec",
    "ExternalBenchmarkAdapter",
    "ExternalItem",
    "ExternalResult",
    "ExternalStatus",
    "JsonlExternalAdapter",
    "QualificationSummary",
    "candidate_specs",
    "summarize_qualification",
]
