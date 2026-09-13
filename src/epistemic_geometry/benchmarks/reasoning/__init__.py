"""Q1 V3 reasoning-agent procedural suite.

This package is deliberately separate from the closed E3 direct-readout
instrument.  Its outputs are exact final answers parsed from generated
reasoning trajectories rather than candidate logits.
"""

from .base import (
    FINAL_ANSWER_INSTRUCTION,
    GENERATOR_VERSION,
    SUITE_VERSION,
    ReasoningItem,
    ReasoningView,
)
from .families import FAMILY_CELLS, generate_item, oracle_for
from .novelty import (
    DIVERSIFICATION_CONDITIONS,
    DIVERSIFICATION_COVERAGE_NAMESPACE,
    DIVERSIFICATION_COVERAGE_SPLIT,
    DIVERSIFICATION_EVALUATION_CONDITIONS,
    DIVERSIFICATION_EVALUATION_NAMESPACE,
    DIVERSIFICATION_EVALUATION_PHASE,
    DIVERSIFICATION_QUALIFICATION_CONDITIONS,
    DIVERSIFICATION_QUALIFICATION_NAMESPACE,
    DIVERSIFICATION_QUALIFICATION_PHASE,
    HISTORICAL_REASONING_SPLIT_NAMES,
    DiversificationCoveragePlan,
    DiversificationManifest,
    DiversificationNoveltyReport,
    audit_diversification_novelty,
    build_diversification_coverage_plan,
    build_diversification_schedule,
    construct_diversification_coverage_plan,
    generate_diversification_manifest,
    plan_diversification_coverage,
)
from .semantic_audit import (
    SemanticAudit,
    audit_reasoning_view,
    audit_reasoning_views,
    validate_reasoning_view,
)

__all__ = [
    "FAMILY_CELLS",
    "FINAL_ANSWER_INSTRUCTION",
    "GENERATOR_VERSION",
    "ReasoningItem",
    "ReasoningView",
    "SUITE_VERSION",
    "SemanticAudit",
    "audit_reasoning_view",
    "audit_reasoning_views",
    "generate_item",
    "oracle_for",
    "validate_reasoning_view",
    "DIVERSIFICATION_CONDITIONS",
    "DIVERSIFICATION_COVERAGE_NAMESPACE",
    "DIVERSIFICATION_COVERAGE_SPLIT",
    "DIVERSIFICATION_EVALUATION_CONDITIONS",
    "DIVERSIFICATION_EVALUATION_NAMESPACE",
    "DIVERSIFICATION_EVALUATION_PHASE",
    "DIVERSIFICATION_QUALIFICATION_CONDITIONS",
    "DIVERSIFICATION_QUALIFICATION_NAMESPACE",
    "DIVERSIFICATION_QUALIFICATION_PHASE",
    "DiversificationCoveragePlan",
    "DiversificationManifest",
    "DiversificationNoveltyReport",
    "HISTORICAL_REASONING_SPLIT_NAMES",
    "audit_diversification_novelty",
    "build_diversification_coverage_plan",
    "build_diversification_schedule",
    "construct_diversification_coverage_plan",
    "generate_diversification_manifest",
    "plan_diversification_coverage",
]
