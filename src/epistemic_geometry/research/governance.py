"""Typed, pure validation for future research-gate lifecycle records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActionClass(StrEnum):
    AUTONOMOUS_ENGINEERING_RECOVERY = "A"
    PROSPECTIVE_INSTRUMENT_AMENDMENT = "B"
    OFFLINE_POST_OUTCOME_FORENSIC_REPAIR = "C"
    PRINCIPAL_RESEARCHER_REQUIRED = "D"


class GateState(StrEnum):
    PREPARE = "PREPARE"
    PREMORTEM = "PREMORTEM"
    PROSPECTIVE_LOCK = "PROSPECTIVE_LOCK"
    ENGINEERING = "ENGINEERING"
    COLLECTION = "COLLECTION"
    OFFLINE_ANALYSIS = "OFFLINE_ANALYSIS"
    FORENSIC_AUDIT = "FORENSIC_AUDIT"
    CLOSED = "CLOSED"
    BLOCKED_RECOVERABLE = "BLOCKED_RECOVERABLE"
    BLOCKED_SCIENTIFIC_REVIEW = "BLOCKED_SCIENTIFIC_REVIEW"


class IncidentReason(StrEnum):
    INFRASTRUCTURE_RECOVERABLE = "INFRASTRUCTURE_RECOVERABLE"
    ENVIRONMENT_RECOVERABLE = "ENVIRONMENT_RECOVERABLE"
    JOURNAL_RESUME = "JOURNAL_RESUME"
    MECHANICAL_ATTRITION = "MECHANICAL_ATTRITION"
    INSTRUMENTATION_BUG = "INSTRUMENTATION_BUG"
    SPEC_IMPLEMENTATION_MISMATCH = "SPEC_IMPLEMENTATION_MISMATCH"
    MEASUREMENT_INTEGRITY_CONCERN = "MEASUREMENT_INTEGRITY_CONCERN"
    SCIENTIFIC_GATE_FAIL = "SCIENTIFIC_GATE_FAIL"
    SCIENTIFIC_DESIGN_DECISION_REQUIRED = "SCIENTIFIC_DESIGN_DECISION_REQUIRED"
    HOLDOUT_FIREWALL = "HOLDOUT_FIREWALL"


_TRANSITIONS: dict[GateState, frozenset[GateState]] = {
    GateState.PREPARE: frozenset(
        {GateState.PREMORTEM, GateState.BLOCKED_RECOVERABLE, GateState.BLOCKED_SCIENTIFIC_REVIEW}
    ),
    GateState.PREMORTEM: frozenset(
        {
            GateState.PROSPECTIVE_LOCK,
            GateState.BLOCKED_RECOVERABLE,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    GateState.PROSPECTIVE_LOCK: frozenset(
        {
            GateState.ENGINEERING,
            GateState.BLOCKED_RECOVERABLE,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    GateState.ENGINEERING: frozenset(
        {
            GateState.COLLECTION,
            GateState.OFFLINE_ANALYSIS,
            GateState.BLOCKED_RECOVERABLE,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    GateState.COLLECTION: frozenset(
        {
            GateState.OFFLINE_ANALYSIS,
            GateState.BLOCKED_RECOVERABLE,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    GateState.OFFLINE_ANALYSIS: frozenset(
        {
            GateState.FORENSIC_AUDIT,
            GateState.CLOSED,
            GateState.BLOCKED_RECOVERABLE,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    GateState.FORENSIC_AUDIT: frozenset(
        {GateState.CLOSED, GateState.BLOCKED_RECOVERABLE, GateState.BLOCKED_SCIENTIFIC_REVIEW}
    ),
    GateState.BLOCKED_RECOVERABLE: frozenset(
        {
            GateState.PREPARE,
            GateState.PREMORTEM,
            GateState.PROSPECTIVE_LOCK,
            GateState.ENGINEERING,
            GateState.COLLECTION,
            GateState.OFFLINE_ANALYSIS,
            GateState.FORENSIC_AUDIT,
            GateState.BLOCKED_SCIENTIFIC_REVIEW,
        }
    ),
    # Principal review must return through a lock/audit/close boundary; it may
    # not jump directly into collection.
    GateState.BLOCKED_SCIENTIFIC_REVIEW: frozenset(
        {
            GateState.PROSPECTIVE_LOCK,
            GateState.OFFLINE_ANALYSIS,
            GateState.FORENSIC_AUDIT,
            GateState.CLOSED,
        }
    ),
    GateState.CLOSED: frozenset(),
}

_ALWAYS_A = {
    IncidentReason.INFRASTRUCTURE_RECOVERABLE,
    IncidentReason.ENVIRONMENT_RECOVERABLE,
    IncidentReason.JOURNAL_RESUME,
}
_AMENDABLE = {
    IncidentReason.MECHANICAL_ATTRITION,
    IncidentReason.INSTRUMENTATION_BUG,
    IncidentReason.SPEC_IMPLEMENTATION_MISMATCH,
    IncidentReason.MEASUREMENT_INTEGRITY_CONCERN,
}
_ALWAYS_D = {
    IncidentReason.SCIENTIFIC_GATE_FAIL,
    IncidentReason.SCIENTIFIC_DESIGN_DECISION_REQUIRED,
    IncidentReason.HOLDOUT_FIREWALL,
}


def validate_transition(source: GateState, target: GateState) -> None:
    """Reject an invalid prospective transition; never update stored state."""

    if target not in _TRANSITIONS[source]:
        raise ValueError(f"invalid gate transition: {source.value} -> {target.value}")


def allowed_targets(source: GateState) -> frozenset[GateState]:
    """Return the immutable prospective transition set for validation tooling."""

    return _TRANSITIONS[source]


def classify_incident(reason: IncidentReason, *, affected_outcomes_observed: bool) -> ActionClass:
    """Classify an incident by timing without consulting scientific outcomes."""

    if reason in _ALWAYS_A:
        return ActionClass.AUTONOMOUS_ENGINEERING_RECOVERY
    if reason in _AMENDABLE:
        return (
            ActionClass.OFFLINE_POST_OUTCOME_FORENSIC_REPAIR
            if affected_outcomes_observed
            else ActionClass.PROSPECTIVE_INSTRUMENT_AMENDMENT
        )
    if reason in _ALWAYS_D:
        return ActionClass.PRINCIPAL_RESEARCHER_REQUIRED
    raise AssertionError(f"unclassified incident reason: {reason}")


@dataclass(frozen=True)
class ActionRequest:
    action_class: ActionClass
    affected_outcomes_observed: bool
    scientific_semantics_changed: bool = False
    hypothesis_changed: bool = False
    estimand_changed: bool = False
    model_or_benchmark_changed: bool = False
    scientific_thresholds_changed: bool = False
    outcome_based_selection: bool = False
    amendment_locked_before_new_outcomes: bool = False
    collect_additional_model_outputs: bool = False
    authorized_by_frozen_decision_tree: bool = False
    principal_researcher_approved: bool = False


def validate_action(request: ActionRequest) -> None:
    """Enforce the autonomy boundary for a proposed future action."""

    scientific_changes = (
        request.scientific_semantics_changed
        or request.hypothesis_changed
        or request.estimand_changed
        or request.model_or_benchmark_changed
        or request.scientific_thresholds_changed
        or request.outcome_based_selection
    )
    if request.action_class is ActionClass.AUTONOMOUS_ENGINEERING_RECOVERY:
        if scientific_changes:
            raise ValueError("Class A recovery cannot change scientific semantics")
    elif request.action_class is ActionClass.PROSPECTIVE_INSTRUMENT_AMENDMENT:
        if request.affected_outcomes_observed:
            raise ValueError("Class B amendment must precede affected outcomes")
        if scientific_changes:
            raise ValueError(
                "Class B must preserve hypothesis, estimand, model, benchmark, thresholds"
            )
        if not request.amendment_locked_before_new_outcomes:
            raise ValueError("Class B amendment must be prospectively locked")
    elif request.action_class is ActionClass.OFFLINE_POST_OUTCOME_FORENSIC_REPAIR:
        if not request.affected_outcomes_observed:
            raise ValueError("Class C is reserved for post-outcome forensic work")
        if scientific_changes:
            raise ValueError(
                "Class C reanalysis must remain condition-symmetric and outcome-independent"
            )
        if (
            request.collect_additional_model_outputs
            and not request.authorized_by_frozen_decision_tree
        ):
            raise ValueError("Class C cannot collect new outputs without frozen authorization")
    elif request.action_class is ActionClass.PRINCIPAL_RESEARCHER_REQUIRED:
        if not request.principal_researcher_approved:
            raise ValueError("Class D action requires principal researcher approval")
