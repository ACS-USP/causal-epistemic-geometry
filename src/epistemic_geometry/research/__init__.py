"""Research-governance and execution-reliability contracts.

The package is deliberately independent of experiment runners.  It validates
prospective records and local infrastructure without mutating historical state
or performing model inference.
"""

from .governance import (
    ActionClass,
    GateState,
    IncidentReason,
    classify_incident,
    validate_action,
    validate_transition,
)

__all__ = [
    "ActionClass",
    "GateState",
    "IncidentReason",
    "classify_incident",
    "validate_action",
    "validate_transition",
]
