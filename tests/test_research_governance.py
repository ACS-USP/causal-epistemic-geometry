from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from epistemic_geometry.research.governance import (
    ActionClass,
    ActionRequest,
    GateState,
    IncidentReason,
    allowed_targets,
    classify_incident,
    validate_action,
    validate_transition,
)

ROOT = Path(__file__).resolve().parents[1]


def test_machine_policy_matches_typed_states_reasons_and_transitions() -> None:
    policy = yaml.safe_load((ROOT / "research_policy.yaml").read_text(encoding="utf-8"))
    assert set(policy["states"]) == {state.value for state in GateState}
    assert set(policy["incident_reasons"]) == {reason.value for reason in IncidentReason}
    for source in GateState:
        assert set(policy["transitions"][source.value]) == {
            target.value for target in allowed_targets(source)
        }


def test_transition_validator_blocks_collection_jump_and_closed_reopen() -> None:
    validate_transition(GateState.PREPARE, GateState.PREMORTEM)
    validate_transition(GateState.BLOCKED_SCIENTIFIC_REVIEW, GateState.PREMORTEM)
    validate_transition(GateState.BLOCKED_SCIENTIFIC_REVIEW, GateState.PROSPECTIVE_LOCK)
    with pytest.raises(ValueError, match="invalid gate transition"):
        validate_transition(GateState.PREMORTEM, GateState.COLLECTION)
    with pytest.raises(ValueError, match="invalid gate transition"):
        validate_transition(GateState.CLOSED, GateState.PREPARE)
    for forbidden in (
        GateState.COLLECTION,
        GateState.OFFLINE_ANALYSIS,
        GateState.FORENSIC_AUDIT,
        GateState.CLOSED,
    ):
        with pytest.raises(ValueError, match="invalid gate transition"):
            validate_transition(GateState.BLOCKED_SCIENTIFIC_REVIEW, forbidden)


def test_incident_timing_routes_amendment_to_b_before_and_c_after_outcomes() -> None:
    assert classify_incident(
        IncidentReason.INSTRUMENTATION_BUG, affected_outcomes_observed=False
    ) is ActionClass.PROSPECTIVE_INSTRUMENT_AMENDMENT
    assert classify_incident(
        IncidentReason.INSTRUMENTATION_BUG, affected_outcomes_observed=True
    ) is ActionClass.OFFLINE_POST_OUTCOME_FORENSIC_REPAIR
    assert classify_incident(
        IncidentReason.HOLDOUT_FIREWALL, affected_outcomes_observed=False
    ) is ActionClass.PRINCIPAL_RESEARCHER_REQUIRED


def test_autonomy_action_invariants_are_fail_closed() -> None:
    validate_action(
        ActionRequest(
            ActionClass.AUTONOMOUS_ENGINEERING_RECOVERY,
            affected_outcomes_observed=True,
        )
    )
    with pytest.raises(ValueError, match="scientific semantics"):
        validate_action(
            ActionRequest(
                ActionClass.AUTONOMOUS_ENGINEERING_RECOVERY,
                affected_outcomes_observed=False,
                scientific_thresholds_changed=True,
            )
        )
    validate_action(
        ActionRequest(
            ActionClass.PROSPECTIVE_INSTRUMENT_AMENDMENT,
            affected_outcomes_observed=False,
            amendment_locked_before_new_outcomes=True,
        )
    )
    with pytest.raises(ValueError, match="frozen authorization"):
        validate_action(
            ActionRequest(
                ActionClass.OFFLINE_POST_OUTCOME_FORENSIC_REPAIR,
                affected_outcomes_observed=True,
                collect_additional_model_outputs=True,
            )
        )
    with pytest.raises(ValueError, match="principal researcher"):
        validate_action(
            ActionRequest(
                ActionClass.PRINCIPAL_RESEARCHER_REQUIRED,
                affected_outcomes_observed=False,
            )
        )


def test_contract_templates_and_schemas_are_machine_readable() -> None:
    pairs = (
        ("PREMORTEM.json", "premortem.schema.json", "PREMORTEM"),
        ("CLOSEOUT_AUDIT.json", "closeout_audit.schema.json", "CLOSEOUT_AUDIT"),
    )
    for template_name, schema_name, contract in pairs:
        template = json.loads(
            (ROOT / "templates" / "research" / template_name).read_text(encoding="utf-8")
        )
        schema = json.loads(
            (ROOT / "schemas" / "research" / schema_name).read_text(encoding="utf-8")
        )
        assert template["contract"] == contract
        assert schema["properties"]["contract"]["const"] == contract
        assert set(schema["required"]) <= set(template)
        required_questions = set(schema["properties"]["questions"]["required"])
        assert required_questions == set(template["questions"])
