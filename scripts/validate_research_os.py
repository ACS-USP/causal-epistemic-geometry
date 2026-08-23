#!/usr/bin/env python3
"""Validate Research OS v1 policy, contracts, and environment specification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.research.governance import (  # noqa: E402
    ActionClass,
    GateState,
    IncidentReason,
    allowed_targets,
    classify_incident,
)
from epistemic_geometry.research.preflight import load_environment_spec  # noqa: E402


def _contract_errors(template_name: str, schema_name: str) -> list[str]:
    errors: list[str] = []
    template = json.loads(
        (ROOT / "templates" / "research" / template_name).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / "schemas" / "research" / schema_name).read_text(encoding="utf-8")
    )
    missing = set(schema["required"]) - set(template)
    if missing:
        errors.append(f"{template_name} lacks schema-required keys: {sorted(missing)}")
    required_questions = set(schema["properties"]["questions"]["required"])
    if required_questions != set(template.get("questions", {})):
        errors.append(f"{template_name} questions do not exactly match its schema")
    if schema["properties"]["contract"]["const"] != template.get("contract"):
        errors.append(f"{template_name} contract discriminator does not match schema")
    return errors


def main() -> int:
    errors: list[str] = []
    policy = yaml.safe_load((ROOT / "research_policy.yaml").read_text(encoding="utf-8"))
    if set(policy.get("states", [])) != {state.value for state in GateState}:
        errors.append("research policy state set differs from typed GateState")
    if set(policy.get("incident_reasons", {})) != {reason.value for reason in IncidentReason}:
        errors.append("research policy incident set differs from typed IncidentReason")
    for state in GateState:
        expected = {target.value for target in allowed_targets(state)}
        if set(policy.get("transitions", {}).get(state.value, [])) != expected:
            errors.append(f"research policy transitions differ for {state.value}")
    for reason in IncidentReason:
        record = policy.get("incident_reasons", {}).get(reason.value, {})
        before = classify_incident(reason, affected_outcomes_observed=False).value
        after = classify_incident(reason, affected_outcomes_observed=True).value
        if record.get("pre_outcome_class") != before or record.get("post_outcome_class") != after:
            errors.append(f"research policy incident classification differs for {reason.value}")
    if set(policy.get("action_classes", {})) != {item.value for item in ActionClass}:
        errors.append("research policy action classes differ from typed ActionClass")

    errors.extend(_contract_errors("PREMORTEM.json", "premortem.schema.json"))
    errors.extend(_contract_errors("CLOSEOUT_AUDIT.json", "closeout_audit.schema.json"))
    for profile in ("CORE_QWEN", "CORE_MINISTRAL3", "RFM_COMPAT"):
        try:
            load_environment_spec(ROOT / "remote_environment.yaml", profile)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("research OS: valid (policy, lifecycle, contracts, environment profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
