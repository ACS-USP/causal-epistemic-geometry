"""CPU-only prompt provenance primitives for the aborted Q2 V3 freeze.

The two templates in this module are historical reconstructions.  Keeping them
separately named prevents a namespaced legacy digest from being mistaken for a
raw SHA-256 digest again.  This module performs no inference and contains no
behavioral outcomes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

LEGACY_TEMPLATE_VERSION = "external-benchmark-cruxeval-prompt-v1"
CURRENT_TEMPLATE_VERSION = "gate7-cruxeval-semantic-task-prompt-v1"
GATE7_CANONICAL_TEMPLATE_VERSION = CURRENT_TEMPLATE_VERSION
LEGACY_HASH_SCHEMA = "stable-digest-v1:EXTERNAL-PROMPT"
RAW_HASH_SCHEMA = "sha256-utf8-raw-v1"
PROPOSED_CONTRACT_SCHEMA = "q2-v3-prompt-provenance-contract-v2-candidate"
AMENDMENT1_PROVENANCE_SCHEMA = "q2-v3-amendment1-prompt-provenance-v1"
AMENDMENT1_TASK_NAMESPACE = "Q2-V3-CRUXEVAL"


def legacy_task_prompt(code: str, value: str) -> str:
    """Reconstruct the historical external-qualification CRUXEval prompt."""

    return (
        "Solve the following code-output prediction problem.\n\n"
        "Python function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Reason carefully, then end with exactly one line in the form "
        "FINAL: <the exact Python output>. Do not add text after FINAL."
    )


def current_task_prompt(code: str, value: str) -> str:
    """Reconstruct the Gate-7-and-later CRUXEval semantic task prompt."""

    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def canonical_q2_v3_task_prompt(code: str, value: str) -> str:
    """Return the principal-authorized Gate-7 prompt for every Q2 V3 purpose."""

    return current_task_prompt(code, value)


def raw_utf8_sha256(text: str) -> str:
    """Hash exact UTF-8 bytes with no namespace or normalization."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_external_prompt_digest(text: str) -> str:
    """Reproduce ``stable_digest('EXTERNAL-PROMPT', text)`` exactly."""

    return hashlib.sha256(b"EXTERNAL-PROMPT\x1f" + text.encode("utf-8")).hexdigest()


def source_record_sha256(*, item_id: str, code: str, value: str, reference: str) -> str:
    """Hash the exact public source fields under an explicit canonical JSON schema."""

    payload = {
        "schema_version": "q2-v3-cruxeval-source-record-v1",
        "item_id": item_id,
        "code": code,
        "input": value,
        "output": reference,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def amendment1_contract(
    *,
    item_id: str,
    purpose: str,
    code: str,
    value: str,
    reference: str,
    official_index: int,
    dataset_repo: str,
    dataset_revision: str,
    historical_prompt_hash: str,
    historical_prompt_schema: str,
) -> dict[str, Any]:
    """Build the authorized Amendment-1 exact-byte provenance record."""

    prompt = canonical_q2_v3_task_prompt(code, value)
    prompt_bytes = prompt.encode("utf-8")
    payload: dict[str, Any] = {
        "provenance_schema_version": AMENDMENT1_PROVENANCE_SCHEMA,
        "task_namespace": AMENDMENT1_TASK_NAMESPACE,
        "item_id": item_id,
        "purpose": purpose,
        "template_version": GATE7_CANONICAL_TEMPLATE_VERSION,
        "model_visible_prompt": {
            "encoding": "UTF-8",
            "unicode_normalization": "NONE",
            "newline_convention": "LF",
            "utf8_base64": base64.b64encode(prompt_bytes).decode("ascii"),
            "byte_length": len(prompt_bytes),
            "prompt_bytes_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        },
        "reference": {
            "identity": "CRUXEval.output",
            "utf8_sha256": raw_utf8_sha256(reference),
        },
        "source_artifact": {
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "official_index": official_index,
            "source_record_sha256": source_record_sha256(
                item_id=item_id,
                code=code,
                value=value,
                reference=reference,
            ),
        },
        "historical_freeze_provenance": {
            "prompt_hash": historical_prompt_hash,
            "classified_schema": historical_prompt_schema,
        },
    }
    digest_payload = {
        "provenance_schema_version": payload["provenance_schema_version"],
        "template_version": payload["template_version"],
        "purpose": purpose,
        "item_id": item_id,
        "exact_model_visible_utf8_base64": payload["model_visible_prompt"]["utf8_base64"],
    }
    encoded = json.dumps(
        digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload["provenance_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def canonical_contract(
    *,
    item_id: str,
    purpose: str,
    prompt: str,
    prompt_template_version: str,
    reference: str,
    dataset_repo: str,
    dataset_revision: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build the proposed, explicitly versioned provenance envelope.

    This is a migration candidate, not an authorized Q2 V3 scientific prompt
    choice.  The exact bytes are embedded as base64 so round trips do not rely
    on JSON newline or Unicode behavior.
    """

    prompt_bytes = prompt.encode("utf-8")
    system_bytes = system_prompt.encode("utf-8") if system_prompt is not None else None
    payload: dict[str, Any] = {
        "schema_version": PROPOSED_CONTRACT_SCHEMA,
        "task_namespace": "Q2-V3-CRUXEVAL",
        "purpose": purpose,
        "item_id": item_id,
        "source": {
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "reference_sha256": raw_utf8_sha256(reference),
        },
        "encoding": "UTF-8",
        "unicode_normalization": "NONE",
        "line_ending_policy": "PRESERVE_EXACT_BYTES",
        "prompt_template_version": prompt_template_version,
        "user_prompt_utf8_base64": base64.b64encode(prompt_bytes).decode("ascii"),
        "user_prompt_bytes_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "system_prompt_utf8_base64": (
            base64.b64encode(system_bytes).decode("ascii")
            if system_bytes is not None
            else None
        ),
        "system_prompt_bytes_sha256": (
            hashlib.sha256(system_bytes).hexdigest() if system_bytes is not None else None
        ),
        "rendering": {
            "mode": "chat",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "enable_thinking": False,
            "rendered_prompt_bytes_sha256": None,
            "status": "REMOTE_RENDER_MUST_MATCH_BEFORE_INFERENCE",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def decode_contract_prompt(contract: dict[str, Any]) -> tuple[bytes, bytes | None]:
    """Round-trip exact user/system bytes from a candidate contract."""

    user = base64.b64decode(contract["user_prompt_utf8_base64"], validate=True)
    encoded_system = contract.get("system_prompt_utf8_base64")
    system = base64.b64decode(encoded_system, validate=True) if encoded_system else None
    return user, system


__all__ = [
    "AMENDMENT1_PROVENANCE_SCHEMA",
    "AMENDMENT1_TASK_NAMESPACE",
    "CURRENT_TEMPLATE_VERSION",
    "GATE7_CANONICAL_TEMPLATE_VERSION",
    "LEGACY_HASH_SCHEMA",
    "LEGACY_TEMPLATE_VERSION",
    "PROPOSED_CONTRACT_SCHEMA",
    "RAW_HASH_SCHEMA",
    "amendment1_contract",
    "canonical_contract",
    "canonical_q2_v3_task_prompt",
    "current_task_prompt",
    "decode_contract_prompt",
    "legacy_external_prompt_digest",
    "legacy_task_prompt",
    "raw_utf8_sha256",
    "source_record_sha256",
]
