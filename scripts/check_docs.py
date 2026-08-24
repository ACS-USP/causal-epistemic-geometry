#!/usr/bin/env python3
"""Check canonical documentation links, classification, and state boundaries."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "DOCUMENT_INDEX.md"
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _local_target(source: Path, raw: str) -> Path | None:
    target = raw.split("#", 1)[0]
    if not target or "://" in target or target.startswith("mailto:"):
        return None
    return (source.parent / target).resolve()


def main() -> int:
    state = yaml.safe_load((ROOT / "project_state.yaml").read_text(encoding="utf-8"))
    errors: list[str] = []
    canonical = state.get("canonical_documents", {})
    for name, relative in canonical.items():
        if not (ROOT / relative).exists():
            errors.append(f"missing canonical document {name}: {relative}")
    sources = [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]
    for source in sources:
        content = source.read_text(encoding="utf-8")
        for raw in LINK.findall(content):
            target = _local_target(source, raw)
            if target is not None and not target.exists():
                errors.append(
                    f"broken local link in {source.relative_to(ROOT)}: {raw}"
                )
    index_text = INDEX.read_text(encoding="utf-8")
    for document in sorted(DOCS.glob("*.md")):
        relative = str(document.relative_to(ROOT))
        if relative not in index_text:
            errors.append(f"unclassified document: {relative}")
    current = state.get("current", {})
    obsolete_actions = ("completion-cap diagnostic", "stage b run", "resume v1")
    next_action = str(current.get("next_authorized_action", "")).lower()
    if any(value in next_action for value in obsolete_actions):
        errors.append("project_state contains an obsolete next action")
    dockerfile = (ROOT / "infra" / "evalplus_sandbox" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    sandbox_status = state["infrastructure"]["dense_code_sandbox"]["status"]
    if "REPLACE_WITH_REVIEWED_DIGEST" in dockerfile and sandbox_status == "PRODUCTION_READY":
        errors.append("placeholder Docker digest cannot be production-ready")
    if state["scientific_firewall"]["confirmatory_holdout"] not in {
        "UNTOUCHED",
        "SEALED_ASSIGNED_UNACCESSED",
        "CONSUMED_CONFIRMATORY_CLOSED",
    }:
        errors.append("confirmatory holdout firewall has an unknown state")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"documentation: valid ({len(sources)} Markdown files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
