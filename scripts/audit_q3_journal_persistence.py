"""Read-only, output-opaque incident audit. Never instantiate a journal class."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(journal, schedule, seal):
    raw = journal.read_bytes()
    schedule_bytes = schedule.read_bytes()
    planned = json.loads(schedule_bytes)["rows"]
    complete_seal = json.loads(seal.read_bytes())
    fields = ("family_id", "condition", "rollout_index")

    def key(row):
        return tuple(row[f] for f in fields)

    expected = {key(r): (i, r) for i, r in enumerate(planned)}
    seen, malformed, errors, duplicate = {}, [], [], []
    identity_hashes, identities, durations = set(), set(), []
    total_tokens = 0
    prefix = b""
    for physical, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            wrapper = json.loads(line)
            row = wrapper["row"]
            k = key(row)
            if list(k) != wrapper["key"] or wrapper["key_fields"] != list(fields):
                errors.append([physical, "KEY_BINDING"])
            if wrapper["version"] != "research-os-jsonl-v1":
                errors.append([physical, "VERSION"])
            identity = wrapper["identity"]
            canon = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
            identity_expected = hashlib.sha256(
                ("RESEARCH-OS-JOURNAL\x1f" + canon).encode()
            ).hexdigest()
            if identity_expected != wrapper["identity_hash"]:
                errors.append([physical, "IDENTITY_HASH"])
            identities.add(canon)
            identity_hashes.add(wrapper["identity_hash"])
            if k in seen:
                duplicate.append(dict(key=k, identical=(seen[k][1] == wrapper)))
            else:
                seen[k] = (physical, wrapper)
            if k not in expected:
                errors.append([physical, "UNEXPECTED"])
            else:
                i, frozen = expected[k]
                if row["seed"] != frozen["seed"] or row["schedule_index"] != i:
                    errors.append([physical, "SEED_OR_INDEX"])
                if any(row.get(f) != v for f, v in frozen.items()):
                    errors.append([physical, "SCHEDULE_FIELD"])
            if identity["schedule_sha256"] != hashlib.sha256(schedule_bytes).hexdigest():
                errors.append([physical, "SCHEDULE_HASH"])
            total_tokens += row["generated_token_count"]
            durations.append(row["elapsed_seconds"])
            if physical <= 35:
                prefix += line
        except (ValueError, KeyError, TypeError):
            malformed.append(physical)
    missing = []
    for k, (i, row) in expected.items():
        if k not in seen:
            missing.append(
                dict(
                    family_id=k[0],
                    condition=k[1],
                    rollout_index=k[2],
                    schedule_index=i,
                    schedule_position=i + 1,
                    seed=row["seed"],
                    classification="UNRESOLVED",
                    original_record_recovered=False,
                )
            )
    return dict(
        schema_version="q3-persistence-incident-readonly-v1",
        journal_sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        physical_lines=len(raw.splitlines()),
        newline_count=raw.count(b"\n"),
        ends_with_newline=raw.endswith(b"\n"),
        complete_wrappers=len(seen) + len(duplicate),
        unique_keys=len(seen),
        duplicate_records=duplicate,
        malformed_lines=malformed,
        validation_errors=errors,
        unexpected_keys=len(set(seen) - set(expected)),
        missing_count=len(missing),
        missing=missing,
        condition_missing_counts=dict(Counter(r["condition"] for r in missing)),
        rollout_missing_counts=dict(Counter(r["rollout_index"] for r in missing)),
        identity_variants=len(identities),
        identity_hashes=sorted(identity_hashes),
        generated_tokens_durable=total_tokens,
        generated_tokens_seal=complete_seal["generated_tokens"],
        seal_completed=complete_seal["completed"],
        seal_hash_matches=complete_seal["journal_sha256"] == hashlib.sha256(raw).hexdigest(),
        seal_size_matches=complete_seal["journal_bytes"] == len(raw),
        first_35_sha256=hashlib.sha256(prefix).hexdigest(),
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        collection_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
        raw_output_displayed=False,
        correctness_inspected=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("seal", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.journal, args.schedule, args.seal), indent=2))
