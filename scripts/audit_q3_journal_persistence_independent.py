import collections
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
j = next(base.glob("*-journal.jsonl"))
s = next(base.glob("*-Q3_FRESH_QUALIFICATION_SCHEDULE.json"))
seal = next(base.glob("*-COLLECTION_COMPLETE_SEAL.json"))
p = next(base.glob("*-PREOPEN_SEAL.json"))


def key(r):
    return (r["family_id"], r["condition"], r["rollout_index"])


plan = json.loads(s.read_bytes())["rows"]
expected = {key(r): (i, r) for i, r in enumerate(plan)}
counts = collections.Counter()
bad = 0
identities = set()
token_total = 0
md5lines = []
sha = hashlib.sha256()
physical = 0
newline = 0
with j.open("rb") as f:
    for line in f:
        sha.update(line)
        physical += 1
        newline += line.endswith(b"\n")
        w = json.loads(line)
        r = w["row"]
        k = key(r)
        counts[k] += 1
        ident = w["identity"]
        identities.add(json.dumps(ident, sort_keys=True))
        frozen = expected.get(k)
        bad += int(
            frozen is None or frozen[0] != r["schedule_index"] or frozen[1]["seed"] != r["seed"]
        )
        bad += int(w["key"] != list(k))
        bad += int(ident["experiment"] != "Q3_FRESH_INSTRUMENT_QUALIFICATION")
        bad += int(ident["code_commit"] != "dda4f6b40d371eaa93cde575838451d98b953fc6")
        bad += int(r["model_revision"] != "b968826d9c46dd6066d109eabc6255188de91218")
        token_total += r["generated_token_count"]
pre = json.loads(p.read_bytes())
complete = json.loads(seal.read_bytes())
log = next(base.glob("*-collection.log")).read_bytes()
progress = []
for line in log.splitlines():
    try:
        v = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        continue
    if isinstance(v, dict) and "completed" in v:
        progress.append({k: v[k] for k in ("completed", "pending", "status", "expected") if k in v})
result = dict(
    independent_reader="stdlib_streaming_no_shared_audit_imports",
    journal_sha256=sha.hexdigest(),
    physical_lines=physical,
    newline_terminated_lines=newline,
    unique_keys=len(counts),
    duplicate_keys=sum(v - 1 for v in counts.values()),
    missing_schedule_indices=sorted(expected[k][0] for k in expected.keys() - counts.keys()),
    unexpected=len(counts.keys() - expected.keys()),
    identity_variants=len(identities),
    validation_errors=bad,
    schedule_hash_matches_preopen=hashlib.sha256(s.read_bytes()).hexdigest()
    == pre["frozen"]["schedule_sha256"],
    seal_hash_matches=sha.hexdigest() == complete["journal_sha256"],
    progress_events=len(progress),
    last_progress_events=progress[-3:],
    error_keyword_counts={
        word: log.count(word.encode())
        for word in (
            "No space left",
            "Input/output error",
            "BrokenPipeError",
            "MemoryError",
            "KeyboardInterrupt",
            "Traceback",
        )
    },
    raw_content_emitted=False,
    scoring_performed=False,
)
print(json.dumps(result, indent=2))
