# Q3.4 Journal Persistence and Completion-Seal Incident Review

Status: **PERSISTENCE_INTEGRITY_BLOCKED — SCORING_NOT_STARTED**.
This is an operational forensic incident review, not a qualification result.
No missing row is converted into an incorrect answer, and no scientific gate
is evaluated on the incomplete collection.

## 1. Preserved state

The campaign collector had exited before this audit. The campaign process
search returned no collector, scorer or campaign tmux process. The current
user's open-file inspection found no descriptor for the original journal or
collection seal. System-wide visibility is limited: lsof reported inaccessible
unrelated kernel/container mounts; this is not a claim to a privileged full-host
process audit. No process was killed and no original was removed or renamed.
The heartbeat with automatic continuation/scoring was deleted at the incident
pause and was not recreated.

Ten initial evidence files were copied privately, byte-for-byte, with restrictive
permissions: journal, original collection seal, preopen seal, both operational
logs, engine-validation artifact, schedule, execution lock, executed collector
source, and shared journal source. Three bytecode files and a hash-authenticated
35-record pause prefix were additionally preserved. Original source hashes,
mtime and ctime were rechecked unchanged after the audit. Read access may update
atime on the filesystem; no content, mtime, ctime or inode mutation was used for
inspection. The full resolved paths, device/inode numbers, ownership, modes,
sizes and nanosecond timestamps remain in private manifests, not public Git.

- Initial private preservation manifest SHA-256:
  `1f31b5fc16dfd7126c5c68ec67ba5401a1cd03a6be64c6c8c982d730fb20d2a5`
- Supplemental private preservation manifest SHA-256:
  `b4a520f0e54457851ecf495cf3ad634c589d2556f6b87cab60071a8ac2de0ba0`
- Public content-free inventory: [PRESERVATION_PUBLIC_HASHES.json](PRESERVATION_PUBLIC_HASHES.json).

No campaign `.truncated.*`, `.recovered`, `.tmp`, alternate journal or backup
was found in the scoped campaign, local worktree or temporary-file inventory.
Synthetic test journals and unrelated historical journals were excluded from
recovery. No claim is made about unavailable host snapshots or unmounted
backups. The 35-record prefix exactly reproduces the prior pause hash
`ec838d88494adc856deefca2ab35e2db230bdfdb84b5b3975108bcca4d56c154`;
it contains none of the ten missing records.

## 2. Disk audit versus the memory-derived seal

Two independent standard-library readers, operating on preserved copies and
without constructing `CrashSafeJournal`, agree:

| Quantity | Durable journal | Original seal |
|---|---:|---:|
| Complete JSON wrappers / unique keys | 5,990 | 6,000 |
| Missing expected keys | 10 | 0 |
| Duplicates / unexpected keys | 0 / 0 | 0 / 0 |
| Malformed / partial final records | 0 / 0 | not reread |
| Generated-token aggregate | 657,334 | 658,254 |
| Bytes | 80,032,031 | 80,032,031 |

All 5,990 durable records match their frozen schedule fields, seeds, schedule
indices, wrapper keys and experiment identity. There is one identity/hash
variant. No records have been semantically evaluated.

Journal SHA-256:
`ae65de79f99f6ef12b423c6e3604b0afea952b9d6cf835bb1668786cf15ed811`.

Original collection-seal SHA-256:
`2b1c848af719d8be923949abc70a9fcdc809121166d4f72cfaf4faea61756e1c`.

The original seal's hash and byte count correctly identify the **incomplete**
file. They do not authenticate its stated count of 6,000. Its operational token
aggregate exceeds the durable aggregate by 920. The original seal is preserved
unchanged as incident evidence and is not accepted as proof of completion.

Evidence: [primary read-only audit](READONLY_JOURNAL_AUDIT.json) and
[independent read-only audit](INDEPENDENT_READONLY_AUDIT.json).

## 3. Executed code and causal reconstruction

The remote checkout remained clean at
`dda4f6b40d371eaa93cde575838451d98b953fc6`.
Collector source SHA-256:
`31789b3d159303f23c73d256cc17ade1cfdbc7e0a8fc9c9cc106157de89bbf9b`.
Journal-library source SHA-256:
`5c29533c7489077a30f0c7d8e67f10df3c3379403b59d0560226ed5d24f307c4`.
The collector puts its own `src` first on the import path. Preserved cached
code for the collector, reliability and reproducibility modules matches the
corresponding compiled source recursively, including bytecode, constants,
names, flags and line tables. Raw marshal serialization differed because code
object serialization can encode object references differently; code-object
equality and normalized structural comparison both pass. See
[bytecode comparison](BYTECODE_STRUCTURAL_COMPARISON.json). The process has
exited, so this is on-disk code/cache and launch-provenance evidence, not a live
`sys.modules` or memory dump.

**Demonstrated completion defect:** `collect()` validates
`list(journal.rows.values())`, an in-memory mapping, then hashes the file
separately. Thus successful in-memory coverage can produce a seal that reports
6,000 but hashes a file with 5,990. The persisted progress log contains the
6,000/0 progress event and final `COLLECTION_COMPLETE_RAW_UNSCORED` event.
This establishes the flaw in completeness verification.

**Demonstrated shared-library vulnerability:** `_load()` can normalize or
quarantine a final line and replace the journal from its earlier byte snapshot.
It holds no interprocess lock. `append()` separately opens the path, writes,
flushes and fsyncs, then updates memory. A second process can subsequently
replace the path with an older snapshot. Flush/fsync makes a particular write
durable; it does not prevent a later replacement of its directory entry.

A deterministic synthetic race reproduces two acknowledged, fsynced appends
being lost after the old constructor normalizes a valid final JSON record
captured before its newline. This path leaves no `.truncated.*` file. It proves
possibility and explains why absence of quarantine does not rule out a race.
It does **not** prove that a monitor performed that operation in this campaign.

| Hypothesis/check | Evidence and limit |
|---|---|
| A. Status/helper constructed the old journal | Known recent monitors used counts/jq; no constructor invocation tied to a loss was recovered. Accessible session records did not supply a complete executable tool-call history. Unresolved. |
| B. `_load()` raced a writer | Mechanically possible and reproduced synthetically; no campaign syscall/rename trace proves it occurred. |
| C. Writer/recovery mutual exclusion | Absent in frozen library; verified in source and cached code. |
| D. Truncation/replacement/sync/restore | Missing interior records require a persistence discrepancy; the responsible operation is not identified. No campaign recovery artifact was found. |
| E. Different paths/filesystems | Current resolved journal, preserved source and seal path agree; filesystem is ext4. No scoped campaign symlink was found. Historical inode continuity was not recorded. |
| F. Multiple writers / stale closeout copy | No current writer exists. Final hash matches the current path; no second writer was established historically. |
| G. Write failure / disk full | No corresponding error keyword in the preserved operational log, and current storage has ample free space. No kernel-level historical I/O log was obtained. |

The physical deletion/replacement cause remains **UNRESOLVED**. It is not
attributed to buffering, an identified monitor, or a specific external command.

## 4. Missing-key inventory and recoverability

Indices below are zero-based, exactly as stored in the schedule; human ordinal
position is index + 1. Full family IDs, exact integer seeds and bindings are in
the [machine-readable audit](READONLY_JOURNAL_AUDIT.json).

| Schedule index | Family ID suffix | Condition | Rollout | Category |
|---:|---|---|---:|---|
| 1380 | 7bacb5c3fb464dff2dd7 | V4_DIRECTION_31_MEDIUM | 0 | UNRESOLVED |
| 1381 | 7bacb5c3fb464dff2dd7 | V4_DIRECTION_10_MEDIUM | 0 | UNRESOLVED |
| 1565 | abe204a74569c06a79da | Q2_OOS_V2_DIRECTION_03_MEDIUM | 0 | UNRESOLVED |
| 1566 | abe204a74569c06a79da | Q2_OOS_V2_DIRECTION_16_MEDIUM | 0 | UNRESOLVED |
| 1855 | a21e6265e3c5bcf8335e | Q2_OOS_V2_DIRECTION_03_MEDIUM | 1 | UNRESOLVED |
| 3043 | 0f9c76fb0d9d1525b5c8 | Q2_OOS_V2_DIRECTION_13_MEDIUM | 0 | UNRESOLVED |
| 4481 | 48ed43d3005e382d47c8 | V4_DIRECTION_10_MEDIUM | 0 | UNRESOLVED |
| 4482 | 48ed43d3005e382d47c8 | V4_DIRECTION_32_MEDIUM | 0 | UNRESOLVED |
| 4678 | df31ea3344f62761f10a | V4_DIRECTION_02_MEDIUM | 1 | UNRESOLVED |
| 5989 | fea4d5cc44c790d9aae9 | ONLINE_ROUTED | 0 | UNRESOLVED |

The ten records span seven families and seven separated index intervals:
three adjacent pairs and four singletons. Eight are rollout 0 and two rollout
1. They are all after the preserved 35-record pause and are not the interrupted
36th attempt. Later schedule indices are durable even after the final gap.

The frozen loop and successful memory-derived closeout strongly suggest all
ten generations completed and entered the writer's in-memory mapping. However,
there is no surviving per-key start/end/append receipt, original output or
writer memory dump to authenticate that chain individually. Their conservative
category is therefore UNRESOLVED, with COMPLETED_OUTPUT_LOST as a strongly
supported possibility, not NEVER_EXECUTED. Log progress events have no per-key
timestamps; monitoring times cannot be aligned causally to individual losses.

- ORIGINAL_RECORD_RECOVERABLE: 0 missing records.
- NEVER_EXECUTED: 0 established.
- INTERRUPTED_BEFORE_PERSISTENCE: 0 established for these ten keys.
- COMPLETED_OUTPUT_LOST: not independently established per key; ten at risk.
- UNRESOLVED: 10.

No recovered 6,000-row candidate was constructed. No synthetic output or
metadata was fabricated. The authenticated pause prefix is preservation of
already-present evidence, not recovery of a missing record.

## 5. Isolated correction and tests

Branch: `codex/q3-journal-persistence-incident`, based on the closed focal-audit
commit `8cc874406b8d3cb6fe86137952e15e2db2d35a1c`.
No code was deployed to the original execution checkout.

The additive `durable_journal` module separates:

1. `read_status`: immutable byte snapshot, no constructor recovery; partial
   appends are reported without editing bytes/inode/content.
2. `SingleWriterJournal`: context-managed sidecar and data-file locks, strict
   resume, append/fsync, inode and external-mutation detection. An uncertain
   append raises an integrity exception, not a retryable generation exception.
3. `offline_tail_candidate`: explicit offline action under writer exclusion,
   separate candidate only, source and omitted-tail hashes, original untouched.
4. `seal`: persisted-file reread, wrapper/identity/schedule/seed validation,
   hash of those same bytes, no overwriting of an existing seal.

The Q3 collector in the isolated branch uses the new context before backend
construction. Completion totals come from validated persisted records. Its
generation-loop AST is exactly equal to the frozen loop; prompt, steering,
router, model, termination, generation seeds and scoring are unchanged.
Historical `reliability.py` and Q1/Q2 callers were not edited.

Locks are advisory. All cooperating status/writer/recovery/sealer tools must
use this contract; arbitrary external filesystem modification is detected,
not made impossible by advisory locking. A deployment must exclude old
recovering readers. This patch does not authorize inference or incident repair.

Validation: **47 focused tests passed**, including the 6,000-memory/5,990-disk
case; partial final line; concurrent monitor/append; recovery and second-writer
exclusion; inode replacement; actual child-process crash/resume; identical and
conflicting duplicates; provenance errors; independent post-seal reread;
uncertain fsync; and exact 64-bit seed preservation in the incident manifest.
Ruff, compile checks and `git diff --check` pass.

The automatic write review caught a proposed JSON roundtrip that would have
rounded large seeds in JavaScript. That proposed artifact was rejected before
write. The accepted artifact preserves the audit's exact text, and the new
regression test compares every missing seed against the frozen schedule using
integer-preserving Python. No frozen seed or source artifact was changed.

## 6. Other potentially affected uses

The shared old constructor is also used by Q2 V3, Q2 V4 presemantic, Q2 V4.1
semantic, controller-held-out V1/V2, Q2 OOS V2 presemantic/semantic, and a
controller-held-out analysis loader. The complete call-site inventory is
recorded separately. This is an implementation exposure inventory, not evidence
that any other journal lost rows. Their sealed files and independent coverage
audits would need campaign-specific review. No Q1/Q2 or historical result was
reclassified, reopened or rescored.

## 7. Minimum completion plan and principal decision

The present data fail completeness. Scoring 5,990 rows, treating absence as
error, lowering the required count, or silently rerunning ten keys is not an
allowed completion route.

If additional authenticated backups become available, compare them opaquely
against exact identity/key/seed bindings; build a separate candidate only from
original complete records; preserve all existing bytes and reject conflicts;
then independently audit all 6,000 keys before requesting acceptance of the
recovered object. This audit found no such additional originals.

**Recommendation:** preserve this attempt as execution-incomplete due to
persistence integrity and do not classify the ten keys as ordinary operational
retries. The decision remaining is whether to accept that closure, or authorize
a separate, explicitly documented recovery amendment for these exact ten keys
despite unresolved prior completion. Such an amendment requires principal
scientific judgment; the current retry rule does not itself establish eligibility.
No entire-campaign restart is proposed.

## 8. Resource and scientific boundaries

- New Qwen forwards/generations: 0; GPU use in this audit: NO.
- Scoring / correctness inspection: NOT_STARTED / NO.
- Original journal and original seals modified: NO.
- Confirmation/reserve model access: 0 / 0.
- Dataset, schedule, seeds, router, portfolio and champion changed: NO.
- Recovered missing records / rerun keys: 0 / 0.
- Spark 1: file/CPU operations only; Spark 2 / RunPod: NO.
- Q1/Q2 or historical classifications changed: NO.
- Auto-resume/auto-scoring monitor: disabled; no follow-up execution launched.

Final operational status: **Q3_FRESH_QUALIFICATION_PERSISTENCE_INTEGRITY_BLOCKED**.
