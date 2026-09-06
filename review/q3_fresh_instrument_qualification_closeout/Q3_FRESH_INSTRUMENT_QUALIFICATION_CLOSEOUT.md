# Q3.4 Fresh Instrument Construction and Qualification

## 1. Immutable system and provenance

Q3.4 evaluated the exact development-selected system frozen by Q3.3: the
ordered A0-maximin bank of eight policies, the geometry-blind prompt-feature
plus learned-policy-identity router, and external champion
`V4_DIRECTION_02_MEDIUM`. No policy, vector, amplitude, router parameter,
prompt, model revision, seed, parser, or qualification gate changed.

The qualification used Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, layer 27,
sustained-current-token steering, BF16, and SDPA on Spark 1. The candidate
system SHA-256 is
`d8128e4ef4bf9459977cc46a3c9698b36c96afb8a2a388428f5daf03ac6e78f0`;
the private fitted-router SHA-256 remains
`269dc116c70b64dd47cf59340b07dbe558ec8c0f13be8410ed97017310ebad3d`.

## 2. Executable generator prelock

The restricted-Python generator, independent restricted-AST interpreter,
pinned CPython reference path, family law, canonicalization, split rules,
serialization contract, and qualification protocol were frozen before the
scientific generation stream. The generator prelock SHA-256 is
`673b63632c3b446760d70064514a105a32d9c4a2c97c9c0725a283183574f364`;
the effective prospective amendment SHA-256 is
`4021ba5a7d089eca171990187f50f05b5d06438dbadb036b52af815be71104e9`.

## 3. Target population and family definition

The target is the population induced by the frozen restricted-Python
generator, not CRUXEval and not arbitrary Python programs. One independent
unit is one canonical program skeleton/family. Variable renaming, formatting,
and superficial literal changes do not create new families. The deterministic
stream accepted 1,600 families: 300 qualification, 1,000 confirmation, and 300
reserve. It recorded zero structural candidate rejections. Distinct hashes
alone were not used as evidence of family independence.

## 4. Independent reference verification and sandboxing

The restricted-AST interpreter and isolated pinned-CPython worker agreed on
all 1,600 typed references and repeated deterministically. Parser/reference
roundtrip was 1.0. Reference workers had bounded resources and no network,
credential, project-write, or private-model-data access. No general `eval` of
model output was used.

## 5. Structural diversity and deduplication

All canonical family IDs and skeletons were unique. Cross-split collisions
were zero and the frozen structural near-duplicate rate was 0.0, below the
0.01 maximum. Excluded engineering fixtures were not members of any split.

## 6. Qualification, confirmation, and reserve manifests

The dataset seal SHA-256 is
`1b889c3c1de9b6d20d93fca96e866322d4a75f672577dcffdc3e49571ea0da72`.
The release-safe split-manifest hashes are:

| Split | Families | Manifest SHA-256 | Qwen access |
|---|---:|---|---:|
| Qualification | 300 | `9a01142e4825efad36c9ede99cacf88ec6c8cc42d37f24c2ba213bb6c4a790a1` | 6,000 generations |
| Confirmation | 1,000 | `3b021291b20a4faf961e40fbb343b0671aa83f28fd8e986025924d72a4885920` | 0 |
| Reserve | 300 | `acbabbbb8bd79b3df6c1db4f3f8b440745e66c49915ec034be48fff0dff0fd00` | 0 |

Private prompts and references remain hash-pinned and absent from the public
release.

## 7. Same-forward router engine validation

Two excluded fixtures exercised four engineering generations and 44 model
forwards. The live engine captured the unsteered layer-27 block input,
transformed it with the frozen feature pipeline, selected one policy once
during prefill, and kept that policy active through decoding. Online and
offline routing agreed, deterministic fixed-policy replay agreed, hooks were
cleaned between items, and router computation did not consume sampling RNG.

## 8. Qualification schedule and collection integrity

The frozen schedule SHA-256 is
`edba56fc8435cdc34b6f7551fc2d1b4a6d4cc3d87fc34127a5096526d670a635`;
the execution-lock SHA-256 is
`3a51a8d6d9fe57722f9ca740e1c5281e0645031f1641cb824c469ab4dc36635f`.
It contained 300 families, two rollouts, eight bank policies, the external
champion, and the online routed system: 6,000 logical generations.

An operational persistence incident left 5,990 rows on disk while the
historical completion seal recorded 6,000 in collector memory. The original
journal, inconsistent seal, and incident audit remain immutable. Before any
correctness inspection, the principal authorized reexecution of only the ten
prespecified missing persisted keys with their original prompts, conditions,
rollouts, seeds, model, and generation contract. These rows are explicitly
marked `REEXECUTED_MISSING_PERSISTED_KEY`; they are not represented as recovered
original bytes. The other 5,990 rows were preserved byte-for-byte.

The hardened disk-first closeout independently reread the resulting journal:

- expected/completed: 6,000/6,000;
- original persisted/reexecuted missing: 5,990/10;
- missing/unexpected/duplicates/conflicts/replacements: 0/0/0/0/0;
- retries/runtime errors: 0/0;
- repetition stops/hard caps: 44/6;
- generated tokens: 658,254;
- final journal bytes: 80,077,505;
- final journal SHA-256:
  `2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431`;
- recovery completion-seal SHA-256:
  `e7eaf43da51690bd388c191283287374f3b45b7cb8f8015e33b790ff5a6e79ba`.

The original collection occupied approximately 16 h 39 min; the ten-row
reexecution added 84.0 seconds after model load.

## 9. Frozen qualification metrics and ruling

Scoring used the frozen `external-semantic-v3` parser only after the raw-data
seal. Invalid, unevaluable, repetition-stop, and hard-cap outcomes counted as
incorrect exactly as predeclared. Routed-minus-champion accuracy was
descriptive and was not a gate.

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Dual-evaluator agreement | 1.0 | = 1.0 | PASS |
| Reference repetition determinism | 1.0 | = 1.0 | PASS |
| Parser/reference roundtrip | 1.0 | = 1.0 | PASS |
| Cross-split family/skeleton collisions | 0 | = 0 | PASS |
| Structural near-duplicate rate | 0.0 | <= 0.01 | PASS |
| Router commitment validity | 0.94 (564/600) | >= 0.95 | **FAIL** |
| Router semantic evaluability | 0.94 (564/600) | >= 0.95 | **FAIL** |
| Champion commitment validity | 1.0 (600/600) | >= 0.95 | PASS |
| Champion semantic evaluability | 1.0 (600/600) | >= 0.95 | PASS |
| Champion accuracy | 0.133333 (80/600) | [0.25, 0.90] | **FAIL** |
| Frozen-bank oracle headroom | 0.015 | >= 0.05 | **FAIL** |
| Maximum per-condition repetition rate | 0.018333 | <= 0.10 | PASS |

Router accuracy was 0.135 (81/600); frozen-bank oracle accuracy was
0.148333. The routed-minus-champion difference was +0.001667, but this was not
used for qualification. The instrument failed mandatory answer-channel,
difficulty, and oracle-opportunity gates. The mechanical ruling is
`Q3_FRESH_INSTRUMENT_NOT_QUALIFIED`.

This is an instrument-development failure. It is not a negative confirmatory
test of routed utility, because the confirmation population was never opened.

## 10. Independent qualification audit

An independent implementation reconstructed all 6,000 identities, parser
decisions, condition summaries, gates, and the terminal classification from
the sealed journal. Parser disagreements were zero, maximum aggregate metric
difference was 0.0, and classification agreement was true. The forensic state
is `Q3_FRESH_INSTRUMENT_QUALIFICATION_FORENSIC_CLEAN`.

## 11. Runtime forecast for confirmation

The new program distribution was materially slower than the inherited
CRUXEval-based planning forecast. A 100,000-resample family-cluster operational
forecast based on qualification elapsed times estimates:

| Confirmation replication | Generations | P50 | P80 | P95 |
|---|---:|---:|---:|---:|
| R=2 | 4,000 | 20.54 h | 21.42 h | 22.28 h |
| R=4 | 8,000 | 41.08 h | 42.85 h | 44.55 h |
| R=6 | 12,000 | 61.62 h | 64.27 h | 66.83 h |

These are conditional operational extrapolations, not guaranteed runtime and
not an authorization to execute. Since qualification failed, they do not
support opening confirmation.

## 12. Confirmation lock prepared but not executed

The prospective confirmation design remains a router-versus-champion,
family-level paired-mean test over the sealed 1,000-family split, with its
existing safety and missingness rules. No confirmation seed, replication
change, or execution is authorized by this closeout. Qualification remained
R=2. Confirmation and reserve received zero Qwen forwards and zero Qwen
generations.

## 13. Repository and resource state

- Qualification families: 300; confirmation: 1,000; reserve: 300.
- Structural candidates rejected: 0.
- Engineering Qwen forwards/generations: 44/4.
- Qualification semantic generations: 6,000.
- Confirmation Qwen forwards/generations: 0/0.
- Reserve Qwen forwards/generations: 0/0.
- Correctness first inspected only after raw seal: YES.
- Router/bank/champion modified: NO.
- Routed gain used for qualification: NO.
- Spark 1 was used only for the authorized engineering and qualification work.
- Spark 2 / RunPod: NO / NO.
- Q3 confirmatory result: `NOT_RUN`.
- Historical Q1, Q2, and Q3.0-Q3.3 classifications changed: NO.

`Q3_FRESH_INSTRUMENT_NOT_QUALIFIED`
