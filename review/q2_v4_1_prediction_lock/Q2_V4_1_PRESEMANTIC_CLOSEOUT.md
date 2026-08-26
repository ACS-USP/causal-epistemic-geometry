# Q2 V4.1 — final presemantic freeze and prediction lock closeout

## Decision

`Q2_V4_1_READY_FOR_PRINCIPAL_SEMANTIC_EXECUTION_REVIEW`

The historical V4 result remains `Q2_V4_SAFE_BANK_INSUFFICIENT` and `Q2_V4_PRESEMANTIC_FORENSIC_CLEAN`. V4.1 retains all 31 directions that passed both original safety shells, in original order. The Q2 relational hypothesis remains untested.

## Label-free qualification

A1: `Q2_V4_1_A1_INSTRUMENT_QUALIFIED`; fit hash `1c67c482096db4ad6ad7671eae80300c5ca1d8833da070af13dd9ac0cfeca2a1`; effective rank `7.603726089151967`; condition `11718.972854437636`.

A2: `Q2_V4_1_A2_INSTRUMENT_QUALIFIED` for MEDIUM and STRONG. MEDIUM minimum Gram eigenvalue `0.00023122892144200995`; STRONG minimum Gram eigenvalue `0.0024947079924260377`. Both repeat archives are byte-identical to the raw archives and every frozen algebraic check passes.

A2 uses natural-log full-vocabulary JS, `0.5 KL(p||m) + 0.5 KL(q||m)`, and an equal-weight mean over 48 probe/checkpoint rows. An independent reference check on one probe passed before accepting the complete A2 metrics.

## Frozen future experiment

The future panel is N=300 with 63 conditions (31 MEDIUM, 31 STRONG, and baseline), two rollouts, and 37,800 prospective semantic rows. The 50,000-map controller-label QAP and maxT structure are frozen. A1/A2/D2 are now materialized, but semantic execution is not authorized by this closeout.

## Provenance and firewall

Spark 1 fingerprint: `8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386`; model revision: `b968826d9c46dd6066d109eabc6255188de91218`; access: direct SSH to Spark 1. The local dstack troubleshooting server was terminated and is not protocol infrastructure. Spark 2 and RunPod were not used.

Raw A2 files: `24/24`; aggregate SHA-256: `ee1e215f19d22914d5a7c36e68c7754c0064425f934056541f02cf2b11072bbf`. Bundle SHA-256: `5ee5b9f388651e677a4eab24867ef284c24f8139bd1213065b3ce04aee7e0b02`.

New semantic outcomes: `0`; correctness inspected: `NO`; Q2: `UNTESTED`; Q3: `NOT_RUN`.

Status: `DRAFT / AWAITING PRINCIPAL SEMANTIC EXECUTION REVIEW`.
