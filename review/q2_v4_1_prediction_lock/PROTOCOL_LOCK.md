# Q2 V4.1 final presemantic protocol lock

Status: `Q2_V4_1_PRESEMANTIC_PROTOCOL_LOCKED`. This commit freezes the immutable 31-safe bank, inherited shell deployments, label-free M1/M2 manifests, A0, 300-item future panel, 63-condition/37,800-row future schedule, QAP, estimands, bootstrap, and G3 characterization. A1/A2 will be materialized on Spark 1 only after this lock. Semantic execution is not authorized and semantic outcomes remain zero.

Post-lock label-free materialization is pinned to consolidator commit `367190e58ec853a70d23ba0d423d75014044cb61`. The 24 raw A2 files are preserved with per-file hashes in `A2_RAW_ARCHIVE_HASHES.json` and aggregate SHA-256 `ee1e215f19d22914d5a7c36e68c7754c0064425f934056541f02cf2b11072bbf`. The independent CPU reference check is recorded in `A2_OFFLINE_REFERENCE_VALIDATION.json`; JS uses natural logarithms, `0.5/0.5` weighting, and an equal-weight mean over 48 probe/checkpoint rows. No semantic outcome or correctness label was inspected.
