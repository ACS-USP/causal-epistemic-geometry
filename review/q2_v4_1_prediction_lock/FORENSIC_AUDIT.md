# Q2 V4.1 label-free forensic audit

Classification: `Q2_V4_1_LABEL_FREE_FORENSIC_CLEAN`.

This audit independently recomputed A0, A1, A2, and D2 from the persisted arrays. It used an explicit max-shift natural-log JS implementation, equal weighting over 48 probe/checkpoint rows, and did not import the primary A2 consolidator.

Maximum matrix absolute difference: `6.02462524313e-12` (frozen audit tolerance `1e-08`).

Raw files verified: `24/24`; raw/repeat byte identity: `PASS`; environment: `PASS`; scientific firewall: `PASS`.

Scientific items processed: `0`. Semantic outcomes: `0`. Correctness inspected: `False`.
