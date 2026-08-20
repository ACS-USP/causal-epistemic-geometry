# Gate 5 — Source Validity and Temporal Persistence

This is a pre-outcome development lock. It reuses the Gate-4 Qwen3-8B direction, alpha, layer, and CRUXEval substrate; it adds only the frozen source-check, sustained-current-token semantics, and R0–R3 random bank.

- Source commit before model outcomes: `6de1261f0f1fc9b588e7c2928c936b5bccbbbaba`
- Fresh items: 40 source-check + 20 manipulation + 60 evaluation
- Historical exclusion digest: `db55699b14721c6369e5a22dc80dd811c87e8362cedf99d93d38e8ae6c196103`
- Model: Qwen/Qwen3-8B, BF16, full non-thinking, sampled
- No Q2, RFM/AGOP, multilayer steering, character count, or holdout access

All machine-readable thresholds, conditions, IDs, prompt text, controller hashes, seed regimes, and estimands are frozen in `PROTOCOL_LOCK.json`.
