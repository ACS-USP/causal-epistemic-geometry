# Gate 12 adversarial premortem

Classification: `PREMORTEM_PASS`

- **Local Vs Finite** — JVP is at alpha=0; Gate-11.1 D75 KL is a finite target only.
- **Trajectory Semantics** — final prompt plus every continuation input token is shifted.
- **Full Sequence Equivalence** — must pass remote KV-cache equivalence before collection.
- **Jvp Exactness** — forward-mode autograd JVP primary; finite differences validation only.
- **Output Geometry** — categorical Fisher q and q/4 Hellinger convention both reported.
- **Utility Functional** — globally canonical minimal correct FINAL continuation.
- **Outcome Blindness** — runner reads frozen manifests and vectors but no historical journals.
- **Direction Matching** — all meaningful/random hashes imported from Gate 9/10/11.
- **Item Selection** — manifest-only SHA256 selection with Gate-11 exclusions.
- **Raw Persistence** — float32 complete logits/JVPs with masks, positions and hashes.
- **Claim Boundary** — one-dimensional sustained-control pullback, not full matrix/Gramian.
- **Q2 Firewall** — no semantic-error geometry or direction-pair matrix claim.
- **Storage** — lossless per-item shards; local verified archive required before Pod deletion.

The collection path is outcome-blind and performs no free generation.
