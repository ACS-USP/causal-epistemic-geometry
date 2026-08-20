# Gate 6 source phase — technical incompleteness

The frozen SOURCE phase was started once on the migrated A40 Pod and stopped
after the second source item failed the pre-specified source-pair completion
requirement. Item `sample_169` under the frozen CAREFUL instruction reached
`max_new_tokens=4096` without an unambiguous `FINAL` marker.

This is not a scientific steering outcome. No controller was constructed, no
manipulation rows were collected, and no evaluation rows were collected. The
cap, prompt, item allocation, model, revision, and source protocol were not
changed. The two completed source rows and raw log are preserved in the review
directory with the hashes recorded in `SOURCE_PHASE_FAILURE.json`.

The run is classified as `GATE6_SOURCE_PHASE_INCOMPLETE`. It must not be
repaired by increasing the cap, selecting a replacement item, or rerunning
only the failed item after observing this result. Any redesign requires a new
prospective protocol and principal review.
