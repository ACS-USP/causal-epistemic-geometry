# Q1 confirmatory Phase Zero — sealed holdout provenance audit

Classification: `Q1_CONFIRMATORY_BLOCKED_HOLDOUT_IDENTITY`.

No model weights were loaded, no RunPod resource was started, and no confirmatory prompt, reference answer, model output, correctness label, or condition metric was inspected. The offline power analysis was not run because Phase Zero did not pass.

## What the 57 IDs are

The exact ordered list is stored in `HOLDOUT_PROVENANCE_AUDIT.json`. Its canonical source is Gate 9's `REMAINING_FRESH_AVAILABILITY.json`, where the IDs are described as 57 `remaining_unseen_unallocated` CRUXEval records after Gate 9 allocated 100 of 157 eligible unseen items. The ordered ID-list SHA-256 is `a012b4d203d88d807a146ebbe8429c55a1834c6b8e0df5751a12b677ff7b2462`.

## Untouched-status finding

Repository metadata supports that these 57 IDs remained untouched through Gate 13.1:

- they have zero overlap with Gate 9's 643-ID historical exclusion union, whose declared scope covers preserved manifests, journals, reserves, drafts, and allocations;
- they have zero overlap with Gate 13's 320 allocated development items;
- they have zero overlap with the 140 unique Gate-13.1 causal-sweep, layer-dose, and final-evaluation items;
- Gate 13 and Gate 13.1 independently preserve the count of 57 untouched CRUXEval IDs.

This is strong provenance that the IDs are unused. It is not, by itself, a confirmatory-holdout designation.

## Identity failure

The canonical records do not establish that the 57 IDs are the Q1 `confirmatory_holdout`:

- Gate 9 calls them only remaining unseen and unallocated;
- Gate 13 and Gate 13.1 record `untouched_cruxeval_ids: 57` and `confirmatory_holdout: UNTOUCHED` as separate firewall fields;
- no CRUXEval confirmatory-holdout manifest was found that maps the holdout label to these 57 IDs;
- no larger separately sealed CRUXEval confirmatory pool was found;
- other repository objects named confirmatory holdout belong to different historical instruments and do not define a CRUXEval set relation.

Therefore the set relation is `AMBIGUOUS_UNMAPPED`: identity, subset, disjointness, or unrelatedness cannot be established because the CRUXEval `confirmatory_holdout` has no canonical ID-bearing object.

The frozen protocol requires an immediate stop when the 57 IDs are not clearly authorized as the Q1 confirmatory pool or their relationship to the confirmatory holdout is ambiguous. The 57 IDs remain untouched. A principal-reviewed, prospective identity record would be required before restarting design review; this audit does not itself authorize that designation or any holdout access.
