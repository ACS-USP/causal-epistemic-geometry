# Q2 V4.1 Semantic Execution

Primary frozen relational classification: `Q2_V4_1_G2`.

The Q2 V4.1 semantic panel was executed only after the principal authorization, with 31 frozen controllers, two shells, N=300 items, and two independent rollouts. The original V4 classification remains `Q2_V4_SAFE_BANK_INSUFFICIENT`; this is a distinct V4.1 result.

## Frozen endpoint

For each controller pair, the analysis uses `D_shape_superpopulation = N/(N-1) * (D_total - m0*m1)` with binary error `e=1` for every non-correct outcome. Negative finite-sample estimates are retained.

## Classification

A0/A1/A2 qualification inputs and the G3 superiority contrasts are in `ESTIMANDS.json`; the 50,000-map QAP and 10,000 item-cluster bootstrap are frozen and recorded there. G3 planning power remains planning-only and was not used to modify the decision rule.

## Firewall

No Q3 was run. No controller, shell, item, seed, endpoint, metric, QAP map, or threshold was changed after opening.
