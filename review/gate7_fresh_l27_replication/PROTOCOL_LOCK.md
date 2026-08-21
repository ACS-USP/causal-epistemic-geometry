# Gate 7 prospective protocol lock

Status: `FROZEN_PRE_OUTCOME`. Lifecycle: `PROSPECTIVE_LOCK`.

Gate 7 reuses the exact frozen `BEST_SINGLE_MEAN_PLUS` L27 paired-mean controller, eta, reference scale, sustained current-token hook, Qwen revision, and external-semantic-v3 evaluator. It allocates a fresh deterministic CRUXEval sample of 120 items and compares baseline, textual CAREFUL, the meaningful controller, and four new architecture-matched random controllers, with two independent rollouts per item-condition.

Meaningful canonical vector SHA-256: `e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838`.

Evaluation manifest hash: `046a36d051da21977bcc4aaa736d49a8097d8b999a314b7629ece1b30cb6ebd5`.

Schedule file SHA-256: `9351168e7824272a43fafbf02304d7a4d324a244a64cc75dea9490b5c3c9eceb`.

Semantic V3 module SHA-256: `d3082512531b1105fb555333c131170d009fa20b4e1edb5eb62dfcfe2702750c`.

No controller, layer, dose, condition, threshold, parser, or sample choice may change after model outcomes. Q2 and character count are NOT RUN; confirmatory holdout is UNTOUCHED.
