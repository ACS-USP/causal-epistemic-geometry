# Development protocol

The current codebase is explicitly DEVELOPMENT. Its purpose is to make a
small, reproducible kill-test easy to run and easy to falsify.

During development we may:

- debug prompts and exact-label parsers;
- test transformer layer paths and hook lifecycle;
- inspect mock outputs and generated artifacts;
- run tiny exploratory alpha sweeps;
- compare random, orthogonal, and simple contrast-vector controls;
- repair implementation or provenance problems.

Before any future confirmatory campaign, freeze in a reviewable record:

- model identifier, revision, dtype, device and quantization choice if any;
- benchmark, task split, item IDs, and ground-truth parser;
- steering-vector construction, source examples, normalization, and layer;
- intervention token scope and alpha values;
- primary and secondary metrics;
- null and destructive controls;
- hypotheses, exclusion rules, and random seeds;
- artifact and analysis version.

No confirmatory experiment should be silently created by changing a development
YAML file. Alpha lists are accepted only for development-stage configs and are
never run automatically by the default scalar mock config.

The initial report should show the full paired 2×2 outcome table. It should not
select a favorable alpha or vector after seeing the results and call that choice
confirmatory. The software does not implement a magic “useful diversity score.”

