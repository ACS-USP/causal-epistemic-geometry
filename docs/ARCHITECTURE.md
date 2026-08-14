# Architecture

The experiment runner depends on two narrow boundaries:

```text
YAML -> RunConfig
          |
    Benchmark adapter  <---- immutable BenchmarkItem + parser
          |
    ModelBackend       <---- predict / extract_activation / steer(context)
          |
    paired experiment  ----> Prediction records + metrics
          |
    artifact writer    ----> JSONL / JSON / YAML / Markdown / manifest
```

`MockBackend` and `HuggingFaceBackend` implement the same backend interface.
The runner does not know whether a prediction came from NumPy or Transformers.
The mock backend uses deterministic latent representations plus a fixed linear
readout. The HuggingFace backend loads `AutoTokenizer` and
`AutoModelForCausalLM` only when selected, freezes parameters, uses inference
mode, and applies a temporary forward hook.

`TinyRandomTransformerBackend` injects a randomly initialized GPT-2-style
decoder and deterministic test tokenizer into the same HuggingFace backend.
It downloads nothing and exists solely to exercise real Torch/Transformers
mechanics. This is the required local seam between the mock fixture and a
future pretrained model.

The hook accepts a tensor or a tuple/list whose first element is the hidden
tensor, updates only the selected token positions, and returns all untouched
outputs. Its context manager removes the handle on normal and exceptional exit.
Layer discovery tries common decoder-only paths, but an explicit
`backend.layer_path` is available for architecture-specific revisions.

Vector values are stored as compressed `.npz`; JSON stores dimension, layer,
constructor, normalization, provenance, and content hash. Pickle is not used.
Activations for the first contrast constructor use one explicit policy: the
mock latent representation or the last non-padding token in the transformer
backend. Intervention token scope is a separate config variable.

Batching is intentionally not optimized yet. Correctness and hook semantics
come first; generation is currently one item at a time. A future batching path
must preserve paired IDs and make token-position behavior explicit.

Prompt rendering is explicit: plain mode uses the benchmark prompt, while chat
mode requires a tokenizer `apply_chat_template`. Prediction rows retain parser
status and stable rendered-prompt hashes. Run sessions append item-condition
rows, atomically update status, validate identity on resume, quarantine a
truncated final JSONL tail, and recompute final metrics from canonical rows.
