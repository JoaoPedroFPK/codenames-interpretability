"""Model loaders. One file per model; each exposes a `load_<name>()` function
that returns `(model, tokenizer, metadata)` where `metadata` is a dict with
keys `num_layers`, `hidden_dim`, `device`, `model_name`, `prefix`,
`chat_template_strategy`, `supports_generation`, plus any per-model extras
(e.g., ModernBERT's GLOBAL_LAYERS / LOCAL_LAYERS).

Model files contain no methodology logic; they are pure model-loading wrappers
that capture each notebook's Cell 2 verbatim.
"""
