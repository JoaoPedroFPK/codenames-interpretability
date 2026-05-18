# Methodology (brief)

This is a one-page operational summary of the experimental methodology. The
full version, with motivations, mathematical definitions, and architectural
rationale, is in Chapter 4 of the thesis.

---

## What the experiment measures

For a sample of 2,000 Codenames Duet clue-giving turns, the package
extracts hidden state representations at every layer of seven transformer
language models, computes per-layer cosine similarities between the hint
word and each candidate word, and summarises the resulting geometry
through a small set of scalar metrics. The metrics are compared across
architectural axes (attention pattern, positional encoding, training vs
random init) that distinguish the seven models.

---

## The five-step procedure

The package's module layout mirrors this procedure intentionally — a reader
of the methodology chapter can map "Step N" directly to a module:

1. **Sample (`data.py`).** 2,000 turns are drawn from CULTURAL CODES with
   `random_state=42`. The same turns are sampled in every model's run.
2. **Prompt (`prompts.py`).** A turn becomes a prompt whose instruction body
   is byte-identical across models; only the chat-template wrapping differs
   per model (Mistral `[INST] ... [/INST]`, Qwen/Random_Qwen ChatML, BERT /
   T5 / ModernBERT / BERT-Random plain text).
3. **Forward + span detection (`extraction.py`, `spans.py`).** The model
   runs a forward pass on the prompt; hidden states at every layer are
   retained. Character-level offset mappings locate the token span of each
   word in the prompt. Two pooling procedures (mean, max-norm) produce a
   vector per (board, layer, word).
4. **Per-layer metrics (`extraction.py`, `sanity.py`).** Cosine
   similarities between the hint vector and each candidate vector are
   computed at every layer. Ranks, MRR, Hit@K, raw margin, anisotropy-
   adjusted margin, all-pairs anisotropy are derived from the cosines.
5. **Confound checks (`sanity.py`, SC6–SC7).** A residual confound is
   addressed by an ordering-perturbation procedure: candidates are presented
   in three orderings per turn (alphabetical and two random permutations).
   Variance of each candidate's cosine-to-hint across orderings is
   decomposed into position-driven and identity-driven components,
   yielding the per-layer semantic signal ratio.

---

## The two experimental conditions

The two conditions differ only in the presence of a demographic preamble
describing the clue-giver. The prompt is otherwise byte-identical between
conditions for a given turn:

- **`no_social`** — instruction body without the giver-features preamble.
- **`with_social`** — same body with the preamble prepended after the
  "You are playing the game Codenames." opener.

All seven models are evaluated under both conditions. Random-init models
omit the generation phase but produce metrics under both conditions.

---

## Where the values come from

| Concept | Module | Function / variable |
|---|---|---|
| Sample size, seed, pooling methods | `contract.py` | `Contract` dataclass |
| Dataset loading, candidate construction | `data.py` | `load_dataset`, `build_candidates_fixed_order` |
| Prompt body, chat templates | `prompts.py` | `build_instruction_body`, `build_prompt` |
| Span detection, pooling, cosine | `spans.py` | `find_token_spans`, `mean_pool_span`, `max_norm_pool_span`, `cosine_similarity_np` |
| Per-board forward + metrics | `extraction.py` | `run_instance` |
| Main loop (both conditions, shuffles, sharding) | `loop.py` | `run_extraction` |
| Causal generation + concordance | `generation.py` | `generate_response` |
| SC1–SC7 sanity outputs | `sanity.py` | `sc1_prompt_structure` … `sc7_shuffle_decomposition` |
| Output filenames, NPZ matrix | `persistence.py` | `save_*` helpers |

The thesis's Methodology chapter (§4.3–§4.8) is the source of record for
why these choices were made. This package is the operational realisation of
that methodology, nothing more.
