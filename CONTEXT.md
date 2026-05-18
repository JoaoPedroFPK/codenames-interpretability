# `codenames-interpretability` — Build Specification

This document is the complete specification for refactoring seven Google
Colab notebooks into a single Python package that runs the same experiments
from Colab via a `git clone`-based workflow. It is written for a code agent
that will execute the build.

**The seven canonical notebooks are present in this repository at
`reference_notebooks/`.** They are the authoritative source for every
implementation detail of the existing experiments. This specification
describes the *architecture* of the refactor (what files exist, what
modules contain what, what the function signatures look like); the
notebooks contain the *implementation* (the actual prompt strings, the
exact metric computations, the precise hot-loop logic, the byte-level
text of every print statement). When this specification and the
notebooks disagree on an implementation detail, **the notebooks win**.
The notebooks produced the existing N=2000 runs that the thesis Results
chapter depends on; the refactor must preserve their behaviour exactly.

Read the **entire** document before writing any file, then read the
seven notebooks in `reference_notebooks/` before writing any production
code. Several sections of this spec constrain each other (the
bit-identity requirements in Section 9 constrain how you write Section
6, etc.) and skipping ahead will produce subtle violations. Section 0
defines a preflight step the agent must complete before beginning the
build proper.

---

## Table of contents

0. **Preflight: reading the canonical notebooks** *(do this first)*
1. Goal and non-goals
2. Audience and review context
3. The runtime model (how this code is meant to be executed)
4. Repository layout
5. Package layout
6. The contract: what must remain byte-identical across the refactor
7. Module-by-module specification
8. Notebook specification
9. The bit-identity validation procedure
10. Documentation deliverables
11. Implementation order
12. Things that are explicitly out of scope
13. Open questions you must surface, not resolve

---

## 0. Preflight: reading the canonical notebooks

**Do this before writing any file.** The notebooks at
`reference_notebooks/` are the authoritative source for every
implementation detail. The agent must read them first.

### Step 0.1 — Inventory

Confirm that `reference_notebooks/` contains the following seven files
(filenames may vary slightly; map to the canonical model the user
documents):

- The Mistral-7B-Instruct-v0.2 notebook
- The Qwen2.5-7B-Instruct notebook
- The Qwen2.5-7B Random Baseline notebook
- The BERT-base-uncased notebook
- The BERT-base Random Baseline notebook
- The T5-base (encoder) notebook
- The ModernBERT-base notebook

If any of these is missing, **stop and report** the missing notebook to
the user. Do not infer its contents from other notebooks.

### Step 0.2 — Read every notebook

Read each of the seven notebooks in full. For each notebook, note:

- The exact cells and their order
- The exact prompt strings produced
- The exact print statements (including spacing, decimal precision,
  separators)
- The exact column orders of every output DataFrame
- The exact filenames produced for outputs
- The exact random-state-consuming operations and their order
- Any model-specific quirks (different system messages, different
  chat-template wrapping, different fp16 handling, different
  initialization patterns)

### Step 0.3 — Map the notebooks to the spec

For each module specified in Section 7, identify which notebook cells
correspond to it. Build an internal map: "Section 7.3 `prompts.py`
corresponds to Cell 5 of every notebook (titled 'Prompt Builder' or
similar); the chat-template strategy differs per model and is captured
in the model file."

If two notebooks have inconsistent implementations of what should be
the same logic (a metric, a print statement, a column order), **stop
and surface the inconsistency** under Section 13's "open questions"
procedure. Do not silently pick one.

### Step 0.4 — Report back

Before beginning the build, the agent must produce a short report to
the user covering:

1. Confirmation that all seven notebooks are present and readable.
2. The mapping of notebooks to spec modules (a table is fine).
3. Any inter-notebook inconsistencies discovered, listed under Section
   13's open-questions format.
4. Any spec sections that the canonical notebooks do not match (where
   the spec says one thing but the notebooks do another) — for these,
   the notebook wins, and the agent should propose the spec amendment.

Only after the user responds to this report does the agent begin the
build proper, starting at Section 11 Step 1.

### Authority order during the build

Throughout the build, when a question arises about an implementation
detail:

1. **Reference notebook for that model** is authoritative. The notebook
   produced the existing experimental output that the thesis depends
   on; the refactor must match.
2. **This specification** describes the desired architecture. It is
   authoritative on file organisation, module boundaries, and function
   signatures.
3. **The agent's own judgment** applies only when (1) and (2) leave a
   genuine gap (e.g., the name of a private helper function the
   notebook didn't have because it was inline). The agent's judgment
   should never override what's in the notebooks.

If the notebook for one model differs from another notebook in a way
that would make a unified package implementation impossible (different
metric definitions, different output schemas), this is an inconsistency
that must be surfaced under Section 13, not papered over.

### Treatment of the reference notebooks during and after the build

The reference notebooks are **read-only** for the agent. The agent
does not edit them, move them, rename them, or delete them. After the
build is complete, `reference_notebooks/` remains in the repository as
the historical record of the experiments-as-originally-run, alongside
`notebooks/` which contains the new thin shells. The user may decide
later to remove `reference_notebooks/` from the repo; that decision is
not part of this build.

---

## 1. Goal and non-goals

### Goal

Take the seven Codenames-interpretability experiment notebooks
(`mistral.ipynb`, `qwen.ipynb`, `qwen_random.ipynb`, `bert.ipynb`,
`bert_random.ipynb`, `t5.ipynb`, `modernbert.ipynb`) — which currently
contain ~80% duplicated code across models — and produce a Python package
plus seven thin orchestration notebooks. The package is the single source
of truth for the experimental methodology. The notebooks become short
shells that import from the package, configure model-specific details,
and run the experiment with full visibility into intermediate outputs.

### Non-goals

This refactor does **not**:

- Change experimental methodology in any way
- Re-run any of the existing N=2000 experiments
- Add new metrics, new pooling methods, new conditions, or new models
  beyond the seven already present
- Convert outputs to a different format
- Replace Colab as the execution environment
- Set up CI/CD beyond an optional GitHub Actions placeholder
- Distribute the package via PyPI

The user has six existing N=2000 runs whose outputs are saved to Drive and
consumed by a separate synthesis notebook. Those existing outputs must
remain valid. Any future run of the refactored code must produce outputs
that are interchangeable with the existing ones for analysis purposes.

---

## 2. Audience and review context

The repository will be **public** on GitHub. It is a submission artifact
for a Computer Science undergraduate thesis. The likely readers are:

- The student's thesis advisor (already familiar with the methodology)
- The thesis examination committee (may inspect code but won't run it)
- Future researchers searching for related work

This affects code style decisions:

- Function and module docstrings are part of the submission, not internal
  notes. Write them as if a committee member is reading.
- The README is the front page of the repo. A reader who clicks the GitHub
  link should understand the project's scope, structure, and how to run it
  within two minutes.
- Variable names match the published methodology chapter where possible.
  If the methodology refers to "the separation margin," do not name the
  variable `m` — name it `separation_margin` or `margin`. The student's
  methodology uses specific terminology consistently.
- Comments explain *why*, not *what*. A reader can read Python; they need
  to know which design decisions are load-bearing.

---

## 3. The runtime model

This is the most important section to understand before writing any code.

### The constraint

Google Colab provides a Linux container with a GPU attached, but the
container is ephemeral: anything outside `/content/drive/` evaporates at
runtime shutdown. The user cannot install editable packages "locally"
because Colab is the only runtime. The user edits Python files on their
laptop (with a text editor, no Python runtime locally), pushes to GitHub,
then pulls from inside Colab.

### The intended workflow

For every experiment run, the user:

1. Edits package code locally with their text editor.
2. Commits and pushes to GitHub.
3. Opens the target notebook in Colab (the notebooks live in the
   `notebooks/` directory of the repo and can be opened via
   "GitHub" in Colab's file dialog).
4. Runs cell 1, which clones (or pulls) the repo into `/content/`.
5. Runs cell 2, which `pip install -e`'s the package.
6. Runs cell 3, which mounts Drive.
7. Runs subsequent cells, which import from the package and execute the
   experiment, writing outputs to Drive.

When the user wants to re-run with updated code, they push the new code
to GitHub, run cell 1 again (which `git pull`s), and the next cell that
imports from the package picks up the new code (because of the
`autoreload` cell, see below).

### Implication for package design

The package must work when installed via `pip install -e .` in a Colab
container that already has its own pinned versions of `torch`,
`transformers`, `numpy`, `pandas`, `tqdm`, `scipy`. The package's
`pyproject.toml` must specify these as dependencies *without pinning
versions*, so Colab's existing versions are respected. The package adds
no version constraints that would force a reinstall.

Specifically: do not pin `torch`, `transformers`, `numpy`, `pandas`,
`tqdm`, `scipy`, or `accelerate`. Use the form `"torch"` not
`"torch>=2.0"`. The user's experiments already work with whatever Colab
ships; do not introduce version drift.

### Implication for notebook design

Every notebook in `notebooks/` begins with the same setup boilerplate
(clone, install, mount, autoreload). The boilerplate is identical across
all seven notebooks except for the notebook-specific imports below it.
The boilerplate must be **copy-pasted verbatim** across notebooks, not
factored out, because each notebook must be runnable as a standalone
artifact opened directly from GitHub. There is no shared "setup script"
that the notebooks all source — each notebook is self-contained from the
user's perspective.

---

## 4. Repository layout

```
codenames-interpretability/
├── README.md                       # Front page, see Section 10
├── LICENSE                         # MIT, see Section 10
├── .gitignore                      # Standard Python ignore
├── pyproject.toml                  # Package metadata, see Section 7.0
├── codenames_interpretability/     # The package
│   ├── __init__.py
│   ├── contract.py                 # Section 7.1
│   ├── data.py                     # Section 7.2
│   ├── prompts.py                  # Section 7.3
│   ├── spans.py                    # Section 7.4
│   ├── extraction.py               # Section 7.5
│   ├── loop.py                     # Section 7.6
│   ├── generation.py               # Section 7.7
│   ├── sanity.py                   # Section 7.8
│   ├── persistence.py              # Section 7.9
│   ├── cli.py                      # Section 7.11 (NEW)
│   └── models/
│       ├── __init__.py
│       ├── mistral.py              # Section 7.10
│       ├── qwen.py                 # Section 7.10
│       ├── qwen_random.py          # Section 7.10
│       ├── bert.py                 # Section 7.10
│       ├── bert_random.py          # Section 7.10
│       ├── t5.py                   # Section 7.10
│       └── modernbert.py           # Section 7.10
├── notebooks/                      # NEW thin orchestration notebooks
│   ├── 00_validation.ipynb         # Bit-identity check, see Section 9
│   ├── 01_mistral.ipynb            # Section 8
│   ├── 02_qwen.ipynb               # Section 8
│   ├── 03_qwen_random.ipynb        # Section 8
│   ├── 04_bert.ipynb               # Section 8
│   ├── 05_bert_random.ipynb        # Section 8
│   ├── 06_t5.ipynb                 # Section 8
│   └── 07_modernbert.ipynb         # Section 8
├── reference_notebooks/            # CANONICAL SOURCE — read-only for the agent
│   ├── <mistral notebook>          # The original Mistral notebook
│   ├── <qwen notebook>             # The original Qwen notebook
│   ├── <qwen_random notebook>      # The original Random Qwen notebook
│   ├── <bert notebook>             # The original BERT notebook
│   ├── <bert_random notebook>      # The original Random BERT notebook
│   ├── <t5 notebook>               # The original T5 notebook
│   └── <modernbert notebook>       # The original ModernBERT notebook
├── tests/
│   ├── __init__.py
│   ├── test_prompts.py
│   ├── test_spans.py
│   └── test_metrics.py
└── docs/
    ├── contract.md                 # Human-readable contract
    ├── runtime.md                  # The Colab-from-GitHub workflow
    └── methodology.md              # Brief — points to thesis chapter
```

The `reference_notebooks/` directory is the authoritative source for
every implementation detail. The agent reads from it but does not write
to it. The agent's job is to produce everything in `codenames_interpretability/`
and `notebooks/` so that the methodology preserved in `reference_notebooks/`
becomes maintainable and reviewable as a package.

---

## 5. Package layout philosophy

The package follows a strict separation:

- **`contract.py`** — frozen parameters only, no logic.
- **`data.py`, `prompts.py`, `spans.py`** — pure functions, no model
  dependencies, no I/O.
- **`extraction.py`** — the per-board function that runs one forward pass
  and computes all per-board metrics. Depends on a model, a tokenizer,
  and pure-function utilities.
- **`loop.py`** — the main loop over boards and conditions. Depends on
  `extraction.py`. Handles sharding and partial saves.
- **`generation.py`** — the causal-only generation utility. Optional;
  not imported by encoder-only notebooks.
- **`sanity.py`** — the seven SC functions. Each is a standalone callable
  that takes a results dict and prints its findings. Functions return
  values for testability but the primary interface is stdout.
- **`persistence.py`** — file-writing utilities: CSV, parquet, NPZ, with
  the exact path conventions the existing notebooks use.
- **`models/<name>.py`** — one file per model. Each exports a single
  `load_<name>()` function that returns `(model, tokenizer)`. No other
  logic lives in model files.

**Why this separation matters:** the methodology chapter of the thesis
describes the experiment as a five-step procedure. The package's module
structure should mirror that procedure so a reader can map "Step 2 of the
methodology = pooling" to "spans.py" without searching.

---

## 6. The contract: what must remain byte-identical

This is the load-bearing requirement. **The refactored code must produce
outputs that are byte-identical, or as close to byte-identical as
floating-point allows, to the outputs the canonical notebooks in
`reference_notebooks/` would produce on the same inputs.** This is not
aspirational; it is required.

The canonical notebooks at `reference_notebooks/` are the empirical
ground truth for this contract. Every section below that references "the
existing notebooks" or "the original" is referring to the files at
`reference_notebooks/`. When this spec describes a function's behavior
and the corresponding notebook cell differs, the notebook cell is
correct.

### Why bit-identity matters

The user has six existing N=2000 runs whose outputs are saved to Drive
and are the basis for the thesis Results chapter. The seventh model
(Random Qwen) is the immediate next run and will be produced by the
refactored code. If the refactored code produces *slightly* different
outputs than the originals — even a 1-ULP difference in a fp16 cosine
value, even a row-order difference in a CSV — then the new run is not
directly comparable to the existing six in the cross-model synthesis.

The bit-identity requirement applies to: cosine values, ranks, MRR,
Hit@k, anisotropy mean and std, Spearman ρ, semantic signal ratio, raw
margin, adjusted margin (where computed), and any saved vector.

### Specific things that can cause drift

The agent must be vigilant about these because they are easy to violate
unintentionally:

**(a) Random state consumption.** The existing notebooks call
`torch.manual_seed(42)` once in Cell 2 and `np.random.seed(42)` once,
and never re-seed thereafter. If the refactored code re-seeds at module
import time, or seeds a different generator before extraction begins, the
shuffle permutations will differ. The refactor must seed at *exactly* the
same point in execution as the originals, which is "once, at the start
of the run, before the dataset is sampled, with seed value pulled from
the contract."

**(b) Operation order in the hot loop.** The existing notebooks compute
cosine similarities and pool vectors in a specific order (cell 16
`run_instance` function). Reordering an inner loop, or computing
pairwise cosines via vectorized matrix ops instead of nested Python
loops, can produce fp16 results that differ by 1 ULP. **Do not
"optimize" the hot loop in this refactor.** Copy the logic verbatim from
the existing notebooks into `extraction.py`, only changing what is
necessary to make it a function rather than top-level code.

**(c) Dictionary iteration order.** Python 3.7+ guarantees dict insertion
order, but if the refactored code builds a result dict by populating
keys in a different order than the original, downstream CSV columns may
appear in a different order. Match the original column orders exactly.

**(d) Pandas concat order.** The existing notebooks accumulate records
into a list of dicts, then call `pd.DataFrame(records)`. The resulting
column order depends on the order of keys in the first record. Preserve.

**(e) Tokenizer initialization arguments.** Every model loader in the
existing notebooks calls `AutoTokenizer.from_pretrained(MODEL_NAME)` with
specific arguments (some set `pad_token = eos_token` post-hoc, some
don't). The model-file refactor must preserve these exact patterns.

**(f) Float16 conversion timing.** The existing notebooks compute metrics
at full precision and convert pooled vectors to float16 *after* metrics
are computed but *before* vectors are stored. Do not change when the
conversion happens.

**(g) The `find_token_spans` function** has subtle character-offset
logic for finding candidate words after a designated anchor. Copy
verbatim; do not refactor.

### The validation procedure

Before declaring the refactor complete, the agent **must** describe how
the user can validate bit-identity. See Section 9 for the procedure. The
agent does not perform the validation; the agent produces the validation
notebook and the comparison script.

---

## 7. Module-by-module specification

### 7.0 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "codenames-interpretability"
version = "1.0.0"
description = "Layer-wise word representation geometry across transformer architectures, evaluated on Codenames Duet"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "<author name from notebooks if present, else 'TCC'>" }]
keywords = ["interpretability", "transformers", "geometric-analysis", "codenames"]

dependencies = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "tqdm",
    "scipy",
    "pyarrow",
    "accelerate",
]

[project.scripts]
codenames-experiment = "codenames_interpretability.cli:main"

[tool.setuptools.packages.find]
include = ["codenames_interpretability*"]
exclude = ["tests*", "notebooks*", "docs*"]
```

The `[project.scripts]` entry creates a `codenames-experiment` command on
the user's PATH after `pip install -e .`. The CLI is documented in
Section 7.11.

No version pinning. No optional dependencies. No dev dependencies block
unless the agent has a strong reason; this is a research package, not a
library.

### 7.1 `contract.py`

Frozen experimental parameters. The existing notebooks define these in
"Cell 2 — Global Configuration" of each notebook, and the values are
identical across all seven notebooks (with `MODEL_NAME`, `MODEL_PREFIX`,
`BASE_DIR` being the per-model differences, which belong elsewhere).

The agent must produce a single immutable dataclass `Contract` with one
default instance `CONTRACT_V1` exported. Fields:

```python
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class Contract:
    """Frozen experimental parameters. See methodology chapter for rationale."""
    sample_size: int = 2000
    candidate_order: str = "fixed"
    pooling_methods: Tuple[str, ...] = ("mean", "max_norm")
    vector_subsample_size: int = 100
    n_shuffles: int = 2
    generation_max_tokens: int = 30
    shard_boards: int = 200
    random_seed: int = 42
    max_seq_len: int = 512


CONTRACT_V1 = Contract()
```

Do **not** add helper methods to this class. Do not add validation. The
frozen dataclass is by design a static record.

The agent should document each field briefly in the class docstring,
referencing the methodology chapter where the field is defined.

### 7.2 `data.py`

Functions for loading and sampling the CULTURAL CODES dataset. **Read
the corresponding cells of every notebook in `reference_notebooks/`**
(specifically the dataset-loading cell, typically titled "Load and
Prepare Dataset" or similar). All seven notebooks should implement
dataset loading identically; if they differ, surface this under Section
13.

The functions to expose:

```python
def load_dataset(path: str) -> pd.DataFrame: ...
def sample_turns(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame: ...
def build_candidates_fixed_order(row) -> List[str]: ...
def extract_giver_features(row, giver_cols) -> Dict[str, object]: ...

GIVER_COLS: List[str]  # exported, value from notebooks
```

The `GIVER_COLS` constant has nine entries: `giver.marriage`,
`giver.education`, `giver.race`, `giver.continent`, `giver.language`,
`giver.religion`, `giver.gender`, `giver.country`, `giver.political`.

**Note on T5:** the T5 notebook historically had a typo `giver.politics`
which was fixed. The canonical key is `giver.political`. Verify against
the T5 notebook in `reference_notebooks/` what value it currently uses.
The `_FEATURE_LABEL_MAP` in `prompts.py` (see 7.3) maps both keys to
"Politics" for backwards compatibility — preserve this mapping if it
exists in any of the canonical notebooks.

### 7.3 `prompts.py`

The byte-identical prompt builder. **Read the prompt-builder cell of
every notebook in `reference_notebooks/`** (typically titled "Prompt
Builder" or similar). This is a critical module — every model must
produce byte-identical prompts (modulo the chat-template wrapping, which
is model-specific).

Compare the prompt-builder logic across all seven notebooks. The
`instruction_body` portion (everything before the chat-template
wrapping) must be byte-identical across models for a given turn — this
is what the methodology promises and what the notebooks deliver.
**Verify this by reading the corresponding cells side-by-side.** If two
notebooks produce subtly different instruction bodies, surface this
under Section 13.

Expose:

```python
def _format_feature_value(v) -> str: ...

_FEATURE_LABEL_MAP: Dict[str, str]  # the giver-key → display-label map

def build_instruction_body(
    hint: str,
    candidates: List[str],
    giver_features: Optional[Dict[str, object]],
    use_social_context: bool,
) -> Tuple[str, Dict[str, str]]: ...

def build_prompt(
    hint: str,
    candidates: List[str],
    giver_features: Optional[Dict[str, object]],
    use_social_context: bool,
    tokenizer,
    chat_template_strategy: str,  # see below
) -> Tuple[str, Dict[str, str]]: ...
```

The `chat_template_strategy` argument is the only difference between
model families. Three strategies:

- `"chatml"` — apply `tokenizer.apply_chat_template(messages, ..., add_generation_prompt=True)` with a "system: You are a helpful assistant" + "user: <body>" message list. Used for Mistral, Qwen, Qwen Random. **Verify the exact system-message string by reading each ChatML notebook in `reference_notebooks/`.** If the strings differ between Mistral and Qwen, this is a model-level configuration the model loader must capture; do not coerce them to a single value.
- `"raw"` — return `instruction_body` directly with no wrapping. Used for BERT, BERT Random, T5, ModernBERT.
- The function dispatches on this argument.

The `instruction_body` itself must be **byte-identical** across all
models for a given turn. This is what the methodology promises and what
the existing notebooks deliver. Verify against the canonical notebooks
by extracting the prompt-builder logic from each and confirming that, on
an identical input row, every notebook produces an identical
`instruction_body`.

### 7.4 `spans.py`

Token span detection and pooling utilities. **Read the span-detection
cell of any notebook in `reference_notebooks/`** (typically titled
"Token Span Detection and Pooling Utilities"). This module is reused
without modification across all models, so reading one notebook is
sufficient, but verify the cell is identical across all seven before
declaring this. Expose:

```python
def find_token_spans(
    full_text: str,
    offset_mapping: List[Tuple[int, int]],
    spans_to_find: Dict[str, str],
    candidate_anchor: str = "The possible words are:",
) -> Dict[str, Tuple[int, int]]: ...

def mean_pool_span(layer_hidden_states, span) -> Optional[np.ndarray]: ...
def max_norm_pool_span(layer_hidden_states, span) -> Optional[np.ndarray]: ...
def pool_span(layer_hidden_states, span, method: str = "mean") -> Optional[np.ndarray]: ...
def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float: ...
```

Copy verbatim from the canonical notebook in `reference_notebooks/`.
The pooling functions return `np.float16` arrays as in the original. Do
not "improve" the cosine similarity function; the existing one returns
0.0 on zero-norm vectors and this behavior must be preserved.

### 7.5 `extraction.py`

The per-board extraction function. **Read the extraction cell of every
notebook in `reference_notebooks/`** (typically titled "Core Instance
Processing Function" and containing the `extract_giver_features` and
`run_instance` definitions). This is the hottest piece of code in the
package and the one most subject to bit-identity risk.

Compare `run_instance` across all seven notebooks. The function should
be functionally identical; if any notebook implements it differently,
surface this under Section 13. Expose:

```python
def run_instance(
    row: pd.Series,
    giver_cols: List[str],
    use_social_context: bool,
    candidates_order: List[str],
    permutation_id: int,
    save_vectors: bool,
    *,
    model,
    tokenizer,
    device: str,
    pooling_methods: Tuple[str, ...],
    num_layers: int,
    hidden_dim: int,
    chat_template_strategy: str,
) -> Tuple[Dict, List[Dict], Optional[List[Dict]]]: ...
```

**The function body is copied verbatim from one of the canonical
notebooks in `reference_notebooks/`** (any notebook is fine if they're
identical; if they differ, surface and ask). The notebook contains the
authoritative implementation. The only modifications allowed are:

1. The `model`, `tokenizer`, `DEVICE`, `POOLING_METHODS`, `NUM_LAYERS`,
   `HIDDEN_DIM` references — which in the notebook are module-globals —
   become keyword-only arguments.
2. The `build_prompt` call now takes the `chat_template_strategy`
   argument introduced in 7.3.

No other changes. No "cleanup." No "let me make this more Pythonic." The
hot loop must match the canonical notebook in `reference_notebooks/`
byte-for-byte.

### 7.6 `loop.py`

The main extraction loop. **Read the main-loop cell of every notebook
in `reference_notebooks/`** (typically titled "Main Extraction Loop").
The loop is structurally similar across notebooks, with differences
limited to the `HAS_GENERATION` branch (causal only) and the model-prefix
substitution. Expose:

```python
def run_extraction(
    *,
    model,
    tokenizer,
    df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    contract: Contract,
    chat_template_strategy: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    has_generation: bool = False,
    generation_fn: Optional[Callable] = None,
) -> Dict[str, Dict]:
    """Run the full extraction for both conditions, saving outputs to base_dir.

    Returns a dict keyed by condition name (no_social / with_social) with
    sub-dicts containing general_df, metrics_df, generation_df, error_log.
    """
```

The function body copies the main loop from the canonical notebook in
`reference_notebooks/` with these adaptations:

- `MODEL_PREFIX`, `BASE_DIR`, `SAMPLE_SIZE`, etc. come from arguments.
- The `HAS_GENERATION` flag becomes the `has_generation` keyword arg,
  resolved by the caller (the notebook) based on whether `generation_fn`
  was passed in.
- The generation call site uses the passed `generation_fn` instead of a
  globally-named `generate_response`.
- All prints, all tqdm bars, all sharding logic remain.

**Do not collapse the prints.** Every existing `print(...)` from the
notebook's main loop must remain in `run_extraction()`. The user reads
these prints during a run; they are the live feedback channel.

### 7.7 `generation.py`

The causal-only generation utility. **Read the generation-utility cell
of the Mistral and Qwen notebooks in `reference_notebooks/`** (typically
titled "Generation Utility"). The cell is not present in encoder-only
notebooks. Compare Mistral's and Qwen's implementations; if they
differ, surface this under Section 13.

Expose:

```python
def generate_response(
    *,
    prompt: str,
    candidates: List[str],
    max_new_tokens: int,
    model,
    tokenizer,
    device: str,
) -> Dict:
    """Generate a continuation and parse the first in-candidate word."""
```

Copy verbatim. The parsing logic (whole-word matching, hint exclusion,
first-by-character-position) is methodologically committed in the thesis
and must not change.

### 7.8 `sanity.py`

The seven SC functions. **Read the SC cells of every notebook in
`reference_notebooks/`** ("SC1 — Prompt Structure Verification" through
"SC7 — Shuffle Confound Decomposition"). The SC cells should be
functionally identical across notebooks except where model-specific
parameters intervene (e.g., `NUM_LAYERS`); verify by reading at least
two notebooks for each SC and reconciling differences.

Expose seven functions, one per SC, named:

```python
def sc1_prompt_structure(df_sample, tokenizer, chat_template_strategy: str): ...
def sc2_span_coverage(results): ...
def sc3_anisotropy(results, num_layers: int): ...
def sc4_behavioral_accuracy(results, pooling_methods, has_generation: bool): ...
def sc5_layer_margin_curve(results, base_dir: str, prefix: str,
                           num_layers: int, pooling_methods): ...
def sc6_positional_confound(results, base_dir: str, prefix: str,
                            num_layers: int): ...
def sc7_shuffle_decomposition(results, base_dir: str, prefix: str,
                              num_layers: int, n_shuffles: int): ...
```

There is no SC0 in the new package. The original SC0 was a generation
diagnostic specific to causal models with an early-development purpose
that is no longer relevant (it predates the current generation pipeline
in the thesis). If any notebook in `reference_notebooks/` still contains
an SC0 cell, do not port it; surface this under Section 13 only if the
notebook's SC0 contains logic that none of SC1-SC7 covers.

Each SC function preserves its original print output exactly. The
**printed tables, the column headers, the "PASS" / "WARN" markers, the
formatting strings** must be byte-identical to the originals in
`reference_notebooks/` because the user reads these to verify a run.

Each function also writes its CSV outputs to disk as the originals do
(SC5 → `{prefix}_layer_margins_*.csv`, SC6 →
`{prefix}_position_confound_by_layer.csv`, SC7 →
`{prefix}_shuffle_decomposition_by_layer.csv`).

Each SC function takes the results dict from `run_extraction()` plus
whatever auxiliary arguments it needs. It does not need a reference to
the model or tokenizer (except SC1, which uses the tokenizer to verify
prompt structure).

### 7.9 `persistence.py`

File I/O helpers. **Read the save logic of the main-loop cell** ("Main
Extraction Loop") **and the final "Save Outputs Summary" cell of every
notebook in `reference_notebooks/`.** Expose helpers for:

```python
def save_general_csv(general_df: pd.DataFrame, base_dir: str, prefix: str, mode_name: str) -> str: ...
def save_metrics_parquet(metrics_df: pd.DataFrame, base_dir: str, prefix: str, mode_name: str) -> str: ...
def save_generation_csv(generation_df: pd.DataFrame, base_dir: str, prefix: str, mode_name: str) -> str: ...
def save_vector_subsample(vector_records, base_dir: str, prefix: str, mode_name: str, hidden_dim: int) -> Tuple[str, str]: ...
def save_error_log(error_log: List[Dict], base_dir: str, prefix: str, mode_name: str) -> Optional[str]: ...
def print_output_summary(base_dir: str, prefix: str, contract: Contract, has_generation: bool, pooling_methods): ...
```

Each helper produces files at the **exact path** the original notebook
produces, with the **exact filename format**:

```
{base_dir}/{prefix}_general_{mode_name}.csv
{base_dir}/{prefix}_metrics_{mode_name}.parquet
{base_dir}/{prefix}_generation_{mode_name}.csv
{base_dir}/{prefix}_vectors_subsample_index_{mode_name}.csv
{base_dir}/{prefix}_vectors_subsample_{mode_name}_f16.npz
{base_dir}/{prefix}_errors_{mode_name}.csv
```

Plus the aggregate files produced by SC5, SC6, SC7 (which can either
live in `persistence.py` as save helpers or stay inside the SC
functions; copy the existing pattern).

The NPZ matrix is saved with `np.savez_compressed(path, vectors=matrix)`
where `matrix` has shape `(n_records, hidden_dim)` and dtype
`np.float16`. The NPZ integrity check (load and verify shape) from the
original is preserved.

### 7.10 `models/<name>.py`

Each model file exports exactly one function: `load_<name>()`. **For
each model, read the corresponding notebook in `reference_notebooks/`**
to extract the exact model-loading logic, including the precise
arguments to `from_pretrained` (or `from_config`), the dtype, the
device-mapping strategy, and any post-load configuration. The model
file is the model-loading cell of that notebook, packaged as a function.

The function returns `(model, tokenizer, metadata)` where `metadata` is
a dict containing model-specific information the notebooks need:

```python
{
    "num_layers": int,
    "hidden_dim": int,
    "device": str,
    "model_name": str,
    "prefix": str,                      # the file-naming prefix
    "chat_template_strategy": str,      # "chatml" or "raw"
    "supports_generation": bool,
}
```

The seven model files are:

- **`mistral.py`** — `load_mistral_instruct()` → loads
  `mistralai/Mistral-7B-Instruct-v0.2`, fp16, `chat_template_strategy="chatml"`,
  `supports_generation=True`, `prefix="mistral"`.
- **`qwen.py`** — `load_qwen_instruct()` → loads `Qwen/Qwen2.5-7B-Instruct`,
  fp16, `chat_template_strategy="chatml"`, `supports_generation=True`,
  `prefix="qwen"`.
- **`qwen_random.py`** — `load_qwen_random()` → uses the `from_config` +
  `accelerate.init_empty_weights` GPU-init pattern, fp16,
  `chat_template_strategy="chatml"`, `supports_generation=False`,
  `prefix="random_qwen"`. See the existing Random Qwen notebook for the
  full init procedure including the RMSNorm re-init fix.
- **`bert.py`** — `load_bert_base()` → loads `bert-base-uncased`, fp32,
  `chat_template_strategy="raw"`, `supports_generation=False`,
  `prefix="bert"`.
- **`bert_random.py`** — `load_bert_random()` → uses `BertModel(BertConfig())`
  for random init (no pretrained weights), `chat_template_strategy="raw"`,
  `supports_generation=False`, `prefix="random_bert"`.
- **`t5.py`** — `load_t5_encoder()` → loads encoder of `t5-base`, fp16,
  `chat_template_strategy="raw"`, `supports_generation=False`,
  `prefix="t5"`.
- **`modernbert.py`** — `load_modernbert()` → loads
  `answerdotai/ModernBERT-base`, `chat_template_strategy="raw"`,
  `supports_generation=False`, `prefix="modernbert"`. Note: requires
  `transformers>=4.48.0` which Colab has by default; do not pin.

Each model file is **20–40 lines**. They do not contain methodology
logic. They are pure model-loading wrappers.

**Critical for `qwen_random.py`:** the init pattern is non-obvious.
**The agent must read the Random Qwen notebook in `reference_notebooks/`
carefully** before writing this loader. The pattern uses
`AutoConfig.from_pretrained` to fetch the config (not the weights),
then `accelerate.init_empty_weights()` context manager to construct the
architecture on the meta device, then `to_empty(device=...)` to
materialize on GPU, then `model.apply(model._init_weights)` to populate
weights, then an **explicit RMSNorm re-init loop** because Qwen2's
`_init_weights` does not handle RMSNorm and the constructor-set values
of 1.0 are wiped by `to_empty()`. The pre-flight diagnostic also lives
in this notebook and must be ported to a separate module
(`codenames_interpretability.diagnostics`) — see Section 8 for how the
Random Qwen notebook calls it. **Do not improvise this loader; the
canonical notebook is the only correct reference.**

### 7.11 `cli.py`

A command-line interface for running experiments. The CLI is the
**batch-run** interface; the notebooks are the **interactive
verification** interface. Both call the same underlying package
functions; the CLI is a thin orchestration layer that takes command-line
arguments, dispatches to the right model loader, and invokes the same
pipeline the notebooks invoke at cell granularity. **`cli.py` must not
contain any experimental logic that is not already in the package's
other modules.** If a behavior cannot be achieved by calling existing
package functions, it does not belong in the CLI.

#### Entry point

The module exposes a `main()` function registered as a console script in
`pyproject.toml`. After `pip install -e .`, the command
`codenames-experiment` is available on the user's PATH. The same module
is also runnable via `python -m codenames_interpretability` for cases
where the script is not on PATH (Colab `!` invocations).

#### Model dispatch

The CLI accepts `--model <name>` where `<name>` is one of seven strings:
`mistral`, `qwen`, `qwen_random`, `bert`, `bert_random`, `t5`,
`modernbert`. The module contains a single hardcoded dispatch dict
mapping each string to its loader function:

```python
MODEL_REGISTRY = {
    "mistral": ("codenames_interpretability.models.mistral", "load_mistral_instruct"),
    "qwen": ("codenames_interpretability.models.qwen", "load_qwen_instruct"),
    "qwen_random": ("codenames_interpretability.models.qwen_random", "load_qwen_random"),
    "bert": ("codenames_interpretability.models.bert", "load_bert_base"),
    "bert_random": ("codenames_interpretability.models.bert_random", "load_bert_random"),
    "t5": ("codenames_interpretability.models.t5", "load_t5_encoder"),
    "modernbert": ("codenames_interpretability.models.modernbert", "load_modernbert"),
}
```

The loader is imported lazily (only when the chosen model is invoked) so
the CLI startup does not load all seven model libraries.

#### Subcommands

The CLI has four subcommands:

##### `run` — full experiment for one model

```
codenames-experiment run \
    --model <model_name> \
    --dataset <path-to-clue_generation.csv> \
    --output-dir <path-to-output-directory> \
    [--sample-size <int>] \
    [--conditions no_social,with_social] \
    [--skip-sanity-checks] \
    [--no-generation]
```

Behavior:
1. Calls `data.load_dataset()` and `data.sample_turns()` with the
   contract's sample size (or `--sample-size` if overridden).
2. Calls the dispatched model loader.
3. Calls `loop.run_extraction()` with the assembled arguments.
4. Unless `--skip-sanity-checks` is passed, calls each `sanity.scN_*`
   function in sequence, printing their output to stdout.
5. Calls `persistence.print_output_summary()` at the end.

The `--no-generation` flag, when passed to a causal model, disables the
generation phase (useful for debugging the geometric pipeline without
paying generation cost). For encoder-only models the flag has no effect.

##### `preflight` — pre-flight diagnostic only (Random Qwen)

```
codenames-experiment preflight \
    --model qwen_random \
    --dataset <path-to-clue_generation.csv>
```

Behavior: loads the model, samples 5 boards, runs the pre-flight
diagnostic from `diagnostics.preflight_random_init()`, prints the
NaN/Inf/norm/anisotropy report, exits without running extraction.

For models other than `qwen_random`, this subcommand prints a message
explaining that pre-flight is only defined for random-init models and
exits without error.

##### `validate` — bit-identity check against existing outputs

```
codenames-experiment validate \
    --model <model_name> \
    --dataset <path-to-clue_generation.csv> \
    --against <path-to-existing-output-directory> \
    [--n <int>]
```

Behavior:
1. Samples `--n` boards (default 50) from the dataset under the contract
   seed.
2. Runs `loop.run_extraction()` to a temporary output directory.
3. Loads the corresponding files from `--against`, filters to the same
   row_ids, and compares cell-by-cell within tolerance.
4. Prints "Validation PASSED" or "Validation FAILED: <details>".

This subcommand is the headless equivalent of `notebooks/00_validation.ipynb`.

##### `sanity` — re-run SC functions on already-extracted results

```
codenames-experiment sanity \
    --model <model_name> \
    --results-dir <path-to-existing-output-directory> \
    [--checks sc1,sc2,...]
```

Behavior: loads previously-extracted results from `--results-dir` and
runs the specified sanity checks against them. Useful for re-running SC
output formatting without re-running the (expensive) extraction.

Default `--checks` is all seven (sc1 through sc7). SC1 requires the
tokenizer; for this check the CLI loads the model loader's tokenizer
(but not the full model). SC2 through SC7 only need the saved DataFrames.

#### Output behavior

The CLI's output **must be identical** to the notebook's cell-by-cell
output. The CLI does not introduce its own logging framework, does not
suppress prints, does not redirect tqdm. Every print produced by the
underlying package functions appears on stdout in the same order it
would appear in a notebook. The only difference is that all output
appears under one shell-cell in Colab if the CLI is invoked via `!`,
rather than under multiple notebook cells.

The user should be able to verify that:

```bash
codenames-experiment run --model qwen --dataset ... --output-dir ...
```

produces stdout that, concatenated, matches what they would see by
running the seven cells (load → sample → run → SC1..SC7) of
`02_qwen.ipynb` in order.

#### Argument parsing

Use `argparse` from the standard library. Do not introduce `click`,
`typer`, or any other CLI framework — Colab already has argparse, and
this package does not need third-party CLI tooling.

Subcommands are implemented as separate `argparse.ArgumentParser`
instances under a top-level `argparse.add_subparsers()` dispatcher. Each
subcommand's argument set is defined in one short function
(`_make_run_parser`, `_make_preflight_parser`, `_make_validate_parser`,
`_make_sanity_parser`) for readability.

#### Error handling

The CLI should print friendly error messages and exit with status 1 on
recoverable errors (missing files, invalid model names, missing
required arguments). On unrecoverable errors (an exception during
extraction), the CLI should let the exception propagate so the
traceback is visible — this is research code and the traceback is
information the user needs.

---

## 8. Notebook specification

Each notebook supports **two ways to run the experiment**, both
documented in the notebook itself with a markdown cell at the top
explaining when to use which:

- **Path A — Full CLI invocation (batch run).** A single shell-cell that
  calls `!codenames-experiment run --model <name> ...`. Produces all
  output in one cell block. Use this when you want to run an experiment
  end-to-end without interactive inspection — for example, overnight
  runs, regeneration after a code fix, or runs on a model you've already
  verified extensively.

- **Path B — Cell-granular invocation (interactive verification).** The
  experiment is decomposed across multiple cells (load → sample →
  extraction → SC1 → SC2 → ... → SC7), each calling package functions
  directly. Use this when you want per-cell visibility into intermediate
  outputs, when debugging an anomaly, or when verifying a new model run
  step by step.

Both paths call the same underlying package functions. The difference is
purely how the user sees the output. Each notebook contains both paths
as alternative blocks; the user runs either Path A *or* Path B for a
given session.

The per-model differences across notebooks are limited to one line (the
`--model` argument in Path A, the model-loader import in Path B). The
remainder of each notebook is identical to the other six.

### Canonical notebook structure

The first three cells of every notebook are the same setup boilerplate.
After setup, the notebook contains Path A (CLI) and Path B (cell-by-cell)
as alternative blocks separated by markdown headers. The user runs the
setup cells once, then runs *either* Path A *or* Path B.

#### Setup (cells 1-3, identical across all notebooks)

**Cell 1 — Pull repo** (markdown above explaining "this clones or
updates the package code from GitHub"):

```python
import os
REPO_URL = "https://github.com/<TBD>/codenames-interpretability.git"
REPO_DIR = "/content/codenames-interpretability"

if os.path.exists(REPO_DIR):
    !git -C {REPO_DIR} pull
else:
    !git clone {REPO_URL} {REPO_DIR}
```

The agent leaves `<TBD>` as a literal placeholder. The user fills in
their GitHub username when they create the repo.

**Cell 2 — Install package**:

```python
!pip install -q -e {REPO_DIR}
```

**Cell 3 — Autoreload + Mount Drive**:

```python
%load_ext autoreload
%autoreload 2

from google.colab import drive
drive.mount("/content/drive")
```

#### Path A — CLI invocation (batch run)

A markdown header above this cell explains: "Run this cell to execute
the full experiment end-to-end via the CLI. All output (extraction
progress, SC tables, summary) appears in this cell's output block."

**Cell A — Full experiment**:

```python
!codenames-experiment run \
    --model <MODEL_NAME> \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv \
    --output-dir /content/drive/MyDrive/TCC/<MODEL_PREFIX>_outputs
```

The placeholders `<MODEL_NAME>` and `<MODEL_PREFIX>` are filled per
notebook (e.g., `qwen` and `qwen` for the trained Qwen notebook,
`qwen_random` and `random_qwen` for the Random Qwen notebook).

For Random Qwen specifically, a preflight CLI call precedes the run
cell:

```python
!codenames-experiment preflight \
    --model qwen_random \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv
```

The user runs preflight first, inspects its output, then proceeds to
the full run cell only if preflight passes.

#### Path B — Cell-by-cell invocation (interactive verification)

A markdown header above this block explains: "Run these cells one at a
time for interactive verification of the experiment. Each cell produces
its own output for inspection."

**Cell 4 — Imports and config**:

```python
from codenames_interpretability.contract import CONTRACT_V1
from codenames_interpretability.data import (
    load_dataset, sample_turns, GIVER_COLS, extract_giver_features
)
from codenames_interpretability.models.<NAME> import load_<NAME>
from codenames_interpretability.loop import run_extraction
from codenames_interpretability.sanity import (
    sc1_prompt_structure, sc2_span_coverage, sc3_anisotropy,
    sc4_behavioral_accuracy, sc5_layer_margin_curve,
    sc6_positional_confound, sc7_shuffle_decomposition,
)
# Only for causal models:
from codenames_interpretability.generation import generate_response

DATASET_PATH = "/content/drive/MyDrive/TCC/clue_generation.csv"
```

**Cell 5 — Load model**:

```python
model, tokenizer, meta = load_<NAME>()
print(f"Model loaded: {meta['model_name']}")
print(f"  Layers: {meta['num_layers']}, Hidden dim: {meta['hidden_dim']}")
```

**Cell 6 — Load and sample dataset**:

```python
df = load_dataset(DATASET_PATH)
df_sample = sample_turns(df, n=CONTRACT_V1.sample_size, seed=CONTRACT_V1.random_seed)
print(f"Sample size: {len(df_sample)} boards")
print(f"First 10 row_ids: {sorted(df_sample['row_id'].tolist())[:10]}")
```

**Cell 7 — Run extraction**:

```python
BASE_DIR = f"/content/drive/MyDrive/TCC/{meta['prefix']}_outputs"

results = run_extraction(
    model=model,
    tokenizer=tokenizer,
    df=df_sample,
    base_dir=BASE_DIR,
    prefix=meta["prefix"],
    contract=CONTRACT_V1,
    chat_template_strategy=meta["chat_template_strategy"],
    device=meta["device"],
    has_generation=meta["supports_generation"],
    generation_fn=generate_response if meta["supports_generation"] else None,
)
```

**Cell 8 onwards — Sanity checks, one per cell**:

```python
# Cell 8 — SC1
sc1_prompt_structure(df_sample, tokenizer, meta["chat_template_strategy"])

# Cell 9 — SC2
sc2_span_coverage(results)

# Cell 10 — SC3
sc3_anisotropy(results, num_layers=meta["num_layers"])

# Cell 11 — SC4
sc4_behavioral_accuracy(
    results,
    pooling_methods=CONTRACT_V1.pooling_methods,
    has_generation=meta["supports_generation"],
)

# Cell 12 — SC5
sc5_layer_margin_curve(
    results, base_dir=BASE_DIR, prefix=meta["prefix"],
    num_layers=meta["num_layers"], pooling_methods=CONTRACT_V1.pooling_methods,
)

# Cell 13 — SC6
sc6_positional_confound(
    results, base_dir=BASE_DIR, prefix=meta["prefix"],
    num_layers=meta["num_layers"],
)

# Cell 14 — SC7
sc7_shuffle_decomposition(
    results, base_dir=BASE_DIR, prefix=meta["prefix"],
    num_layers=meta["num_layers"], n_shuffles=CONTRACT_V1.n_shuffles,
)
```

**Cell 15 — Output summary**:

```python
from codenames_interpretability.persistence import print_output_summary
print_output_summary(
    base_dir=BASE_DIR, prefix=meta["prefix"], contract=CONTRACT_V1,
    has_generation=meta["supports_generation"],
    pooling_methods=CONTRACT_V1.pooling_methods,
)
```

### Notebook-specific notes

- **`03_qwen_random.ipynb`** has an additional cell in Path B between
  Cell 5 (load model) and Cell 6 (load dataset): the **pre-flight
  diagnostic** from the Random Qwen notebook in `reference_notebooks/`.
  This cell runs a forward pass on 5 boards, checks for NaN/Inf, reports
  per-layer norm growth, and reports L0 anisotropy. If pre-flight fails,
  the user does not proceed to Cell 6. **Read the pre-flight cell of the
  reference Random Qwen notebook** to extract the diagnostic logic. The
  function should be exposed as
  `codenames_interpretability.diagnostics.preflight_random_init` (the
  agent creates this module as well, with the diagnostic as a single
  function, ported verbatim from the reference notebook).
  In Path A, the pre-flight check is invoked separately as
  `!codenames-experiment preflight --model qwen_random ...` before the
  full `run` command.
- **Encoder notebooks** (BERT, BERT Random, T5, ModernBERT) omit the
  `from codenames_interpretability.generation import generate_response`
  import in Path B's Cell 4 and pass `generation_fn=None` to
  `run_extraction`. In Path A, no flag is needed — the CLI dispatches
  generation based on the model's `supports_generation` metadata.

---

## 9. The bit-identity validation procedure

The agent does not run experiments. The agent produces a validation
notebook at `notebooks/00_validation.ipynb` that the user runs in Colab
to verify the refactored code produces bit-identical output to the
existing N=2000 runs already in Drive (which were produced by the
notebooks now in `reference_notebooks/`).

### Validation notebook structure

The validation notebook:

1. Pulls the repo, installs, mounts Drive (cells 1–3).
2. Loads one model (default: `bert.py` — fastest model, smallest hardware
   requirement). The user can change which model to validate against.
3. Loads the full N=2000 dataset, samples the first 50 rows under the
   contract seed.
4. Runs `run_extraction()` against this 50-row sample with a temporary
   output prefix (e.g., `bert_validation`).
5. Loads the corresponding existing N=2000 outputs from Drive (e.g.,
   `bert_outputs/bert_general_no_social.csv`).
6. **Filters** the existing N=2000 outputs to the same 50 row_ids that
   the validation run used.
7. **Compares** the validation outputs to the filtered N=2000 outputs:
   - For CSVs: assert all numeric columns equal within `1e-6` tolerance
     (allowing for fp32 rounding differences from how pandas reads/writes).
   - For parquet: same.
   - For NPZ: assert the float16 matrices are exactly equal where both
     are valid.
8. Prints a summary: "Validation PASSED" or "Validation FAILED: <list of
   differing columns / rows>."

The agent provides this notebook with the comparison logic explicit and
copy-pasteable. The user runs it and reports back if anything diverges.

### What the agent does on validation failure

If the user runs the validation notebook and it fails, the agent's
responsibility is to:

1. Read the failure output.
2. Identify the source of the drift (likely one of the categories in
   Section 6).
3. Fix the package code.
4. Tell the user to push and re-run validation.

This is iterative. The agent should expect 1–2 rounds of validation
debugging is realistic for a refactor of this scope.

---

## 10. Documentation deliverables

### `README.md`

The front page. Sections, in order:

1. **One-paragraph project description.** What this is, what it studies,
   what's in the repo.
2. **Citation.** A bibtex block citing the thesis (placeholder if the
   thesis is not yet published).
3. **Repository structure.** Directory tree with one-line descriptions.
4. **Reproducing the experiments.** Four subsections:
   - **"Quick start with the CLI"** — for batch runs. Documents the four
     subcommands (`run`, `preflight`, `validate`, `sanity`) with one
     example invocation each. State that this is the right path for
     end-to-end runs where interactive inspection is not needed.
   - **"Interactive runs with notebooks"** — for cell-by-cell
     verification. Points the reader to `notebooks/` and explains that
     each notebook contains both Path A (CLI) and Path B (cell-by-cell)
     blocks. State that this is the right path for debugging, verifying
     a new run, or producing the notebook artifact that gets saved to
     Drive as a record.
   - **"Required data."** The CULTURAL CODES dataset is not redistributed;
     point the reader to the original source (Shaikh 2023).
   - **"Drive layout."** The `BASE_DIR` convention for outputs.
5. **The seven experiments.** A table listing the seven models, their
   notebooks, their CLI `--model` argument values, and their `prefix`
   values for output files.
6. **Methodology.** A two-paragraph summary of what the experiment does,
   pointing the reader to the thesis chapter for the full methodology.
7. **License.** MIT.

The README does **not** include:

- Installation instructions for non-Colab environments (the package is
  designed for Colab).
- Development instructions (there is no local dev workflow).
- Detailed metric definitions (those live in the thesis).
- Results numbers (those live in the thesis Results chapter).

### `LICENSE`

Standard MIT license text. Year 2026. Author from the package metadata.

### `.gitignore`

Standard Python `.gitignore` (pycache, .pyc, .ipynb_checkpoints,
.pytest_cache, dist/, build/, *.egg-info/). No special entries.

### `docs/contract.md`

A human-readable description of the experimental contract — the same
information as `contract.py` but expanded with prose. One short paragraph
per field. Reference the methodology chapter.

### `docs/runtime.md`

A description of the Colab-from-GitHub workflow. Explain:
- How notebooks find the package (clone + install).
- How autoreload makes edits flow through.
- How outputs flow to Drive.
- How to validate against existing N=2000 outputs.

### `docs/methodology.md`

A *brief* one-page summary of the experimental methodology, *pointing to
the thesis chapter* for the full version. The package is a piece of the
thesis, not the other way around.

---

## 11. Implementation order

The agent must build in this order to enable progressive validation:

**Step 0 — Preflight (Section 0).** Inventory the canonical notebooks
in `reference_notebooks/`, read each in full, build the mapping of
notebook cells to spec modules, surface inter-notebook inconsistencies,
and report back to the user. **Do not proceed to Step 1 until the user
responds to the preflight report.**

**Step 1 — Repo skeleton.** `pyproject.toml`, `README.md`, `LICENSE`,
`.gitignore`, empty `__init__.py` files. The user can `git init` and push
at this point; nothing executes yet.

**Step 2 — Pure-function modules.** `contract.py`, `data.py`, `prompts.py`,
`spans.py`. No model dependency. Read from the canonical notebooks at
`reference_notebooks/`. The user can `pip install` the package at this
point and the imports will work.

**Step 3 — Model loaders.** All seven `models/<name>.py` files. Each
corresponds to the model-loading cell of one notebook in
`reference_notebooks/`. Build them in parallel as a batch.

**Step 4 — Extraction core.** `extraction.py`, `generation.py`. This is
where bit-identity risk concentrates. Build slowly and copy from the
canonical notebooks at `reference_notebooks/` verbatim. The `diagnostics.py`
module for Random Qwen's pre-flight check also goes here.

**Step 5 — Main loop and persistence.** `loop.py`, `persistence.py`. Same
verbatim-copy discipline as Step 4.

**Step 6 — Sanity checks.** `sanity.py`. Each SC function is independent;
build them in any order, copying from the SC cells of any canonical
notebook in `reference_notebooks/`.

**Step 7 — CLI.** `cli.py`. Implement the four subcommands (`run`,
`preflight`, `validate`, `sanity`) as orchestration over the modules
built in Steps 2-6. The CLI must not contain new logic; verify this by
checking that every behavior the CLI exposes is implemented in one of
the earlier modules. Test that `codenames-experiment --help` lists the
subcommands and `codenames-experiment run --help` documents the run
arguments, after a fresh `pip install -e .`.

**Step 8 — Notebooks.** All seven model notebooks plus the validation
notebook (`notebooks/00_validation.ipynb`). Each notebook contains the
three setup cells, then both Path A (CLI) and Path B (cell-by-cell)
blocks. Build them as a batch by templating from the first one. The new
notebooks go in `notebooks/`; the canonical reference notebooks at
`reference_notebooks/` are not touched.

**Step 9 — Documentation.** README, docs/*.md, tests skeleton.

The agent should produce output after each step so the user can review
incrementally. After Step 8, the user runs `notebooks/00_validation.ipynb`
in Colab and either accepts the refactor or reports drift.

---

## 12. Things that are explicitly out of scope

The agent must **not**:

- Add new metrics (e.g., compute the `margin_adj` that is defined in the
  methodology but not in the current saved outputs).
- Add new pooling methods, conditions, or shuffles.
- Restructure the output schema.
- Replace fp16 with bf16 anywhere except as documented in the existing
  Random Qwen notebook.
- Add type hints beyond what makes the code readable; don't try to make
  this a fully-typed codebase.
- Run black, isort, or any formatter on the copy-from-notebook code if
  doing so would risk a subtle behavior change. Format only new code.
- Put any experimental logic in `cli.py`. The CLI must be a thin
  orchestration layer over the package's other modules. Any behavior
  the CLI exposes must already exist as a function call in one of the
  modules built in Steps 2-6. If the agent finds itself adding logic
  to `cli.py` that doesn't have a function-level counterpart elsewhere,
  the agent has put the logic in the wrong place.
- Add a third-party CLI framework. Use `argparse` from the standard
  library.
- Convert any notebook to a Python script. The notebooks are an
  interactive interface; the CLI is the batch interface. Both exist;
  neither replaces the other.
- Implement the synthesis notebook (`00_synthesis.ipynb`). The user has
  this elsewhere.
- Implement GitHub Actions or any CI configuration.
- Pin dependency versions.

---

## 13. Open questions you must surface, not resolve

When the agent encounters an ambiguity that the canonical notebooks at
`reference_notebooks/` do not resolve, the agent must **stop and ask the
user**, not guess.

The most common source of ambiguity will be **inter-notebook
inconsistency**: two or more reference notebooks implement the same
logical step in different ways. For each such inconsistency, the agent
must:

1. State which notebooks disagree.
2. Quote the disagreeing code from each.
3. Identify which behaviour the existing N=2000 runs depend on, if known.
4. Ask the user how to reconcile.

Likely inconsistencies to watch for:

- "The Mistral and Qwen notebooks both use `chat_template_strategy='chatml'`,
  but the exact system message string differs slightly between them.
  Should the package use one canonical system message or preserve the
  per-model strings?" [Likely answer: preserve per-model strings as
  model-specific configuration in the model loader.]
- "The BERT Random notebook initializes via `BertModel(BertConfig())` —
  do you want this to use the same `accelerate.init_empty_weights`
  pattern as Qwen Random for consistency?" [Likely answer: no, keep the
  existing pattern, because the existing N=2000 Random BERT run uses
  this pattern and refactoring its init risks bit-identity.]
- "The T5 notebook has fp16 overflow warning behaviour — should the
  refactored T5 loader switch to fp32 or bf16?" [Likely answer: no,
  keep fp16 to preserve bit-identity with the existing N=2000 T5 run.]
- "I see `MAX_SEQ_LEN = 512` referenced in some notebooks but not all —
  should this be enforced in the contract?" [Ask the user.]
- "The T5 notebook had a typo `giver.politics` — has it been fixed in
  the version in `reference_notebooks/`?" [Inspect and report.]
- "Two notebooks produce slightly different formatted prints in SC5
  (different decimal precision, different column alignment) — which
  formatting should the package adopt?" [Ask the user.]

Default behavior on ambiguity: **preserve the corresponding notebook's
behavior exactly for that specific model**. If a feature differs across
notebooks in a way that affects only one model's output, that
difference is a per-model configuration and lives in the model loader,
not in shared code. If a feature differs across notebooks in a way that
should be shared, ask the user which version is canonical.

---

## Final note to the agent

The user has emphasized that the package must preserve full access to
the results — every print, every tqdm bar, every SC table — that they
currently have when running the existing notebooks cell-by-cell. This
is a **load-bearing** requirement, not a nice-to-have. If a refactor
choice would collapse multiple prints into one less-readable block, or
suppress a progress bar, or change the format of a printed table, that
choice is wrong and must be reconsidered.

The package is meant to be invisible at runtime: a user who knows the
canonical notebooks in `reference_notebooks/` should be able to open a
refactored notebook in `notebooks/`, run the cells, and see output that
is **indistinguishable from the canonical notebook's output** except for
the four setup cells at the top.

The single most important discipline throughout the build is:
**`reference_notebooks/` is the source of truth.** When in doubt, open
the notebook, read the cell, and copy what's there. This specification
describes the architecture; the canonical notebooks describe the
implementation. Build accordingly.