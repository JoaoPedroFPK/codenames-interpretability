# Visualization support — implementation report

This documents the visualization pipeline added to turn the experiments' raw
layer-wise word vectors into **inspectable, publication-formatted figures**, and
records the design decisions, the dimensionality-reduction validation, and the
findings from the first run on the BERT outputs.

## 1. Goal

The experiments already produce, per board (a Codenames Duet turn), one pooled
hidden-state vector per `(word, layer, pooling_method)` plus a battery of scalar
metrics (cosine-to-hint, ranks, per-layer anisotropy). What was missing was a way
to **see** that geometry, to confirm quantitative findings against real examples.
Two figure families were requested:

1. a **2D projection** of the word vectors at fixed board+layer, to read off
   spatial proximity layer by layer;
2. a **word×word cosine-similarity heatmap** at fixed board+layer, including the
   giver's demographic words in the social condition, shown without the redundant
   (symmetric) half.

A hard requirement: because projection reduces dimensionality, it must be
**cosine-aware** and its faithfulness must be **measured**, not assumed.

## 2. What was built

A self-contained `codenames/viz/` package plus a `visualize` CLI subcommand.

| File | Responsibility |
|---|---|
| `codenames/viz/loader.py` | Discover model dirs, auto-detect file prefix, join the index CSV with the `_f16.npz` matrix by `record_idx`, filter valid vectors + pooling, upcast to f32, sample boards (seed 42), parse `giver_features`. |
| `codenames/viz/metrics.py` | Cosine-aware DR-quality metrics: trustworthiness (sklearn, `metric="cosine"`), continuity (custom dual), Shepard Spearman correlation (HD cosine vs 2D Euclidean). |
| `codenames/viz/embedding.py` | Run UMAP/t-SNE/PCA (all cosine-aware), score each per board×layer, render the multi-panel projection with the fixed reducer (UMAP). |
| `codenames/viz/heatmap.py` | Symmetric cosine matrix, lower-triangle masking, the `no_social`/`with_social` paired heatmap. |
| `codenames/viz/style.py` | Publication style (Okabe-Ito palette, sans-serif, despined, PDF+PNG @300 DPI), the canonical word-type → colour/marker map, layer-depth labels, representative-layer selection. |
| `codenames/viz/pipeline.py` | Orchestration: per model, per sampled board, emit both figure families + the DR-quality CSV. |

CLI: `codenames-experiment visualize --model bert` (or `--all`). The heavy
plotting/reduction libraries are imported **lazily** inside the command handler,
so the experiment `run`/`doctor` paths — and the Colab install — never require
them. They live in a new optional dependency group:

```toml
[project.optional-dependencies]
viz = ["matplotlib==3.9.2", "seaborn==0.13.2", "scikit-learn==1.5.2",
       "umap-learn==0.5.7", "adjusttext==1.3.0"]
```
(`adjustText` repels word labels off the points so projection panels stay
legible.)

Install locally with `pip install -e ".[viz]"`. Verified: these resolve cleanly
against the existing pins (numpy 2.0.2, scipy 1.14.1, pandas 2.2.2 unchanged).

## 3. Figures

### Heatmap (`heatmap_L{layer}.{pdf,png}`)
- Word×word cosine at fixed board+layer, **`no_social` vs `with_social` side by
  side**.
- **Lower triangle only** (matrix is symmetric; the trivial unit diagonal is
  masked too) — no redundant grid.
- Colorblind-safe **diverging `RdBu_r` centred at 0**, symmetric vmin/vmax shared
  across the pair, so positive vs negative association reads directly.
- **Identical word ordering across both panels** (type blocks hint → target →
  assassin → neutral, alphabetical within), so the shared board cells can be
  diffed cell-by-cell. Axis labels coloured by type.
- The social panel contains **all giver-feature words the board specifies**
  (2–9; see §5); these append **only at the end** of the with-social panel, after
  the shared board words, so the shared block stays aligned.

### Projection (`umap_{condition}_layers.{pdf,png}`)
- Multi-panel, one panel per representative layer (≈6 spread across depth,
  endpoints always included).
- Coloured by **true word type**; **hint = diamond**, **targets tagged `[T]`**,
  arrow from the hint to its true nearest neighbour in cosine space — only the
  arrow's DIRECTION is meaningful, its projected 2D length is not (stated in the
  caption). The six layers span network depth evenly (embeddings, quarter, mid,
  three-quarter, final); this even-depth spread is also stated in the caption.
- Rendered with the fixed reducer **UMAP (cosine)** (see §4.6 for the choice);
  each panel prints its **T / C / Shepard scores**, and the full UMAP/t-SNE/PCA
  comparison is written to `dr_quality_{condition}.csv` (column `selected` marks
  the rendered method).

### 3.3 Figure-construction details (for reproduction)

Common style (`viz/style.py`, `apply_publication_style`): sans-serif (Arial →
Helvetica → DejaVu Sans fallback), base font 8 pt, despined axes, white
background, `savefig.dpi=300`, `pdf.fonttype=42` (editable embedded text). Word
types use the **Okabe-Ito** colorblind-safe palette with all-distinct marker
shapes (so type survives in grayscale): hint = vermillion diamond, target =
bluish-green circle, assassin = black ✕, neutral = orange square, giver feature =
blue (#0072B2) triangle. Giver-feature uses the darker blue rather than sky-blue
so its axis-label colour does not collapse against Neutral amber in grayscale
(luminance ~87 vs ~162); see the Task 2 grayscale/colorblind audit.

- **Heatmap** (`viz/heatmap.py`): one shared word ordering across both panels
  (type blocks then alphabetical; giver features appended last in the social
  panel) so cells diff directly; matrix = clipped `X̂ X̂ᵀ` on L2-normalised rows; upper triangle
  **and** diagonal masked (`np.triu(..., k=0)`); `cmap="RdBu_r"`, `center=0`, and
  **symmetric limits** `vmin=−v, vmax=+v` where `v = max(|off-diagonal|)` across
  both panels (≥0.1) so the pair is directly comparable; cells annotated when
  `n ≤ 28`; one shared colorbar; axis labels coloured by word type plus a patch
  legend.
- **Projection** (`viz/embedding.py`): markers carry a thin **white edge**
  (hint: black edge, larger) to separate overlapping points; the hint→nearest
  connector is an arrow drawn *under* the markers; word labels are repelled with
  **`adjustText`** (`expand=(1.15, 1.3)`, leader lines in light grey) so text does
  not overlap points or other text; ≈6 layers per figure in a 3-column grid;
  per-panel score box and a shared word-type legend.
- **Determinism:** board sampling, t-SNE, and UMAP all take the same `--seed`
  (default 42); reruns are identical.

## 4. Dimensionality-reduction validation (the key requirement)

This section is written to be reusable in the thesis methodology chapter: it
states the problem, the metrics with their definitions, the experimental
protocol, the full results, and the resulting design decision.

### 4.1 The problem

A 2D projection is only worth showing if the *neighbourhood structure* of the
original space survives the reduction. Our original space is compared by **cosine
similarity** (each word vector is the pooled sub-token hidden state; magnitude is
not the quantity of interest). Two failure modes must be controlled:

- **false neighbours** — points placed close in 2D that were far apart in cosine
  space (the projection *invents* proximity);
- **missing neighbours** — points that were close in cosine space but are torn
  apart in 2D (the projection *destroys* proximity).

A single global statistic (e.g. a correlation of distances) cannot separate
these; we therefore use rank-based local metrics plus one global metric.

### 4.2 Metrics (definitions)

Let `n` be the number of words on a board at a fixed layer. For point *i*, let
`r(i, j)` be the rank of *j* among *i*'s neighbours (1 = nearest) in a given
space. High-dimensional ranks use **cosine distance** `d_cos = 1 − cosθ`;
2D ranks use **Euclidean distance**. Fix a neighbourhood size *k* (default 5,
clamped to `(n−1)//2` for small boards).

- **Trustworthiness** `T(k) ∈ [0,1]` (Venna & Kaski, 2001), via scikit-learn
  with `metric="cosine"`:

  ```
  T(k) = 1 − (2 / (n·k·(2n − 3k − 1))) · Σ_i Σ_{j∈U_k(i)} (r_high(i,j) − k)
  ```

  where `U_k(i)` is the set of points among *i*'s *k* nearest in **2D** that were
  **not** among its *k* nearest in cosine space. Penalises false neighbours.

- **Continuity** `C(k) ∈ [0,1]` — the dual, implemented in `metrics.py`: same
  formula with the roles of the two spaces swapped (`V_k(i)` = true cosine
  neighbours demoted out of the 2D neighbourhood, weighted by their 2D rank).
  Penalises missing neighbours.

- **Shepard correlation** `ρ ∈ [−1,1]` — the Spearman rank correlation between
  all `n(n−1)/2` pairwise **cosine** distances (HD) and **Euclidean** distances
  (2D); the numeric summary of the Shepard diagram. Measures *global* monotone
  distance preservation.

A perfect isometric embedding scores `T = C = 1`, `ρ ≈ 1`; this is asserted in
`tests/test_viz.py::test_metrics_isometric_embedding_is_near_perfect`.

### 4.3 Candidate reducers (all cosine-aware)

| Reducer | Configuration | Why cosine-aware |
|---|---|---|
| **t-SNE** | `sklearn.manifold.TSNE`, `metric="cosine"`, `init="pca"`, `perplexity = clip((n−1)/3, 2, 30)`, `random_state=seed` | neighbour affinities computed directly from cosine distances |
| **UMAP** | `umap.UMAP`, `metric="cosine"`, `n_neighbors = clip(n−1, 2, 15)`, `min_dist=0.1`, `random_state=seed` | fuzzy simplicial set built on cosine distances |
| **PCA** | `sklearn.decomposition.PCA(2)` on **L2-normalised** vectors | on the unit sphere, Euclidean distance is a monotone function of cosine distance, so linear PCA operates in cosine geometry |

`perplexity`/`n_neighbors` are adapted to the small board sizes so the reducers
remain well defined; vectors are upcast f16→f32 and L2-normalised before every
fit.

### 4.4 Protocol

For each (board, layer, condition) the pipeline fits **all three** reducers once
(`embedding.embed_all`), computes `T`, `C`, `ρ` for each, and records every row
to `dr_quality_{condition}.csv` (with a `selected` flag). A single comparable
score is the mean of the three diagnostics, treating NaN as 0 and clamping the
Shepard term to its non-negative part:

```
combined = mean( T, C, max(0, ρ) )
```

This protocol is the *audit*; the **rendered** reducer is fixed (§4.6).

### 4.5 Results (BERT, first run: 5 boards × 6 layers × 2 conditions = 60 cases)

Reproduce with: `codenames-experiment visualize --model bert --n-boards 5`,
then aggregate the `dr_quality_*.csv` files.

**(a) Which reducer scores best, per case**

| Method | Times best (of 60) |
|---|---|
| t-SNE (cosine) | 34 |
| PCA (normalised) | 17 |
| UMAP (cosine) | 9 |

**(b) Mean diagnostics per method, over all 60 cases**

| Method | Trustworthiness | Continuity | Shepard ρ | Combined |
|---|---|---|---|---|
| **t-SNE** | **0.849** | **0.848** | 0.632 | **0.776** |
| PCA | 0.808 | 0.834 | **0.658** | 0.767 |
| UMAP | 0.824 | 0.820 | 0.541 | 0.728 |

**(c) Mean diagnostics by layer (best-per-case method)**

| Layer | n | Trustworthiness | Continuity | Shepard ρ |
|---|---|---|---|---|
| 0 (embeddings) | 10 | 0.854 | 0.820 | 0.554 |
| 2 | 10 | 0.846 | 0.831 | 0.561 |
| 5 | 10 | 0.852 | 0.871 | 0.694 |
| 7 | 10 | 0.854 | 0.880 | 0.693 |
| 10 | 10 | 0.847 | 0.869 | 0.744 |
| 12 (final) | 10 | 0.870 | 0.879 | 0.783 |

### 4.6 Decision and its justification

**The figures render a single reducer — UMAP with `metric="cosine"`.** A fixed
method (rather than the per-panel best) is used for **comparability across
panels**: letting the reducer vary mixes incomparable layouts (PCA axes are
linear; t-SNE/UMAP are not), which misleads when scanning a board across layers.
UMAP is the chosen fixed method for this work, giving one consistent visual
grammar and aligning with the projection technique used elsewhere in the thesis.

**Honest caveat (kept for the thesis):** on the comparison above, UMAP is *not*
the top scorer on these small per-board sets — t-SNE has the best mean
trustworthiness (0.849) and continuity (0.848) and is the per-case best in 34/60
cases, while UMAP has the weakest Shepard (0.541) and is rarely the single best.
With ≈17–26 points per board UMAP operates below its comfortable density regime.
The rendering choice therefore trades a small amount of measured local fidelity
for consistency with the wider thesis. Because all three reducers are still
scored, each rendered UMAP panel **prints its own `T / C / ρ`** and the full
audit lives in `dr_quality_{condition}.csv` — so any panel's reliability can be
checked, and switching back is a one-line change (`PREFERRED_METHOD` in
`codenames/viz/embedding.py`).

### 4.7 How to read the numbers (caveats for the thesis)

- **Local fidelity is good** (`T, C ≈ 0.82–0.88` for the selected method): the
  near/far relationships in the figures are largely trustworthy.
- **Global distance preservation is modest at shallow layers** (`ρ ≈ 0.55` at the
  embedding layer) and **improves monotonically with depth** (`ρ ≈ 0.78` at the
  final layer): deeper representations flatten into a structure a 2D map captures
  more honestly. **Do not** read absolute inter-cluster distances off shallow-
  layer panels.
- Small `n` makes every metric higher-variance; treat per-panel scores as
  indicative, and prefer the aggregated tables above for general claims.

## 5. Findings from the BERT run

- **Giver-feature clustering (social condition).** On the demographically rich
  boards the giver-feature words form a **distinct cluster** in the projection,
  separated from the board vocabulary across all layers — a clean visual of the
  social context occupying its own region of representation space.
- **High global anisotropy.** The heatmaps are overwhelmingly warm (most cosines
  positive, ~0.3–0.7) with few near-zero/negative cells, the visual signature of
  the high anisotropy the scalar metrics report for BERT.
- **Depth-dependent legibility.** Both the rising Shepard correlation and the
  cleaner deep-layer projections corroborate that geometry becomes more
  2D-faithful with depth.

### 5.4 Worked example — board 2521 (`cream → mammoth`), a low-quality human clue

A useful validation case surfaced during review. Board **2521** is a single-target
turn whose human clue is `hint = "cream"`, `target = "mammoth"` — a semantically
odd association. The figures let us interrogate it directly:

- In the projection (`umap_no_social_layers`), the **arrow from the hint
  (`cream`) to its nearest word in cosine space points to `drop` / `wake` /
  `slip`, never to `mammoth`**, at every layer. The model does not reconstruct
  the human association.
- This matches the scalar outputs: `predicted_word_mean = "drop"`,
  `correct_mean = False` — i.e. BERT's nearest-to-hint candidate is not the
  intended target.

So the visualization corroborates a finding rather than contradicting one: this
is an example of an idiosyncratic human clue whose association is **not** encoded
geometrically by the model. (For contrast, board 1263 `elephant → mammoth` is
predicted correctly — `mammoth` is the nearest candidate to `elephant`.) Such
cases are worth flagging in the thesis as the lower tail of human-clue quality in
CULTURAL CODES.

> Note: the same word can play different roles on different boards. `mammoth` is
> the **target** on boards 2521/1263/4054 but an **assassin** on board 565 (whose
> clue is `bright → {genius, light}`). The figures encode role by colour/marker,
> so an assassin is never mistaken for a target once labels are read.

### 5.5 Data provenance and faithfulness

`row_id` is assigned in `codenames/data.py::load_dataset` as the 0-based position
in `clue_generation.csv` after `pd.read_csv(...).reset_index(drop=True)` — no
rows are dropped first — so `row_id = N` corresponds to the *(N+2)*-th line of the
CSV (header + 0-based). The visualization carries `hint`, `targets`, `black`,
`tan`, and `giver_features` straight from that row; it performs **no
re-derivation** of board roles. A surprising play (like `cream → mammoth`) is
therefore faithful to the source dataset, not an artefact of the pipeline; it can
be confirmed against the corresponding `clue_generation.csv` row.

## 6. Reproducibility & tests

- Board sampling, t-SNE, and UMAP are all seeded (default 42); reruns are
  deterministic.
- `tests/test_viz.py` adds 18 unit tests (cosine-matrix symmetry/limits, triangle
  masking, type-block ordering, index↔NPZ join + invalid-vector dropping,
  reproducible sampling, `giver_features` parsing, DR metrics on a known
  isometric embedding, reducer selection). The reducer-dependent tests are
  `importorskip`-guarded so the **core suite stays green without `[viz]`
  installed**. Full suite: **60 passed**.
- Confirmed the core `codenames.cli` import pulls in **none** of
  matplotlib/umap/sklearn/seaborn (lazy-import guard intact).

## 7. Scope, limitations, things to note

- **Only `output/bert/` exists locally**, so figures were produced for BERT. The
  pipeline is model-agnostic and auto-discovers `mistral`, `qwen`, etc. the
  moment their output folders are present (`--all` sweeps all of them).
- **Giver-feature count is a property of the source board, not the pipeline.**
  Boards specify 2–9 demographic attributes (mean 7.5); the figures show exactly
  those that have stored vectors. Boards with few attributes (e.g. board 565 has
  2) are faithful, not truncated.
- **Vectors are the retained subsample only** (the 100 boards with raw vectors),
  so board sampling necessarily draws from that subsample.
- **Rendered images are not committed** (`.gitignore`); the per-board
  `dr_quality_*.csv` and this report are. Regenerate images with the command
  below.

## 8. Usage

```bash
pip install -e ".[viz]"
codenames-experiment visualize --model bert            # 5 boards, mean pooling
codenames-experiment visualize --model bert --n-boards 8 --layers 0,6,12
codenames-experiment visualize --all                   # every model under output/
codenames-experiment visualize --all --boards 565,2521 # same fixed boards, every model
```

Output: `visualization/{model}/board_{row_id}/` containing `heatmap_L*.{pdf,png}`,
`umap_{condition}_layers.{pdf,png}`, and `dr_quality_{condition}.csv`.

## 9. Cross-model comparability (same boards across models)

To compare models on the *same* board, the chosen boards must be identical
across model runs. Board selection is deterministic in `(available id-set,
seed)`, but the available id-set is only the same across models when they were
extracted with the same contract **seed (42) and run size**; differing run sizes
or extraction errors make the per-model subsamples diverge. Two mechanisms make
the comparison explicit rather than incidental:

- **`--boards 565,2521,...`** — pin exact board `row_id`s; the same set is used
  for every model (any board not available for a given model is skipped with a
  warning). Recommended for figures that appear side by side in the thesis.
- **`--all` without `--boards`** — boards are sampled from the **intersection**
  of boards available across *all* discovered models, so the selection is
  guaranteed present everywhere and identical for every model. If the models have
  no common boards (e.g. different run sizes), the pipeline warns loudly and
  falls back to per-model sampling (not comparable).
