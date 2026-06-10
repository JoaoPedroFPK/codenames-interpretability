# Qualitative analysis of the layer-wise geometry figures

**Scope.** This report reads the figures rendered by
`codenames-experiment visualize --all --n-boards 10` — 7 models × the same 10
shared boards × per-layer cosine **heatmaps** and **UMAP projections** (1,120
images under `visualization/`). Every claim below is anchored in a specific
figure *and* in the behavioural/geometry numbers from the `*_general_*.csv`
outputs, so the pictures are read as evidence rather than described as pixels.

> The referenced PNGs live under `visualization/` (gitignored local artifacts).
> Paths are relative to this file; open them alongside the report.

**The 10 shared boards** (`hint → target(s) | assassins`), identical across every
model by construction (cross-model board intersection, seed 42):

| board | hint → target | board | hint → target |
|------:|---------------|------:|---------------|
| 352  | novelty → novel | 4802 | mark → pass |
| 370  | scam → capital, racket | 5096 | tour → trip |
| 735  | planets → space | 6288 | **insurance → agent** |
| 2159 | place → club, sub | 6983 | **bio → war** |
| 3641 | **test → check** | 7425 | role → lead |

Reading key for every figure: **red ◆ = hint**, **green ▲ = target [T]**,
**✕ = assassin**, **tan ■ = neutral**, **blue ◆ = giver-feature word** (social
condition only). In the UMAP panels a green line connects hint→target; **its
length is the single most informative signal in the whole figure set.**

---

## Finding 1 — The hint→target connector length is a visual proxy for *both* binding and behavioural success

The clearest pattern across every UMAP figure: in models/boards that solve the
clue, the target migrates toward the hint as depth increases (the connector
*shortens*); where they fail, it stays long.

**Trained model, solvable board — qwen, board 6288 (`insurance → agent`):**

![qwen 6288 no_social UMAP](../visualization/qwen_outputs/board_6288/umap_no_social_layers.png)

`insurance` (◆) and `agent` (▲) sit far apart at **Layer 0**, then bind tightly
by **Layer 11 (Mid)** and stay close through the deep layers. This is the
`raw_margin = cos(hint,target) − cos(hint,non-targets) = +0.291` (the largest in
the whole set) made spatial — qwen answers this board correctly.

**Random control, same board — random_qwen, board 6288:**

![random_qwen 6288 no_social UMAP](../visualization/random_qwen_outputs/board_6288/umap_no_social_layers.png)

The connector stays **long and arbitrary at every layer** — by Layer 28
`insurance` is far-left and `agent` far-right (maximally separated). No binding
ever emerges; margin = −0.003.

**Trained model, *unsolvable* board — qwen & bert, board 6983 (`bio → war`):**

![qwen 6983 no_social UMAP](../visualization/qwen_outputs/board_6983/umap_no_social_layers.png)
![bert 6983 no_social UMAP](../visualization/bert_outputs/board_6983/umap_no_social_layers.png)

Here the *trained* models look like the random control: `war` never binds to
`bio`, the connector stays long at every layer (qwen margin −0.063, bert
−0.135). The `bio→war` association is weak/indirect, and the geometry says so.

> **Insight.** Trained models do not bind universally — they bind when the
> hint↔target association is strong (6288, 3641) and fail when it is weak (6983),
> at which point their geometry is indistinguishable from a random network.
> *The connector length is the behaviour.*

---

## Finding 2 — Trained vs random is a difference of *structure/variance*, not cosine magnitude

The heatmaps reveal that the random controls fail in **two opposite-looking
ways**, yet for the *same* underlying reason.

**random_bert, board 6288, Layer 12 — anisotropic cone:**

![random_bert 6288 heatmap L12](../visualization/random_bert_outputs/board_6288/heatmap_L12.png)

The entire lower triangle is uniformly **red**: every word pair has cosine ≈0.65.
All representations have collapsed into a narrow cone.

**random_qwen, board 6288, Layer 28 — orthogonal collapse:**

![random_qwen 6288 heatmap L28](../visualization/random_qwen_outputs/board_6288/heatmap_L28.png)

The entire triangle is uniformly **white**: every pair has cosine ≈0.0. Random
Qwen's weights produce near-orthogonal representations.

**bert (trained), board 6288, Layer 5 — textured structure:**

![bert 6288 heatmap L05](../visualization/bert_outputs/board_6288/heatmap_L05.png)

Intermediate red, but **differentiated** — some cells visibly darker than others.
The model assigns *different* similarities to different word pairs.

> **Insight.** The diagnostic of a trained network is not whether the heatmap is
> hot or cold — it is whether it has **texture**. Random networks are *uniform*
> (every pair equal ⇒ targets are indistinguishable from non-targets ⇒
> `raw_margin ≈ 0`), regardless of whether the cone is hot (random_bert,
> cos≈0.65) or cold (random_qwen, cos≈0.0). Trained networks carve variance into
> the similarity matrix. This single idea unifies the two random failure modes
> and explains why both yield ~zero margin despite opposite cosine levels.

The DR-quality metrics confirm it quantitatively: mean Shepard correlation
(global distance preservation) is **0.57 for trained qwen/bert** versus **0.34
(random_qwen) and 0.21 (random_bert, the worst)** — the uniform cone has almost
no distance structure to preserve.

---

## Finding 3 — Semantic binding emerges with depth, and peaks at architecture-characteristic layers

Reading the UMAP panels left-to-right (shallow→deep) shows *when* each
architecture binds the target.

- **Encoders bind early and hold.** bert binds `agent` to `insurance` already at
  **Layer 2** (tiny connector) and maintains it; t5 keeps the pair adjacent at
  *every* layer.

  ![t5 6288 no_social UMAP](../visualization/t5_outputs/board_6288/umap_no_social_layers.png)

  t5 shows the **tightest, most stable** binding in the entire set (margin
  +0.214), with assassins (`soul`, `center`, `unicorn`) held at the periphery
  throughout.

- **Causal LMs bind in the middle.** qwen binds at **Layer 11** (Finding 1);
  mistral binds early but modestly (+0.068); modernbert binds with a visible
  **deep-layer wobble** — the target drifts away around Layer 18 and returns by
  Layer 22.

This is corroborated by the DR-quality depth trend (combined embeddability by
normalized depth):

| model | L0–early | early–mid | mid–deep | final |
|-------|:--------:|:---------:|:--------:|:-----:|
| qwen (trained) | 0.52 | 0.53 | **0.65** | 0.60 |
| bert (trained) | 0.52 | **0.60** | 0.59 | 0.61 |
| random_qwen | 0.39 | 0.33 | 0.34 | 0.31 |
| random_bert | 0.19 | 0.22 | 0.22 | 0.22 |

Trained models' structure **peaks at the layer where binding happens** (qwen
mid-deep, bert early-and-sustained); random models are **flat-low (random_bert)
or *decline* with depth (random_qwen)** — structure never emerges.

---

## Finding 4 — Architecture leaves a geometric signature

- **Anisotropy regime differs by family.** Encoders (bert, modernbert) live in a
  high-cosine cone (typical pair cos ≈0.5–0.8); causal LMs (qwen, mistral) sit
  lower (≈0.1–0.35). Visible directly as the overall warmth of the no-social
  heatmaps, and in the absolute `mean_cos_hint_*` columns.
- **All models grow more anisotropic with depth.** qwen board 6288 heatmaps go
  from near-white at Layer 0 to saturated red at Layer 28:

  ![qwen 6288 heatmap L00](../visualization/qwen_outputs/board_6288/heatmap_L00.png)
  ![qwen 6288 heatmap L28](../visualization/qwen_outputs/board_6288/heatmap_L28.png)

- **Projection preference encodes intrinsic geometry.** Of the three reducers
  scored per layer (UMAP is rendered; t-SNE/PCA also scored), the
  *best*-scoring is **t-SNE 46% / PCA 44% / UMAP 9%**. Causal LMs prefer **PCA**
  (mistral 64%, qwen 60% of layers) — their geometry is dominated by a few
  global linear axes. **random_qwen prefers t-SNE 69%** — its near-orthogonal
  simplex has only local neighbourhood structure to recover. (Caveat: the
  rendered UMAP panels are for visual consistency and are rarely the
  metrically-best projection; trust the `dr_quality_*.csv`, not the 2-D
  coordinates, for quantitative claims.)

---

## Finding 5 — Social context adds a self-contained semantic cluster and mildly blurs binding

**qwen, board 6288, with_social:**

![qwen 6288 with_social UMAP](../visualization/qwen_outputs/board_6288/umap_with_social_layers.png)

The giver-feature words (blue ◆: *"united states"*, *"master's degree"*,
*"constitutional monarchy"*, …) form **their own cluster**, partly separated from
the board words. The hint→target binding still appears in the mid layers but is
**noisier** than in the no-social condition — the social distractors pull on the
geometry. The same effect is visible in the with-social heatmaps as a dense
dark-red block among the giver-feature rows (those words are highly mutually
similar).

> **Insight.** Adding the giver's social context does not reorganise the board's
> semantics; it appends a coherent secondary neighbourhood (the giver features)
> and slightly reduces the crispness of hint→target binding.

---

## Finding 6 — The patterns generalise across boards

The same reading holds on a second solvable board — **qwen, board 3641
(`test → check`):**

![qwen 3641 no_social UMAP](../visualization/qwen_outputs/board_3641/umap_no_social_layers.png)

`check` tightens toward `test` through the mid-deep layers (peak ≈ Layer 17),
with a slight final-layer wobble — the same causal-LM "bind in the middle"
signature seen on board 6288. Across all 10 boards the behavioural numbers tell
the consistent story: trained models produce positive margins on boards with
strong hint↔target links (6288, 3641, 352, 6288) and ~zero-or-negative margins
on weak ones (6983, 4802), while **the random controls hover at margin ≈ 0 on
every board**.

---

## Summary of extracted insights

1. **Connector length = binding = behaviour.** The hint→target distance in the
   UMAP panels predicts whether the model solves the board; it shortens with
   depth on solvable boards and stays long on hard ones (Findings 1, 6).
2. **Trained ≠ random is about *variance*, not cosine level.** Random networks
   produce *uniform* similarity matrices (two flavours: anisotropic hot cone vs
   orthogonal cold collapse); training introduces *texture* — the ability to
   make targets closer than non-targets (Finding 2).
3. **Binding emerges with depth, on an architecture-specific schedule:**
   encoders early (bert L2, t5 throughout), causal LMs mid (qwen L11), with
   deep-layer wobble in mistral/modernbert (Finding 3).
4. **Architecture is geometrically legible:** encoder high-cosine cones vs causal
   moderate cones; universal depth-anisotropy growth; PCA-friendly causal
   geometry vs t-SNE-only random_qwen (Finding 4).
5. **Social context appends a giver-feature cluster** and mildly blurs binding
   rather than reorganising it (Finding 5).
6. **Honest limit:** even trained models fail on weak associations, and when they
   do, their geometry is indistinguishable from a random network — which is
   exactly why the random-init controls are the right baseline (Findings 1, 6).

**Methods note.** Quantitative anchors (`raw_margin`, `mean_cos_hint_targets`,
`correct`, Shepard/trustworthiness/continuity) come from the per-board
`*_general_*.csv` and `dr_quality_*.csv`; the figures visualise the same
quantities. UMAP coordinates are for inspection only — the report never derives
a number from a 2-D position.
