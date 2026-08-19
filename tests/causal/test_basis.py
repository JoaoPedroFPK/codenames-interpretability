import numpy as np
import pytest
from transformers import AutoTokenizer

from codenames.causal.basis import (
    ROLES,
    ROLE_INDEX,
    collapse_grid,
    role_of_each_token,
    role_positions,
)
from codenames.prompts import build_prompt

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TINY)


def _prompt(tok, hint, candidates):
    text, _ = build_prompt(hint=hint, candidates=candidates, giver_features={},
                           use_social_context=False, tokenizer=tok,
                           chat_template_strategy="raw")
    return text


def _roles(tok, hint, candidates, clean_target, donor_target):
    prompt = _prompt(tok, hint, candidates)
    return prompt, role_of_each_token(
        tok, prompt, hint=hint, candidates=candidates,
        clean_target=clean_target, donor_target=donor_target)


# --- the basis labels every token ------------------------------------------

def test_every_token_carries_a_role(tok):
    prompt, roles = _roles(tok, "water", ["moon", "sea", "ship"], "sea", "moon")
    n = len(tok(prompt, return_offsets_mapping=True,
                add_special_tokens=False)["offset_mapping"])
    assert roles.shape == (n,)
    assert set(np.unique(roles)).issubset(set(ROLE_INDEX.values()))


def test_role_counts_sum_to_the_sequence_length(tok):
    """Coverage must be total: a token silently dropped from every role would
    be a hole in the grid that no aggregate could reveal."""
    _, roles = _roles(tok, "water", ["moon", "sea", "ship"], "sea", "moon")
    pos = role_positions(roles)
    assert sum(len(v) for v in pos.values()) == roles.shape[0]


def test_the_hint_and_both_contrast_words_are_located(tok):
    _, roles = _roles(tok, "ocean", ["moon", "sea", "ship"], "sea", "moon")
    pos = role_positions(roles)
    assert pos.get("hint"), "the corrupted span must be locatable"
    assert pos.get("cand_target"), "the clean target must be its own role"
    assert pos.get("cand_donor"), "the donor target must be its own role"


def test_target_and_donor_are_not_pooled_with_the_other_candidates(tok):
    """The symmetric counterfactual contrasts exactly these two words; pooling
    them into `cand_other` would average away the contrast under study."""
    _, roles = _roles(tok, "ocean", ["moon", "sea", "ship"], "sea", "moon")
    pos = role_positions(roles)
    assert set(pos["cand_target"]).isdisjoint(pos.get("cand_other", []))
    assert set(pos["cand_donor"]).isdisjoint(pos.get("cand_other", []))
    assert set(pos["cand_target"]).isdisjoint(pos["cand_donor"])


def test_list_numbering_is_not_pooled_into_the_preamble(tok):
    """The ordinal markers ("1. ", "\\n2. ") fell through to `prefix` at first,
    which both mislabelled ~45 tokens of a real prompt and hid the one signal
    that diagnoses the §3.3.3 ordinal-slot confound."""
    cands = ["moon", "sea", "ship", "castle", "engine"]
    prompt = _prompt(tok, "ocean", cands)
    roles = role_of_each_token(tok, prompt, hint="ocean", candidates=cands,
                               clean_target="sea", donor_target="moon")
    pos = role_positions(roles)
    ids = tok(prompt)["input_ids"]
    assert pos.get("list_scaffold"), "the numbering must have its own role"
    scaffold = tok.decode([ids[i] for i in pos["list_scaffold"]])
    assert "2." in scaffold or "3." in scaffold
    prefix = tok.decode([ids[i] for i in pos["prefix"]])
    assert "2." not in prefix, "numbering leaked into the preamble"


def test_the_generating_position_is_its_own_role(tok):
    _, roles = _roles(tok, "water", ["moon", "sea", "ship"], "sea", "moon")
    assert roles[-1] == ROLE_INDEX["final"]
    assert role_positions(roles)["final"] == [roles.shape[0] - 1]


def test_hint_precedes_the_candidates(tok):
    """Anchors are resolved by search, so a regression that matched the wrong
    occurrence would show up as an out-of-order layout."""
    _, roles = _roles(tok, "ocean", ["moon", "sea", "ship"], "sea", "moon")
    pos = role_positions(roles)
    assert max(pos["hint"]) < min(pos["cand_target"])


def test_a_candidate_word_also_appearing_before_the_list_is_not_mislabelled(tok):
    """`hint` is searched from the start and `cand:` from the list anchor, so a
    hint that repeats a candidate word must not steal that candidate's span."""
    _, roles = _roles(tok, "sea", ["moon", "sea", "ship"], "sea", "moon")
    pos = role_positions(roles)
    assert pos.get("hint") and pos.get("cand_target")
    assert max(pos["hint"]) < min(pos["cand_target"])


def test_a_candidate_inside_a_longer_candidate_is_not_stolen(tok):
    """The trap that would have corrupted the grid silently: a bare search for
    "sea" matches the first three characters of "seashore". Codenames boards
    really carry such pairs (ICE / ICE CREAM, NEW / NEW YORK), so the span must
    be anchored to the numbered list entry."""
    cands = ["seashore", "sea", "moonlight", "moon"]
    prompt = _prompt(tok, "ocean", cands)
    roles = role_of_each_token(tok, prompt, hint="ocean", candidates=cands,
                               clean_target="sea", donor_target="moon")
    ids = tok(prompt)["input_ids"]
    pos = role_positions(roles)
    assert tok.decode([ids[i] for i in pos["cand_target"]]).strip() == "sea"
    assert tok.decode([ids[i] for i in pos["cand_donor"]]).strip() == "moon"


def test_roles_index_the_same_tokens_the_forward_pass_sees(tok):
    """Offsets are taken with the forward pass's own add_special_tokens
    default, so role index i is hidden-state index i. If these ever diverge the
    grid is misaligned by a constant and nothing else would reveal it."""
    prompt = _prompt(tok, "ocean", ["moon", "sea", "ship"])
    n_forward = len(tok(prompt)["input_ids"])
    roles = role_of_each_token(tok, prompt, hint="ocean",
                               candidates=["moon", "sea", "ship"],
                               clean_target="sea", donor_target="moon")
    assert roles.shape[0] == n_forward


def test_n_positions_guard_pads_rather_than_misaligning(tok):
    prompt = _prompt(tok, "ocean", ["moon", "sea", "ship"])
    base = role_of_each_token(tok, prompt, hint="ocean",
                              candidates=["moon", "sea", "ship"],
                              clean_target="sea", donor_target="moon")
    longer = role_of_each_token(tok, prompt, hint="ocean",
                                candidates=["moon", "sea", "ship"],
                                clean_target="sea", donor_target="moon",
                                n_positions=base.shape[0] + 2)
    assert longer.shape[0] == base.shape[0] + 2
    hint_base = np.flatnonzero(base == ROLE_INDEX["hint"])
    hint_long = np.flatnonzero(longer == ROLE_INDEX["hint"])
    assert list(hint_base) == list(hint_long), "spans must not shift"


@pytest.mark.parametrize("strategy,social", [
    ("raw", False), ("mistral_inst", False), ("chatml", False), ("raw", True),
])
def test_every_template_and_condition_locates_the_anchors(tok, strategy, social):
    """One basis must serve all three chat templates and both conditions; the
    anchors live in the shared instruction body, which is why it can."""
    cands = ["moon", "sea", "ship"]
    feats = {"giver.gender": "Female", "giver.country": "Brazil"} if social else {}
    text, _ = build_prompt(hint="ocean", candidates=cands, giver_features=feats,
                           use_social_context=social, tokenizer=tok,
                           chat_template_strategy=strategy)
    roles = role_of_each_token(tok, text, hint="ocean", candidates=cands,
                               clean_target="sea", donor_target="moon",
                               n_positions=len(tok(text)["input_ids"]))
    pos = role_positions(roles)
    assert pos.get("hint") and pos.get("cand_target") and pos.get("cand_donor")
    assert sum(len(v) for v in pos.values()) == roles.shape[0]


# --- the basis is what makes cross-turn aggregation defined ----------------

def test_different_length_prompts_collapse_to_the_same_width(tok):
    """The bug this basis exists for: two turns with different candidate counts
    produce grids of different width, and `total + grid` raised
    `operands could not be broadcast together with shapes (33,118) (33,108)`."""
    short = _roles(tok, "water", ["moon", "sea"], "sea", "moon")
    long = _roles(tok, "water", ["moon", "sea", "ship", "castle", "engine"],
                  "sea", "moon")
    n_short, n_long = short[1].shape[0], long[1].shape[0]
    assert n_short != n_long, "fixture must actually differ in length"

    g_short = collapse_grid(np.random.default_rng(0).normal(size=(4, n_short)), short[1])
    g_long = collapse_grid(np.random.default_rng(1).normal(size=(4, n_long)), long[1])
    assert g_short.shape == g_long.shape == (4, len(ROLES))
    assert np.isfinite(g_short + g_long).any()


def test_collapse_keeps_the_signed_extreme_not_the_sum(tok):
    """A sum would let a wide role outscore a narrow one by arithmetic alone;
    a mean would dilute a sharp single-token effect."""
    roles = np.array([ROLE_INDEX["hint"], ROLE_INDEX["hint"], ROLE_INDEX["final"]])
    grid = np.array([[0.5, -2.0, 0.1]])
    out = collapse_grid(grid, roles)
    assert out[0, ROLE_INDEX["hint"]] == pytest.approx(-2.0)
    assert out[0, ROLE_INDEX["final"]] == pytest.approx(0.1)


def test_absent_roles_are_nan_not_zero(tok):
    """An absent role must not be averaged in as a measured null effect."""
    roles = np.array([ROLE_INDEX["hint"], ROLE_INDEX["final"]])
    out = collapse_grid(np.array([[1.0, 2.0]]), roles)
    assert np.isnan(out[0, ROLE_INDEX["cand_donor"]])
    assert not np.isnan(out[0, ROLE_INDEX["hint"]])


def test_collapse_tolerates_a_grid_wider_than_the_role_vector(tok):
    roles = np.array([ROLE_INDEX["hint"], ROLE_INDEX["final"]])
    out = collapse_grid(np.zeros((3, 5)), roles)
    assert out.shape == (3, len(ROLES))


def test_roles_are_a_fixed_vocabulary():
    """The grid's second axis is a constant of the study, not a property of a
    draw, so conditions and models stay directly comparable."""
    assert ROLES[0] == "prefix"
    assert {"hint", "cand_target", "cand_donor", "final", "generation"} <= set(ROLES)
    assert len(set(ROLES)) == len(ROLES)


# --- per-candidate token spans (T3d: the equalisation intervention) ---------

def test_span_positions_gives_one_token_range_per_candidate_and_the_hint(tok):
    """The equalisation intervention rotates each candidate span separately, so
    it needs per-candidate ranges, not the collapsed cand_other role."""
    from codenames.causal.basis import span_positions

    candidates = ["moon", "sea", "ship"]
    prompt = _prompt(tok, "water", candidates)
    spans = span_positions(tok, prompt, hint="water", candidates=candidates)

    assert set(spans) == {"hint", "moon", "sea", "ship"}
    for name, (lo, hi) in spans.items():
        assert 0 <= lo < hi, name
    ids = tok(prompt)["input_ids"]
    for word in candidates:
        lo, hi = spans[word]
        assert word in tok.decode(ids[lo:hi]).lower()


def test_span_positions_does_not_match_inside_a_longer_candidate(tok):
    """ICE must not land on the first three characters of ICE CREAM."""
    from codenames.causal.basis import span_positions

    candidates = ["ice cream", "ice", "sea"]
    prompt = _prompt(tok, "cold", candidates)
    spans = span_positions(tok, prompt, hint="cold", candidates=candidates)
    assert spans["ice"] != spans["ice cream"]
    assert spans["ice"][0] >= spans["ice cream"][1]
