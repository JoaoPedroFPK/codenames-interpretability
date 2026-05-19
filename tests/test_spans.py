"""Tests for spans.py — pooling, cosine, span detection.

No model dependency; uses torch tensors directly.
"""

import numpy as np
import pytest
import torch

from codenames.spans import (
    cosine_similarity_np,
    find_token_spans,
    max_norm_pool_span,
    mean_pool_span,
    pool_span,
)


def test_cosine_zero_norm_returns_zero():
    """Zero-norm vectors return 0.0, not NaN. Part of the bit-identity contract."""
    a = np.zeros(8, dtype=np.float32)
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert cosine_similarity_np(a, b) == 0.0
    assert cosine_similarity_np(b, a) == 0.0


def test_cosine_identical_returns_one():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity_np(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_returns_zero():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity_np(a, b) == pytest.approx(0.0)


def test_mean_pool_returns_float16():
    """Mean pool returns float16. Part of the bit-identity contract."""
    hs = torch.randn(5, 8)
    out = mean_pool_span(hs, (1, 4))
    assert out is not None
    assert out.dtype == np.float16
    assert out.shape == (8,)


def test_mean_pool_empty_span_returns_none():
    hs = torch.randn(5, 8)
    assert mean_pool_span(hs, (2, 2)) is None


def test_max_norm_pool_single_token():
    """Span of length 1 returns that token (as float16)."""
    hs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = max_norm_pool_span(hs, (0, 1))
    assert out is not None
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), [1.0, 2.0, 3.0, 4.0])


def test_max_norm_pool_picks_largest_norm():
    """Of three tokens, the one with the highest L2 norm wins."""
    hs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],   # norm 1
        [3.0, 4.0, 0.0, 0.0],   # norm 5
        [0.0, 0.0, 1.0, 0.0],   # norm 1
    ])
    out = max_norm_pool_span(hs, (0, 3))
    assert out is not None
    np.testing.assert_allclose(out.astype(np.float32), [3.0, 4.0, 0.0, 0.0])


def test_pool_span_dispatcher():
    hs = torch.randn(5, 8)
    assert pool_span(hs, (0, 3), method="mean") is not None
    assert pool_span(hs, (0, 3), method="max_norm") is not None
    with pytest.raises(ValueError, match="Unknown pooling method"):
        pool_span(hs, (0, 3), method="nonsense")


def test_find_token_spans_basic():
    """A simple case: each character is a token; span detection finds substrings."""
    text = "hello world"
    # one offset per character
    offsets = [(i, i + 1) for i in range(len(text))]
    spans = find_token_spans(text, offsets, {"hint": "world"})
    assert "hint" in spans
    assert spans["hint"] == (6, 11)


def test_find_token_spans_candidate_anchor_avoids_false_match():
    """A 'cand:'-prefixed key only matches after the candidate anchor."""
    text = "Marriage: married\nThe possible words are:\nmarried\nblue"
    offsets = [(i, i + 1) for i in range(len(text))]
    spans = find_token_spans(text, offsets, {"cand:married": "married"})
    # The match must be after 'The possible words are:', not the 'married' in
    # the social block.
    anchor_char = text.find("The possible words are:")
    assert anchor_char != -1
    assert spans["cand:married"][0] >= anchor_char


def test_find_token_spans_missing_substring_omitted():
    text = "hello world"
    offsets = [(i, i + 1) for i in range(len(text))]
    spans = find_token_spans(text, offsets, {"hint": "absent"})
    assert "hint" not in spans
