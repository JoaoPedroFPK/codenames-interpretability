"""Token span detection and pooling utilities.

Architecture-agnostic. Verbatim from Cell 6 of every reference notebook —
this module is the closest thing in the package to a "do not touch" zone for
bit-identity. The cosine and pooling functions intentionally return ``np.float16``
arrays as in the originals, and ``cosine_similarity_np`` returns 0.0 (not NaN)
on zero-norm vectors. Both behaviors are load-bearing for the bit-identity
contract.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def find_token_spans(
    full_text: str,
    offset_mapping: List[Tuple[int, int]],
    spans_to_find: Dict[str, str],
    candidate_anchor: str = "The possible words are:",
) -> Dict[str, Tuple[int, int]]:
    """Locate token-level spans for target substrings in the prompt.

    Parameters
    ----------
    full_text
        The full prompt string.
    offset_mapping
        Character-level offsets for each token (from a Fast tokenizer).
    spans_to_find
        Maps span name -> substring. Span names with the ``"cand:"`` prefix
        search starting from after the candidate-list anchor, so that the
        social block can't produce false matches against candidate words.
    candidate_anchor
        Delimiter marking the start of the candidate list.

    Returns
    -------
    found
        Maps span name -> ``(token_start, token_end)`` index tuple. Missing
        substrings are omitted from the result rather than raising.
    """
    found: Dict[str, Tuple[int, int]] = {}
    cand_start_char = full_text.find(candidate_anchor)
    if cand_start_char == -1:
        cand_start_char = 0

    for name, substring in spans_to_find.items():
        if name.startswith("cand:"):
            char_start = full_text.find(substring, cand_start_char)
        else:
            char_start = full_text.find(substring)

        if char_start == -1:
            continue

        char_end = char_start + len(substring)

        token_start, token_end = None, None
        for idx, (s, e) in enumerate(offset_mapping):
            if s < char_end and e > char_start:
                if token_start is None:
                    token_start = idx
                token_end = idx + 1

        if token_start is not None and token_end > token_start:
            found[name] = (token_start, token_end)

    return found


def mean_pool_span(
    layer_hidden_states: torch.Tensor,
    span: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Mean-pool hidden states across a token span. Returns float16."""
    s, e = span
    tokens = layer_hidden_states[s:e]
    if tokens.shape[0] == 0:
        return None
    pooled = tokens.mean(dim=0).detach().float().cpu().numpy()
    return pooled.astype(np.float16)


def max_norm_pool_span(
    layer_hidden_states: torch.Tensor,
    span: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Select the token with the highest L2 norm from a span. Returns float16.

    Avoids creating synthetic composite vectors. The highest-norm token
    typically corresponds to the root morpheme.
    """
    s, e = span
    tokens = layer_hidden_states[s:e]
    if tokens.shape[0] == 0:
        return None
    if tokens.shape[0] == 1:
        return tokens[0].detach().float().cpu().numpy().astype(np.float16)
    norms = torch.norm(tokens, p=2, dim=1)
    best_idx = torch.argmax(norms).item()
    pooled = tokens[best_idx].detach().float().cpu().numpy()
    return pooled.astype(np.float16)


def pool_span(
    layer_hidden_states: torch.Tensor,
    span: Tuple[int, int],
    method: str = "mean",
) -> Optional[np.ndarray]:
    """Dispatcher for pooling methods. Supports 'mean' and 'max_norm'."""
    if method == "mean":
        return mean_pool_span(layer_hidden_states, span)
    elif method == "max_norm":
        return max_norm_pool_span(layer_hidden_states, span)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D arrays. Returns 0.0 on zero norm.

    Zero-norm protection returns 0.0 (not NaN). This behaviour is part of
    the bit-identity contract — downstream metrics depend on it.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
