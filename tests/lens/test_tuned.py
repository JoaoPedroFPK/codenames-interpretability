import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.lens.tuned import TunedLens, TunedLensConfig, train_tuned_lens

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

TEXTS = ["the quick brown fox jumps over the lazy dog. " * 40,
         "codenames is a word association board game. " * 40]


@pytest.mark.network
def test_training_reduces_loss_and_roundtrips(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    cfg = TunedLensConfig(seq_len=32, n_steps=8, seqs_per_step=2,
                          positions_per_seq=8, lr=1e-3, val_every=4)
    lens = train_tuned_lens(model, tok, TEXTS, cfg, device="cpu")
    assert len(lens.history) == 8
    assert np.mean(lens.history[-2:]) < np.mean(lens.history[:2])
    assert len(lens.val_history) > 0          # held-out KL was tracked

    p = str(tmp_path / "translators.npz")
    lens.save(p)
    lens2 = TunedLens.load(p)
    np.testing.assert_allclose(lens.A, lens2.A, atol=0)
    np.testing.assert_allclose(lens2.val_history, lens.val_history)

    H = np.random.default_rng(0).standard_normal(
        (5, lens.A.shape[1])).astype(np.float32)
    # Final hidden state (layer index >= L) passes through untranslated.
    np.testing.assert_allclose(lens2.translate(H, layer=lens.A.shape[0]), H)


@pytest.mark.network
def test_weight_decay_pulls_toward_identity_not_zero():
    """Decay acts on the deviation D (A = I + D): under absurd decay the
    translators must collapse to the identity, never toward zero."""
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    cfg = TunedLensConfig(seq_len=32, n_steps=8, seqs_per_step=2,
                          positions_per_seq=8, lr=1e-2, weight_decay=50.0)
    lens = train_tuned_lens(model, tok, TEXTS, cfg, device="cpu")
    diag = np.diagonal(lens.A, axis1=1, axis2=2)
    assert diag.mean() > 0.5   # old decay-on-A would leave ~0.004


def test_load_old_artifact_without_val_history(tmp_path):
    d, L = 4, 3
    p = str(tmp_path / "old.npz")
    np.savez_compressed(p, A=np.stack([np.eye(d, dtype=np.float32)] * L),
                        b=np.zeros((L, d), dtype=np.float32),
                        history=np.zeros(5))
    lens = TunedLens.load(p)
    assert lens.val_history == []             # backward compatible


def test_identity_init_translate_is_identity():
    d, L = 4, 3
    lens = TunedLens(A=np.stack([np.eye(d, dtype=np.float32)] * L),
                     b=np.zeros((L, d), dtype=np.float32), history=[])
    H = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(lens.translate(H, 1), H)
