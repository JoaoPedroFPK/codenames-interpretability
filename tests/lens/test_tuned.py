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
                          positions_per_seq=8, lr=1e-3)
    lens = train_tuned_lens(model, tok, TEXTS, cfg, device="cpu")
    assert len(lens.history) == 8
    assert np.mean(lens.history[-2:]) < np.mean(lens.history[:2])

    p = str(tmp_path / "translators.npz")
    lens.save(p)
    lens2 = TunedLens.load(p)
    np.testing.assert_allclose(lens.A, lens2.A, atol=0)

    H = np.random.default_rng(0).standard_normal(
        (5, lens.A.shape[1])).astype(np.float32)
    # Final hidden state (layer index >= L) passes through untranslated.
    np.testing.assert_allclose(lens2.translate(H, layer=lens.A.shape[0]), H)


def test_identity_init_translate_is_identity():
    d, L = 4, 3
    lens = TunedLens(A=np.stack([np.eye(d, dtype=np.float32)] * L),
                     b=np.zeros((L, d), dtype=np.float32), history=[])
    H = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(lens.translate(H, 1), H)
