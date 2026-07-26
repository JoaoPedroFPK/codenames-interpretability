import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.lens.raw import RawLens, dump_readout_weights

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    return model, tok


def test_final_layer_raw_lens_equals_model_logits(tiny):
    """The calibration identity: Unembed(FinalLN(h_final)) == model logits."""
    model, tok = tiny
    inputs = tok("the hint is spring", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    lens = RawLens(model)
    lens_logits = lens.logits(out.hidden_states[-1][0, -1])
    torch.testing.assert_close(lens_logits, out.logits[0, -1], atol=1e-4, rtol=1e-4)


def test_dump_readout_weights_roundtrip(tiny, tmp_path):
    model, _ = tiny
    p = str(tmp_path / "readout.npz")
    dump_readout_weights(model, p)
    z = np.load(p, allow_pickle=False)
    d = model.config.hidden_size
    assert z["norm_weight"].shape == (d,)
    assert z["lm_head_weight"].shape[1] == d
    assert float(z["eps"]) > 0
