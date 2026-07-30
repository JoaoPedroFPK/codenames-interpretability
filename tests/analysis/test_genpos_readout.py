import numpy as np
import pandas as pd

from codenames.analysis.genpos_readout import genpos_readout_curve


def _fixture():
    """2 boards x 2 layers x 4 dims. Layer 0: the generating state points at a
    non-target; layer 1: at the target. Expected curve: top-1 = 0 then 1."""
    D = 4
    e = np.eye(D, dtype=np.float16)
    # gen_hidden[board, layer, dim]
    gen_hidden = np.stack([
        np.stack([e[0], e[1]]),          # board 0: layer0 -> dim0, layer1 -> dim1
        np.stack([e[2], e[3]]),          # board 1: layer0 -> dim2, layer1 -> dim3
    ])
    gen_index = pd.DataFrame({
        "board_idx": [0, 1], "row_id": [10, 11], "ok": [True, True]})

    rows, vecs = [], []
    def add(row_id, layer, word, word_type, vec):
        rows.append({"record_idx": len(vecs), "row_id": row_id, "layer": layer,
                     "word": word, "word_type": word_type,
                     "pooling_method": "mean", "vector_valid": True})
        vecs.append(vec)
    # board 10: target 'sea' on dim1, tan 'moon' on dim0
    for layer in (0, 1):
        add(10, layer, "sea", "target", e[1])
        add(10, layer, "moon", "tan", e[0])
    # board 11: target 'ship' on dim3, black 'castle' on dim2
    for layer in (0, 1):
        add(11, layer, "ship", "target", e[3])
        add(11, layer, "castle", "black", e[2])
    vec_index = pd.DataFrame(rows)
    vec_matrix = np.stack(vecs)
    return gen_hidden, gen_index, vec_matrix, vec_index


def test_curve_tracks_where_the_generating_state_points():
    gen_hidden, gen_index, vec_matrix, vec_index = _fixture()
    curve = genpos_readout_curve(gen_hidden, gen_index, vec_matrix, vec_index)
    curve = curve.set_index("layer")
    assert curve.loc[0, "top1"] == 0.0     # both boards point at non-targets
    assert curve.loc[1, "top1"] == 1.0     # both boards point at targets
    assert curve.loc[0, "n_boards"] == 2


def test_hint_rows_and_other_poolings_are_excluded():
    gen_hidden, gen_index, vec_matrix, vec_index = _fixture()
    # A hint vector aligned with every layer-0 state would fake a hit if the
    # candidate pool wrongly included hints; a max_norm duplicate would
    # double-count.
    extra = pd.DataFrame([
        {"record_idx": len(vec_matrix), "row_id": 10, "layer": 0,
         "word": "internet", "word_type": "hint", "pooling_method": "mean",
         "vector_valid": True},
        {"record_idx": len(vec_matrix) + 1, "row_id": 10, "layer": 0,
         "word": "sea", "word_type": "target", "pooling_method": "max_norm",
         "vector_valid": True},
    ])
    vec_index = pd.concat([vec_index, extra], ignore_index=True)
    vec_matrix = np.concatenate([
        vec_matrix,
        np.stack([vec_matrix[1], vec_matrix[0]])])  # hint = e0 (layer-0 state)
    curve = genpos_readout_curve(gen_hidden, gen_index, vec_matrix,
                                 vec_index).set_index("layer")
    assert curve.loc[0, "top1"] == 0.0


def test_invalid_vectors_and_failed_boards_are_dropped():
    gen_hidden, gen_index, vec_matrix, vec_index = _fixture()
    gen_index.loc[1, "ok"] = False
    curve = genpos_readout_curve(gen_hidden, gen_index, vec_matrix,
                                 vec_index).set_index("layer")
    assert curve.loc[1, "n_boards"] == 1
