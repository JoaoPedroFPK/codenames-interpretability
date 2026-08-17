"""analysis_concordance_by_layer must carry the geometry-only base decoders
(T5): top-1-by-layer needs no generations; concordance stays NaN for them."""

import numpy as np
import pandas as pd

from codenames.analysis.tables import build_concordance_by_layer


def _metrics(rows):
    recs = []
    for row_id, layer, ranks in rows:
        for word, wt, r in ranks:
            recs.append({"row_id": row_id, "layer": layer, "word": word, "word_type": wt,
                         "rank_mean": r, "rank_max_norm": r, "permutation_id": 0})
    return pd.DataFrame(recs)


def test_base_decoder_gets_top1_without_generations(tmp_path):
    for prefix in ("mistral", "mistral_base"):
        d = tmp_path / f"{prefix}_outputs"
        d.mkdir()
        _metrics([(0, 0, [("SEA", "target", 1), ("MOON", "tan", 2)]),
                  (0, 1, [("SEA", "target", 2), ("MOON", "tan", 1)])]).to_parquet(
            d / f"{prefix}_metrics_no_social.parquet", index=False)
    pd.DataFrame({"row_id": [0], "generated_word": ["SEA"]}).to_csv(
        tmp_path / "mistral_outputs" / "mistral_generation_no_social.csv", index=False)
    missing = []
    out = build_concordance_by_layer(str(tmp_path), missing=missing)
    base = out[(out.model == "mistral_base") & (out.pooling == "mean")].sort_values("layer")
    assert list(base.top1_accuracy) == [1.0, 0.0]
    assert base.concordance.isna().all()
    inst = out[(out.model == "mistral") & (out.pooling == "mean")].sort_values("layer")
    assert list(inst.concordance) == [1.0, 0.0]
    assert not any("mistral_base" in m and "generation" in m for m in missing)
