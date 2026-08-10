import numpy as np
from calibrate_similarity_scale import (
    paraphrase_base, quantiles, cos_pairs, percentile_of,
)


def test_paraphrase_base_extracts_source_id():
    assert paraphrase_base("core_058_v1_para1") == "core_058_v1"
    assert paraphrase_base("core_115_v1_para12") == "core_115_v1"


def test_paraphrase_base_none_for_plain_id():
    assert paraphrase_base("core_058_v1") is None
    assert paraphrase_base("dae_X3_001") is None
    assert paraphrase_base("ms3_0000") is None


def test_paraphrase_base_rejects_non_numeric_suffix():
    assert paraphrase_base("core_1_paraX") is None


def test_quantiles_known_values():
    q = quantiles([0.0, 0.25, 0.5, 0.75, 1.0])
    assert abs(q["median"] - 0.5) < 1e-9
    assert abs(q["q25"] - 0.25) < 1e-9 and abs(q["q75"] - 0.75) < 1e-9
    assert q["n"] == 5


def test_quantiles_drops_nan():
    q = quantiles([1.0, float("nan"), 3.0])
    assert q["n"] == 2 and abs(q["mean"] - 2.0) < 1e-9


def test_cos_pairs_identical_rows_is_one():
    V = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    got = cos_pairs(V, [(0, 1), (0, 2)])
    assert abs(got[0] - 1.0) < 1e-9   # 同一方向
    assert abs(got[1] - 0.0) < 1e-9   # 直交


def test_cos_pairs_empty():
    assert cos_pairs(np.eye(2), []).size == 0


def test_percentile_of():
    vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert abs(percentile_of(0.2, vals) - 0.6) < 1e-9   # 0.0,0.1,0.2 の3つが以下
    assert abs(percentile_of(-1.0, vals) - 0.0) < 1e-9
    assert abs(percentile_of(9.9, vals) - 1.0) < 1e-9
