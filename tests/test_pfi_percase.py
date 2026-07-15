import math
import numpy as np
from analyze_pfi import per_case_signal, sep_for_feature


def test_per_case_signal_full_flip():
    # base=1.0、全置換で 1.0 未満 → flip=1, flip_rate=1.0
    s = per_case_signal(1.0, [0.5, 0.0, 0.5, 0.5])
    assert abs(s["perm_R_mean"] - 0.375) < 1e-9
    assert abs(s["drop"] - 0.625) < 1e-9
    assert s["flip"] == 1
    assert abs(s["flip_rate"] - 1.0) < 1e-9


def test_per_case_signal_minority_no_flip():
    # base=1.0、過半未満(2/5)しか崩れない → flip=0
    s = per_case_signal(1.0, [1.0, 1.0, 1.0, 0.5, 0.5])
    assert s["flip"] == 0
    assert abs(s["flip_rate"] - 0.4) < 1e-9


def test_per_case_signal_not_solved_never_flips():
    # base<1.0 は flip 対象外（定義上 flip=0）だが drop は測る
    s = per_case_signal(0.5, [0.0, 0.0])
    assert s["flip"] == 0
    assert abs(s["drop"] - 0.5) < 1e-9


def test_sep_for_feature_basic():
    # col0: 正解行(cands 2,4)=[0.8,0.6] 平均0.7、不正解行=[0.1,0.3] 平均0.2 → sep=0.5
    feats = np.array([[0.1], [0.8], [0.3], [0.6]], dtype=np.float32)
    cands = [1, 2, 3, 4]
    corr = {2, 4}
    assert abs(sep_for_feature(feats, cands, corr, 0) - 0.5) < 1e-6


def test_sep_for_feature_nan_when_one_side_empty():
    feats = np.array([[0.1], [0.8]], dtype=np.float32)
    assert math.isnan(sep_for_feature(feats, [1, 2], {1, 2}, 0))  # 不正解行なし
