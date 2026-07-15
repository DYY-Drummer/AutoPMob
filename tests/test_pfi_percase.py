import math
import numpy as np
import torch
from two_stage_query_conditioned import Reranker
from analyze_pfi import per_case_signal, sep_for_feature, score_to_RK


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


def _tiny_cache():
    # 2ケース、候補5件ずつ、1特徴列。corr は cands の一部。
    rng = np.random.RandomState(0)
    cache = []
    for cid, corr in [("c1", {10}), ("c2", {20, 21})]:
        cands = [10, 11, 12] if cid == "c1" else [20, 21, 22, 23]
        feats = rng.rand(len(cands), 3).astype(np.float32)
        cache.append({"feats": feats, "corr": set(corr), "cands": cands,
                      "variant": "original", "case_id": cid,
                      "n_input": 3, "n_output": 1, "n_sources": 1})
    return cache


def test_score_to_RK_per_case_mean_equals_aggregate():
    torch.manual_seed(0)
    cache = _tiny_cache()
    model = Reranker(3, 8)
    agg, pc = score_to_RK(model, cache, return_per_case=True)
    assert len(pc) == len(cache)
    assert all(0.0 <= v <= 1.0 for v in pc)
    assert abs(float(np.mean(pc)) - agg) < 1e-9


def test_score_to_RK_default_returns_float():
    torch.manual_seed(0)
    out = score_to_RK(Reranker(3, 8), _tiny_cache())
    assert isinstance(out, float)
