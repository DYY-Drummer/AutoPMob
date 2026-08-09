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


def test_per_case_signal_exact_half_flip_rate_flips():
    # flip_rate がちょうど0.5の境界値。演算子は >= なので flip=1 になる。
    s = per_case_signal(1.0, [1.0, 1.0, 0.5, 0.5])
    assert abs(s["flip_rate"] - 0.5) < 1e-9
    assert s["flip"] == 1


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


def test_sep_for_feature_nan_when_correct_side_empty():
    feats = np.array([[0.1], [0.8]], dtype=np.float32)
    assert math.isnan(sep_for_feature(feats, [1, 2], set(), 0))  # 正解行なし


def _tiny_cache():
    # 2ケース、候補3〜4件、特徴列3（旧コメントは「候補5件・特徴列1」で実装と不一致だった）。
    # c1: 正解が候補と完全一致（cands==corr）→ 順位に関わらず Recall@K_correct は必ず1.0（solvable）。
    # c2: 正解の一部（99）が候補に含まれない → miss_rank 扱いとなり Recall@K_correct は必ず1.0未満
    #     （not fully solvable）。モデルの重み初期化や dropout の乱数に依存せず、
    #     cands/corr の構造だけで2ケースが異なる値になることを保証する。
    rng = np.random.RandomState(0)
    cache = []
    for cid, cands, corr in [
        ("c1", [10, 11, 12], {10, 11, 12}),
        ("c2", [20, 21, 22, 23], {20, 99}),
    ]:
        feats = rng.rand(len(cands), 3).astype(np.float32)
        cache.append({"feats": feats, "corr": set(corr), "cands": cands,
                      "variant": "original", "case_id": cid,
                      "n_input": 3, "n_output": 1, "n_sources": 1})
    return cache


def test_score_to_RK_per_case_mean_equals_aggregate():
    torch.manual_seed(0)
    cache = _tiny_cache()
    model = Reranker(3, 8)
    model.eval()  # 本番の呼び出しパス（train_and_cache は eval() 済みモデルを返す）を再現する。
    # dropout が有効だと同一入力でも forward の度に結果が変わり、後段の
    # 「個別採点との突き合わせ」によるアライメント検証が意味を持たなくなる。
    agg, pc = score_to_RK(model, cache, return_per_case=True)

    assert len(pc) == len(cache)
    assert all(0.0 <= v <= 1.0 for v in pc)
    assert abs(float(np.mean(pc)) - agg) < 1e-9

    # フィクスチャは2ケースが必ず異なる値になるよう構成している。
    # 全ケースが同じ値（旧フィクスチャでは両方1.0）だと mean(list)==aggregate は
    # 並び替えや誤重み付けの下でも自明に成り立ち、検証として空虚になる。
    assert len(set(pc)) == 2
    assert pc[0] == 1.0   # c1: 候補が全て正解 → 必ず1.0
    assert pc[1] < 1.0    # c2: 正解の一部が候補に無い → 必ず1.0未満

    # per_case[i] が cache[i] に対応する（ずれていない）ことを、各ケースを単独で
    # 採点した結果との突き合わせで検証する。aggregate は case の並びに依らない
    # 平均なので、これがないと per_case リストが cache とずれていても検知できない。
    for i, rec in enumerate(cache):
        _, solo_pc = score_to_RK(model, [rec], return_per_case=True)
        assert abs(solo_pc[0] - pc[i]) < 1e-9


def test_score_to_RK_default_returns_float():
    torch.manual_seed(0)
    out = score_to_RK(Reranker(3, 8), _tiny_cache())
    assert isinstance(out, float)
