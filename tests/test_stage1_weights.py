"""stage1 の重みパラメタ化（A1: 第1段再設計実験）のテスト.

検証すること:
  1. stage1 が w_text / w_var を受け取り、既定値 (0.7, 0.3) は従来挙動と一致する
  2. 重みを変えると意図した向きに順位が変わる（変数寄り・文章寄り）
  3. set_aware_reranker.run_mode が重みを stage1 まで配線する
  4. baseline モード（全 DB を混合式で順位付け）も同じ重みに従う
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from two_stage_query_conditioned import stage1  # noqa: E402
import set_aware_reranker  # noqa: E402


# ---------------------------------------------------------------------------
# 1--2. stage1 単体
# ---------------------------------------------------------------------------

def _toy_stage1_inputs():
    """式 3 本の玩具データ.

    ts = cos(query, eq) = [1.0, 0.0, 1/sqrt(2)]
    vs = jaccard(io, vars) = [0.0, 1.0, 0.5]
    既定の混合 0.7*ts + 0.3*vs = [0.700, 0.300, 0.645] → 順位 [0, 2, 1]
    """
    X_eq = sparse.csr_matrix(np.array([[1.0, 0.0],
                                       [0.0, 1.0],
                                       [1.0, 1.0]]))
    X_ctx = sparse.csr_matrix(np.array([[1.0, 0.0]]))
    eq_vars_list = [set(), {"a", "b"}, {"a"}]
    io_set = {"a", "b"}
    return X_ctx, X_eq, io_set, eq_vars_list


def test_stage1_default_matches_explicit_070_030():
    X_ctx, X_eq, io_set, evl = _toy_stage1_inputs()
    default = stage1(0, X_ctx, X_eq, io_set, evl, 3)
    explicit = stage1(0, X_ctx, X_eq, io_set, evl, 3, w_text=0.7, w_var=0.3)
    assert default == explicit == [0, 2, 1]


def test_stage1_pure_variable_weight_ranks_by_jaccard():
    X_ctx, X_eq, io_set, evl = _toy_stage1_inputs()
    order = stage1(0, X_ctx, X_eq, io_set, evl, 3, w_text=0.0, w_var=1.0)
    assert order == [1, 2, 0]  # vs = [0.0, 1.0, 0.5]


def test_stage1_pure_text_weight_ranks_by_cosine():
    X_ctx, X_eq, io_set, evl = _toy_stage1_inputs()
    order = stage1(0, X_ctx, X_eq, io_set, evl, 3, w_text=1.0, w_var=0.0)
    assert order == [0, 2, 1]  # ts = [1.0, 0.0, 0.707]


def test_stage1_keeps_top_k():
    X_ctx, X_eq, io_set, evl = _toy_stage1_inputs()
    order = stage1(0, X_ctx, X_eq, io_set, evl, 2, w_text=0.0, w_var=1.0)
    assert order == [1, 2]


# ---------------------------------------------------------------------------
# 3--4. set_aware_reranker への配線
# ---------------------------------------------------------------------------

def _toy_run_mode_data():
    """run_mode を最小構成で回すデータ（5 ソース × 2 式、ケース 5 件）.

    各ソース s{i} に:
      good_i: 変数がケース i の入出力と完全一致、文章は無関係（"zzz qqq"）
      bad_i : 文章がケース文脈と一致（"alpha beta gamma"）、変数は無関係
    → 文章寄り重みでは bad が、変数寄り重みでは good が上位に来る。
    """
    eq_keys, eq_texts, eq_vars_l, eq_domains, eq_sources = [], [], [], [], []
    cases, correct_lists, case_srcs = [], [], []
    for i in range(5):
        src = f"s{i}"
        # run_mode は 256 次元の TruncatedSVD を組むので、式ごとに固有の
        # 埋め草語を足して TF-IDF 語彙を 256 次元以上にする（コサインの
        # 大小関係は変えない：good は文脈語を含まず、bad は含む）。
        pad_g = " ".join(f"padg{i}w{j}" for j in range(30))
        pad_b = " ".join(f"padb{i}w{j}" for j in range(30))
        eq_keys += [f"{src}__good", f"{src}__bad"]
        eq_texts += [f"zzz qqq {pad_g}", f"alpha beta gamma {pad_b}"]
        eq_vars_l += [{f"v{i}a", f"v{i}b"}, {f"u{i}x"}]
        eq_domains += ["", ""]
        eq_sources += [src, src]
        cases.append({
            "case_id": f"case_{i}",
            "context": "alpha beta gamma",
            "input_variables": [f"v{i}a"],
            "output_variables": [f"v{i}b"],
            "variant_type": "original",
        })
        correct_lists.append([2 * i])       # good_i の index
        case_srcs.append(src)
    return (cases, eq_keys, eq_texts, eq_vars_l, eq_domains, eq_sources,
            correct_lists, case_srcs)


def _run(mode, monkeypatch=None, recorder=None, **kw):
    (cases, ek, et, ev, ed, es, cl, cs) = _toy_run_mode_data()
    if monkeypatch is not None and recorder is not None:
        def fake_stage1(ci, X_ctx, X_eq, io_set, evl, k,
                        w_text=0.7, w_var=0.3):
            recorder.append((w_text, w_var, k))
            return list(range(min(k, len(evl))))
        monkeypatch.setattr(set_aware_reranker, "stage1", fake_stage1)
    return set_aware_reranker.run_mode(
        mode, [42], cases, ek, et, ev, ed, es, cl, cs,
        top_k=4, epochs=1, lr=1e-3, **kw)


def test_run_mode_forwards_weights_to_stage1(monkeypatch):
    calls: list = []
    _run("reranker-2", monkeypatch, calls,
         stage1_w_text=0.2, stage1_w_var=0.8)
    assert calls, "stage1 が一度も呼ばれていない"
    assert all(c[:2] == (0.2, 0.8) for c in calls), \
        f"重みが配線されていない: {set(c[:2] for c in calls)}"


def test_run_mode_default_weights_are_070_030(monkeypatch):
    calls: list = []
    _run("reranker-2", monkeypatch, calls)
    assert calls and all(c[:2] == (0.7, 0.3) for c in calls)


# ---------------------------------------------------------------------------
# 5. 被覆率スイープ（analyze_stage1_weights.py）の純関数
# ---------------------------------------------------------------------------

def test_wlabel_format():
    from analyze_stage1_weights import wlabel
    assert wlabel((0.7, 0.3)) == "w70-30"
    assert wlabel((0.0, 1.0)) == "w00-100"
    assert wlabel((0.5, 0.5)) == "w50-50"


def test_weight_orders_ranks_each_blend():
    from analyze_stage1_weights import weight_orders
    ts = np.array([1.0, 0.0, 0.5])
    vs = np.array([0.0, 1.0, 0.5])
    orders = weight_orders(ts, vs, [(0.7, 0.3), (0.0, 1.0)])
    assert orders["w70-30"] == [0, 2, 1]   # 0.7*ts+0.3*vs = [0.70, 0.30, 0.50]
    assert orders["w00-100"] == [1, 2, 0]  # vs のみ


def test_sweep_parse_name():
    from analyze_stage1_sweep import parse_name
    assert parse_name("reranker-10S_w30-70__42.json") == ("reranker-10S", "w30-70", 42)
    assert parse_name("baseline_w00-100__9999.json") == ("baseline", "w00-100", 9999)
    assert parse_name("reranker-10S__42.json") is None      # 重みラベル無し（xd 形式）
    assert parse_name("sweep_k50.done") is None


def test_baseline_blend_follows_weights():
    # 文章のみ: bad 系（文章一致・変数不一致）が上位 → 正解を取れない
    r_text = _run("baseline", stage1_w_text=1.0, stage1_w_var=0.0)
    # 変数のみ: good 系（変数完全一致）が上位 → 正解が 1 位
    r_var = _run("baseline", stage1_w_text=0.0, stage1_w_var=1.0)
    assert r_text["Recall@K_correct"]["mean"] == 0.0
    assert r_var["Recall@K_correct"]["mean"] == 1.0
