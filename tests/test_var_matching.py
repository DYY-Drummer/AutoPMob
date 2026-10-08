"""var_matching.py（照合・第 1 段の近似・τ・特徴量）の単体テスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.4, §4
"""
import numpy as np

from var_matching import (
    SYMBOL_TAU, max_matching, occurrence_incidence, stage1_match_counts,
    jaccard_from_counts, youden_tau, domain_flags, case_features,
)
from two_stage_query_conditioned import jaccard


def test_max_matching_crossing_case_finds_two():
    # 行 0 は列 0・1、行 1 は列 0 だけ。貪欲に (0,0) を取ると 1 本だが、最大は 2 本
    adj = np.array([[True, True], [True, False]])
    m = max_matching(adj)
    assert (m >= 0).sum() == 2
    assert m[0] == 1 and m[1] == 0


def test_max_matching_two_queries_one_target():
    assert (max_matching(np.array([[True], [True]])) >= 0).sum() == 1


def test_max_matching_empty_inputs():
    assert list(max_matching(np.zeros((3, 0), dtype=bool))) == [-1, -1, -1]
    assert list(max_matching(np.zeros((2, 2), dtype=bool))) == [-1, -1]


def test_stage1_exact_counts_equal_jaccard():
    # 式 0: {a, b}、式 1: 変数なし、式 2: {b, c, d}
    occ_sym = ["a", "b", "b", "c", "d"]
    eq_ptr = np.array([0, 2, 2, 5])
    inc = occurrence_incidence(eq_ptr)
    eq_sets = [{"a", "b"}, set(), {"b", "c", "d"}]
    for q in (["a"], ["b", "c"], ["a", "b", "c", "d", "e"], ["z"]):
        hit = np.array([[s == o for o in occ_sym] for s in q])
        got = jaccard_from_counts(stage1_match_counts(hit, inc), len(q), np.diff(eq_ptr))
        want = np.array([jaccard(set(q), s) for s in eq_sets], dtype=np.float32)
        assert got.dtype == np.float32
        assert np.array_equal(got, want)


def test_stage1_approx_is_upper_bound_and_exact_with_single_edges():
    rng = np.random.default_rng(0)
    eq_ptr = np.array([0, 3, 7, 7, 12])
    inc = occurrence_incidence(eq_ptr)
    for _ in range(300):
        hit = rng.random((4, 12)) < 0.3
        m = stage1_match_counts(hit, inc)
        for j in range(4):
            a, b = eq_ptr[j], eq_ptr[j + 1]
            exact = int((max_matching(hit[:, a:b]) >= 0).sum())
            assert m[j] >= exact
            if (hit[:, a:b].sum(axis=1) <= 1).all():   # 各 q の辺が 1 本以下なら一致する
                assert m[j] == exact


def test_youden_tau_separates_two_groups():
    sims = np.array([0.95, 0.92, 0.90, 0.70, 0.60, 0.50])
    labels = np.array([1, 1, 1, 0, 0, 0])
    tau = youden_tau(sims, labels)
    assert 0.70 < tau <= 0.90


def test_domain_flags_substring_rule():
    flags = domain_flags([0, 1, 2], ["CSTR", "", "Distillation"], "A CSTR with a jacket")
    assert list(flags) == [1.0, 0.0, 0.0]


def test_case_features_symbol_toy():
    # 照会 Q = (C 出力, F 入力, T 入力)。候補 0 = {F, T, V}、候補 1 = {C, T}。参照集合 = 上位 5 件 = 両方
    q = np.array(["C", "F", "T"], dtype=object)
    q_in = np.array([False, True, True])
    q_out = np.array([True, False, False])
    cv = np.array(["F", "T", "V", "C", "T"], dtype=object)
    cptr = np.array([0, 3, 5])
    QC = (q[:, None] == cv[None, :]).astype(np.float32)
    CR = (cv[:, None] == cv[None, :5]).astype(np.float32)
    feats, m = case_features(np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([1.0, 0.0], dtype=np.float32),
                             q_in, q_out, QC, CR, cptr, ["cstr", "cstr"], SYMBOL_TAU)
    assert list(m) == [2, 2]
    # 候補 0: |Q∩V|=2 (F,T)、|Q∪V|=4 → f1=0.5、f3=2/2、f4=0/1、f5=2/3
    assert feats[0, 1] == np.float32(0.5) and feats[0, 3] == 1.0 and feats[0, 4] == 0.0
    assert feats[0, 5] == np.float32(2 / 3)
    # 参照（自身を除く）= 候補 1 {C, T}: gComp は F だけ → 1/3、gCoh は V={F,T,V} のうち T → 1/3、gDom = 1
    assert feats[0, 7] == np.float32(1 / 3) and feats[0, 8] == np.float32(1 / 3) and feats[0, 9] == 1.0
    assert feats[0, 0] == np.float32(0.1) and feats[0, 2] == np.float32(0.3) and feats[0, 6] == 1.0
    # 候補 1: f1 = 2/3、f3 = 1/2（T）、f4 = 1、f5 = 1。参照 = 候補 0 {F,T,V}: gComp は C → 1/3、gCoh は T → 1/2
    assert feats[1, 1] == np.float32(2 / 3) and feats[1, 3] == np.float32(0.5) and feats[1, 4] == 1.0
    assert feats[1, 5] == 1.0 and feats[1, 7] == np.float32(1 / 3) and feats[1, 8] == np.float32(0.5)
