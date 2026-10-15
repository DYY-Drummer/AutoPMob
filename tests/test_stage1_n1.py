"""analyze_stage1_n1.py のテスト（加藤先生 2026-10-14 の指示 3：全正解式が候補に入る N1）.

同点の扱い（同点の中で最も後ろの順位）、ベクトル化した第 1 段の点数が stage1() と
同じ順位を返すこと、曲線と分位の数え方を、手で数えられる小さな例で確かめる。
"""
import numpy as np
import pytest
from scipy import sparse

from analyze_stage1_n1 import (
    coverage_curve, jaccard_matrix, min_n1_full, n1_for_fraction,
    stage1_score_matrix, worst_tie_ranks,
)
from two_stage_query_conditioned import jaccard, stage1


def test_worst_tie_rank_takes_last_position_among_ties():
    s = np.array([0.5, 0.9, 0.5, 0.1, 0.0])
    # 0.5 の 2 本は 2 位と 3 位を分け合う → どちらも後ろの 3 位
    assert worst_tie_ranks(s, [0, 2]) == [3, 3]
    assert worst_tie_ranks(s, [1]) == [1]


def test_worst_tie_rank_of_zero_score_is_whole_db():
    s = np.array([0.0, 0.3, 0.0, 0.0])
    # 点数 0 の正解式は、点数 0 の式すべての後ろ（＝DB の最後）
    assert worst_tie_ranks(s, [0]) == [4]


def test_min_n1_full_is_worst_rank():
    assert min_n1_full([3, 17, 5]) == 17


def test_jaccard_matrix_matches_python_jaccard():
    rng = np.random.default_rng(0)
    vocab = [f"v{i}" for i in range(12)]
    eqs = [set(rng.choice(vocab, size=rng.integers(0, 5), replace=False)) for _ in range(30)]
    qs = [set(rng.choice(vocab, size=rng.integers(0, 6), replace=False)) for _ in range(7)]
    J = jaccard_matrix(qs, eqs)
    want = np.array([[jaccard(q, e) for e in eqs] for q in qs], dtype=np.float32)
    assert J.dtype == np.float32
    assert np.array_equal(J, want)


def test_stage1_score_matrix_reproduces_stage1_order():
    """stage1() の argsort（同じ dtype の点数）と、行列版の上位 k 件が一致する."""
    rng = np.random.default_rng(1)
    n_eq, n_q, n_term = 40, 5, 25
    X_eq = sparse.csr_matrix(rng.random((n_eq, n_term)) * (rng.random((n_eq, n_term)) < 0.3))
    X_ctx = sparse.csr_matrix(rng.random((n_q, n_term)) * (rng.random((n_q, n_term)) < 0.4))
    vocab = [f"v{i}" for i in range(10)]
    eqs = [set(rng.choice(vocab, size=rng.integers(1, 4), replace=False)) for _ in range(n_eq)]
    qs = [set(rng.choice(vocab, size=rng.integers(1, 5), replace=False)) for _ in range(n_q)]
    S = stage1_score_matrix(X_ctx, X_eq, qs, eqs, w_text=0.3, w_var=0.7)
    assert S.dtype == np.float64
    for ci in range(n_q):
        want = stage1(ci, X_ctx, X_eq, qs[ci], eqs, k=n_eq, w_text=0.3, w_var=0.7)
        got = np.argsort(-S[ci]).tolist()
        assert got == want


def test_coverage_curve_counts_full_inclusion_and_mean_coverage():
    # ケース 1: 正解の順位 [1, 4]、ケース 2: [2]
    ranks = [[1, 4], [2]]
    cur = coverage_curve(ranks, [1, 2, 4])
    assert cur["full"] == pytest.approx([0.0, 0.5, 1.0])
    assert cur["coverage"] == pytest.approx([0.25, 0.75, 1.0])


def test_n1_for_fraction_smallest_n1_reaching_share():
    mins = [1, 2, 5, 40]
    assert n1_for_fraction(mins, 1.0) == 40
    assert n1_for_fraction(mins, 0.75) == 5
    assert n1_for_fraction(mins, 0.5) == 2
