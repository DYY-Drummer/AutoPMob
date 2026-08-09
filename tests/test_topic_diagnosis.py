import numpy as np
import pytest
from diagnose_topic_features import (
    mean_pairwise_cos, sample_pairwise_cos, auc_from_groups, sep_groups,
    within_candidate_spread, stage1_coverage, top_k_by_score, rank_auc_full_db,
)


def _norm_rows(V):
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def test_mean_pairwise_cos_matches_bruteforce():
    rng = np.random.RandomState(0)
    V = _norm_rows(rng.randn(20, 5))
    got = mean_pairwise_cos(V)
    S = V @ V.T
    n = V.shape[0]
    want = (S.sum() - n) / (n * (n - 1))
    assert abs(got - want) < 1e-9


def test_mean_pairwise_cos_identical_rows_is_one():
    V = np.tile(_norm_rows(np.array([[1.0, 2.0, 3.0]])), (7, 1))
    assert abs(mean_pairwise_cos(V) - 1.0) < 1e-9


def test_sample_pairwise_cos_deterministic_and_bounded():
    rng = np.random.RandomState(1)
    V = _norm_rows(rng.randn(30, 8))
    a = sample_pairwise_cos(V, n_pairs=100, seed=42)
    b = sample_pairwise_cos(V, n_pairs=100, seed=42)
    assert np.allclose(a, b)
    assert a.min() >= -1.0 - 1e-9 and a.max() <= 1.0 + 1e-9
    assert a.size == 100


def test_auc_perfect_separation():
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.1, 0.2, 0.3])
    assert abs(auc_from_groups(pos, neg) - 1.0) < 1e-9


def test_auc_no_separation_is_half():
    v = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(auc_from_groups(v[:2], v[2:]) - 0.5) < 1e-9


def test_auc_empty_group_is_nan():
    assert np.isnan(auc_from_groups(np.array([]), np.array([1.0])))


def test_sep_groups_known_value():
    assert abs(sep_groups(np.array([1.0, 0.8]), np.array([0.2, 0.0])) - 0.8) < 1e-9
    assert np.isnan(sep_groups(np.array([]), np.array([0.1])))


def test_within_candidate_spread_ratio():
    full = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])   # std大
    cand = np.array([2.4, 2.5, 2.6])                   # std小
    r = within_candidate_spread(full, cand)
    assert r["std_full"] > r["std_cand"]
    assert 0.0 < r["ratio"] < 0.2


def test_stage1_coverage():
    assert abs(stage1_coverage([1, 2, 3], {2, 3, 9}) - 2 / 3) < 1e-9
    assert np.isnan(stage1_coverage([1], set()))


def test_top_k_by_score_orders_desc():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k_by_score(s, 2) == [1, 3]


def test_rank_auc_full_db_matches_group_auc():
    rng = np.random.RandomState(3)
    scores = rng.randn(50)
    corr = {3, 10, 20}
    got = rank_auc_full_db(scores, corr)
    pos = np.array([scores[j] for j in sorted(corr)])
    neg = np.array([scores[j] for j in range(50) if j not in corr])
    assert abs(got - auc_from_groups(pos, neg)) < 1e-9
