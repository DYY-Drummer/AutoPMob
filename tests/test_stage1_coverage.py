import numpy as np
from analyze_stage1_coverage import coverage, mean_sem


def test_coverage_partial():
    assert abs(coverage([5, 1, 9, 2], {1, 2, 7}, 2) - 1 / 3) < 1e-9
    assert abs(coverage([5, 1, 9, 2], {1, 2, 7}, 4) - 2 / 3) < 1e-9


def test_coverage_monotone_in_k():
    order = list(range(20))
    corr = {0, 7, 15}
    vals = [coverage(order, corr, k) for k in (1, 8, 16, 20)]
    assert vals == sorted(vals)          # k を増やすと減らない
    assert abs(vals[-1] - 1.0) < 1e-9    # 全件見れば必ず 1.0


def test_coverage_empty_truth_is_nan():
    assert np.isnan(coverage([1, 2], set(), 2))


def test_mean_sem_known_values():
    r = mean_sem([1.0, 1.0, 1.0])
    assert abs(r["mean"] - 1.0) < 1e-9 and r["sem"] == 0.0 and r["n"] == 3


def test_mean_sem_drops_nan():
    r = mean_sem([1.0, float("nan"), 0.0])
    assert r["n"] == 2 and abs(r["mean"] - 0.5) < 1e-9


def test_mean_sem_empty():
    r = mean_sem([float("nan")])
    assert r["n"] == 0 and np.isnan(r["mean"])
