"""kato1014_common.py のテスト（2026-10-14 の指示 4：Recall@20 と標準の MAP で数え直す）.

指標の定義（報告書 §3.5）:
  Recall@20 = |R_q ∩ Top_20| / K
  AP（標準）= (1/K) Σ_{正解式 e} P(e)、P(e) は e の順位までの正解式の割合。並べた中にない正解式は 0。
"""
import pytest

from kato1014_common import ap_std, paired, recall_at, seed_summary


def test_recall_at_20_counts_ranks_up_to_20():
    assert recall_at([1, 20, 21], 20) == pytest.approx(2 / 3)
    assert recall_at([10_000], 20) == 0.0


def test_ap_std_all_found():
    # 1 位と 3 位：(1/1 + 2/3) / 2
    assert ap_std([3, 1], n_list=50) == pytest.approx((1 + 2 / 3) / 2)


def test_ap_std_missing_counts_zero_and_divides_by_K():
    # 候補 50 件の外（印 10,000）は 0、割る数は K = 2
    assert ap_std([2, 10_000], n_list=50) == pytest.approx(0.5 / 2)


def test_ap_std_full_db_ranks_above_10000_are_real():
    # 全式を並べる baseline では 10,000 位を超える順位も本当の順位
    assert ap_std([5, 10_500], n_list=11_146) == pytest.approx((1 / 5 + 2 / 10_500) / 2)


def test_seed_summary_mean_and_sample_sd():
    s = seed_summary({1: 0.5, 2: 0.7})
    assert s["mean"] == pytest.approx(0.6)
    assert s["std"] == pytest.approx(0.1414213562)  # 不偏標準偏差（ddof = 1）


def test_paired_all_wins_gives_wilcoxon_min_p():
    a = {s: 0.80 + 0.01 * i for i, s in enumerate(range(10))}
    b = {s: 0.70 + 0.011 * i for i, s in enumerate(range(10))}
    r = paired(a, b)
    assert r["n_seeds"] == 10 and r["wins"] == 10
    assert r["p_wilcoxon"] == pytest.approx(2 / 1024)
    assert r["mean_delta"] == pytest.approx(sum(a[s] - b[s] for s in a) / 10)


from kato1014_common import ap_at


def test_map_at_20_truncates_at_rank_20_and_divides_by_K():
    # 正解式 3 本：1 位、5 位、25 位 → 25 位は 0、K = 3 で割る
    assert ap_at([1, 5, 25], 20) == pytest.approx((1 / 1 + 2 / 5) / 3)
    # 20 位までに全部あれば標準の AP と同じ
    assert ap_at([2, 3], 20) == pytest.approx(ap_std([2, 3], n_list=50))


def test_r_precision_is_the_old_recall_at_k():
    """2026-10-20：旧 Recall@K（K = 正解式の数）を R-Precision と呼ぶ（加藤先生のコメント）."""
    from kato1014_common import METRIC_FNS
    rec = {"ranks": [1, 3, 4]}
    assert METRIC_FNS["R-Precision"](rec, 50) == METRIC_FNS["Recall@K"](rec, 50) == 2 / 3
