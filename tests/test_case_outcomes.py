import numpy as np
from analyze_case_outcomes import (
    rank_candidates, failure_split, topic_rescue_fraction, within_case_corr,
)


def test_rank_candidates_orders_by_score_desc():
    cands = [10, 20, 30]
    scores = np.array([0.1, 0.9, 0.5])
    assert rank_candidates(cands, scores) == [20, 30, 10]


def test_failure_split_separates_stage1_and_reranker():
    # 正解 {1,2,3}, 候補に 1,2 のみ、上位K=3 に 1 だけ入った
    ranked = [1, 9, 8, 2]
    out = failure_split(ranked, cands=[1, 2, 8, 9], corr={1, 2, 3}, k_c=3)
    assert out["hit"] == {1}
    assert out["reranker_miss"] == {2}   # 候補にいたが上位K外
    assert out["stage1_miss"] == {3}     # 候補に入らなかった
    assert abs(out["recall"] - 1 / 3) < 1e-9


def test_topic_rescue_fraction_counts_pairs():
    # 見逃し正解 m の特徴値 > 誤選択 w のペア割合
    vals = {1: 0.9, 2: 0.1, 8: 0.5, 9: 0.5}
    # missed={1,2}, wrong_above={8,9} → 比較4ペア中、1>8, 1>9 の2つだけ
    frac = topic_rescue_fraction(missed={1, 2}, wrong_above={8, 9}, value_of=vals)
    assert abs(frac - 0.5) < 1e-9


def test_topic_rescue_fraction_empty_is_nan():
    assert np.isnan(topic_rescue_fraction(set(), {1}, {1: 0.0}))


def test_within_case_corr_sign():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert within_case_corr(x, -x) < -0.99
    assert within_case_corr(x, x) > 0.99
    assert np.isnan(within_case_corr(x, np.full(4, 2.0)))  # 定数列は nan
